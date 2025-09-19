import torch
import argparse
import glob
from pathlib import Path
import os
import numpy as np
import json

from sklearn.cluster import SpectralClustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from typing import List, Optional

def calculate_lora_weight_sum(state_dict, task_id):
    """
    计算每个task的LoRA权重和
    Args:
        state_dict: 模型状态字典
        task_id: 任务ID
    Returns:
        dict: 每层的权重和统计信息
    """
    results = {}
    
    # 提取历史方向和缩放参数
    historical_directions = {}
    historical_scalings = {}
    
    # 解析state_dict中的历史方向和缩放参数
    for key, value in state_dict.items():
        if 'num' in key:
            task_id = value.item() if hasattr(value, 'item') else int(value)
        if 'historical_directions' in key:
            # 解析层名和方向索引
            parts = key.split('.')
            layer_path = []
            direction_info = []
            
            for i, part in enumerate(parts):
                if part == 'historical_directions':
                    layer_path = '.'.join(parts[:i])
                    direction_info = parts[i+1:]
                    break
            
            if len(direction_info) >= 3:  # adapter.dir_X.A/B.weight
                dir_key = direction_info[0]  # dir_0, dir_1, etc.
                component = direction_info[1]  # A or B
                
                if layer_path not in historical_directions:
                    historical_directions[layer_path] = {}
                if dir_key not in historical_directions[layer_path]:
                    historical_directions[layer_path][dir_key] = {}
                
                historical_directions[layer_path][dir_key][component] = value
        
        elif 'shared_historical_scalings' in key:
            # 解析缩放参数
            parts = key.split('.')
            layer_path = []
            scaling_info = []
            
            for i, part in enumerate(parts):
                if part == 'shared_historical_scalings':
                    layer_path = '.'.join(parts[:i])
                    scaling_info = parts[i+1:]
                    break
            
            if scaling_info:  # dir_X
                dir_key = scaling_info[0]
                historical_scalings[dir_key] = value
    
    # 计算每层的权重和
    weight = {}# 每层合并后的lora权重
    for layer_path in historical_directions:
        layer_results = {}
        
        total_weight_sum = 0
        direction_count = 0
        for dir_key in historical_directions[layer_path]:
            direction_data = historical_directions[layer_path][dir_key]
            
            if 'A' in direction_data and 'B' in direction_data:
                A_weight = direction_data['A'].weight if hasattr(direction_data['A'], 'weight') else direction_data['A']
                B_weight = direction_data['B'].weight if hasattr(direction_data['B'], 'weight') else direction_data['B']
                
                # 获取对应的缩放参数
                scaling = 1.0
                if dir_key in historical_scalings:
                    scaling_param = historical_scalings[dir_key]
                    scaling = scaling_param.item() if hasattr(scaling_param, 'item') else float(scaling_param)
                
                # 计算归一化因子
                norm_A = torch.norm(A_weight)
                norm_B = torch.norm(B_weight)
                norm_factor = norm_A * norm_B + 1e-8
                
                # 计算权重矩阵: B @ A
                weight_matrix = torch.mm(B_weight, A_weight)
                
                # 应用缩放和归一化: scaling * weight_matrix / norm_factor
                if task_id!=dir_key.split('_')[-1]:
                    scaled_weight = scaling * weight_matrix / norm_factor
                else:
                    scaled_weight = weight_matrix * scaling

                weight[layer_path] = scaled_weight if layer_path not in weight else weight[layer_path] + scaled_weight
                
                # 计算权重和
                weight_sum = torch.sum(torch.abs(scaled_weight)).item()
                total_weight_sum += weight_sum
                direction_count += 1
                
                # layer_results[dir_key] = {
                #     'weight_sum': weight_sum,
                #     'scaling': scaling,
                #     'norm_factor': norm_factor.item(),
                #     'A_shape': list(A_weight.shape),
                #     'B_shape': list(B_weight.shape)
                # }
        
        layer_results['total_weight_sum'] = total_weight_sum
        layer_results['direction_count'] = direction_count
        # layer_results['avg_weight_sum'] = total_weight_sum / direction_count if direction_count > 0 else 0
        
        results[layer_path] = layer_results
    
    return results,weight


def _flatten_weight_dict_to_vector(weight_dict: dict) -> np.ndarray:
    """将每层的 2D 权重矩阵按层名排序后展平并串联为一个长向量"""
    if not weight_dict:
        return None
    # 固定层顺序，保证不同任务拼接一致
    layers = sorted(weight_dict.keys())
    flat_list = []
    for layer in layers:
        w = weight_dict[layer]
        if isinstance(w, torch.Tensor):
            w_np = w.detach().cpu().float().numpy()
        else:
            w_np = np.array(w, dtype=np.float32)
        flat_list.append(w_np.reshape(-1))
    vec = np.concatenate(flat_list, axis=0).astype(np.float32)
    return vec


def _infer_dataset_name(task_dir_name: str) -> str:
    """从目录名中解析数据集名称（尽量鲁棒）"""
    name = task_dir_name.lower()
    # 常见格式: '0-dbpeida', '1-amazon', etc.
    if '-' in name:
        name = name.split('-', 1)[1]
    # 去除可能的后缀
    name = name.replace('_', '').replace(' ', '')
    # 与已知 family 做匹配
    families = ["dbpedia", "amazon", "agnews", "yahoo"]
    for fam in families:
        if fam in name:
            return fam
    return name


def _tsne_2d(vectors: np.ndarray, metric: str = 'cosine', random_state: int = 42) -> np.ndarray:
    n = vectors.shape[0]
    # 经验设定: perplexity < n_samples/3, 且不小于 5
    perplexity = max(5, min(30, max(5, n // 3)))
    if metric == 'cosine':
        # 使用余弦距离矩阵：1 - 余弦相似度
        sim = cosine_similarity(vectors)
        dist = 1.0 - np.clip(sim, -1.0, 1.0)
        tsne = TSNE(n_components=2, metric='precomputed', random_state=random_state, perplexity=perplexity, init='pca')
        return tsne.fit_transform(dist)
    else:
        tsne = TSNE(n_components=2, metric='euclidean', random_state=random_state, perplexity=perplexity, init='pca')
        return tsne.fit_transform(vectors)


def _spectral_cluster_cosine(vectors: np.ndarray, n_clusters: int, random_state: int = 42) -> np.ndarray:
    # 先做 L2 归一化，再用点积得到 cos 相似
    eps = 1e-12
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + eps
    normed = vectors / norms
    affinity = np.clip(normed @ normed.T, -1.0, 1.0)
    # 转为非负，供 precomputed affinity 使用
    affinity_pos = (affinity + 1.0) / 2.0
    sc = SpectralClustering(n_clusters=n_clusters, affinity='precomputed', assign_labels='kmeans', random_state=random_state)
    labels = sc.fit_predict(affinity_pos)
    return labels


def _plot_tsne(points_2d: np.ndarray, color_labels: List, title: str, out_path: Path, annotations: Optional[List] = None):
    plt.figure(figsize=(7, 6))
    # 统一颜色映射
    uniq = sorted(list(set(color_labels)))
    cmap = plt.get_cmap('tab10')
    color_map = {lab: cmap(i % 10) for i, lab in enumerate(uniq)}
    for i, (x, y) in enumerate(points_2d):
        c = color_map[color_labels[i]]
        plt.scatter(x, y, c=[c], s=30, alpha=0.8, edgecolors='none')
        if annotations is not None:
            plt.text(x + 0.5, y + 0.5, annotations[i], fontsize=7, alpha=0.8)
    plt.title(title)
    plt.axis('off')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path.as_posix(), dpi=200)
    plt.close()

def print_weight_statistics(results, task_name):
    """打印权重统计信息"""
    print(f"\n  === {task_name} 权重统计 ===")
    
    total_sum = 0
    total_layers = 0
    
    for layer_path, layer_data in results.items():
        if 'total_weight_sum' in layer_data:
            layer_sum = layer_data['total_weight_sum']
            direction_count = layer_data['direction_count']
            # avg_sum = layer_data['avg_weight_sum']
            
            print(f"    {layer_path}:")
            print(f"      总权重和: {layer_sum:.6f}")
            print(f"      方向数量: {direction_count}")
            # print(f"      平均权重和: {avg_sum:.6f}")
            
            total_sum += layer_sum
            total_layers += 1
            
            # 打印每个方向的详细信息（可选）
            # for dir_key, dir_data in layer_data.items():
            #     if isinstance(dir_data, dict) and 'weight_sum' in dir_data:
            #         print(f"        {dir_key}: 权重和={dir_data['weight_sum']:.6f}, "
            #                 f"缩放={dir_data['scaling']:.4f}, 归一化={dir_data['norm_factor']:.6f}")
    
    print(f"    总体统计:")
    print(f"      所有层总权重和: {total_sum:.6f}")
    print(f"      处理层数: {total_layers}")
    print(f"      平均每层权重和: {total_sum/total_layers:.6f}" if total_layers > 0 else "      平均每层权重和: 0")

import debugpy
# debugpy.listen(5678)
# print("Waiting for debugger attach...")
# debugpy.wait_for_client()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="/home/yongxi/work/O-LoRA/exp/sdlora/", 
                        help="Base path to search for adapter models")
    parser.add_argument('--output', type=str, default="/home/yongxi/work/O-LoRA/analyze/lora_weights_analysis.txt",
                        help="Output file to save analysis results (text, optional)")
    parser.add_argument('--figdir', type=str, default="/home/yongxi/work/O-LoRA/analyze/figs",
                        help="Directory to save t-SNE plots and clustering results")
    args = parser.parse_args()
    
    base_path = Path(args.path)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig_dir = Path(args.figdir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = {}
    # 收集所有任务的向量与元信息
    all_task_entries = []  # {order, task_name, dataset, vector}
    per_order_entries = {}
    
    # 遍历每个order目录
    for order_dir in sorted(base_path.glob('order_*')):
        order_name = order_dir.name
        outputs_dir = order_dir / 'outputs'
        if not outputs_dir.exists():
            continue
        
        print(f"\n处理 {order_name}...")
        order_results = {}
        
        # 按任务顺序处理
        task_dirs = sorted([d for d in outputs_dir.glob('*') if d.is_dir()], 
                          key=lambda x: int(x.name.split('-')[0]))
        
        for task_idx, task_dir in enumerate(task_dirs):
            task_name = task_dir.name
            adapter_path = task_dir / 'adapter' / 'adapter_model.bin'
            
            if not adapter_path.exists():
                print(f"  跳过 {task_name}，未找到 adapter_model.bin")
                continue
            
            print(f"  加载 {task_name} 的 adapter_model.bin...")
            
            try:
                state_dict = torch.load(adapter_path, map_location='cpu', weights_only=True)
                
                # 计算权重和
                weight_results, weight = calculate_lora_weight_sum(state_dict, task_idx)
                
                # # 打印统计信息
                # print_weight_statistics(weight_results, task_name)
                
                # order_results[task_name] = weight_results
                vec = _flatten_weight_dict_to_vector(weight)
                if vec is None or vec.size == 0:
                    print(f"  跳过 {task_name}，未提取到有效的 LoRA 权重向量")
                    continue
                dataset = _infer_dataset_name(task_name)
                entry = {
                    'order': order_name, #order_1
                    'task_name': task_name,# 1-dbpedia
                    'dataset': dataset,# dbpedia
                    'vector': vec, # weight vector
                }
                all_task_entries.append(entry)
                per_order_entries.setdefault(order_name, []).append(entry)
                
            except Exception as e:
                print(f"  处理 {task_name} 时出错: {e}")
                continue
        
        all_results[order_name] = order_results
    
    # 若未收集到任何任务，直接结束
    if not all_task_entries:
        print("未收集到任何任务向量，结束。")
        return

    # 组合所有任务进行一次总体聚类与可视化
    print("\n构建总体任务矩阵并使用余弦相似度进行聚类与 t-SNE...")
    all_vectors = np.stack([e['vector'] for e in all_task_entries], axis=0)
    all_datasets = [e['dataset'] for e in all_task_entries]
    all_annotations = [f"{e['order']}\n{e['task_name']}" for e in all_task_entries]
    n_clusters_overall = len(set(all_datasets))
    # t-SNE (cosine metric)
    all_tsne = _tsne_2d(all_vectors, metric='cosine', random_state=42)
    _plot_tsne(all_tsne, all_datasets, f"All Orders t-SNE by Dataset (cosine)", fig_dir / "all_orders_tsne_by_dataset.png", annotations=all_annotations)
    # 谱聚类（cosine affinity）
    overall_clusters = _spectral_cluster_cosine(all_vectors, n_clusters=n_clusters_overall, random_state=42)
    # 以聚类标签着色的 t-SNE
    _plot_tsne(all_tsne, [str(c) for c in overall_clusters], f"All Orders t-SNE by Cluster (cosine)", fig_dir / "all_orders_tsne_by_cluster.png", annotations=all_annotations)

    # 保存总体聚类结果
    overall_csv = fig_dir / 'overall_cluster_assignments.csv'
    with open(overall_csv, 'w', encoding='utf-8') as f:
        f.write('order,task_name,dataset,cluster,tsne_x,tsne_y\n')
        for i, (e, c) in enumerate(zip(all_task_entries, overall_clusters)):
            x, y = all_tsne[i]
            f.write(f"{e['order']},{e['task_name']},{e['dataset']},{int(c)},{x:.6f},{y:.6f}\n")
    print(f"总体 t-SNE 图已保存: {(fig_dir / 'all_orders_tsne_by_dataset.png').as_posix()} 和 {(fig_dir / 'all_orders_tsne_by_cluster.png').as_posix()}")
    print(f"总体聚类结果 CSV 已保存: {overall_csv.as_posix()}")

    # 每个 order 单独聚类与可视化
    for order_name, entries in per_order_entries.items():
        if len(entries) < 2:
            continue
        print(f"处理 {order_name} 的单独聚类与可视化...")
        vectors = np.stack([e['vector'] for e in entries], axis=0)
        datasets = [e['dataset'] for e in entries]
        annotations = [e['task_name'] for e in entries]
        n_clusters = len(set(datasets))
        tsne_points = _tsne_2d(vectors, metric='cosine', random_state=42)
        _plot_tsne(tsne_points, datasets, f"{order_name} t-SNE by Dataset (cosine)", fig_dir / f"{order_name}_tsne_by_dataset.png", annotations=annotations)
        clusters = _spectral_cluster_cosine(vectors, n_clusters=n_clusters, random_state=42)
        _plot_tsne(tsne_points, [str(c) for c in clusters], f"{order_name} t-SNE by Cluster (cosine)", fig_dir / f"{order_name}_tsne_by_cluster.png", annotations=annotations)
        csv_path = fig_dir / f"{order_name}_cluster_assignments.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('task_name,dataset,cluster,tsne_x,tsne_y\n')
            for i, (e, c) in enumerate(zip(entries, clusters)):
                x, y = tsne_points[i]
                f.write(f"{e['task_name']},{e['dataset']},{int(c)},{x:.6f},{y:.6f}\n")
        print(f"  {order_name} t-SNE 图已保存: {(fig_dir / f'{order_name}_tsne_by_dataset.png').as_posix()} 和 {(fig_dir / f'{order_name}_tsne_by_cluster.png').as_posix()}")
        print(f"  {order_name} 聚类结果 CSV 已保存: {csv_path.as_posix()}")

    # 保存结果到文件
    # with open(args.output, 'w', encoding='utf-8') as f:
    #     f.write("LoRA权重和分析结果\n")
    #     f.write("=" * 50 + "\n\n")
        
    #     for order_name, order_data in all_results.items():
    #         f.write(f"{order_name}:\n")
    #         f.write("-" * 30 + "\n")
            
    #         for task_name, task_data in order_data.items():
    #             f.write(f"  {task_name}:\n")
                
    #             total_sum = 0 
    #             total_layers = 0
                
    #             for layer_path, layer_data in task_data.items():
    #                 for adapter_name, adapter_data in layer_data.items():
    #                     if 'total_weight_sum' in adapter_data:
    #                         layer_sum = adapter_data['total_weight_sum']
    #                         direction_count = adapter_data['direction_count']
                            
    #                         f.write(f"    {layer_path}: 权重和={layer_sum:.6f}, 方向数={direction_count}\n")
    #                         total_sum += layer_sum
    #                         total_layers += 1
                
    #             f.write(f"    总权重和: {total_sum:.6f}\n")
    #             f.write(f"    平均每层: {total_sum/total_layers:.6f}\n" if total_layers > 0 else "    平均每层: 0\n")
    #             f.write("\n")
            
    #         f.write("\n")
    
    # print(f"\n分析结果已保存到: {args.output}")

if __name__ == "__main__":
    main()