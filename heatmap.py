
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
                # print(f"Layer: {layer_path}, Direction: {dir_key}, Scaling: {scaling:.4f}, Norm_A: {norm_A:.4f}, Norm_B: {norm_B:.4f}, Norm_Factor: {norm_factor:.4f}")
                # 计算权重矩阵: B @ A
                weight_matrix = torch.mm(B_weight, A_weight)
                
                # 应用缩放和归一化: scaling * weight_matrix / norm_factor
                if task_id!=dir_key.split('_')[-1]:
                    scaled_weight = scaling * weight_matrix 
                else:
                    scaled_weight = weight_matrix * scaling / norm_factor

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

def calculate_weight_olora(state_dict) -> dict:
    """基于 O-LoRA：直接使用每层的 lora_B.weight @ lora_A.weight 作为该层的权重矩阵。
    从 state_dict 中解析出所有层的 lora_A / lora_B，并按层合并为字典。
    返回: weight_dict[layer_path] = Tensor(out_features, in_features)
    """
    A_map = {}
    B_map = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            continue
        if key.endswith('.weight'):
            parts = key.split('.')
            # 寻找 lora_A / lora_B 的位置
            if 'lora_A' in parts:
                i = parts.index('lora_A')
                layer_path = '.'.join(parts[:i])
                # 期望形式: <layer_path>.lora_A.<adapter_name>.weight 或 <layer_path>.lora_A.weight
                A_map[layer_path] = value
            elif 'lora_B' in parts:
                i = parts.index('lora_B')
                layer_path = '.'.join(parts[:i])
                B_map[layer_path] = value

    weight = {}
    for layer_path in sorted(set(list(A_map.keys()) + list(B_map.keys()))):
        if layer_path in A_map and layer_path in B_map:
            A_w = A_map[layer_path]
            B_w = B_map[layer_path]
            if hasattr(A_w, 'weight'):
                A_w = A_w.weight
            if hasattr(B_w, 'weight'):
                B_w = B_w.weight
            # LoRA 有效权重为 B @ A
            try:
                W = torch.matmul(B_w, A_w)
            except Exception:
                # 若维度不匹配，尝试转置其中之一（保底处理）
                if B_w.shape[0] == A_w.shape[0]:
                    W = torch.matmul(B_w.T, A_w)
                else:
                    W = torch.matmul(B_w, A_w.T)
            # scaling = 32/8 =4
            W = W * 4
            weight[layer_path] = W
    return weight

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



parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, default="/home/yongxi/work/O-LoRA/exp/sdlora/", help="Base path to search for adapter models")
parser.add_argument('--output', type=str, default="/home/yongxi/work/O-LoRA/analyze/",help="Output file to save analysis results (text, optional)")

parser.add_argument('--type', type=str, default='sdlora', choices=['sdlora', 'olora'],help='Select extraction mode: sdlora uses historical_directions; olora uses lora_A/B directly')
args = parser.parse_args()

base_path = Path(args.path)
all_task_entries = []  # 存储所有任务的向量信息

# 遍历每个order目录
for order_dir in sorted(base_path.glob('order_*')):
    order_name = order_dir.name
    outputs_dir = order_dir / 'outputs'
    if not outputs_dir.exists():
        continue
    print(f"\n处理 {order_name}...")
    
    # 按任务顺序处理
    task_dirs = sorted([d for d in outputs_dir.glob('*') if d.is_dir()], key=lambda x: int(x.name.split('-')[0]))
    for task_idx, task_dir in enumerate(task_dirs):
        task_name = task_dir.name
        adapter_path = task_dir / 'adapter' / 'adapter_model.bin'
        if not adapter_path.exists():
            print(f"  跳过 {task_name}，未找到 adapter_model.bin")
            continue
        print(f"  加载 {task_name} 的 adapter_model.bin...")
        
        try:
            state_dict = torch.load(adapter_path, map_location='cpu', weights_only=True)
            
            # 计算权重（根据 type 切换）
            if args.type.lower() == 'olora':
                weight = calculate_weight_olora(state_dict)
            else:
                weight_results, weight = calculate_lora_weight_sum(state_dict, task_idx)
            
            vec = _flatten_weight_dict_to_vector(weight)
            if vec is None or vec.size == 0:
                print(f"  跳过 {task_name}，未提取到有效的 LoRA 权重向量")
                continue
            
            entry = {
                'order': order_name,
                'task_name': task_name,
                'task_idx': task_idx,
                'vector': vec,
            }
            all_task_entries.append(entry)
            
        except Exception as e:
            print(f"  处理 {task_name} 时出错: {e}")
            continue

# 按 order 分组并计算热力图
orders = {}
for entry in all_task_entries:
    order_name = entry['order']
    if order_name not in orders:
        orders[order_name] = []
    orders[order_name].append(entry)

# 为每个 order 生成热力图
for order_name, entries in orders.items():
    if len(entries) < 2:
        continue
    
    print(f"\n生成 {order_name} 的热力图...")
    
    # 构建向量矩阵
    vectors = np.stack([e['vector'] for e in entries], axis=0)  # shape: (n_tasks, dim)
    task_names = [e['task_name'] for e in entries]
    n_tasks = len(entries)
    
    # 计算相似度矩阵: 2||w_i - w_j|| / (||w_i|| + ||w_j||)
    similarity_matrix = np.zeros((n_tasks, n_tasks))
    
    for i in range(n_tasks):
        for j in range(n_tasks):
            if i == j:
                similarity_matrix[i, j] = 0.0  # 自己与自己的距离为0
            else:
                w_i = vectors[i]
                w_j = vectors[j]
                
                # 计算 L2 范数
                norm_i = np.linalg.norm(w_i)
                norm_j = np.linalg.norm(w_j)
                
                # 计算差值的范数
                diff_norm = np.linalg.norm(w_i - w_j)
                
                # 计算相似度: 2||w_i - w_j|| / (||w_i|| + ||w_j||)
                if norm_i + norm_j > 1e-8:
                    similarity_matrix[i, j] = 2 * diff_norm / (norm_i + norm_j)
                else:
                    similarity_matrix[i, j] = 0.0
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    
    # 创建热力图
    im = plt.imshow(similarity_matrix, cmap='Blues', aspect='equal')
    
    # 设置坐标轴标签
    plt.xticks(range(n_tasks), [f"Task {i+1}" for i in range(n_tasks)])
    plt.yticks(range(n_tasks), [f"Task {i+1}" for i in range(n_tasks)])
    plt.xlabel('Task Index i')
    plt.ylabel('Task Index j')
    
    # 设置标题
    plt.title(f'Task Similarity Heatmap - {order_name}\n' + 
              r'$2 \times ||W_i - W_j||_F / (||W_i||_F + ||W_j||_F)$')
    
    # 添加颜色条
    cbar = plt.colorbar(im,ticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_label('Similarity Score')
    
    # 在每个单元格中添加数值
    for i in range(n_tasks):
        for j in range(n_tasks):
            text = plt.text(j, i, f'{similarity_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black" if similarity_matrix[i, j] < 0.5 else "white")
    
    plt.tight_layout()
    
    # 保存图片
    output_dir = Path(args.output)
    # output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{args.type}_{order_name}_similarity_heatmap.png"
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"  热力图已保存: {output_path}")
    
    # 打印相似度矩阵
    print(f"  相似度矩阵:")
    print(f"    任务名称: {task_names}")
    for i in range(n_tasks):
        row_str = "    " + " ".join([f"{similarity_matrix[i, j]:.3f}" for j in range(n_tasks)])
        print(row_str)

print("\n所有热力图生成完成！")