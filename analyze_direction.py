import argparse
import logging
import os
from typing import Dict, List, Tuple
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle


def load_state_dict(adapter_dir: str) -> Dict:
    """加载适配器状态字典"""
    path = os.path.join(adapter_dir, 'adapter_model.bin')
    return torch.load(path, map_location="cpu", weights_only=True)


def parse_directions(sd: Dict, task_id: int = 0) -> List[Dict]:
    """提取 historical_directions 条目"""
    rows: List[Dict] = []
    for k, v in sd.items():
        if "historical_directions" not in k:
            continue
        if 'num' in k:
            continue
        print(f"Found direction key: {k}")
        print(f"Value type: {type(v)}, shape: {getattr(v, 'shape', 'N/A')}")
        # 解析键名：layer.historical_direction.dir_X
        parts = k.split("historical_directions.")

        if len(parts) < 2:
            continue
            
        dir_key = parts[1]  # dir_0, dir_1, etc.
        layer = parts[0].rstrip(".")  # 层名
        
        # 提取方向编号
        try:
            dir_num = int(dir_key.split("_")[-1])
        except:
            continue
            
        # 只处理当前任务及之前的方向
        if dir_num > task_id:
            continue
            
        # 转换张量为numpy数组
        try:
            if hasattr(v, 'numpy'):
                direction_vec = v.cpu().numpy()
            else:
                direction_vec = np.array(v)
        except:
            continue
            
        rows.append({
            "task": task_id,
            "layer": layer,
            "direction": dir_key,
            "direction_num": dir_num,
            "vector": direction_vec,
            "norm": np.linalg.norm(direction_vec),
            "dim": len(direction_vec)
        })
    
    return rows


def auto_discover(root: str) -> List[str]:
    """自动发现适配器目录"""
    dirs = [f.name for f in os.scandir(root) if f.is_dir()]
    dirs = [os.path.join(root, d, 'adapter') for d in sorted(dirs)]
    return [d for d in dirs if os.path.exists(os.path.join(d, 'adapter_model.bin'))]


def compute_direction_similarities(all_rows: List[Dict]) -> pd.DataFrame:
    """计算不同任务间方向向量的相似度"""
    similarity_data = []
    
    # 按层分组
    layer_groups = {}
    for row in all_rows:
        layer = row['layer']
        if layer not in layer_groups:
            layer_groups[layer] = {}
        
        key = (row['task'], row['direction_num'])
        layer_groups[layer][key] = row['vector']
    
    # 计算每层内不同任务方向的相似度
    for layer, directions in layer_groups.items():
        keys = list(directions.keys())
        for i, key1 in enumerate(keys):
            for key2 in keys[i:]:
                task1, dir1 = key1
                task2, dir2 = key2
                
                vec1 = directions[key1]
                vec2 = directions[key2]
                
                # 计算余弦相似度
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                
                similarity_data.append({
                    'layer': layer,
                    'task1': task1,
                    'dir1': dir1,
                    'task2': task2,
                    'dir2': dir2,
                    'cosine_similarity': cos_sim,
                    'same_direction': dir1 == dir2,
                    'same_task': task1 == task2
                })
    
    return pd.DataFrame(similarity_data)


def plot_direction_norms(df: pd.DataFrame, output_dir: str):
    """绘制方向向量模长的变化"""
    plt.figure(figsize=(12, 8))
    
    # 按层分组绘制
    layers = sorted(df['layer'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(layers)))
    
    for i, layer in enumerate(layers):
        layer_data = df[df['layer'] == layer]
        
        for dir_num in sorted(layer_data['direction_num'].unique()):
            dir_data = layer_data[layer_data['direction_num'] == dir_num]
            dir_data = dir_data.sort_values('task')
            
            plt.plot(dir_data['task'], dir_data['norm'], 
                    marker='o', linewidth=2, markersize=6,
                    color=colors[i], alpha=0.7,
                    label=f'{layer} - dir_{dir_num}' if len(layers) <= 3 else None)
    
    plt.xlabel('Task ID', fontsize=12)
    plt.ylabel('Direction Vector Norm', fontsize=12)
    plt.title('Historical Direction Vector Norms Over Tasks', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    if len(layers) <= 3:
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'direction_norms.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_similarity_heatmap(sim_df: pd.DataFrame, output_dir: str):
    """绘制方向相似度热图"""
    # 筛选相同方向但不同任务的相似度
    same_dir_diff_task = sim_df[(sim_df['same_direction'] == True) & (sim_df['same_task'] == False)]
    
    if same_dir_diff_task.empty:
        logging.warning("No cross-task same-direction similarities found")
        return
    
    # 创建透视表
    pivot_data = same_dir_diff_task.pivot_table(
        values='cosine_similarity', 
        index=['layer', 'dir1'], 
        columns=['task1', 'task2'],
        aggfunc='mean'
    )
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_data, annot=True, cmap='coolwarm', center=0, 
                fmt='.3f', cbar_kws={'label': 'Cosine Similarity'})
    plt.title('Cross-Task Direction Similarity (Same Direction Index)', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'similarity_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_direction_evolution(df: pd.DataFrame, output_dir: str, max_layers: int = 3):
    """绘制方向向量的演化过程"""
    layers = sorted(df['layer'].unique())[:max_layers]
    
    fig, axes = plt.subplots(len(layers), 1, figsize=(15, 5*len(layers)))
    if len(layers) == 1:
        axes = [axes]
    
    for idx, layer in enumerate(layers):
        layer_data = df[df['layer'] == layer].copy()
        layer_data = layer_data.sort_values(['direction_num', 'task'])
        
        directions = sorted(layer_data['direction_num'].unique())
        colors = plt.cm.Set1(np.linspace(0, 1, len(directions)))
        
        for i, dir_num in enumerate(directions):
            dir_data = layer_data[layer_data['direction_num'] == dir_num]
            
            # 绘制向量的前几个维度作为示例
            max_dims = min(5, dir_data.iloc[0]['dim'])
            
            for dim in range(max_dims):
                values = [vec[dim] for vec in dir_data['vector']]
                axes[idx].plot(dir_data['task'], values, 
                             color=colors[i], alpha=0.6, linewidth=1,
                             label=f'dir_{dir_num}_dim_{dim}' if dim < 3 else None)
        
        axes[idx].set_title(f'Direction Evolution - {layer}', fontsize=12)
        axes[idx].set_xlabel('Task ID')
        axes[idx].set_ylabel('Vector Component Value')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'direction_evolution.png'), dpi=150, bbox_inches='tight')
    plt.close()


def plot_direction_diversity(sim_df: pd.DataFrame, output_dir: str):
    """绘制方向多样性分析"""
    # 计算每个任务内不同方向间的平均相似度
    within_task_sim = sim_df[
        (sim_df['same_task'] == True) & (sim_df['same_direction'] == False)
    ].groupby(['task1', 'layer'])['cosine_similarity'].mean().reset_index()
    
    plt.figure(figsize=(10, 6))
    
    for layer in within_task_sim['layer'].unique():
        layer_data = within_task_sim[within_task_sim['layer'] == layer]
        plt.plot(layer_data['task1'], layer_data['cosine_similarity'], 
                marker='o', linewidth=2, markersize=6, label=layer)
    
    plt.xlabel('Task ID', fontsize=12)
    plt.ylabel('Average Inter-Direction Similarity', fontsize=12)
    plt.title('Direction Diversity Within Each Task', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'direction_diversity.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze historical_direction changes in SD-LoRA")
    parser.add_argument("--root", default="exp/sdlora/order_1/outputs", 
                       help="Root directory to discover adapter directories")
    parser.add_argument("--adapter-dirs", nargs="*", default=[], 
                       help="Specific adapter directories to analyze")
    parser.add_argument("--output-dir", default="analyze/directions", 
                       help="Output directory for plots and analysis")
    parser.add_argument("--max-layers", type=int, default=3, 
                       help="Maximum number of layers to analyze in detail")
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
    log = logging.getLogger("analyze_directions")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 发现或使用指定的适配器目录
    adapter_dirs = args.adapter_dirs if args.adapter_dirs else auto_discover(args.root)
    if not adapter_dirs:
        log.error("No adapter directories found")
        return
    
    log.info(f"Found {len(adapter_dirs)} adapter directories")
    
    # 解析所有方向数据
    all_rows = []
    for idx, adapter_dir in enumerate(adapter_dirs):
        try:
            sd = load_state_dict(adapter_dir)
            rows = parse_directions(sd, task_id=idx)
            all_rows.extend(rows)
            log.info(f"Processed task {idx}: found {len(rows)} direction vectors")
        except Exception as e:
            log.warning(f"Failed to process {adapter_dir}: {e}")
    
    if not all_rows:
        log.error("No direction data found")
        return
    
    # 转换为DataFrame
    df = pd.DataFrame(all_rows)
    log.info(f"Loaded {len(df)} direction vectors across {df['task'].nunique()} tasks")
    
    # 保存原始数据
    df_save = df.drop('vector', axis=1)  # 移除向量列以便保存
    df_save.to_csv(os.path.join(args.output_dir, 'direction_data.csv'), index=False)
    
    # 计算相似度
    log.info("Computing direction similarities...")
    sim_df = compute_direction_similarities(all_rows)
    sim_df.to_csv(os.path.join(args.output_dir, 'direction_similarities.csv'), index=False)
    
    # 生成可视化
    log.info("Generating visualizations...")
    
    plot_direction_norms(df, args.output_dir)
    log.info("Generated direction norms plot")
    
    plot_similarity_heatmap(sim_df, args.output_dir)
    log.info("Generated similarity heatmap")
    
    plot_direction_evolution(df, args.output_dir, args.max_layers)
    log.info("Generated direction evolution plot")
    
    plot_direction_diversity(sim_df, args.output_dir)
    log.info("Generated direction diversity plot")
    
    # 输出统计信息
    log.info("\n=== Direction Analysis Summary ===")
    log.info(f"Total tasks analyzed: {df['task'].nunique()}")
    log.info(f"Total layers: {df['layer'].nunique()}")
    log.info(f"Direction vector dimensions: {df['dim'].iloc[0] if len(df) > 0 else 'N/A'}")
    log.info(f"Average direction norm: {df['norm'].mean():.4f}")
    log.info(f"Direction norm std: {df['norm'].std():.4f}")
    
    if not sim_df.empty:
        same_dir_cross_task = sim_df[(sim_df['same_direction'] == True) & (sim_df['same_task'] == False)]
        if not same_dir_cross_task.empty:
            log.info(f"Average cross-task same-direction similarity: {same_dir_cross_task['cosine_similarity'].mean():.4f}")
    
    log.info(f"All results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()