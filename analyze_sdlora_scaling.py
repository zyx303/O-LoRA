#!/usr/bin/env python3
"""
分析SDLoRA中的scaling机制和历史方向存储
"""

import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns

def load_adapter_weights(adapter_path):
    """加载adapter权重"""
    weights_path = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.exists(weights_path):
        return torch.load(weights_path, map_location='cpu')
    return None

def analyze_scaling_mechanisms():
    """分析SDLoRA的scaling机制"""
    base_path = "/home/yongxi/work/O-LoRA/logs_and_outputs/sdlora/order_1/outputs"
    
    # 任务列表
    tasks = ['1-dbpedia', '2-amazon', '3-yahoo', '4-agnews']
    
    print("=== SDLoRA Scaling Mechanisms Analysis ===\n")
    
    for task in tasks:
        task_path = os.path.join(base_path, task, "adapter")
        print(f"分析任务: {task}")
        print("-" * 60)
        
        # 加载权重
        weights = load_adapter_weights(task_path)
        if weights is None:
            print(f"没有找到 {task} 的权重文件")
            continue
        
        print("1. 权重键分析:")
        print("   当前任务LoRA权重 (loranew_*):")
        loranew_keys = [k for k in weights.keys() if 'loranew_' in k]
        print(f"      共 {len(loranew_keys)} 个loranew权重")
        
        print("   历史LoRA权重 (lora_*):")
        lora_keys = [k for k in weights.keys() if 'lora_' in k and 'loranew_' not in k]
        print(f"      共 {len(lora_keys)} 个历史lora权重")
        
        print("   历史方向权重 (historical_directions):")
        historical_dir_keys = [k for k in weights.keys() if 'historical_directions' in k]
        print(f"      共 {len(historical_dir_keys)} 个历史方向权重")
        
        print("   历史scaling权重 (historical_scalings):")
        historical_scale_keys = [k for k in weights.keys() if 'historical_scalings' in k]
        print(f"      共 {len(historical_scale_keys)} 个历史scaling参数")
        
        # 分析权重形状和值
        print("\n2. 权重形状和值分析:")
        
        # 分析loranew权重
        print("   2.1 当前任务LoRA权重 (loranew_*):")
        loranew_stats = analyze_loranew_weights(weights)
        
        # 分析历史权重
        print("   2.2 历史LoRA权重 (lora_*):")
        historical_stats = analyze_historical_weights(weights)
        
        # 分析历史方向
        print("   2.3 历史方向存储 (historical_directions):")
        historical_dir_stats = analyze_historical_directions(weights)
        
        # 分析历史scaling
        print("   2.4 历史scaling参数 (historical_scalings):")
        historical_scale_stats = analyze_historical_scalings(weights)
        
        print("\n" + "="*80 + "\n")

def analyze_loranew_weights(weights):
    """分析当前任务的LoRA权重"""
    loranew_A_keys = [k for k in weights.keys() if 'loranew_A' in k and 'weight' in k]
    loranew_B_keys = [k for k in weights.keys() if 'loranew_B' in k and 'weight' in k]
    
    print(f"      loranew_A权重: {len(loranew_A_keys)} 个")
    print(f"      loranew_B权重: {len(loranew_B_keys)} 个")
    
    if loranew_A_keys:
        # 分析第一个loranew_A权重
        sample_A = weights[loranew_A_keys[0]]
        sample_B = weights[loranew_B_keys[0]] if loranew_B_keys else None
        
        print(f"      样例loranew_A形状: {sample_A.shape}")
        print(f"      样例loranew_A统计: mean={sample_A.mean():.6f}, std={sample_A.std():.6f}, norm={torch.norm(sample_A):.6f}")
        
        if sample_B is not None:
            print(f"      样例loranew_B形状: {sample_B.shape}")
            print(f"      样例loranew_B统计: mean={sample_B.mean():.6f}, std={sample_B.std():.6f}, norm={torch.norm(sample_B):.6f}")
        
        # 检查是否有非零值
        non_zero_A = (sample_A != 0).sum().item()
        total_A = sample_A.numel()
        print(f"      loranew_A非零比例: {non_zero_A}/{total_A} ({non_zero_A/total_A*100:.2f}%)")
    
    return {
        'loranew_A_count': len(loranew_A_keys),
        'loranew_B_count': len(loranew_B_keys)
    }

def analyze_historical_weights(weights):
    """分析历史LoRA权重"""
    lora_A_keys = [k for k in weights.keys() if 'lora_A' in k and 'loranew_' not in k and 'weight' in k]
    lora_B_keys = [k for k in weights.keys() if 'lora_B' in k and 'loranew_' not in k and 'weight' in k]
    
    print(f"      lora_A权重: {len(lora_A_keys)} 个")
    print(f"      lora_B权重: {len(lora_B_keys)} 个")
    
    if lora_A_keys:
        # 分析第一个lora_A权重
        sample_A = weights[lora_A_keys[0]]
        sample_B = weights[lora_B_keys[0]] if lora_B_keys else None
        
        print(f"      样例lora_A形状: {sample_A.shape}")
        print(f"      样例lora_A统计: mean={sample_A.mean():.6f}, std={sample_A.std():.6f}, norm={torch.norm(sample_A):.6f}")
        
        if sample_B is not None:
            print(f"      样例lora_B形状: {sample_B.shape}")
            print(f"      样例lora_B统计: mean={sample_B.mean():.6f}, std={sample_B.std():.6f}, norm={torch.norm(sample_B):.6f}")
        
        # 检查空维度（r_sum=0的情况）
        if sample_A.shape[0] == 0 or sample_A.shape[1] == 0:
            print(f"      ⚠️  lora_A有空维度，表示没有历史方向")
        else:
            non_zero_A = (sample_A != 0).sum().item()
            total_A = sample_A.numel()
            print(f"      lora_A非零比例: {non_zero_A}/{total_A} ({non_zero_A/total_A*100:.2f}%)")
    
    return {
        'lora_A_count': len(lora_A_keys),
        'lora_B_count': len(lora_B_keys)
    }

def analyze_historical_directions(weights):
    """分析历史方向存储"""
    historical_dir_keys = [k for k in weights.keys() if 'historical_directions' in k]
    
    if not historical_dir_keys:
        print("      ⚠️  未找到historical_directions参数")
        return {}
    
    print(f"      找到 {len(historical_dir_keys)} 个历史方向参数:")
    
    # 按adapter和方向分组
    direction_groups = defaultdict(list)
    for key in historical_dir_keys:
        # 解析键名，例如: base_model.model.encoder.block.0.layer.0.SelfAttention.q.historical_directions.default.dir_0.A.weight
        parts = key.split('.')
        if 'dir_' in key:
            # 找到dir_的位置
            dir_idx = next(i for i, part in enumerate(parts) if part.startswith('dir_'))
            dir_name = parts[dir_idx]
            direction_type = parts[dir_idx + 1] if dir_idx + 1 < len(parts) else 'unknown'
            layer_path = '.'.join(parts[:dir_idx-2])  # 去掉historical_directions.default.dir_x
            
            direction_groups[layer_path].append({
                'key': key,
                'dir_name': dir_name,
                'type': direction_type,
                'weight': weights[key]
            })
    
    print(f"      涉及 {len(direction_groups)} 个不同的层")
    
    # 分析每个层的历史方向
    for layer_path, directions in list(direction_groups.items())[:3]:  # 只显示前3个层
        print(f"        层 {layer_path}:")
        
        # 按方向分组
        dir_dict = defaultdict(dict)
        for item in directions:
            dir_dict[item['dir_name']][item['type']] = item['weight']
        
        print(f"          包含 {len(dir_dict)} 个历史方向:")
        for dir_name, dir_weights in dir_dict.items():
            if 'A' in dir_weights and 'weight' in dir_weights:
                A_weight = dir_weights['A']
                print(f"            {dir_name}: A形状={A_weight.shape}, norm={torch.norm(A_weight):.6f}")
            
            if 'B' in dir_weights and 'weight' in dir_weights:
                B_weight = dir_weights['B']
                print(f"            {dir_name}: B形状={B_weight.shape}, norm={torch.norm(B_weight):.6f}")
    
    return {
        'total_directions': len(historical_dir_keys),
        'layers_with_directions': len(direction_groups)
    }

def analyze_historical_scalings(weights):
    """分析历史scaling参数"""
    historical_scale_keys = [k for k in weights.keys() if 'historical_scalings' in k]
    
    if not historical_scale_keys:
        print("      ⚠️  未找到historical_scalings参数")
        return {}
    
    print(f"      找到 {len(historical_scale_keys)} 个历史scaling参数:")
    
    # 收集所有scaling值
    scaling_values = []
    scaling_by_layer = defaultdict(list)
    
    for key in historical_scale_keys:
        scaling_value = weights[key]
        scaling_values.append(scaling_value.item())
        
        # 解析层路径
        parts = key.split('.')
        if 'dir_' in key:
            dir_idx = next(i for i, part in enumerate(parts) if part.startswith('dir_'))
            layer_path = '.'.join(parts[:dir_idx-2])  # 去掉historical_scalings.default.dir_x
            dir_name = parts[dir_idx]
            
            scaling_by_layer[layer_path].append({
                'dir_name': dir_name,
                'value': scaling_value.item()
            })
    
    if scaling_values:
        print(f"        scaling值统计:")
        print(f"          平均值: {np.mean(scaling_values):.6f}")
        print(f"          标准差: {np.std(scaling_values):.6f}")
        print(f"          范围: [{np.min(scaling_values):.6f}, {np.max(scaling_values):.6f}]")
        print(f"          唯一值: {set(scaling_values)}")
    
    # 显示每层的scaling
    print(f"        按层分析 (显示前3层):")
    for layer_path, scalings in list(scaling_by_layer.items())[:3]:
        layer_name = layer_path.split('.')[-1] if '.' in layer_path else layer_path
        print(f"          {layer_name}:")
        for item in scalings:
            print(f"            {item['dir_name']}: {item['value']:.6f}")
    
    return {
        'total_scalings': len(historical_scale_keys),
        'scaling_values': scaling_values,
        'layers_with_scalings': len(scaling_by_layer)
    }

def compare_scaling_mechanisms():
    """比较不同任务间的scaling机制"""
    print("=== 跨任务Scaling机制比较 ===\n")
    
    base_path = "/home/yongxi/work/O-LoRA/logs_and_outputs/sdlora/order_1/outputs"
    tasks = ['1-dbpedia', '2-amazon', '3-yahoo', '4-agnews']
    
    all_stats = {}
    
    for task in tasks:
        task_path = os.path.join(base_path, task)
        weights = load_adapter_weights(task_path)
        if weights is None:
            continue
        
        # 收集统计信息
        loranew_count = len([k for k in weights.keys() if 'loranew_' in k and 'weight' in k])
        historical_dir_count = len([k for k in weights.keys() if 'historical_directions' in k])
        historical_scale_count = len([k for k in weights.keys() if 'historical_scalings' in k])
        
        # 收集scaling值
        scaling_values = []
        for key in weights.keys():
            if 'historical_scalings' in key:
                scaling_values.append(weights[key].item())
        
        all_stats[task] = {
            'loranew_count': loranew_count,
            'historical_dir_count': historical_dir_count,
            'historical_scale_count': historical_scale_count,
            'scaling_values': scaling_values
        }
    
    # 打印比较表
    print("任务对比表:")
    print(f"{'任务':<12} {'loranew':<10} {'hist_dir':<10} {'hist_scale':<12} {'scaling值':<20}")
    print("-" * 70)
    
    for task, stats in all_stats.items():
        scaling_summary = f"{len(stats['scaling_values'])}个" if stats['scaling_values'] else "无"
        if stats['scaling_values']:
            avg_scaling = np.mean(stats['scaling_values'])
            scaling_summary += f"(均值:{avg_scaling:.3f})"
        
        print(f"{task:<12} {stats['loranew_count']:<10} {stats['historical_dir_count']:<10} "
              f"{stats['historical_scale_count']:<12} {scaling_summary:<20}")
    
    return all_stats

def analyze_scaling_evolution(all_stats):
    """分析scaling的演化"""
    print("\n=== Scaling演化分析 ===\n")
    
    tasks = sorted(all_stats.keys())
    
    print("1. 历史方向数量演化:")
    for task in tasks:
        hist_dir_count = all_stats[task]['historical_dir_count']
        hist_scale_count = all_stats[task]['historical_scale_count']
        print(f"   {task}: {hist_dir_count} 个方向, {hist_scale_count} 个scaling参数")
    
    print("\n2. Scaling值分布:")
    for task in tasks:
        scaling_values = all_stats[task]['scaling_values']
        if scaling_values:
            unique_values = set(scaling_values)
            print(f"   {task}: {len(scaling_values)} 个scaling, 唯一值: {unique_values}")
        else:
            print(f"   {task}: 无scaling参数")
    
    # 检查scaling是否可训练
    print("\n3. Scaling机制分析:")
    print("   根据代码分析:")
    print("   - self.scaling: 当前任务的固定scaling (代码中设为0.8)")
    print("   - self.historical_scalings: 历史方向的可训练scaling参数")
    print("   - historical_directions: 存储历史A和B矩阵，权重被冻结")

def visualize_scaling_analysis(all_stats):
    """可视化scaling分析结果"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        tasks = sorted(all_stats.keys())
        
        # 1. 参数数量对比
        ax1 = axes[0, 0]
        loranew_counts = [all_stats[task]['loranew_count'] for task in tasks]
        hist_dir_counts = [all_stats[task]['historical_dir_count'] for task in tasks]
        hist_scale_counts = [all_stats[task]['historical_scale_count'] for task in tasks]
        
        x = np.arange(len(tasks))
        width = 0.25
        
        ax1.bar(x - width, loranew_counts, width, label='loranew参数', alpha=0.8)
        ax1.bar(x, hist_dir_counts, width, label='历史方向参数', alpha=0.8)
        ax1.bar(x + width, hist_scale_counts, width, label='历史scaling参数', alpha=0.8)
        
        ax1.set_xlabel('任务')
        ax1.set_ylabel('参数数量')
        ax1.set_title('不同类型参数数量对比')
        ax1.set_xticks(x)
        ax1.set_xticklabels([t.split('-')[1] for t in tasks])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Scaling值分布
        ax2 = axes[0, 1]
        all_scaling_values = []
        task_labels = []
        
        for task in tasks:
            scaling_values = all_stats[task]['scaling_values']
            if scaling_values:
                all_scaling_values.extend(scaling_values)
                task_labels.extend([task.split('-')[1]] * len(scaling_values))
        
        if all_scaling_values:
            unique_values = sorted(set(all_scaling_values))
            ax2.hist(all_scaling_values, bins=max(len(unique_values), 10), alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Scaling值')
            ax2.set_ylabel('频次')
            ax2.set_title('历史Scaling值分布')
            ax2.grid(True, alpha=0.3)
        
        # 3. 任务演化趋势
        ax3 = axes[1, 0]
        cumulative_directions = []
        cumulative_scalings = []
        
        for i, task in enumerate(tasks):
            if i == 0:
                cumulative_directions.append(all_stats[task]['historical_dir_count'])
                cumulative_scalings.append(all_stats[task]['historical_scale_count'])
            else:
                # 注意：这里的逻辑可能需要根据实际的累积方式调整
                cumulative_directions.append(all_stats[task]['historical_dir_count'])
                cumulative_scalings.append(all_stats[task]['historical_scale_count'])
        
        ax3.plot(range(len(tasks)), cumulative_directions, 'o-', label='历史方向数量', marker='o')
        ax3.plot(range(len(tasks)), cumulative_scalings, 's-', label='历史scaling数量', marker='s')
        ax3.set_xlabel('任务序号')
        ax3.set_ylabel('参数数量')
        ax3.set_title('历史参数累积趋势')
        ax3.set_xticks(range(len(tasks)))
        ax3.set_xticklabels([t.split('-')[1] for t in tasks])
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 参数比例分析
        ax4 = axes[1, 1]
        proportions = []
        labels = []
        
        for task in tasks:
            total_lora = all_stats[task]['loranew_count'] + all_stats[task]['historical_dir_count']
            if total_lora > 0:
                loranew_prop = all_stats[task]['loranew_count'] / total_lora
                hist_prop = all_stats[task]['historical_dir_count'] / total_lora
                proportions.append([loranew_prop, hist_prop])
                labels.append(task.split('-')[1])
        
        if proportions:
            proportions = np.array(proportions)
            bottom = np.zeros(len(labels))
            
            ax4.bar(labels, proportions[:, 0], label='当前任务LoRA', alpha=0.8)
            ax4.bar(labels, proportions[:, 1], bottom=proportions[:, 0], label='历史方向', alpha=0.8)
            
            ax4.set_xlabel('任务')
            ax4.set_ylabel('比例')
            ax4.set_title('当前vs历史LoRA参数比例')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/yongxi/work/O-LoRA/sdlora_scaling_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("\n可视化结果已保存为 'sdlora_scaling_analysis.png'")
        
    except ImportError:
        print("\n无法导入matplotlib，跳过可视化")

def main():
    """主函数"""
    print("开始分析SDLoRA的scaling机制...\n")
    
    # 1. 分析每个任务的scaling机制
    analyze_scaling_mechanisms()
    
    # 2. 跨任务比较
    all_stats = compare_scaling_mechanisms()
    
    # 3. 演化分析
    analyze_scaling_evolution(all_stats)
    
    # 4. 可视化
    visualize_scaling_analysis(all_stats)
    
    # 5. 保存分析结果
    output_path = "/home/yongxi/work/O-LoRA/sdlora_scaling_analysis.json"
    with open(output_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    
    print(f"\n分析结果已保存到: {output_path}")
    
    print("\n=== 总结 ===")
    print("SDLoRA的三种scaling机制:")
    print("1. self.scaling: 当前任务的固定scaling因子 (代码中设为0.8)")
    print("2. self.historical_directions: 存储历史LoRA方向的A和B矩阵 (权重冻结)")
    print("3. self.historical_scalings: 每个历史方向对应的可训练scaling参数")
    print("\n这种设计允许:")
    print("- 保持历史知识不被遗忘 (通过frozen的historical_directions)")
    print("- 动态调整历史知识的贡献 (通过trainable的historical_scalings)")
    print("- 学习新任务的表示 (通过当前的loranew_A/B)")

if __name__ == "__main__":
    main()
