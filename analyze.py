import torch
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns

def load_adapter_config(adapter_path):
    """加载adapter配置"""
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return json.load(f)
    return None

def load_adapter_weights(adapter_path):
    """加载adapter权重"""
    weights_path = os.path.join(adapter_path, "adapter_model.bin")
    if os.path.exists(weights_path):
        return torch.load(weights_path, map_location='cpu')
    return None

def analyze_lora_weights(weights, layer_name, task_name):
    """分析单个LoRA层的权重"""
    analysis = {}
    
    # 查找该层的所有相关权重
    layer_weights = {k: v for k, v in weights.items() if layer_name in k}
    
    for weight_name, weight_tensor in layer_weights.items():
        if weight_tensor.numel() > 0:  # 只分析非空张量
            analysis[weight_name] = {
                'shape': list(weight_tensor.shape),
                'mean': float(weight_tensor.mean()),
                'std': float(weight_tensor.std()),
                'norm': float(torch.norm(weight_tensor)),
                'max': float(weight_tensor.max()),
                'min': float(weight_tensor.min()),
                'zeros_ratio': float((weight_tensor == 0).sum() / weight_tensor.numel())
            }
    
    return analysis

def analyze_adapters():
    """分析所有adapter"""
    base_path = "/home/yongxi/work/O-LoRA/logs_and_outputs/sdlora/order_1/outputs"
    
    # 任务列表
    tasks = ['1-dbpedia','2-amazon','3-yahoo','4-agnews']
    
    # 存储所有分析结果
    all_analysis = {}
    
    print("=== SDLoRA Adapter Analysis ===\n")
    
    for task in tasks:
        task_path = os.path.join(base_path, task, "adapter")
        print(f"Analyzing task: {task}")
        print("-" * 50)
        
        # 加载配置
        config = load_adapter_config(task_path)
        if config:
            print(f"Config: {json.dumps(config, indent=2)}")
        
        # 加载权重
        weights = load_adapter_weights(task_path)
        if weights is None:
            print(f"No weights found for {task}")
            continue
        
        # 分析权重结构
        print(f"\nWeight keys in {task}:")
        for key in sorted(weights.keys()):
            shape = list(weights[key].shape) if hasattr(weights[key], 'shape') else 'scalar'
            print(f"  {key}: {shape}")
            if 'num' in key:
                # 打印值
                print(f"  {key} value: {weights[key]}")
        
        # 按层分析
        task_analysis = {}
        
        # 获取所有层名
        layer_names = set()
        for key in weights.keys():
            # 提取层名 (例如从 "base_model.model.encoder.block.0.layer.0.SelfAttention.q.lora_A.default.weight" 提取层名)
            parts = key.split('.')
            if 'lora_A' in key or 'lora_B' in key or 'loranew_A' in key or 'loranew_B' in key:
                layer_name = '.'.join(parts[:-4])  # 去掉最后的 lora_A/B.default.weight
                layer_names.add(layer_name)
        
        # for layer_name in sorted(layer_names):
        #     layer_analysis = analyze_lora_weights(weights, layer_name, task)
        #     if layer_analysis:
        #         task_analysis[layer_name] = layer_analysis
        
        # all_analysis[task] = {
        #     'config': config,
        #     'layer_analysis': task_analysis,
        #     'total_params': sum(w.numel() for w in weights.values())
        # }
        
        # print(f"\nTotal parameters: {all_analysis[task]['total_params']:,}")
        print(f"Number of layers with LoRA: {len(task_analysis)}")
        print("\n" + "="*70 + "\n")
    
    return all_analysis

def compare_tasks_analysis(all_analysis):
    """比较不同任务的分析结果"""
    print("=== Task Comparison ===\n")
    
    # 比较总参数数量
    print("Total parameters per task:")
    for task, analysis in all_analysis.items():
        print(f"  {task}: {analysis['total_params']:,}")
    
    # 比较rank参数
    print("\nRank parameters:")
    for task, analysis in all_analysis.items():
        config = analysis['config']
        if config:
            r = config.get('r', 'N/A')
            r_sum = config.get('r_sum', 'N/A')
            print(f"  {task}: r={r}, r_sum={r_sum}")
    
    # 分析权重分布
    print("\nWeight statistics across tasks:")
    
    # 收集所有权重的统计信息
    weight_stats = defaultdict(list)
    
    for task, analysis in all_analysis.items():
        for layer_name, layer_data in analysis['layer_analysis'].items():
            for weight_name, stats in layer_data.items():
                weight_stats[f"{weight_name}_norm"].append(stats['norm'])
                weight_stats[f"{weight_name}_std"].append(stats['std'])
    
    # 打印平均统计信息
    for stat_name, values in weight_stats.items():
        if values:
            print(f"  {stat_name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

def visualize_weight_evolution(all_analysis):
    """可视化权重在不同任务间的变化"""
    tasks = sorted(all_analysis.keys())
    
    # 收集所有层的norm数据
    layer_norms = defaultdict(list)
    
    for task in tasks:
        analysis = all_analysis[task]
        for layer_name, layer_data in analysis['layer_analysis'].items():
            for weight_name, stats in layer_data.items():
                if 'norm' in stats:
                    layer_norms[f"{layer_name}.{weight_name}"].append(stats['norm'])
    
    # 只显示在所有任务中都存在的层
    consistent_layers = {k: v for k, v in layer_norms.items() if len(v) == len(tasks)}
    
    if consistent_layers:
        plt.figure(figsize=(15, 8))
        
        # 选择前10个层进行可视化
        layers_to_plot = list(consistent_layers.keys())[:10]
        
        for i, layer in enumerate(layers_to_plot):
            plt.subplot(2, 5, i+1)
            plt.plot(range(len(tasks)), consistent_layers[layer], 'bo-')
            plt.title(f"{layer.split('.')[-1]}", fontsize=8)
            plt.xticks(range(len(tasks)), [t.split('-')[0] for t in tasks], rotation=45)
            plt.ylabel('Weight Norm')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/yongxi/work/O-LoRA/weight_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Weight evolution plot saved as 'weight_evolution.png'")

def check_lora_directions(all_analysis):
    """检查LoRA方向的演化"""
    print("=== LoRA Directions Analysis ===\n")
    
    tasks = sorted(all_analysis.keys())
    
    for task in tasks:
        analysis = all_analysis[task]
        config = analysis['config']
        
        print(f"Task: {task}")
        if config:
            r = config.get('r', 0)
            r_sum = config.get('r_sum', 0)
            print(f"  r (current): {r}, r_sum (historical): {r_sum}")
        
        # 检查lora_A和lora_B的形状
        layer_analysis = analysis['layer_analysis']
        if layer_analysis:
            sample_layer = list(layer_analysis.keys())[0]
            sample_data = layer_analysis[sample_layer]
            
            for weight_name, stats in sample_data.items():
                if 'lora_A' in weight_name or 'lora_B' in weight_name:
                    print(f"    {weight_name}: {stats['shape']}")
        
        print()

def main():
    # 分析所有adapters
    all_analysis = analyze_adapters()
    
    # 保存分析结果
    output_path = "/home/yongxi/work/O-LoRA/adapter_analysis.json"
    with open(output_path, 'w') as f:
        # 将numpy类型转换为Python原生类型以便JSON序列化
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            return obj
        
        json.dump(convert_numpy(all_analysis), f, indent=2)
    
    print(f"Detailed analysis saved to: {output_path}")
    
    # 比较分析
    compare_tasks_analysis(all_analysis)
    
    # 检查LoRA方向
    check_lora_directions(all_analysis)
    
    # 可视化（如果有matplotlib）
    try:
        visualize_weight_evolution(all_analysis)
    except ImportError:
        print("Matplotlib not available, skipping visualization")
import debugpy
if __name__ == "__main__":
    # debugpy.listen(5678)
    # debugpy.wait_for_client()
    main()