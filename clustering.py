import torch
import argparse
import glob
from pathlib import Path
import os

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
                scaled_weight = scaling * weight_matrix / norm_factor
                
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
    
    return results

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
                        help="Output file to save analysis results")
    args = parser.parse_args()
    
    base_path = Path(args.path)
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    all_results = {}
    
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
                weight_results = calculate_lora_weight_sum(state_dict, task_idx)
                
                # 打印统计信息
                print_weight_statistics(weight_results, task_name)
                
                order_results[task_name] = weight_results
                
            except Exception as e:
                print(f"  处理 {task_name} 时出错: {e}")
                continue
        
        all_results[order_name] = order_results
    
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