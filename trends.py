import json
import os
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def extract_accuracy_from_results(file_path):
    """从结果文件中提取准确率"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 优先使用predict_exact_match作为主要准确率指标
        if 'predict_exact_match' in data:
            return data['predict_exact_match']
        elif 'eval_accuracy' in data:
            return data['eval_accuracy']
        elif 'test_accuracy' in data:
            return data['test_accuracy']
        elif 'accuracy' in data:
            return data['accuracy']
        else:
            # 如果没有直接的accuracy字段，查看所有字段
            for key, value in data.items():
                if 'exact_match' in key.lower() and isinstance(value, (int, float)):
                    return value
                elif 'acc' in key.lower() and isinstance(value, (int, float)):
                    return value
        return None
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def extract_all_task_accuracies(file_path):
    """从结果文件中提取所有任务的准确率"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        task_accuracies = {}
        
        # 查找所有任务特定的准确率字段
        for key, value in data.items():
            if key == 'predict_exact_match':
                task_accuracies['overall'] = value
            if key.startswith('predict_exact_match_for_') and isinstance(value, (int, float)):
                task_name = key.replace('predict_exact_match_for_', '')
                if task_name in ['TC','SC']:
                    continue
                task_accuracies[task_name] = value
        
        return task_accuracies
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return {}

def parse_sdlora_sequential_results(base_path):
    """解析SDLoRA顺序训练结果，追踪每完成一个任务后所有任务的准确率"""
    results = {}
    
    base_path = Path(base_path)
    
    # 遍历每个order
    for order_dir in sorted(base_path.glob('order_*')):
        order_name = order_dir.name
        results[order_name] = {}
        
        outputs_dir = order_dir / 'outputs'
        if not outputs_dir.exists():
            continue
        
        print(f"\n处理 {order_name}...")
        
        # 按任务序号排序（1-task, 2-task, 3-task, 4-task）
        task_dirs = sorted([d for d in outputs_dir.glob('*') if d.is_dir()], 
                          key=lambda x: int(x.name.split('-')[0]))
        
        # 对于每个完成的任务阶段，记录所有任务的准确率
        for i, task_dir in enumerate(task_dirs):
            stage_name = f"after_task_{i+1}"  # after_task_1, after_task_2, etc.
            
            predict_results_file = task_dir / 'predict_results.json'
            if predict_results_file.exists():
                # 获取这个阶段所有任务的准确率
                all_task_accs = extract_all_task_accuracies(predict_results_file)
                results[order_name][stage_name] = all_task_accs
                
                print(f"  完成任务 {i+1} 后的准确率:")
                for task, acc in all_task_accs.items():
                    print(f"    {task}: {acc:.2f}")
    
    return results

def plot_sequential_accuracy_trends(results, save_path=None):
    """绘制顺序训练过程中每个任务准确率的变化"""
    
    # 获取所有任务名称
    all_tasks = set()
    for order_data in results.values():
        for stage_data in order_data.values():
            all_tasks.update(stage_data.keys())
    all_tasks = sorted(list(all_tasks))
    
    # 为每个order创建一个子图
    orders = sorted(results.keys())
    fig, axes = plt.subplots(1, len(orders), figsize=(5*len(orders), 6))
    if len(orders) == 1:
        axes = [axes]
    
    colors = plt.cm.Set1(range(len(all_tasks)))
    
    for order_idx, order in enumerate(orders):
        ax = axes[order_idx]
        order_data = results[order]
        
        # 获取阶段列表（按顺序）
        stages = sorted([s for s in order_data.keys() if s.startswith('after_task_')], 
                       key=lambda x: int(x.split('_')[-1]))
        
        # 为每个任务绘制线条
        for task_idx, task in enumerate(all_tasks):
            task_accs = []
            stage_numbers = []
            
            for stage in stages:
                if task in order_data[stage]:
                    task_accs.append(order_data[stage][task])
                    stage_numbers.append(int(stage.split('_')[-1]))
            
            if task_accs:  # 只绘制有数据的任务
                ax.plot(stage_numbers, task_accs, 
                       marker='o', linewidth=2, markersize=6, 
                       label=task, color=colors[task_idx])
        
        ax.set_xlabel('Completed Tasks', fontsize=12)
        ax.set_ylabel('Accuracy (%)', fontsize=12)
        ax.set_title(f'{order.replace("_", " ").title()}', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels([f'Task {i}' for i in range(1, 5)])
        
        # 只在第一个子图显示图例
        if order_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sequential training accuracy trends saved to: {save_path}")
    
    plt.show()

def create_sequential_summary_table(results):
    """创建顺序训练结果汇总表"""
    all_data = []
    
    for order_name, order_data in results.items():
        for stage_name, stage_data in order_data.items():
            for task_name, accuracy in stage_data.items():
                all_data.append({
                    'Order': order_name,
                    'Stage': stage_name,
                    'Task': task_name,
                    'Accuracy': accuracy
                })
    
    df = pd.DataFrame(all_data)
    
    # 创建透视表
    pivot_df = df.pivot_table(
        index=['Order', 'Stage'], 
        columns='Task', 
        values='Accuracy', 
        fill_value=None
    )
    
    return pivot_df

def plot_accuracy_trends(results, save_path=None):
    """绘制准确率趋势图"""
    plt.figure(figsize=(12, 8))
    
    # 获取所有unique的任务名称
    all_tasks = set()
    for order_data in results.values():
        all_tasks.update(order_data.keys())
    all_tasks = sorted(list(all_tasks))
    
    # 为每个任务创建数据
    task_data = {}
    for task in all_tasks:
        task_data[task] = []
        task_orders = []
        
        for order in sorted(results.keys()):
            if task in results[order]:
                task_data[task].append(results[order][task])
                task_orders.append(order)
        
        if task_data[task]:  # 只绘制有数据的任务
            plt.plot(range(len(task_data[task])), task_data[task], 
                    marker='o', linewidth=2, markersize=6, label=task)
    
    plt.xlabel('Training Order', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Accuracy Trends Across Different Training Orders', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 设置x轴标签
    order_labels = sorted(results.keys())
    plt.xticks(range(len(order_labels)), [f"Order {i+1}" for i in range(len(order_labels))])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()

def create_summary_table(results):
    """创建结果汇总表"""
    # 获取所有任务
    all_tasks = set()
    for order_data in results.values():
        all_tasks.update(order_data.keys())
    all_tasks = sorted(list(all_tasks))
    
    # 创建DataFrame
    df_data = {}
    for order in sorted(results.keys()):
        df_data[order] = [results[order].get(task, None) for task in all_tasks]
    
    df = pd.DataFrame(df_data, index=all_tasks)
    return df

def main():
    # 设置数据路径
    sdlora_path = "/home/yongxi/work/O-LoRA/logs_and_outputs/sdlora"
    
    print("Parsing SDLoRA sequential training results...")
    results = parse_sdlora_sequential_results(sdlora_path)
    
    if not results:
        print("No valid result data found!")
        return
    
    print("\n=== Sequential Training Results Summary ===")
    summary_df = create_sequential_summary_table(results)
    print(summary_df)
    
    # 保存汇总表
    path ='/home/yongxi/work/O-LoRA/analyze/acc_T5_sdlora'
    summary_df.to_csv(f"{path}.csv")
    print(f"\nSummary table saved to: {path}.csv")
    
    print("\nPlotting sequential training accuracy trends...")
    plot_sequential_accuracy_trends(results, f"{path}.png")

if __name__ == "__main__":
    main()