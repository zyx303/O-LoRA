import json
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import glob

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Liberation Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def extract_loss_from_trainer_state(file_path):
    """从trainer_state.json文件中提取loss历史"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        log_history = data.get('log_history', [])
        
        steps = []
        losses = []
        
        for entry in log_history:
            if 'loss' in entry and 'step' in entry:
                steps.append(entry['step'])
                losses.append(entry['loss'])
        
        return steps, losses
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return [], []

def parse_sequential_results(base_path):
    results = {}
    
    base_path = Path(base_path)
    
    # 遍历每个order目录
    for order_dir in sorted(base_path.glob('order*')):
        order_name = order_dir.name
        results[order_name] = {}
        
        outputs_dir = order_dir / 'outputs'
        if not outputs_dir.exists():
            continue
        
        print(f"\n处理 {order_name}...")
        
        # 使用glob查找所有任务目录，按任务序号排序
        task_dirs = sorted(glob.glob(str(outputs_dir / '*')), 
                          key=lambda x: int(Path(x).name.split('-')[0]))
        
        # 对于每个任务，提取loss历史
        for i, task_dir_str in enumerate(task_dirs):
            task_dir = Path(task_dir_str)
            if not task_dir.is_dir():
                continue
                
            task_name = task_dir.name  # 例如: "1-dbpedia"
            trainer_state_file = task_dir / 'trainer_state.json'
            
            if trainer_state_file.exists():
                steps, losses = extract_loss_from_trainer_state(trainer_state_file)
                if steps and losses:
                    results[order_name][task_name] = {
                        'steps': steps,
                        'losses': losses,
                        'task_order': i + 1
                    }
                    
                    print(f"  任务 {task_name}: {len(steps)} 个训练步骤")
                    print(f"    初始loss: {losses[0]:.4f}")
                    print(f"    最终loss: {losses[-1]:.4f}")
                    print(f"    最低loss: {min(losses):.4f}")
    
    return results

def plot_sequential_loss_curves(results, save_path=None):
    """绘制顺序训练过程中每个任务的loss曲线"""
    
    # 为每个order创建一个子图
    orders = sorted(results.keys())
    fig, axes = plt.subplots(1, len(orders), figsize=(8*len(orders), 6))
    if len(orders) == 1:
        axes = [axes]
    
    # 定义任务颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for order_idx, order in enumerate(orders):
        ax = axes[order_idx]
        order_data = results[order]
        
        # 按任务顺序排序
        tasks = sorted(order_data.keys(), key=lambda x: order_data[x]['task_order'])
        
        # 为每个任务绘制loss曲线
        for task_idx, task in enumerate(tasks):
            task_data = order_data[task]
            steps = task_data['steps']
            losses = task_data['losses']
            
            ax.plot(steps, losses, 
                   marker='o', linewidth=2, markersize=3, 
                   label=f'Task {task_data["task_order"]}: {task.split("-")[1]}', 
                   color=colors[task_idx % len(colors)],
                   alpha=0.8)
        
        ax.set_xlabel('Training Steps', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title(f'{order.replace("_", " ").title()}', fontsize=14)
        ax.grid(True, alpha=0.3)
        # ax.set_yscale('log')  # 使用对数刻度
        
        # 只在第一个子图显示图例
        if order_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sequential training loss curves saved to: {save_path}")
    
    plt.show()

def plot_combined_loss_with_task_separation(results, save_path=None):
    """绘制连续的loss曲线，用垂直线分隔不同任务"""
    
    orders = sorted(results.keys())
    fig, axes = plt.subplots(len(orders), 1, figsize=(15, 6*len(orders)))
    if len(orders) == 1:
        axes = [axes]
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for order_idx, order in enumerate(orders):
        ax = axes[order_idx]
        order_data = results[order]
        
        # 按任务顺序排序
        tasks = sorted(order_data.keys(), key=lambda x: order_data[x]['task_order'])
        
        current_step_offset = 0
        task_boundaries = []
        all_steps = []
        all_losses = []
        for task_idx, task in enumerate(tasks):
            task_data = order_data[task]
            steps = task_data['steps']
            losses = task_data['losses']
            
            # 调整步骤以形成连续序列
            adjusted_steps = [step + current_step_offset for step in steps]
            
            all_steps = all_steps + adjusted_steps
            all_losses = all_losses + losses
            # ax.plot(adjusted_steps, losses, 
            #        color=colors[task_idx % len(colors)], 
            #        linewidth=2, 
            #        label=f'Task {task_data["task_order"]}: {task.split("-")[1]}',
            #        alpha=0.8)
            # 记录任务边界
            if task_idx > 0:
                task_boundaries.append(current_step_offset)
            
            # 更新步骤偏移量
            if adjusted_steps:
                current_step_offset = max(adjusted_steps)

        ax.plot(all_steps, all_losses, 
                   color=colors[task_idx % len(colors)], 
                   linewidth=2, 
                   label=f'Task {task_data["task_order"]}: {task.split("-")[1]}',
                   alpha=0.8)
        # 添加任务分隔线
        for boundary in task_boundaries:
            ax.axvline(x=boundary, color='gray', linestyle='--', alpha=0.5)
        
        ax.set_xlabel('Training Steps (Adjusted)', fontsize=12)
        ax.set_ylabel('Training Loss', fontsize=12)
        ax.set_title(f'{order.replace("_", " ").title()} - Sequential Training Loss', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        # ax.set_yscale('log')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Combined sequential loss curves saved to: {save_path}")
    
    plt.show()

def create_loss_summary_table(results):
    """创建loss结果汇总表"""
    all_data = []
    
    for order_name, order_data in results.items():
        for task_name, task_data in order_data.items():
            steps = task_data['steps']
            losses = task_data['losses']
            
            if losses:
                all_data.append({
                    'Order': order_name,
                    'Task': task_name,
                    'Task_Order': task_data['task_order'],
                    'Initial_Loss': losses[0],
                    'Final_Loss': losses[-1],
                    'Min_Loss': min(losses),
                    'Max_Loss': max(losses),
                    'Total_Steps': len(steps),
                    'Loss_Reduction': losses[0] - losses[-1]
                })
    
    df = pd.DataFrame(all_data)
    return df
import argparse
def main():
    # 设置数据路径
    parser = argparse.ArgumentParser(description="Plot training loss curves from trainer_state.json files.")
    parser.add_argument('--path', type=str, default="/home/yongxi/work/O-LoRA/logs_and_outputs/",
                        help="Path to the directory containing order subdirectories.")
    args = parser.parse_args()
    path = args.path
    
    print("Parsing Sequential training results...")
    results = parse_sequential_results(path)

    if not results:
        print("No valid result data found!")
        return
    
    print("\n=== Sequential Training Loss Summary ===")
    summary_df = create_loss_summary_table(results)
    print(summary_df)
    
    # 保存汇总表
    output_path = 'analyze/loss'
    summary_df.to_csv(f"{output_path}.csv", index=False)
    print(f"\nLoss summary table saved to: {output_path}.csv")
    
    print("\nPlotting sequential training loss curves...")
    plot_sequential_loss_curves(results, f"{output_path}_individual.png")
    
    print("\nPlotting combined sequential loss curves...")
    plot_combined_loss_with_task_separation(results, f"{output_path}_combined.png")

if __name__ == "__main__":
    main()