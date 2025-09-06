import pandas as pd
import matplotlib.pyplot as plt
model_name = 'T5'
# 读取数据
olora = pd.read_csv(f'./analyze/acc_{model_name}_olora.csv')
sdlora = pd.read_csv(f'./analyze/acc_{model_name}_sdlora.csv')

# 选择要对比的任务
tasks = ['agnews', 'amazon', 'dbpedia', 'overall', 'yahoo']
orders = sorted(set(olora['Order'].dropna()))

# 为每个任务定义固定颜色
task_colors = {
    'agnews': '#FF6B6B',    # 红色
    'amazon': '#4ECDC4',    # 青色
    'dbpedia': '#45B7D1',   # 蓝色
    'overall': '#96CEB4',   # 绿色
    'yahoo': '#FFEAA7'      # 黄色
}

# 画图
for order in orders:
    plt.figure(figsize=(10, 6))
    
    for task in tasks:
        # 获取任务对应的颜色
        color = task_colors[task]
        
        # 获取该order的数据
        olora_order = olora[olora['Order'] == order].copy()
        sdlora_order = sdlora[sdlora['Order'] == order].copy()
        
        # 为了正确处理x轴，提取任务编号
        if not olora_order.empty:
            olora_order['task_num'] = olora_order['Stage'].str.extract(r'after_task_(\d+)')[0].astype(int)
            olora_task_data = olora_order[olora_order[task].notna()]
            if not olora_task_data.empty:
                plt.plot(olora_task_data['task_num'], olora_task_data[task], 
                        marker='o', linestyle='-', color=color, linewidth=2, markersize=6, 
                        label=f'{task} (O-LoRA)')
        
        if not sdlora_order.empty:
            sdlora_order['task_num'] = sdlora_order['Stage'].str.extract(r'after_task_(\d+)')[0].astype(int)
            sdlora_task_data = sdlora_order[sdlora_order[task].notna()]
            if not sdlora_task_data.empty:
                plt.plot(sdlora_task_data['task_num'], sdlora_task_data[task], 
                        marker='s', linestyle='--', color=color, linewidth=2, markersize=6, 
                        label=f'{task} (SD-LoRA)')
    
    plt.xlabel('Completed Tasks', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Accuracy Comparison: {order.replace("_", " ").title()}', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 5), [f'Task {i}' for i in range(1, 5)])
    plt.tight_layout()
    plt.savefig(f'acc_compare_{model_name}_{order}.png', dpi=300, bbox_inches='tight')
    plt.show()