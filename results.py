import pandas as pd
import io
import numpy as np

# 1. 写入数据
# 将表格数据存储为CSV格式的字符串
# data = """
# method,order-1,order-2,order-3
# o-lora*,75.4,75.7,76.3
# o-lora,74.0724,74.227,76.9013
# sdlora,77.9737,76.1941,77.5428
# l2p*,60.3,61.7,61.1
# ProgPrompt*,75.2,75,75.1
# hideprompt(cr),69.0625,65.4474,67.6447
# """
data = """
method,order-1,order-2,order-3
o-lora*,76.8,75.7,75.7
o-lora,74.4803,71.9276,75.8717
sdlora,77.9638,79.1349,80.1349
"""
# 使用pandas从字符串中读取数据
df = pd.read_csv(io.StringIO(data))

# 2. 计算 avg
# 将所有数值列转换为float类型（处理可能的字符串）
for col in ['order-1', 'order-2', 'order-3']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 计算 'order-1', 'order-2', 和 'order-3' 这三列的平均值
# axis=1 表示按行计算
df['avg'] = df[['order-1', 'order-2', 'order-3']].mean(axis=1)

# 3. 显示结果
# 打印带有计算结果的完整表格
print(df.to_string(index=False))
print("\n格式化输出：")
print(df.to_string(index=False, float_format='%.4f'))
