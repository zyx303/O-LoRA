import os
from transformers import T5ForConditionalGeneration, T5Tokenizer

os.env['HF_HUB_ENABLE_HF_TRANSFER'] = '1'  # 设置为离线模式，防止重复下载
# 1. 定义模型名称和目标路径
model_name = 't5-large'
output_dir = 'initial_model'
# 拼接成最终的模型保存路径
model_save_path = os.path.join(output_dir, model_name)

# 2. 确保目标文件夹存在
# os.makedirs 会创建所有不存在的中间目录
os.makedirs(model_save_path, exist_ok=True)

print(f"开始从 Hugging Face 下载模型: {model_name}")
print(f"将要保存到: {model_save_path}")

# 3. 下载并保存模型和分词器
# .from_pretrained() 会自动从 Hugging Face 下载
# .save_pretrained() 会将下载的文件保存到指定目录
try:
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)

    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    print("模型和分词器下载并保存成功！")
    print(f"最终文件结构: {output_dir}/{model_name}/")

except Exception as e:
    print(f"下载过程中发生错误: {e}")