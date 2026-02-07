#!/usr/bin/env python3
"""
Debug script to check gradient flow in hide_prompt implementation
"""

import torch
import sys
import os
sys.path.append("/root/work/O-LoRA/src")

from peft import get_peft_model, HidePromptConfig, TaskType
from transformers import T5ForConditionalGeneration, T5Tokenizer

def check_gradient_flow(model, inputs, target_params=None):
    """检查梯度流动"""
    print("\n=== Gradient Flow Check ===")
    
    # 前向传播
    model.train()
    loss = model(**inputs).loss
    print(f"Loss: {loss.item()}")
    
    # 清空之前的梯度
    model.zero_grad()
    
    # 反向传播
    loss.backward()
    
    # 检查所有参数的梯度
    params_with_grad = 0
    params_without_grad = 0
    hide_prompt_params = 0
    hide_prompt_with_grad = 0
    
    print("\n=== Parameter Gradient Status ===")
    for name, param in model.named_parameters():
        if param.requires_grad:
            has_grad = param.grad is not None and torch.any(param.grad != 0)
            
            if 'prompt' in name.lower():
                hide_prompt_params += 1
                if has_grad:
                    hide_prompt_with_grad += 1
                print(f"HIDE_PROMPT: {name}")
                print(f"  - requires_grad: {param.requires_grad}")
                print(f"  - has_grad: {param.grad is not None}")
                print(f"  - grad_nonzero: {has_grad}")
                if param.grad is not None:
                    print(f"  - grad_norm: {param.grad.norm().item():.6f}")
                    print(f"  - grad_max: {param.grad.max().item():.6f}")
                    print(f"  - grad_min: {param.grad.min().item():.6f}")
                print()
            
            if has_grad:
                params_with_grad += 1
            else:
                params_without_grad += 1
                if 'prompt' in name.lower():
                    print(f"WARNING: {name} has no gradient!")
    
    print(f"Total params with grad: {params_with_grad}")
    print(f"Total params without grad: {params_without_grad}")
    print(f"Hide-prompt params: {hide_prompt_params}")
    print(f"Hide-prompt params with grad: {hide_prompt_with_grad}")
    
    return hide_prompt_with_grad > 0

def trace_gradient_path(model, inputs):
    """追踪梯度路径"""
    print("\n=== Tracing Gradient Path ===")
    
    model.train()
    model.zero_grad()
    
    # 注册hook来追踪梯度
    grad_info = {}
    
    def make_hook(name):
        def hook(grad):
            grad_info[name] = {
                'has_grad': grad is not None,
                'nonzero': torch.any(grad != 0).item() if grad is not None else False,
                'norm': grad.norm().item() if grad is not None else 0.0
            }
            print(f"Hook {name}: has_grad={grad is not None}, nonzero={torch.any(grad != 0).item() if grad is not None else False}")
        return hook
    
    # 为hide_prompt相关参数注册hook
    for name, param in model.named_parameters():
        if 'prompt' in name.lower() and param.requires_grad:
            param.register_hook(make_hook(name))
    
    # 前向+反向传播
    loss = model(**inputs).loss
    loss.backward()
    
    print("\nGradient Hook Results:")
    for name, info in grad_info.items():
        print(f"{name}: {info}")

def test_simple_forward():
    """测试简单的前向传播"""
    print("\n=== Testing Simple Forward Pass ===")
    
    # 初始化模型 - 使用与debug_hide.sh一致的配置
    model_name = "initial_model/t5-large"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    # 配置HidePrompt - 使用与debug_hide.sh一致的参数
    peft_config = HidePromptConfig(
        num_virtual_tokens=5,
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        prompt_key=False,
        pool_size=10,
        top_k=1
    )
    
    model = get_peft_model(model, peft_config)
    
    # 准备输入 - 使用与debug_hide.sh一致的格式
    input_text = "translate English to German: Hello world"
    target_text = "Hallo Welt"
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    targets = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    
    inputs["labels"] = targets.input_ids
    inputs["task_id"] = 0  # 指定task_id
    inputs["Dataset"] = ["dbpedia"]  # 添加dataset name，与debug_hide.sh一致
    
    print(f"Input shape: {inputs['input_ids'].shape}")
    print(f"Labels shape: {inputs['labels'].shape}")
    
    # 检查模型参数
    print("\n=== Model Parameters ===")
    for name, param in model.named_parameters():
        if 'prompt' in name.lower():
            print(f"{name}: shape={param.shape}, requires_grad={param.requires_grad}")
    
    # 检查梯度流动
    has_gradients = check_gradient_flow(model, inputs)
    
    # 追踪梯度路径
    trace_gradient_path(model, inputs)
    
    return has_gradients

def debug_prompt_selection():
    """调试prompt选择过程"""
    print("\n=== Debugging Prompt Selection ===")
    
    # 初始化模型 - 使用与debug_hide.sh一致的配置
    model_name = "initial_model/t5-large"
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    
    peft_config = HidePromptConfig(
        num_virtual_tokens=5,
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        prompt_key=False,
        pool_size=10,
        top_k=1
    )
    
    model = get_peft_model(model, peft_config)
    
    # 获取eprompt模块
    eprompt = model.prompt_encoder[model.active_adapter]
    print(f"EPrompt module: {type(eprompt)}")
    print(f"Prompt shape: {eprompt.prompt.shape}")
    print(f"Prompt requires_grad: {eprompt.prompt.requires_grad}")
    
    # 准备输入 - 使用与debug_hide.sh一致的格式
    input_text = "translate English to German: Hello world"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs_embeds = model.word_embeddings(inputs.input_ids)
    
    # 测试eprompt forward
    print("\n=== Testing EPrompt Forward ===")
    # 使用dataset_names而不是prompt_idx
    out = eprompt(
        x_embed=inputs_embeds,
        dataset_names=["dbpedia"]  # 使用dataset name，与debug_hide.sh一致
    )
    
    print(f"EPrompt output keys: {out.keys()}")
    if 'batched_prompt' in out:
        batched_prompt = out['batched_prompt']
        print(f"Batched prompt shape: {batched_prompt.shape}")
        print(f"Batched prompt requires_grad: {batched_prompt.requires_grad}")
        
        # 测试反向传播
        if batched_prompt.requires_grad:
            dummy_loss = batched_prompt.sum()
            dummy_loss.backward()
            print(f"After backward, eprompt.prompt.grad is not None: {eprompt.prompt.grad is not None}")
            if eprompt.prompt.grad is not None:
                print(f"Prompt grad norm: {eprompt.prompt.grad.norm().item()}")

if __name__ == "__main__":
    print("=== Hide Prompt Gradient Debug ===")
    
    # 测试1: 简单前向传播
    has_gradients = test_simple_forward()
    
    # 测试2: 调试prompt选择
    debug_prompt_selection()
    
    if not has_gradients:
        print("\n❌ PROBLEM: Hide prompt parameters do not receive gradients!")
        print("Possible causes:")
        print("1. Prompt parameters are detached somewhere in the forward pass")
        print("2. Loss computation doesn't include prompt contributions")
        print("3. Gradient masking or zeroing happening")
        print("4. Incorrect tensor operations that break gradient flow")
    else:
        print("\n✅ SUCCESS: Hide prompt parameters receive gradients!")
