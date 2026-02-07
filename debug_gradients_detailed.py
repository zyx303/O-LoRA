#!/usr/bin/env python3
"""
详细的梯度调试脚本
"""

import torch
import sys
import os
sys.path.append("/root/work/O-LoRA/src")

from peft import get_peft_model, HidePromptConfig, TaskType
from transformers import T5ForConditionalGeneration, T5Tokenizer

def debug_step_by_step():
    """逐步调试前向传播过程"""
    print("\n=== Step-by-Step Gradient Debug ===")
    
    # 初始化模型
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
    model.train()
    
    # 准备输入
    input_text = "translate English to German: Hello world"
    target_text = "Hallo Welt"
    
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    targets = tokenizer(target_text, return_tensors="pt", padding=True, truncation=True, max_length=50)
    
    inputs["labels"] = targets.input_ids
    inputs["task_id"] = 0
    inputs["Dataset"] = ["dbpedia"]
    
    # 获取关键组件
    eprompt = model.prompt_encoder[model.active_adapter]
    print(f"Original prompt requires_grad: {eprompt.prompt.requires_grad}")
    print(f"Original prompt shape: {eprompt.prompt.shape}")
    
    # Step 1: 测试输入嵌入
    print("\n=== Step 1: Input Embeddings ===")
    input_ids = inputs["input_ids"]
    inputs_embeds = model.word_embeddings(input_ids)
    print(f"Inputs embeds shape: {inputs_embeds.shape}")
    print(f"Inputs embeds requires_grad: {inputs_embeds.requires_grad}")
    
    # Step 2: 测试EPrompt输出
    print("\n=== Step 2: EPrompt Output ===")
    model.zero_grad()
    
    # 直接调用eprompt
    out = eprompt(
        x_embed=inputs_embeds,
        dataset_names=["dbpedia"]
    )
    
    batched_prompt = out["batched_prompt"]
    print(f"Batched prompt shape: {batched_prompt.shape}")
    print(f"Batched prompt requires_grad: {batched_prompt.requires_grad}")
    print(f"Batched prompt dtype: {batched_prompt.dtype}")
    print(f"Inputs embeds dtype: {inputs_embeds.dtype}")
    
    # 测试这一步的梯度
    test_loss = batched_prompt.sum()
    test_loss.backward()
    print(f"After EPrompt backward, prompt grad norm: {eprompt.prompt.grad.norm().item() if eprompt.prompt.grad is not None else 'None'}")
    
    # Step 3: 测试dtype转换的影响
    print("\n=== Step 3: Testing dtype conversion ===")
    model.zero_grad()
    
    # 重新获取batched_prompt
    out = eprompt(
        x_embed=inputs_embeds,
        dataset_names=["dbpedia"]
    )
    batched_prompt = out["batched_prompt"]
    
    # 模拟peft_model.py中的dtype转换
    if batched_prompt.dtype != inputs_embeds.dtype:
        print(f"Converting dtype from {batched_prompt.dtype} to {inputs_embeds.dtype}")
        batched_prompt_converted = batched_prompt.to(inputs_embeds.dtype)
    else:
        print("No dtype conversion needed")
        batched_prompt_converted = batched_prompt
    
    print(f"Converted prompt requires_grad: {batched_prompt_converted.requires_grad}")
    
    # 测试转换后的梯度
    test_loss2 = batched_prompt_converted.sum()
    test_loss2.backward()
    print(f"After dtype conversion backward, prompt grad norm: {eprompt.prompt.grad.norm().item() if eprompt.prompt.grad is not None else 'None'}")
    
    # Step 4: 测试K/V重塑的影响
    print("\n=== Step 4: Testing K/V reshaping ===")
    model.zero_grad()
    
    # 重新获取batched_prompt
    out = eprompt(
        x_embed=inputs_embeds,
        dataset_names=["dbpedia"]
    )
    batched_prompt = out["batched_prompt"]
    
    # 模拟peft_model.py中的K/V处理
    num_layers, B, dual, P, H, D = batched_prompt.shape
    print(f"Batched prompt shape breakdown: layers={num_layers}, B={B}, dual={dual}, P={P}, H={H}, D={D}")
    
    # 选择一层进行测试
    li = 0
    k = batched_prompt[li, :, 0]  # (B, P, H, D)
    v = batched_prompt[li, :, 1]  # (B, P, H, D)
    
    print(f"K shape before permute: {k.shape}")
    print(f"K requires_grad before permute: {k.requires_grad}")
    
    # 重塑
    k_reshaped = k.permute(0, 2, 1, 3).contiguous()
    v_reshaped = v.permute(0, 2, 1, 3).contiguous()
    
    print(f"K shape after permute: {k_reshaped.shape}")
    print(f"K requires_grad after permute: {k_reshaped.requires_grad}")
    
    # 测试重塑后的梯度
    test_loss3 = k_reshaped.sum() + v_reshaped.sum()
    test_loss3.backward()
    print(f"After K/V reshape backward, prompt grad norm: {eprompt.prompt.grad.norm().item() if eprompt.prompt.grad is not None else 'None'}")
    
    # Step 5: 测试完整的past_key_values构建
    print("\n=== Step 5: Testing past_key_values construction ===")
    model.zero_grad()
    
    # 准备完整的前向传播，但插入梯度检查点
    decoder_input_ids = targets.input_ids
    decoder_inputs_embeds = model.word_embeddings(decoder_input_ids)
    
    # 使用模型内部逻辑构建past_key_values
    print("Testing the complete forward pass...")
    
    try:
        # 完整前向传播
        outputs = model(**inputs)
        loss = outputs.loss
        print(f"Forward pass successful, loss: {loss.item()}")
        
        # 反向传播
        loss.backward()
        
        # 检查梯度
        if eprompt.prompt.grad is not None:
            grad_norm = eprompt.prompt.grad.norm().item()
            grad_nonzero = torch.any(eprompt.prompt.grad != 0).item()
            print(f"Final gradient norm: {grad_norm}")
            print(f"Final gradient nonzero: {grad_nonzero}")
        else:
            print("Final gradient: None")
            
    except Exception as e:
        print(f"Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_step_by_step()
