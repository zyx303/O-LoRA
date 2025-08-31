#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate a trained adapter on specified datasets.
"""

import os
import json
import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    HfArgumentParser,
    set_seed,
)
from src.peft import PeftModel
from src.arguments import ModelArguments, DataTrainingArguments, TrainingArguments
from src.uie_trainer_lora import UIETrainer
from src.data_collator import DataCollatorForUIE
from src.dataset import UIEDataset
import deepspeed


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained adapter')
    parser.add_argument('--adapter_path', type=str, required=True, 
                       help='Path to the trained adapter')
    parser.add_argument('--base_model', type=str, default='t5-base',
                       help='Base model name or path')
    parser.add_argument('--data_dir', type=str, default='CL_Benchmark',
                       help='Directory containing the datasets')
    parser.add_argument('--task_config_dir', type=str, required=True,
                       help='Directory containing task configurations')
    parser.add_argument('--instruction_file', type=str, 
                       default='configs/instruction_config.json',
                       help='Path to instruction configuration file')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--max_eval_samples', type=int, default=None,
                       help='Maximum number of evaluation samples')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Evaluation batch size')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda:0, etc.)')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup device for evaluation"""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU")
    else:
        device = torch.device(device_arg)
        print(f"Using specified device: {device}")
    
    return device


def load_model_and_adapter(base_model_path, adapter_path, device):
    """Load base model and adapter"""
    print(f"Loading base model from: {base_model_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    
    # Load base model
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,  # 确保数据类型一致
        device_map="auto" if device.type == 'cuda' else None,
    )
    
    print(f"Loading adapter from: {adapter_path}")
    
    # Load adapter
    model = PeftModel.from_pretrained(
        base_model,
        adapter_path,
        torch_dtype=torch.float32,
    )
    
    # Set to evaluation mode
    model.eval()
    
    return model, tokenizer


def prepare_dataset(data_dir, task_config_dir, instruction_file, tokenizer, max_eval_samples=None):
    """Prepare evaluation dataset"""
    print(f"Preparing dataset from: {data_dir}")
    print(f"Task config dir: {task_config_dir}")
    
    # Create mock arguments for dataset creation
    class MockArgs:
        def __init__(self):
            self.data_dir = data_dir
            self.task_config_dir = task_config_dir
            self.instruction_file = instruction_file
            self.instruction_strategy = "single"
            self.max_source_length = 512
            self.max_target_length = 50
            self.add_task_name = True
            self.add_dataset_name = True
            self.max_eval_samples = max_eval_samples
            self.overwrite_cache = True
    
    args = MockArgs()
    
    # Load evaluation dataset
    eval_dataset = UIEDataset(
        args=args,
        tokenizer=tokenizer,
        data_dir=data_dir,
        data_type="test"  # or "dev" depending on your setup
    )
    
    return eval_dataset


def evaluate_model(model, tokenizer, eval_dataset, batch_size, output_dir):
    """Evaluate the model and save results"""
    print("Starting evaluation...")
    
    # Create data collator
    data_collator = DataCollatorForUIE(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        max_length=512,
    )
    
    # Create mock training arguments for trainer
    class MockTrainingArgs:
        def __init__(self):
            self.output_dir = output_dir
            self.per_device_eval_batch_size = batch_size
            self.dataloader_drop_last = False
            self.dataloader_num_workers = 0
            self.remove_unused_columns = False
            self.label_names = ["labels"]
            self.prediction_loss_only = False
            self.generation_max_length = 50
            self.predict_with_generate = True
            self.include_inputs_for_metrics = True
            self.local_rank = -1
            self.device = model.device
    
    training_args = MockTrainingArgs()
    
    # Create trainer
    trainer = UIETrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    # Run evaluation
    eval_results = trainer.evaluate(eval_dataset=eval_dataset)
    
    return eval_results


def save_results(results, output_dir, adapter_path):
    """Save evaluation results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, "eval_results.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Save summary
    summary = {
        "adapter_path": adapter_path,
        "exact_match": results.get("eval_exact_match", 0),
        "rouge1": results.get("eval_rouge1", 0),
        "rougeL": results.get("eval_rougeL", 0),
        "loss": results.get("eval_loss", 0),
    }
    
    # Extract task-specific metrics
    for key, value in results.items():
        if "exact_match_for_" in key or "rouge1_for_" in key or "rougeL_for_" in key:
            summary[key.replace("eval_", "")] = value
    
    summary_file = os.path.join(output_dir, "eval_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {output_dir}")
    print(f"Summary: {summary}")


def main():
    args = parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Setup device
    device = setup_device(args.device)
    
    # Load model and adapter
    model, tokenizer = load_model_and_adapter(
        args.base_model, 
        args.adapter_path, 
        device
    )
    
    # Prepare dataset
    eval_dataset = prepare_dataset(
        args.data_dir,
        args.task_config_dir,
        args.instruction_file,
        tokenizer,
        args.max_eval_samples
    )
    
    print(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Run evaluation
    results = evaluate_model(
        model,
        tokenizer,
        eval_dataset,
        args.batch_size,
        args.output_dir
    )
    
    # Save results
    save_results(results, args.output_dir, args.adapter_path)
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()