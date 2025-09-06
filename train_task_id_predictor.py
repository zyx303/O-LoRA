#!/usr/bin/env python3
"""
Train Task ID Predictor for HiDe-Prompt

This script trains the task_id predictor using real continual learning data.
It supports both original model-based and learned predictor training.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from datasets import load_dataset, Dataset
import numpy as np

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from peft.tuners.task_id_predictor import TaskIDPredictor, TaskIDPredictorWithOriginalModel
from peft.tuners.task_id_predictor_trainer import (
    TaskIDPredictorTrainer, 
    TaskIDPredictorTrainingConfig,
    TaskIDDataset
)

logger = logging.getLogger(__name__)


class ContinualLearningDataProcessor:
    """Process continual learning datasets for task_id predictor training"""
    
    def __init__(self, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def process_uie_data(self, data_path: str, task_mapping: Dict[str, int]) -> List[Dict[str, Any]]:
        """
        Process UIE (Universal Information Extraction) data
        
        Args:
            data_path: Path to the data directory
            task_mapping: Mapping from task name to task_id
        
        Returns:
            List of processed samples
        """
        processed_data = []
        
        for task_name, task_id in task_mapping.items():
            task_file = os.path.join(data_path, f"{task_name}.json")
            if not os.path.exists(task_file):
                logger.warning(f"Task file not found: {task_file}")
                continue
                
            logger.info(f"Processing task: {task_name} (ID: {task_id})")
            
            with open(task_file, 'r', encoding='utf-8') as f:
                task_data = json.load(f)
            
            for item in task_data:
                # Extract text and tokenize
                text = item.get('text', item.get('input', ''))
                if not text:
                    continue
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                
                processed_data.append({
                    'inputs': inputs,
                    'task_id': task_id,
                    'text': text,
                    'task_name': task_name,
                    'original_data': item
                })
        
        logger.info(f"Processed {len(processed_data)} samples total")
        return processed_data
    
    def process_benchmark_data(self, benchmark_path: str) -> List[Dict[str, Any]]:
        """
        Process CL_Benchmark data
        
        Args:
            benchmark_path: Path to the CL_Benchmark directory
        
        Returns:
            List of processed samples
        """
        processed_data = []
        
        # Load benchmark files
        benchmark_files = list(Path(benchmark_path).glob("*.json"))
        
        for i, file_path in enumerate(benchmark_files):
            logger.info(f"Processing benchmark file: {file_path}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract task name from filename
            task_name = file_path.stem
            task_id = i  # Use file index as task_id
            
            for item in data:
                # Extract text
                text = item.get('text', item.get('input', ''))
                if not text:
                    continue
                
                # Tokenize
                inputs = self.tokenizer(
                    text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                
                processed_data.append({
                    'inputs': inputs,
                    'task_id': task_id,
                    'text': text,
                    'task_name': task_name,
                    'original_data': item
                })
        
        logger.info(f"Processed {len(processed_data)} samples from {len(benchmark_files)} tasks")
        return processed_data
    
    def create_synthetic_data(self, num_tasks: int = 5, samples_per_task: int = 200) -> List[Dict[str, Any]]:
        """Create synthetic data for testing"""
        
        # Define task templates
        task_templates = {
            0: ["Extract entities from: {}", "Find named entities in: {}", "Identify entities: {}"],
            1: ["Classify sentiment: {}", "Sentiment analysis: {}", "What's the sentiment of: {}"],
            2: ["Answer the question: {}", "Q: {} A:", "Question: {}"],
            3: ["Summarize this text: {}", "Summary: {}", "Provide a summary: {}"],
            4: ["Translate to Spanish: {}", "Spanish translation: {}", "Convert to Spanish: {}"]
        }
        
        # Sample texts
        sample_texts = [
            "John works at Microsoft in Seattle.",
            "I love this amazing product!",
            "What is the capital of France?",
            "The weather is beautiful today with sunshine.",
            "Hello world, how are you?",
            "The company reported strong quarterly results.",
            "This movie is terrible and boring.",
            "When was the Declaration of Independence signed?",
            "Climate change affects global weather patterns.",
            "Good morning, nice to meet you."
        ]
        
        processed_data = []
        
        for task_id in range(num_tasks):
            templates = task_templates.get(task_id, ["Process: {}"])
            
            for _ in range(samples_per_task):
                # Random template and text
                template = np.random.choice(templates)
                text = np.random.choice(sample_texts)
                full_text = template.format(text)
                
                # Tokenize
                inputs = self.tokenizer(
                    full_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                # Remove batch dimension
                inputs = {k: v.squeeze(0) for k, v in inputs.items()}
                
                processed_data.append({
                    'inputs': inputs,
                    'task_id': task_id,
                    'text': full_text,
                    'task_name': f"task_{task_id}",
                    'template': template,
                    'original_text': text
                })
        
        logger.info(f"Created {len(processed_data)} synthetic samples")
        return processed_data


def setup_logging(output_dir: str, level: str = "INFO"):
    """Setup logging configuration"""
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, "training.log")
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )


def load_original_model(model_name: str, device: str):
    """Load original model for feature extraction"""
    try:
        logger.info(f"Loading original model: {model_name}")
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name, config=config)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        logger.error(f"Failed to load original model: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Train Task ID Predictor")
    
    # Data arguments
    parser.add_argument("--data_path", type=str, help="Path to training data")
    parser.add_argument("--data_type", type=str, choices=["uie", "benchmark", "synthetic"], 
                       default="synthetic", help="Type of data to use")
    parser.add_argument("--task_mapping_file", type=str, help="JSON file with task name to ID mapping")
    parser.add_argument("--num_tasks", type=int, default=5, help="Number of tasks")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="t5-small", 
                       help="Base model name for tokenizer and features")
    parser.add_argument("--predictor_type", type=str, choices=["learned", "original_model"], 
                       default="learned", help="Type of predictor to train")
    parser.add_argument("--use_original_model", action="store_true", 
                       help="Use original model for feature extraction")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, default="./task_id_predictor_output",
                       help="Output directory for checkpoints and logs")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--use_attention", action="store_true", help="Use attention in predictor")
    parser.add_argument("--use_focal_loss", action="store_true", help="Use focal loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1, help="Label smoothing")
    
    # Other arguments
    parser.add_argument("--max_length", type=int, default=128, help="Maximum sequence length")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--log_level", type=str, default="INFO", help="Logging level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup logging
    setup_logging(args.output_dir, args.log_level)
    
    logger.info("Starting Task ID Predictor Training")
    logger.info(f"Arguments: {args}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load original model if needed
    original_model = None
    if args.use_original_model or args.predictor_type == "original_model":
        original_model = load_original_model(args.model_name, device)
        if original_model is None:
            logger.error("Failed to load original model, falling back to learned predictor")
            args.predictor_type = "learned"
    
    # Process data
    data_processor = ContinualLearningDataProcessor(tokenizer, args.max_length)
    
    if args.data_type == "uie" and args.data_path and args.task_mapping_file:
        # Load task mapping
        with open(args.task_mapping_file, 'r') as f:
            task_mapping = json.load(f)
        processed_data = data_processor.process_uie_data(args.data_path, task_mapping)
        args.num_tasks = len(task_mapping)
        
    elif args.data_type == "benchmark" and args.data_path:
        processed_data = data_processor.process_benchmark_data(args.data_path)
        # Update num_tasks based on actual data
        task_ids = set(item['task_id'] for item in processed_data)
        args.num_tasks = len(task_ids)
        
    else:
        # Use synthetic data
        logger.info("Using synthetic data for training")
        processed_data = data_processor.create_synthetic_data(args.num_tasks, 200)
    
    if not processed_data:
        logger.error("No data processed, exiting")
        return
    
    # Create model
    if args.predictor_type == "learned":
        # Determine input dimension
        if original_model:
            input_dim = original_model.config.hidden_size
        else:
            input_dim = 768  # Default
            
        model = TaskIDPredictor(
            input_dim=input_dim,
            num_tasks=args.num_tasks,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            use_attention=args.use_attention
        )
    else:
        model = TaskIDPredictorWithOriginalModel(
            original_model=original_model,
            num_tasks=args.num_tasks
        )
    
    logger.info(f"Created {args.predictor_type} predictor with {args.num_tasks} tasks")
    
    # Create training config
    config = TaskIDPredictorTrainingConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        use_attention=args.use_attention,
        use_original_model_features=args.use_original_model,
        use_focal_loss=args.use_focal_loss,
        label_smoothing=args.label_smoothing,
        validation_split=args.validation_split,
        output_dir=args.output_dir
    )
    
    # Create trainer
    trainer = TaskIDPredictorTrainer(
        config=config,
        model=model,
        original_model=original_model,
        num_tasks=args.num_tasks,
        device=device
    )
    
    # Prepare data
    train_dataset, val_dataset = trainer.prepare_data(processed_data)
    
    # Train
    logger.info("Starting training...")
    history = trainer.train(train_dataset, val_dataset)
    
    # Save final results
    results = {
        'args': vars(args),
        'best_val_accuracy': max(history['val_accuracy']),
        'final_train_accuracy': history['train_accuracy'][-1],
        'final_val_accuracy': history['val_accuracy'][-1],
        'num_samples': len(processed_data),
        'num_tasks': args.num_tasks
    }
    
    results_file = os.path.join(args.output_dir, "training_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Training completed successfully!")
    logger.info(f"Best validation accuracy: {results['best_val_accuracy']:.4f}")
    logger.info(f"Results saved to: {results_file}")


if __name__ == "__main__":
    main()
