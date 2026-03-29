#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
import math
from copy import deepcopy
import logging
import os
import sys
import json
import time
from dataclasses import dataclass, field
from typing import Optional
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
import torch
from peft.tuners.inflora import LoraLayer as InfLoraLayer

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSeq2SeqLM,
    AutoModelForCausalLM,  # add
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    set_seed, )
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig  # add
from peft import SDLoraConfig  # new
from peft import L2PConfig  # new
from peft import PeftType  # new
from peft import HidePromptConfig  # new
from peft import InfLoRAConfig
from peft.utils.save_and_load import (
    set_hide_prompt_task_id,
    get_hide_prompt_task_id,
    update_hide_prompt_after_task,
)
from uie_collator import DataCollatorForUIE
from uie_dataset_lora import gen_cache_path

from uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions, cls_mean, cls_cov, features_all
from compute_metrics import compute_metrics, compute_grouped_metrics
from model.llama import LlamaForCausalLM_with_lossmask


# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)

# try:
#     nltk.data.find("tokenizers/punkt")
# except (LookupError, OSError):
#     if is_offline_mode():
#         raise LookupError(
#             "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
#         )
#     with FileLock(".lock") as lock:
#         nltk.download("punkt", quiet=True)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
                    "with private models)."
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                    "the model's position embeddings."
        },
    )
    # added for AutoCL
    lora_dim: Optional[int] = field(
        default=8,
        metadata={
            "help": "Intrinsic dimension of the latent space."
        },
    )
    peft_type: Optional[str] = field(
        default="LORA",
        metadata={"help": "PEFT adapter type: LORA or SDLORA"},
    )
    num_virtual_tokens: Optional[int] = field(
        default=20,
        metadata={"help": "The number of virtual tokens to use for the task."}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    lang: str = field(default=None, metadata={"help": "Language id for multilingual model."})
    data_dir: str = field(
        default=None, metadata={"help": "The directory for saving the UIE train/dev/test splits."}
    )
    task_config_dir: str = field(
        default=None, metadata={"help": "The json file for config training and testing tasks"}
    )
    instruction_file: str = field(
        default=None, metadata={"help": "The instruction file for different tasks."}
    )
    instruction_strategy: Optional[str] = field(
        default='single', metadata={
            "help": "How many different instructions to use? Support 'single' and 'multiple' mode."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    input_record_file: str = field(
        default=None, metadata={"help": "file to record model input"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    # for decoder model, it means max_new_tokens
    max_target_length: Optional[int] = field(
        default=50,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    repetition_penalty: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Penalty for repeat tokens in decode stage."
        },
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                    "which is used during ``evaluate`` and ``predict``."
        },
    )
    max_num_instances_per_task: int = field(
        default=10000, metadata={"help": "The maximum number of instances we will consider for each training task."}
    )
    max_num_instances_per_eval_task: int = field(
        default=200,
        metadata={"help": "The maximum number of instances we will consider for each validation/test task."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                    "value if set."
        },
    )
    num_examples: Optional[int] = field(
        default=0,
        metadata={"help": "number of in-context positive examples."}
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    add_task_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend task name before the task input."}
    )
    add_dataset_name: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to preappend dataset name before the task input."}
    )
    

@dataclass
class UIETrainingArguments(Seq2SeqTrainingArguments):
    gradient_checkpointing: Optional[bool] = field(
        default=False,
        metadata={"help": "Whether to use computing time to gain more memory"}
    )
    denser_evaluation: Optional[bool] = field(
        default=False,
        metadata={"help": "If specifid, the model will do more evaluation at the beginning of training."}
    )
    do_demo: bool = field(default=False, metadata={"help": "Whether to run the model as a demo in the terminal."})
    lamda_1: float = field(default = 0.5)
    lamda_2: float = field(default = 0)
    regularization: bool = field(default=False)
    # L2P continual learning parameters
    pool_size: int = field(default=10, metadata={"help": "Size of the L2P prompt pool"})
    l2p_top_k: int = field(default=5, metadata={"help": "Number of top prompts to select in L2P"})
    l2p_task_id: Optional[int] = field(default=None, metadata={"help": "Current task ID for L2P continual learning"})
    l2p_num_classes: Optional[int] = field(default=None, metadata={"help": "Number of classes in current task"})
    l2p_known_classes: int = field(default=0, metadata={"help": "Number of known classes from previous tasks"})
    pull_constraint: bool = field(default=True, metadata={"help": "Whether to use pull constraint in L2P"})
    pull_constraint_coeff: float = field(default=0.1, metadata={"help": "Pull constraint coefficient for L2P"})
    logging_strategy: str = field(default="steps", metadata={"help": "Log strategy to use."})
    logging_steps: int = field(default=10)
    # HiDe-Prompt continual learning parameters
    hide_task_id: Optional[int] = field(default=None, metadata={"help": "Current task ID for HiDe-Prompt CL"})
    prompt_momentum: float = field(default=0.01, metadata={"help": "Momentum for post-task HiDe prompt update [0-1]"})
    # Task adaptive prediction parameters
    not_train_ca: bool = field(default=False, metadata={"help": "Whether to skip task adaptive prediction training"})
    crct_epochs: int = field(default=5, metadata={"help": "Number of epochs for task adaptive prediction training"})
    ca_lr: float = field(default=1e-4, metadata={"help": "Learning rate for task adaptive prediction training"})
    ca_storage_efficient_method: str = field(default="covariance", metadata={"help": "Method for storing class statistics: covariance, variance, or multi-centroid"})
    n_centroids: int = field(default=5, metadata={"help": "Number of centroids for multi-centroid method"})

@torch.no_grad()
def _compute_mean(model: torch.nn.Module, device: torch.device, task_id,
                  args=None, method='covariance'):
    """
    计算当前任务的类别统计信息，使用trainer中已收集的features_all
    
    Args:
        model: 模型（未使用，保持接口一致）
        device: 计算设备
        task_id: 任务ID
        args: 训练参数
        method: 统计方法 ('covariance', 'variance', 'multi-centroid')
    """
    # # 只在rank 0进程上计算
    # if args.world_size > 1 and args.local_rank != 0:
    #     return
    if not features_all:
        print(f"Warning: features_all is empty for task {task_id}")
        return
    
    # 合并所有收集的特征
    features_per_cls = torch.cat([f for f in features_all if f is not None], dim=0)
    
    # 分布式训练：收集所有进程的特征
    if args.world_size > 1:
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]
        torch.distributed.barrier()
        torch.distributed.all_gather(features_per_cls_list, features_per_cls)
        features_per_cls = torch.cat(features_per_cls_list, dim=0)
    
    print(f"Computing class statistics for task {task_id} using {features_per_cls.shape[0]} features with method '{method}'")
    
    if method == 'covariance':
        cls_mean[task_id] = features_per_cls.mean(dim=0)
        # TODO 
        # cls_cov[task_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[task_id].shape[-1]) * 1e-4).to(device)
        
    elif method == 'variance':
        cls_mean[task_id] = features_per_cls.mean(dim=0)
        cls_cov[task_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[task_id].shape[-1]) * 1e-4).to(device))
        
    elif method == 'multi-centroid':
        import numpy as np
        from sklearn.cluster import KMeans
        
        n_clusters = args.n_centroids
        features_numpy = features_per_cls.cpu().numpy()
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(features_numpy)
        cluster_labels = kmeans.labels_
        
        cluster_means = []
        cluster_vars = []
        
        for i in range(n_clusters):
            cluster_mask = (cluster_labels == i)
            if np.sum(cluster_mask) > 0:  # 确保簇不为空
                cluster_data = features_numpy[cluster_mask]
                cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float32).to(device)
                cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float32).to(device)
                cluster_means.append(cluster_mean)
                cluster_vars.append(cluster_var)
        
        cls_mean[task_id] = cluster_means
        cls_cov[task_id] = cluster_vars
        
    print(f"Task {task_id} statistics computed. cls_mean keys: {list(cls_mean.keys())}")
    
    # 清空features_all为下一个任务准备
    features_all.clear()

def train_task_adaptive_prediction(model, tokenizer, args, device, task_id=-1, method='covariance'):
    """
    HiDe-Prompt版本的任务自适应预测训练
    
    Args:
        model: HiDe-Prompt模型
        tokenizer: 分词器 
        args: 训练参数
        device: 计算设备
        task_id: 当前任务ID
        method: 统计方法 ('covariance', 'variance', 'multi-centroid')
    """
    # 只在rank 0进程上执行
    if args.world_size > 1 and args.local_rank != 0:
        return
        
    if not cls_mean or task_id == 0:
        print(f"Skipping task adaptive prediction for task {task_id} (no previous tasks or missing statistics)")
        return
    
    print(f"Starting task adaptive prediction training for task {task_id}")
    
    model.train()
    
    # 训练参数设置
    run_epochs = args.crct_epochs
    ca_lr = args.ca_lr
    batch_size = args.per_device_train_batch_size
    num_sampled_per_task = batch_size * 5  # 每个任务采样的数据量
    
    # 获取可训练参数（排除HiDe-Prompt相关参数，只训练分类头）
    param_list = []
    for name, param in model.named_parameters():
        if param.requires_grad and not any(key in name for key in ["prompt", "prompt_key"]):
            param_list.append(param)
    
    if not param_list:
        print("No trainable parameters found for task adaptive prediction")
        return
    
    # 设置优化器和调度器
    network_params = [{'params': param_list, 'lr': ca_lr, 'weight_decay': args.weight_decay}]
    optimizer = torch.optim.AdamW(network_params, lr=ca_lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    print(f"Task adaptive prediction - Epochs: {run_epochs}, LR: {ca_lr}, Trainable params: {len(param_list)}")
    
    for epoch in range(run_epochs):
        print(f"Task adaptive prediction - Epoch {epoch + 1}/{run_epochs}")
        
        # 为每个之前的任务生成合成数据
        all_sampled_features = []
        all_sampled_labels = []
        
        for prev_task_id in range(task_id):
            if prev_task_id not in cls_mean:
                continue
                
            if method in ['covariance', 'variance']:
                mean = cls_mean[prev_task_id][0]
                cov = cls_cov[prev_task_id][0]
                
                if method == 'variance':
                    cov = torch.diag(cov)
                
                from torch.distributions.multivariate_normal import MultivariateNormal
                m = MultivariateNormal(mean.float(), cov.float())
                sampled_features = m.sample(sample_shape=(num_sampled_per_task,))
                
                all_sampled_features.append(sampled_features)
                all_sampled_labels.extend([prev_task_id] * num_sampled_per_task)
                    
            elif method == 'multi-centroid':
                cluster_means = cls_mean[prev_task_id][0]   
                cluster_vars = cls_cov[prev_task_id][0]
                
                for cluster_idx, (cluster_mean, cluster_var) in enumerate(zip(cluster_means, cluster_vars)):
                    if cluster_var.mean() == 0:
                        continue
                        
                    try:
                        cov_matrix = torch.diag(cluster_var) + 1e-4 * torch.eye(cluster_mean.shape[0]).to(cluster_mean.device)
                        m = MultivariateNormal(cluster_mean.float(), cov_matrix.float())
                        sampled_features = m.sample(sample_shape=(num_sampled_per_task // len(cluster_means),))
                        
                        all_sampled_features.append(sampled_features)
                        all_sampled_labels.extend([prev_task_id] * (num_sampled_per_task // len(cluster_means)))
                        
                    except Exception as e:
                        print(f"Error sampling cluster {cluster_idx} for task {prev_task_id}: {e}")
                        continue
        
        if not all_sampled_features:
            print(f"No valid sampled features for epoch {epoch}, skipping")
            continue
            
        # 合并所有采样的特征
        sampled_features = torch.cat(all_sampled_features, dim=0).float().to(device)
        sampled_labels = torch.tensor(all_sampled_labels).long().to(device)
        
        # 随机打乱数据
        shuffle_indices = torch.randperm(sampled_features.size(0))
        sampled_features = sampled_features[shuffle_indices]
        sampled_labels = sampled_labels[shuffle_indices]
        
        print(f"Generated {sampled_features.shape[0]} synthetic samples for {len(set(all_sampled_labels))} previous tasks")
        
        # 分批训练
        num_batches = (sampled_features.size(0) + batch_size - 1) // batch_size
        epoch_loss = 0.0
        epoch_acc = 0.0
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, sampled_features.size(0))
            
            batch_features = sampled_features[start_idx:end_idx]
            batch_labels = sampled_labels[start_idx:end_idx]
            
            # 使用模型的forward方法，但只计算logits
            try:
                # 这里我们直接使用特征通过模型的分类头或语言建模头
                
                # 首先尝试直接使用features作为hidden states
                if hasattr(model, 'base_model') and hasattr(model.base_model, 'lm_head'):
                    # 对于因果语言模型，直接通过lm_head
                    logits = model.base_model.lm_head(batch_features)
                elif hasattr(model, 'lm_head'):
                    # 直接通过lm_head
                    logits = model.lm_head(batch_features)
                elif hasattr(model, 'classifier'):
                    # 对于分类模型
                    logits = model.classifier(batch_features)
                # else:
                #     # 尝试使用inputs_embeds方式
                #     # 为每个特征创建一个假的token序列
                #     batch_size = batch_features.size(0)
                #     seq_len = 1  # 使用单个token
                #     inputs_embeds = batch_features.unsqueeze(1)  # [batch_size, 1, hidden_size]
                    
                #     # 创建attention mask
                #     attention_mask = torch.ones(batch_size, seq_len, device=device)
                    
                #     outputs = model(
                #         inputs_embeds=inputs_embeds,
                #         attention_mask=attention_mask,
                #         return_dict=True
                #     )
                #     logits = outputs.logits
                    
                #     # 取最后一个token的logits
                #     if logits.dim() == 3:  # [batch_size, seq_len, vocab_size]
                #         logits = logits[:, -1, :]  # 取最后一个token
                
                # 确保logits维度正确
                if logits.dim() > 2:
                    logits = logits.view(-1, logits.size(-1))
                
                # 如果logits的词汇表大小与任务数不匹配，我们需要映射,TODO 不合理
                num_tasks = max(all_sampled_labels) + 1
                if logits.size(-1) != num_tasks:
                    # 使用简单的线性变换映射到正确的任务数
                    # 这里我们只使用logits的前num_tasks个维度，或者进行平均池化
                    if logits.size(-1) > num_tasks:
                        # 如果logits维度大于任务数，取前num_tasks个
                        logits = logits[:, :num_tasks]
                    else:
                        # 如果logits维度小于任务数，重复填充或使用线性层
                        # 为了简单起见，我们使用平均值来填充缺失的维度
                        padding_size = num_tasks - logits.size(-1)
                        padding = torch.mean(logits, dim=-1, keepdim=True).repeat(1, padding_size)
                        logits = torch.cat([logits, padding], dim=-1)
                
                loss = criterion(logits, batch_labels)
                
                # 计算准确率
                pred = torch.argmax(logits, dim=-1)
                acc = (pred == batch_labels).float().mean()
                
                # 反向传播
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_acc += acc.item()
                
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                print(f"Model type: {type(model)}")
                print(f"Batch features shape: {batch_features.shape}")
                print(f"Available model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}")
                continue
        
        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_acc = epoch_acc / num_batches if num_batches > 0 else 0.0
        
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f}, Acc: {avg_acc:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        scheduler.step()
    
    print(f"Task adaptive prediction training completed for task {task_id}")

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, UIETrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    
    # get task id 
    import re, os
    m = re.match(r"^(\d+)-", os.path.basename(os.path.normpath(training_args.output_dir)))
    task_id = int(m.group(1)) - 1 if m else None
    print('task_id：', task_id)

    m_name = re.match(r"^\d+-(.*)", os.path.basename(os.path.normpath(training_args.output_dir)))
    task_name = m_name.group(1) if m_name else None
    print('task_name：', task_name)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    data_cache_dir = gen_cache_path(training_args.output_dir, data_args)

    # Get the UIE dataset
    raw_datasets = load_dataset(
        os.path.join(CURRENT_DIR, "uie_dataset_lora.py"),
        data_dir=data_args.data_dir,
        task_config_dir=data_args.task_config_dir,
        instruction_file=data_args.instruction_file,
        instruction_strategy=data_args.instruction_strategy,
        cache_dir=data_cache_dir,  # for debug, change dataset size, otherwise open it
        max_num_instances_per_task=data_args.max_num_instances_per_task,
        max_num_instances_per_eval_task=data_args.max_num_instances_per_eval_task,
        num_examples=data_args.num_examples
    )
    raw_datasets.cleanup_cache_files()

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    if 'adapter' in model_args.model_name_or_path: # load lora-config
        config = PeftConfig.from_pretrained(model_args.model_name_or_path)
        if 'llama' in model_args.model_name_or_path.lower():
            tokenizer = transformers.LlamaTokenizer.from_pretrained(config.base_model_name_or_path)
            config.bos_token_id = 1
            config.eos_token_id = 2
            config.pad_token_id = 1
            tokenizer.bos_token_id = 1
            tokenizer.eos_token_id = 2
            tokenizer.pad_token_id = 1
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        config.bos_token_id = 1
        config.eos_token_id = 2
        config.pad_token_id = 1
        tokenizer = transformers.LlamaTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir = model_args.cache_dir,
            use_fast = model_args.use_fast_tokenizer,
            revision = model_args.model_revision,
            use_auth_token = True if model_args.use_auth_token else None,
        )
        tokenizer.bos_token_id = 1
        tokenizer.eos_token_id = 2
        tokenizer.pad_token_id = 1
    else: # load original config
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

    if 'llama' in model_args.model_name_or_path.lower():  # add llama
        model_class = LlamaForCausalLM_with_lossmask
        tokenizer.padding_side = 'left'
    else: 
        model_class = AutoModelForSeq2SeqLM
        
    if 'adapter' in model_args.model_name_or_path: # add lora-adapter to the original model
        model = model_class.from_pretrained(config.base_model_name_or_path)
        # 先调整token embeddings大小，再加载adapter
        model.resize_token_embeddings(len(tokenizer))
        print(f"len(tokenizer): {len(tokenizer)}")
        model = PeftModel.from_pretrained(model, model_args.model_name_or_path)
    elif 'llama' in model_args.model_name_or_path.lower():
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None
        )
        print(f"Using HiDe-Prompt with {model_args.num_virtual_tokens} virtual tokens.")
        peft_config = HidePromptConfig(
            num_virtual_tokens=model_args.num_virtual_tokens,
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            prompt_key=False,
            pool_size=10,  # 明确设置prompt pool大小
            top_k=1
        )
        model = get_peft_model(model, peft_config)
    else:
        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
        print(f"Using HiDe-Prompt with {model_args.num_virtual_tokens} virtual tokens.")
        peft_config = HidePromptConfig(
            num_virtual_tokens=model_args.num_virtual_tokens,
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            prompt_key=False,
            pool_size=10,  # 明确设置prompt pool大小
            top_k=1
        )
        model = get_peft_model(model, peft_config)
        with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
            json.dump(training_args.to_dict(), f, indent=4)
        # 对于新训练的模型，也需要调整token embeddings大小
        model.resize_token_embeddings(len(tokenizer))
    # 注意：对于从adapter加载的模型，resize_token_embeddings已经在加载前执行了
    print(f"len(tokenizer): {len(tokenizer)}")
    # If using HiDe-Prompt, set task_id based on continual learning order
    # try:
    #     if model_args.peft_type.upper() == "HIDE_PROMPT":
    #         if training_args.hide_task_id is not None:
    #             # 使用命令行指定的task_id
    #             task_id = int(training_args.hide_task_id)
    #         else:
    #             # 从配置目录自动推断task_id
    #             from task_mapping import get_task_order_from_config
    #             task_id = get_task_order_from_config(data_args.task_config_dir)
    #             if task_id is None:
    #                 task_id = 0  # 默认值
    #                 logger.warning(f"Could not determine task_id from config '{data_args.task_config_dir}', using default: {task_id}")
    #             else:
    #                 logger.info(f"Auto-determined task_id from config '{data_args.task_config_dir}': {task_id}")
            
    set_hide_prompt_task_id(model, task_id)
    #         logger.info(f"Successfully set HiDe-Prompt task_id to: {task_id} (config: {data_args.task_config_dir})")
    # except Exception as _e:
    #     logger.warning(f"Failed to set HiDe task id pre-training: {_e}")

    if 'llama' in model_args.model_name_or_path.lower():
        model.generation_config.bos_token_id = 1
        model.generation_config.eos_token_id = 2
        model.generation_config.pad_token_id = 1
        
    # fix lora_A/B (bases of previous LoRA parameters, loaded in "load_adapter"[peft_momdel.py])
    # fine-tune loranew_A/B (initialized in "update_layer"[lora.py])
    # optional: lora_A/B is trainable but should not move too far from lorapre_A/B
    # (constrained in "training_step"[uie_trainer_lora.py])
    # for name, param in model.named_parameters():
    #     if name.find("loranew_") != -1:
    #         # for Inflora, only train loranew_B
    #         if model_args.peft_type.upper() == "INFLORA" and name.find("loranew_B") != -1:
    #             param.requires_grad = True
    #         elif model_args.peft_type.upper() == "INFLORA" and name.find("loranew_A") != -1:
    #             param.requires_grad = False
    #         else:
    #             param.requires_grad = True
    #     elif name.find("lora_") != -1:
    #         param.requires_grad = False
    #     elif name.find("shared_historical_scalings") != -1:
    #         param.requires_grad = True
    #     # this module should always be frozen because we change the vocabulary
    #     elif name.find("shared") != -1:
    #         param.requires_grad = False
    #     elif name.find("historical_directions") != -1:
    #         param.requires_grad = False
    #     elif name.find("scaling") != -1:
    #         param.requires_grad = True
        

    # # L2P specific parameter freezing
    # if model_args.peft_type.upper() == "L2P":
    #     print("Applying L2P parameter freezing...")
    #     trainable_params = 0
    #     frozen_params = 0
        
    #     for name, param in model.named_parameters():
    #         # Only allow L2P prompt pool and prompt key parameters to be trainable
    #         if any(key in name for key in ["prompt", "prompt_key"]):
    #             param.requires_grad = True
    #             trainable_params += param.numel()
    #             print(f"  Trainable: {name} ({param.numel()} params)")
    #         # Also allow classification head to be trainable if it exists
    #         elif any(key in name for key in ["classifier", "lm_head", "score"]):
    #             param.requires_grad = True
    #             trainable_params += param.numel()
    #             print(f"  Trainable: {name} ({param.numel()} params)")
    #         else:
    #             param.requires_grad = False
    #             frozen_params += param.numel()
        
    #     print(f"L2P Parameter Summary:")
    #     print(f"  Trainable parameters: {trainable_params:,}")
    #     print(f"  Frozen parameters: {frozen_params:,}")
    #     print(f"  Trainable percentage: {100 * trainable_params / (trainable_params + frozen_params):.2f}%")

    # HiDe-Prompt specific parameter freezing
    if model_args.peft_type.upper() == "HIDE_PROMPT":
        print("Applying HiDe-Prompt parameter freezing...")
        trainable_params = 0
        frozen_params = 0
        
        for name, param in model.named_parameters():
            # Only allow HiDe-Prompt prompt pool and prompt key parameters to be trainable
            if any(key in name for key in ["prompt", "prompt_key"]):
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"  Trainable: {name} ({param.numel()} params)")
            # Also allow classification head to be trainable if it exists
            elif any(key in name for key in ["classifier", "lm_head", "score","shared"]):
                param.requires_grad = True
                trainable_params += param.numel()
                print(f"  Trainable: {name} ({param.numel()} params)")
            else:
                param.requires_grad = False
                frozen_params += param.numel()
        
        print(f"HiDe-Prompt Parameter Summary:")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Frozen parameters: {frozen_params:,}")
        print(f"  Trainable percentage: {100 * trainable_params / (trainable_params + frozen_params):.2f}%")

    if (
            hasattr(model.config, "max_position_embeddings")
            and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))

    if training_args.do_eval:
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            eval_dataset = eval_dataset.select(range(data_args.max_eval_samples))

    if training_args.do_predict:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForUIE(
        tokenizer,
        model=model,
        padding="longest",
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
        add_task_name=data_args.add_task_name,
        add_dataset_name=data_args.add_dataset_name,
        num_examples=data_args.num_examples,
        input_record_file=data_args.input_record_file
    )
    # we don't want to remove unused columns because we will prepare each batch during training,
    # and some of the information will also be used in evaluation.
    training_args.remove_unused_columns = False

    # Metric
    def compute_rouge_metrics(dataset, preds, save_prefix=None):
        decoded_preds = skip_instructions(model, preds, tokenizer)
        references = [e["Instance"]["label"] for e in dataset]
        result = compute_metrics(predictions=decoded_preds, references=references)
        result_per_task = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                  groups=dataset["Task"])
        result.update(result_per_task)
        categories = dataset["Dataset"]
        result_per_category = compute_grouped_metrics(predictions=decoded_preds, references=references,
                                                      groups=categories)
        result.update(result_per_category)
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        if save_prefix is not None:
            with open(os.path.join(training_args.output_dir, f"{save_prefix}_eval_predictions.jsonl"), "w") as fout:
                for example, pred in zip(dataset, decoded_preds):
                    fout.write(json.dumps({
                        "Task": example["Task"],
                        "Dataset": example["Dataset"],
                        "Instance": example["Instance"],
                        "Prediction": pred
                    }) + "\n")
        return result

    print(f"-----Gradient checkpointing: {training_args.gradient_checkpointing} -----")
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if model_args.peft_type.upper() == "L2P":
        training_args.regularization = False
    if model_args.peft_type.upper() == "LORA":
        training_args.regularization = True
        print("Using LoRA with regularization on loranew_A and loranew_B")
    trainer = UIETrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_rouge_metrics,
        callbacks=[DenserEvalCallback] if training_args.denser_evaluation else None
    )

    all_metrics = {"run_name": training_args.run_name}

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        # if model_args.peft_type.upper() == "SDLORA":
        #     historical_scaling_params = []
        #     other_params = []
            
        #     for name, param in model.named_parameters():
        #         if param.requires_grad:
        #             if "historical_scalings" in name:
        #                 historical_scaling_params.append(param)
        #             else:
        #                 other_params.append(param)
            
        #     # 创建参数组
        #     param_groups = [
        #         {
        #             'params': historical_scaling_params,
        #             'lr': 0.01,
        #             'weight_decay': 0.0
        #         },
        #         {
        #             'params': other_params,
        #             'lr': 1e-3,
        #             'weight_decay': 0.0
        #         }
        #     ]
            
        #     from transformers import AdamW,get_constant_schedule,get_linear_schedule_with_warmup
        #     optimizer = AdamW(param_groups)

        #     scheduler = get_constant_schedule(optimizer)
            
        #     print(f"历史scaling参数数量: {len(historical_scaling_params)}")
        #     print(f"其他参数数量: {len(other_params)}")
        #     # # 创建自定义调度器
        #     # class CustomScheduler:
        #     #     def __init__(self, optimizer, num_training_steps):
        #     #         self.optimizer = optimizer
        #     #         # 其他参数使用 constant scheduler
        #     #         self.scheduler_other = get_constant_schedule(optimizer)
        #     #         # historical_scaling_params 使用线性衰减 scheduler
        #     #         self.scheduler_historical = get_linear_schedule_with_warmup(
        #     #             optimizer, 
        #     #             num_warmup_steps=100,  
        #     #             num_training_steps=num_training_steps
        #     #         )
                    
        #     #     def step(self):
        #     #         current_lr_other = self.scheduler_other.get_last_lr()[0]
        #     #         current_lr_historical = self.scheduler_historical.get_last_lr()[0]
                    
        #     #         self.optimizer.param_groups[0]['lr'] = current_lr_other
        #     #         self.optimizer.param_groups[1]['lr'] = current_lr_historical

        #     #         self.scheduler_other.step()
        #     #         self.scheduler_historical.step()
                    
        #     #     def get_last_lr(self):
        #     #         return [self.scheduler_other.get_last_lr()[0], self.scheduler_historical.get_last_lr()[0]]
        
        #     # # 计算训练步数
        #     # total_steps = len(trainer.get_train_dataloader()) * training_args.num_train_epochs
        #     # custom_scheduler = CustomScheduler(optimizer, total_steps)

        #     # 使用线性预热和线性衰减的调度器
        #     # num_training_steps = len(trainer.get_train_dataloader()) * training_args.num_train_epochs
        #     # print('num_training_steps: ', num_training_steps)
        #     # custom_scheduler = get_linear_schedule_with_warmup(
        #     #     optimizer,
        #     #     num_warmup_steps=20,
        #     #     num_training_steps=num_training_steps
        #     # )
            
        #     trainer.optimizer = optimizer
        #     trainer.lr_scheduler = scheduler
            # trainer.lr_scheduler = custom_scheduler

        # transfer previous prompt key if not the first task
        if task_id > 0:
            cur_idx = (slice(None), slice(None), slice(task_id, task_id+1))
            prev_idx = (slice(None), slice(None),slice(task_id-1, task_id))
            if model.prompt_encoder[model.active_adapter].prompt.grad is not None:
                model.prompt_encoder[model.active_adapter].prompt.grad.zero_()
            with torch.no_grad():
                model.prompt_encoder[model.active_adapter].prompt[cur_idx] = model.prompt_encoder[model.active_adapter].prompt[prev_idx]

        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        # For HiDe-Prompt: apply momentum update on e_prompt after finishing this task, then save
        try:
            if model_args.peft_type.upper() == "HIDE_PROMPT":
                base_model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
                cur_task_id = (
                    int(training_args.hide_task_id)
                    if training_args.hide_task_id is not None
                    else int(get_hide_prompt_task_id(base_model))
                )
                if update_hide_prompt_after_task(base_model, cur_task_id, float(training_args.prompt_momentum)):
                    print(
                        f"HiDe-Prompt: updated e_prompt for task {cur_task_id} with momentum={training_args.prompt_momentum}"
                    )
            
            # 更新mean和cov
            _compute_mean(model, training_args.device, task_id, args=training_args, method=training_args.ca_storage_efficient_method)
            
            # # 如果不是第一个任务，执行任务自适应预测训练
            # if task_id > 0 and not training_args.not_train_ca:
            #     print(f"Starting task adaptive prediction training for task {task_id}")
            #     train_task_adaptive_prediction(
            #         model=trainer.model,
            #         tokenizer=tokenizer,
            #         args=training_args,
            #         device=training_args.device,
            #         task_id=task_id,
            #         method=training_args.ca_storage_efficient_method
            #     )

                
        except Exception as _e:
            logger.warning(f"HiDe-Prompt post-task update failed: {_e}")

        peft_model_id = training_args.output_dir + "/adapter"
        trainer.model.save_pretrained(peft_model_id)  
        tokenizer.save_pretrained(peft_model_id)

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Metrics {metrics}")
        all_metrics.update(metrics)

    # Evaluation
    results = {}
    # in case the batch is shorter than max length, the output should be padded
    max_new_tokens = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.max_target_length
    )

    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    repetition_penalty = data_args.repetition_penalty

    if training_args.do_predict:
        logger.info("*** Prediction ***")
        logger.info("*** Loading CheckPoint ***")

        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))

        predict_results = trainer.predict(
            predict_dataset,
            metric_key_prefix="predict",
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            repetition_penalty=repetition_penalty,
            pad_token_id=tokenizer.pad_token_id
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log(metrics)
        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)
        all_metrics.update(metrics)

    return results

import os
import debugpy
if __name__ == "__main__":
    if os.getenv('debug'):
        debugpy.listen(5678)
        debugpy.wait_for_client()
    main()
