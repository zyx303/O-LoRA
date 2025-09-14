
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
from src.peft.tuners.inflora import LoraLayer as InfLoraLayer

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
from src.peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig  # add
from src.peft import SDLoraConfig  # new
from src.peft import L2PConfig  # new
from src.peft import PeftType  # new
from src.peft import HidePromptConfig  # new
from src.peft import InfLoRAConfig
from src.peft.utils.save_and_load import (
    set_hide_prompt_task_id,
    get_hide_prompt_task_id,
    update_hide_prompt_after_task,
)
from src.uie_collator import DataCollatorForUIE
from src.uie_dataset_lora import gen_cache_path

from src.uie_trainer_lora import UIETrainer, DenserEvalCallback, skip_instructions
from src.compute_metrics import compute_metrics, compute_grouped_metrics
from src.model.llama import LlamaForCausalLM_with_lossmask

# off wandb
os.environ['WANDB_DISABLED'] = "True"
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logger = logging.getLogger(__name__)
CURRENT_DIR = os.path.dirname(__file__)


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


#load 
if 'adapter' in model_args.model_name_or_path: # add lora-adapter to the original model
    model = model_class.from_pretrained(config.base_model_name_or_path)
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
    if model_args.peft_type.upper() == "SDLORA":
        peft_config = SDLoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1,
            save_loranew=True,
        )
    elif model_args.peft_type.upper() == "INFLORA":
        peft_config = InfLoRAConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=model_args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1
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
    if model_args.peft_type.upper() == "SDLORA":
        peft_config = SDLoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=model_args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1,
            save_loranew=True,
        )
    elif model_args.peft_type.upper() == "L2P":
        print(f"Using L2P with {model_args.num_virtual_tokens} virtual tokens.")
        peft_config = L2PConfig(
            num_virtual_tokens=model_args.num_virtual_tokens,
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            pool_size=training_args.pool_size,
            top_k=training_args.l2p_top_k,
            pull_constraint=training_args.pull_constraint,
            pull_constraint_coeff=training_args.pull_constraint_coeff,
            engine=model_args.l2p_engine,
        )
    elif model_args.peft_type.upper() == "HIDE_PROMPT":
        print(f"Using HiDe-Prompt with {model_args.num_virtual_tokens} virtual tokens.")
        peft_config = HidePromptConfig(
            num_virtual_tokens=model_args.num_virtual_tokens,
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            prompt_key=False,
            pool_size=100,  # 明确设置prompt pool大小
            top_k=1,       # 明确设置top_k
            use_prefix_tune_for_e_prompt=True,  # 确保使用prefix tuning
        )
    elif model_args.peft_type.upper() == "INFLORA":
        peft_config = InfLoRAConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            inference_mode=False,
            r=model_args.lora_dim,
            lora_alpha=32,
            lora_dropout=0.1
        )
    else:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=model_args.lora_dim, lora_alpha=32, lora_dropout=0.1
        )
    model = get_peft_model(model, peft_config)
    with open(os.path.join(training_args.output_dir, "config.json"), "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)
model.resize_token_embeddings(len(tokenizer))