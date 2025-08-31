# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
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

import enum
import math
import warnings
from dataclasses import dataclass, field
from typing import Optional, Union

import torch
import torch.nn.functional as F

from ..utils import PeftType, PromptLearningConfig


class L2PInit(str, enum.Enum):
    RANDOM = "RANDOM"
    UNIFORM = "UNIFORM"


@dataclass
class L2PConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`L2PPromptPool`].

    Args:
        pool_size (`int`): The size of the prompt pool.
        selection_size (`int`): The number of prompts to select from the pool for each input.
        prompt_init (Union[[`L2PInit`], `str`]): The initialization method for prompts.
        top_k (`int`): The number of top prompts to select based on similarity.
        shared_prompt_pool (`bool`): Whether to share prompt pool across tasks.
        shared_prompt_key (`bool`): Whether to share prompt keys across tasks.
        prompt_key_init (`str`): Initialization method for prompt keys.
        ortho_mu (`float`): Orthogonal regularization coefficient.
        sim_coefficient (`float`): Similarity loss coefficient.
        pull_constraint (`bool`): Whether to use pull constraint.
        pull_constraint_coeff (`float`): Pull constraint coefficient.
    """

    pool_size: int = field(
        default=10,
        metadata={"help": "The size of the prompt pool"},
    )
    selection_size: int = field(
        default=5,
        metadata={"help": "The number of prompts to select from the pool"},
    )
    prompt_init: Union[L2PInit, str] = field(
        default=L2PInit.UNIFORM,
        metadata={"help": "How to initialize the prompt pool"},
    )
    top_k: int = field(
        default=5,
        metadata={"help": "The number of top prompts to select"},
    )
    shared_prompt_pool: bool = field(
        default=True,
        metadata={"help": "Whether to share prompt pool across tasks"},
    )
    shared_prompt_key: bool = field(
        default=True,
        metadata={"help": "Whether to share prompt keys across tasks"},
    )
    prompt_key_init: str = field(
        default="uniform",
        metadata={"help": "Initialization method for prompt keys"},
    )
    ortho_mu: float = field(
        default=0.1,
        metadata={"help": "Orthogonal regularization coefficient"},
    )
    sim_coefficient: float = field(
        default=0.1,
        metadata={"help": "Similarity loss coefficient"},
    )
    pull_constraint: bool = field(
        default=True,
        metadata={"help": "Whether to use pull constraint"},
    )
    pull_constraint_coeff: float = field(
        default=0.1,
        metadata={"help": "Pull constraint coefficient"},
    )

    def __post_init__(self):
        self.peft_type = PeftType.L2P


class L2PPromptPool(torch.nn.Module):
    """
    The L2P (Learning to Prompt) prompt pool module for continual learning.

    Args:
        config ([`L2PConfig`]): The configuration of the L2P prompt pool.

    Example:

    ```py
    >>> from peft import L2PPromptPool, L2PConfig

    >>> config = L2PConfig(
    ...     peft_type="L2P",
    ...     task_type="SEQ_CLS",
    ...     num_virtual_tokens=5,
    ...     token_dim=768,
    ...     pool_size=10,
    ...     selection_size=5,
    ...     top_k=5,
    ... )
    >>> l2p_pool = L2PPromptPool(config)
    ```

    **Attributes**:
        - **prompt** (`torch.nn.Parameter`) -- The prompt pool parameters.
        - **prompt_key** (`torch.nn.Parameter`) -- The prompt key parameters for selection.

    Input shape: (`batch_size`, `hidden_size`) for query

    Output shape: (`batch_size`, `num_virtual_tokens`, `token_dim`) for selected prompts
    """

    def __init__(self, config):
        super().__init__()
        self.pool_size = config.pool_size
        self.selection_size = config.selection_size
        self.top_k = config.top_k
        self.num_virtual_tokens = config.num_virtual_tokens
        self.token_dim = config.token_dim
        self.ortho_mu = config.ortho_mu
        self.sim_coefficient = config.sim_coefficient
        self.pull_constraint = config.pull_constraint
        self.pull_constraint_coeff = config.pull_constraint_coeff

        # Initialize prompt pool
        if config.prompt_init == L2PInit.UNIFORM:
            val = math.sqrt(6.0 / float(3 * self.token_dim + self.pool_size))
            self.prompt = torch.nn.Parameter(
                torch.zeros(self.pool_size, self.num_virtual_tokens, self.token_dim)
            )
            torch.nn.init.uniform_(self.prompt.data, -val, val)
        else:  # RANDOM
            self.prompt = torch.nn.Parameter(
                torch.randn(self.pool_size, self.num_virtual_tokens, self.token_dim)
            )
            torch.nn.init.xavier_uniform_(self.prompt.data)

        # Initialize prompt keys for selection
        if config.prompt_key_init == "uniform":
            self.prompt_key = torch.nn.Parameter(
                torch.zeros(self.pool_size, self.token_dim)
            )
            torch.nn.init.uniform_(self.prompt_key.data, -1, 1)
        else:
            self.prompt_key = torch.nn.Parameter(
                torch.randn(self.pool_size, self.token_dim)
            )
            torch.nn.init.xavier_uniform_(self.prompt_key.data)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """L2 normalize"""
        square_sum = torch.sum(x ** 2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(torch.maximum(square_sum, torch.tensor(epsilon, device=x.device)))
        return x * x_inv_norm

    def forward(self, x_embed, prompt_mask=None, cls_features=None, task_id=None, train=False):
        """
        Forward pass for L2P prompt selection and retrieval.
        
        Args:
            x_embed: Input embeddings (batch_size, seq_len, hidden_size)
            prompt_mask: Optional mask for prompt selection
            cls_features: CLS token features for prompt selection (batch_size, hidden_size)  
            task_id: Current task ID for continual learning
            train: Whether in training mode
            
        Returns:
            Dict containing:
                - selected_prompts: Selected prompt embeddings
                - reduce_sim: Similarity reduction loss
                - similarities: Similarity scores for analysis
        """
        batch_size = x_embed.shape[0]
        
        # Use CLS features for prompt selection if available, otherwise use mean pooling
        if cls_features is not None:
            query_features = cls_features
        else:
            # Use mean of input embeddings as query
            query_features = torch.mean(x_embed, dim=1)  # (batch_size, hidden_size)
        
        # Normalize query and prompt keys
        query_norm = self.l2_normalize(query_features, dim=1)  # (batch_size, hidden_size)
        prompt_key_norm = self.l2_normalize(self.prompt_key, dim=1)  # (pool_size, hidden_size)
        
        # Compute similarity between query and prompt keys
        similarity = torch.matmul(query_norm, prompt_key_norm.t())  # (batch_size, pool_size)
        
        # Select top-k prompts based on similarity
        if self.top_k == -1:
            # Use all prompts
            top_k = self.pool_size
        else:
            top_k = min(self.top_k, self.pool_size)
            
        if prompt_mask is not None:
            similarity = similarity * prompt_mask
            
        # Get top-k indices and similarities
        top_similarities, top_indices = torch.topk(similarity, top_k, dim=1)  # (batch_size, top_k)
        
        # Compute selection weights using softmax
        selection_weights = F.softmax(top_similarities, dim=1)  # (batch_size, top_k)
        
        # Select and weight prompts
        batch_selected_prompts = []
        for i in range(batch_size):
            selected_prompt_indices = top_indices[i]  # (top_k,)
            selected_prompts = self.prompt[selected_prompt_indices]  # (top_k, num_virtual_tokens, token_dim)
            weights = selection_weights[i].unsqueeze(-1).unsqueeze(-1)  # (top_k, 1, 1)
            weighted_prompts = selected_prompts * weights  # (top_k, num_virtual_tokens, token_dim)
            # Sum weighted prompts
            final_prompt = torch.sum(weighted_prompts, dim=0)  # (num_virtual_tokens, token_dim)
            batch_selected_prompts.append(final_prompt)
        
        selected_prompts = torch.stack(batch_selected_prompts, dim=0)  # (batch_size, num_virtual_tokens, token_dim)
        
        # Compute losses for training
        reduce_sim = 0.0
        if train:
            # Similarity reduction loss to encourage diversity
            if self.sim_coefficient > 0:
                # Compute pairwise similarities between selected prompts
                selected_prompt_keys = prompt_key_norm[top_indices.view(-1)]  # (batch_size * top_k, hidden_size)
                selected_prompt_keys = selected_prompt_keys.view(batch_size, top_k, -1)  # (batch_size, top_k, hidden_size)
                
                # Compute pairwise cosine similarities within each batch
                for i in range(batch_size):
                    keys = selected_prompt_keys[i]  # (top_k, hidden_size)
                    pairwise_sim = torch.matmul(keys, keys.t())  # (top_k, top_k)
                    # Exclude diagonal (self-similarity)
                    mask = torch.eye(top_k, device=pairwise_sim.device).bool()
                    pairwise_sim = pairwise_sim.masked_fill(mask, 0)
                    reduce_sim += torch.sum(torch.abs(pairwise_sim))
                
                reduce_sim = reduce_sim * self.sim_coefficient / batch_size
        
        return {
            "selected_prompts": selected_prompts,
            "reduce_sim": reduce_sim,
            "similarities": similarity,
            "top_indices": top_indices,
            "selection_weights": selection_weights,
        }

    def get_prompt_params(self):
        """Get prompt parameters for optimization"""
        return [self.prompt, self.prompt_key]
        
    def copy_prompts(self, src_indices, dst_indices):
        """Copy prompts from source indices to destination indices"""
        with torch.no_grad():
            self.prompt[dst_indices] = self.prompt[src_indices].clone()
            self.prompt_key[dst_indices] = self.prompt_key[src_indices].clone()
