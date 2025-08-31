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

from .config import PeftType, PromptLearningConfig
import torch


def get_peft_model_state_dict(model, state_dict=None, adapter_name="default"):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.peft_config[adapter_name]
    if state_dict is None:
        state_dict = model.state_dict()
        
        # For SDLoRA: consolidate current task's LoRA directions before saving
        if config.peft_type == PeftType.SDLORA:
            # Consolidate LoRA directions: W ← W ∪ {A_t B_t}
            if hasattr(model, 'consolidate_lora_directions'):
                model.consolidate_lora_directions()
            # Update state_dict after consolidation
            state_dict = model.state_dict()
        
        # Original concatenation logic for backward compatibility
        if config.save_loranew == False and config.peft_type != PeftType.SDLORA:
            flag = 1 # this is a switch represents whether 'r_sum' is written to the config file
            for k in state_dict:
                if "lora_A" in k:
                    for k_ in state_dict:
                        if "loranew_A" in k_ and k.split("lora_A")[0] == k_.split("loranew_A")[0]:
                            state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=0) # [r_sum + r, r]
                            if flag == 1:
                                config.r_sum = state_dict[k].shape[0] 
                                flag = 0
                            break # target modules have been matched
                elif "lora_B" in k:
                    for k_ in state_dict:
                        if "loranew_B" in k_ and k.split("lora_B")[0] == k_.split("loranew_B")[0]:
                            state_dict[k] = torch.cat((state_dict[k], state_dict[k_]), dim=1) # [r, r_sum + r]
                            break # target modules have been matched

                
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.SDLORA):
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias

        # modified
        if bias == "none":
            if config.save_loranew and config.peft_type != PeftType.SDLORA: 
                to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "loranew_" in k} # modified
            else:
                base_keys = {k: state_dict[k] for k in state_dict if "lora_" in k}
                # For SDLoRA with separate storage, also include historical directions and scalings
                if config.peft_type == PeftType.SDLORA:
                    historical_keys = {k: state_dict[k] for k in state_dict if "historical_directions" in k or "historical_scalings" in k}
                    # Also save num_historical_directions for each layer
                    num_directions_keys = {k: state_dict[k] for k in state_dict if "num_historical_directions" in k}
                    to_return = {**base_keys, **historical_keys, **num_directions_keys}
                else:
                    to_return = base_keys
                # Update r_sum for SDLoRA based on consolidated lora_A size
                # if config.peft_type == PeftType.SDLORA:
                #     for k, v in base_keys.items():
                #         if "lora_A" in k and adapter_name in k:
                #             config.r_sum = v.shape[0]  # Update r_sum based on current consolidated size
                #             break

        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k:
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError

        # modified
        to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k) or ("loranew_" in k) or ("historical_directions" in k) or ("historical_scalings" in k) or ("num_historical_directions" in k))}
        
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
                config.rank_pattern = rank_pattern
                to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

    elif config.peft_type == PeftType.ADAPTION_PROMPT:
        to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    elif isinstance(config, PromptLearningConfig):
        to_return = {}
        if config.inference_mode:
            prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
        else:
            prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
        to_return["prompt_embeddings"] = prompt_embeddings
    else:
        raise NotImplementedError
    if model.modules_to_save is not None:
        for key, value in state_dict.items():
            if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
                to_return[key.replace("modules_to_save.", "")] = value

    to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    # torch.save(model.state_dict(), "full_model.pth") # for debug
    return to_return

# 加载lora
def set_peft_model_state_dict(model, peft_model_state_dict, adapter_name="default"):
    """
    Set the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model.
        peft_model_state_dict (`dict`): The state dict of the Peft model.
    """
    config = model.peft_config[adapter_name]
    state_dict = {}
    if model.modules_to_save is not None:
        for key, value in peft_model_state_dict.items():
            if any(module_name in key for module_name in model.modules_to_save):
                for module_name in model.modules_to_save:
                    if module_name in key:
                        key = key.replace(module_name, f"{module_name}.modules_to_save.{adapter_name}")
                        break
            state_dict[key] = value
    else:
        state_dict = peft_model_state_dict

    # For SDLoRA: Pre-create historical direction structures before loading
    if config.peft_type == PeftType.SDLORA:
        # First, collect all historical direction data from state_dict
        historical_data = {}
        for k, v in state_dict.items():
            if "historical_directions" in k:
                # Parse the key to extract layer name and direction info
                # Example key: "base_model.model.encoder.block.0.layer.0.SelfAttention.q.historical_directions.dir_0.A.weight"
                parts = k.split("historical_directions.")
                if len(parts) >= 2:
                    layer_path = parts[0]  # e.g., "base_model.model.encoder.block.0.layer.0.SelfAttention.q."
                    remaining = parts[1]   # e.g., "dir_0.A.weight"
                    
                    remaining_parts = remaining.split(".")
                    if len(remaining_parts) >= 2:
                        direction_key = remaining_parts[0]  # "dir_0"
                        component = remaining_parts[1]  # "A" or "B"
                        
                        if layer_path not in historical_data:
                            historical_data[layer_path] = {}
                        if adapter_name not in historical_data[layer_path]:  # 使用当前的adapter_name
                            historical_data[layer_path][adapter_name] = {}
                        if direction_key not in historical_data[layer_path][adapter_name]:
                            historical_data[layer_path][adapter_name][direction_key] = {}
                        
                        historical_data[layer_path][adapter_name][direction_key][component] = v
        
        # Now create the historical direction structures in the model
        for layer_path, adapter_data in historical_data.items():
            for adapter_key, directions in adapter_data.items():
                # Find the corresponding layer in the model
                layer_parts = layer_path.strip('.').split('.')
                current_module = model
                for part in layer_parts:
                    if hasattr(current_module, part):
                        current_module = getattr(current_module, part)
                    else:
                        break
                
                # Check if this is a LoRA layer with historical directions capability
                if hasattr(current_module, 'historical_directions') and hasattr(current_module, 'historical_scalings'):
                    # Ensure the adapter key exists in historical_directions
                    if adapter_key not in current_module.historical_directions:
                        current_module.historical_directions[adapter_key] = torch.nn.ModuleDict()
                    if adapter_key not in current_module.historical_scalings:
                        current_module.historical_scalings[adapter_key] = torch.nn.ParameterDict()
                    
                    # Create each direction
                    for direction_key, components in directions.items():
                        if 'A' in components and 'B' in components:
                            # Get weight shapes to create placeholder modules
                            A_weight = components['A']
                            B_weight = components['B']
                            
                            # Create placeholder direction module structure
                            direction_module = torch.nn.ModuleDict({
                                'A': torch.nn.Linear(A_weight.shape[1], A_weight.shape[0], bias=False),
                                'B': torch.nn.Linear(B_weight.shape[1], B_weight.shape[0], bias=False)
                            })
                            
                            # Add to historical_directions
                            current_module.historical_directions[adapter_key][direction_key] = direction_module
                            
                            # Create placeholder scaling parameter
                            scaling_param = torch.nn.Parameter(torch.tensor(1.0, dtype=A_weight.dtype))
                            current_module.historical_scalings[adapter_key][direction_key] = scaling_param

                            new_num_directions = max(current_module.num_historical_directions[adapter_key], int(direction_key.split('_')[1]) + 1)
                            current_module.num_historical_directions[adapter_key] = torch.nn.Parameter(
                                torch.tensor(new_num_directions, dtype=torch.long), 
                                requires_grad=False
                            )

    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.SDLORA):
        peft_model_state_dict = {}
        for k, v in state_dict.items():
            if "lora_" in k:
                suffix = k.split("lora_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            
            # modified
            elif "loranew_" in k: 
                suffix = k.split("loranew_")[1]
                if "." in suffix:
                    suffix_to_replace = ".".join(suffix.split(".")[1:])
                    k = k.replace(suffix_to_replace, f"{adapter_name}.{suffix_to_replace}")
                else:
                    k = f"{k}.{adapter_name}"
                peft_model_state_dict[k] = v
            # For SDLoRA: handle historical_directions, historical_scalings, and num_historical_directions
            elif "historical_directions" in k or "historical_scalings" in k or "num_historical_directions" in k:
                k = k.replace("historical_directions", f"historical_directions.{adapter_name}")
                k = k.replace("historical_scalings", f"historical_scalings.{adapter_name}")
                # k = k.replace("num_historical_directions", f"num_historical_directions.{adapter_name}")
                peft_model_state_dict[k] = v
            else:
                peft_model_state_dict[k] = v
        if config.peft_type == PeftType.ADALORA:
            rank_pattern = config.rank_pattern
            if rank_pattern is not None:
                model.resize_modules_by_rank_pattern(rank_pattern, adapter_name)
        elif config.peft_type == PeftType.SDLORA:
            # For SDLoRA, ensure r_sum is updated based on loaded lora_A size
            for k, v in peft_model_state_dict.items():
                if "lora_A" in k and adapter_name in k:
                    config.r_sum = v.shape[0]
                    break
    elif isinstance(config, PromptLearningConfig) or config.peft_type == PeftType.ADAPTION_PROMPT:
        peft_model_state_dict = state_dict
    else:
        raise NotImplementedError

    #保存peft_model_state_dict,debug
    with open("peft_model_state_dict_debug.log","w") as f:
        for k, v in peft_model_state_dict.items():
            f.write(f"{k}: {v}\n")
    model.load_state_dict(peft_model_state_dict, strict=False)
    if isinstance(config, PromptLearningConfig):
        model.prompt_encoder[adapter_name].embedding.load_state_dict(
            {"weight": peft_model_state_dict["prompt_embeddings"]}, strict=True
        )
