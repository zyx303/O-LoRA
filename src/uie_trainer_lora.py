import torch
import os
from transformers import GenerationConfig
from transformers.trainer_seq2seq import Seq2SeqTrainer
from transformers.trainer import *
from transformers.trainer_callback import TrainerCallback

from uie_collator import SUPPORTED_DECODER_MODELS, check_model
from uie_dataset_lora import ANSWER_PREFIX
from peft.tuners.inflora import LoraLayer as InfLoraLayer
import sys
from peft import PeftType


def check_prompt_gradients(model, step=0, log_details=False):
    """检查prompt梯度回传是否正常"""
    prompt_params = {}
    for name, param in model.named_parameters():
        # print(name)
        if param.requires_grad:
            print(f'{name} requires grad')
            # if any(kw in name.lower() for kw in ['prompt', 'hide_prompt', 'e_prompt']):
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                prompt_params[name] = grad_norm
                if log_details:
                    print(f"Step {step} - {name}: grad_norm={grad_norm:.6f}")
            else:
                prompt_params[name] = 0.0
                if log_details:
                    print(f"Step {step} - {name}: No gradient!")
    
    if prompt_params:
        total_norm = sum(prompt_params.values())
        if log_details:
            print(f"Step {step} - Total prompt grad norm: {total_norm:.6f}")
        sys.stdout.flush()
        return total_norm > 1e-8  # 梯度是否正常
    # 刷新std
    return True


# 全局变量用于存储类别统计
global cls_mean
global cls_cov
global features_all
cls_mean = dict()
cls_cov = dict()
features_all = []  # 用于存储所有batch的特征表示


def orth_loss(features, targets=None, device=None, args=None):
    """
    HiDe-Prompt风格的正交损失 (Orthogonal Loss)
    
    Args:
        features: 模型输出的特征表示 [batch_size, feature_dim]
        targets: 目标标签 (未使用，保持接口一致)
        device: 设备
        args: 训练参数，包含reg系数
        
    Returns:
        torch.Tensor: 正交损失值
    """
    if features is None or features.size(0) == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    
    reg_coeff = getattr(args, 'reg', 0.1) if args else 0.1
    temperature = 0.8  # 固定温度系数，与HiDe-Prompt保持一致
    
    if cls_mean:
        # 使用已存储的类别均值
        sample_mean = []
        for k, v in cls_mean.items():
            if isinstance(v, list):
                sample_mean.extend(v)
            else:
                sample_mean.append(v)
        
        if sample_mean:
            sample_mean = torch.stack(sample_mean, dim=0).to(device, non_blocking=True)
            M = torch.cat([sample_mean, features], dim=0)
            sim = torch.matmul(M, M.t()) / temperature
            loss = torch.nn.functional.cross_entropy(sim, torch.arange(sim.shape[0], device=device, dtype=torch.long))
            return reg_coeff * loss
    
    # 如果没有类别统计，使用当前batch的特征
    sim = torch.matmul(features, features.t()) / temperature
    loss = torch.nn.functional.cross_entropy(sim, torch.arange(features.size(0), device=device, dtype=torch.long))
    return reg_coeff * loss


def skip_instructions(model, predictions_ids, tokenizer, ignore_idx=-100):
    predictions_ids = np.where(predictions_ids == ignore_idx, tokenizer.pad_token_id, predictions_ids)

    predictions = tokenizer.batch_decode(
        predictions_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    final_predictions = []
    if check_model(model.config._name_or_path, SUPPORTED_DECODER_MODELS):
        for pred in predictions:

            if ANSWER_PREFIX in pred:
                splits = pred.split(ANSWER_PREFIX)
                final_predictions.append(splits[-1].strip())
            else:
                final_predictions.append('')
    else:
        final_predictions = predictions

    return final_predictions


class DenserEvalCallback(TrainerCallback):

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):

        log_eval_steps = [1, 50, 100, 200]

        # Log
        if args.logging_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_log = True

        # Evaluate
        if args.evaluation_strategy == IntervalStrategy.STEPS and state.global_step in log_eval_steps:
            control.should_evaluate = True

        # Save
        # if args.save_strategy

        return control


class UIETrainer(Seq2SeqTrainer):
    
    def _extract_features(self, outputs, inputs=None):
        """从模型输出中提取特征用于CR loss计算"""
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            return outputs.hidden_states[-1].mean(dim=1)
        elif hasattr(outputs, 'encoder_last_hidden_state') and outputs.encoder_last_hidden_state is not None:
            return outputs.encoder_last_hidden_state.mean(dim=1)
        elif isinstance(outputs, dict):
            if 'hidden_states' in outputs and outputs['hidden_states'] is not None:
                return outputs['hidden_states'][-1].mean(dim=1)
            elif 'encoder_last_hidden_state' in outputs and outputs['encoder_last_hidden_state'] is not None:
                return outputs['encoder_last_hidden_state'].mean(dim=1)
        return None
    
    # def create_optimizer(self):
    #     """创建具有不同参数组学习率的优化器"""
    #     if self.optimizer is None:
    #         historical_scaling_params = []
    #         other_params = []
            
    #         for name, param in self.model.named_parameters():
    #             if param.requires_grad:
    #                 if "historical_scalings" in name:
    #                     historical_scaling_params.append(param)
    #                 else:
    #                     other_params.append(param)
            
    #         # 创建参数组
    #         optimizer_grouped_parameters = []
            
    #         if other_params:
    #             optimizer_grouped_parameters.append({
    #                 'params': other_params,
    #                 'lr': self.args.learning_rate,  # 使用默认学习率
    #                 'weight_decay': self.args.weight_decay,
    #             })
            
    #         if historical_scaling_params:
    #             optimizer_grouped_parameters.append({
    #                 'params': historical_scaling_params, 
    #                 'lr': self.args.learning_rate * 10.0,  # 10倍学习率
    #                 'weight_decay': self.args.weight_decay,
    #             })
            
    #         # 让 DeepSpeed 处理优化器创建
    #         return None  # 返回 None 让 DeepSpeed 自己创建
        
    #     return self.optimizer

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """
        inputs = self._prepare_inputs(inputs)

        
        if (getattr(self.args, "get_cur_feat", False) or getattr(self.args, "get_feat", False)) and getattr(model, "peft_type", "").upper() == "INFLORA":
            model.eval()
            with torch.no_grad():
                _ = model(**inputs)  # get matrix
            return torch.zeros([], device=self.args.device, dtype=torch.float32)
        model.train()

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        
        
        with self.compute_loss_context_manager():
            if 'Dataset' in inputs and getattr(model, "peft_type", "").upper() != "HIDE_PROMPT":
                inputs.pop("Dataset")
            loss,outputs = self.compute_loss(model, inputs,return_outputs=True)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.args.gradient_accumulation_steps > 1 and not self.deepspeed:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.args.gradient_accumulation_steps

        ####################### l2p loss #######################
        if 'reduce_sim' in outputs:
            l2p_loss = outputs['reduce_sim']
            loss -= l2p_loss * self.args.pull_constraint_coeff

        ####################### CR loss (HiDe-Prompt style) #######################

        
        if getattr(model, "peft_type", "").upper() == "HIDE_PROMPT":
            features = self._extract_features(outputs, inputs) # [B,d] t5:d=1024
            ## 更新features_all  
            features_all.append(features)
            if features is not None:
                cr_loss_val = orth_loss(features, None, self.args.device, self.args)
                loss += cr_loss_val
                
                # 记录CR loss
                if self.state.global_step % 10 == 0:
                    print(f"Step {self.state.global_step}: CR Loss = {cr_loss_val.item():.6f}")
                    sys.stdout.flush()

        ########################### Regularization ##########################
        
        if self.args.regularization:
            orthogonal_loss = 0.
            # 关闭ortho
            # orthogonal_loss = torch.tensor(0.).to(self.args.device)
            for name, param in self.model.named_parameters():
                if "lora_A" in name:
                    for name_, param_ in self.model.named_parameters():
                        if "loranew_A" in name_ and name.split("lora_A")[0] == name_.split("loranew_A")[0]:
                            orthogonal_loss += torch.abs(torch.mm(param, param_.T)).sum() # [r * dim] * [dim * r]
                            break # target modules have been matched

            # l2-normalization for loranew_A/B
            l2_loss = 0.
            for name, param in self.model.named_parameters():
                if "loranew_" in name:
                    l2_loss += torch.norm(param, p=2)

            lamda_1 = self.args.lamda_1
            lamda_2 = self.args.lamda_2

            # print(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {lamda_1}; λ2: {lamda_2}")
            # logger.info(f"orthogonal_loss: {orthogonal_loss.item()}; l2_loss: {l2_loss.item()}; accuracy_loss: {loss.item()}; λ1: {lamda_1}; λ2: {lamda_2}")
            loss = loss + orthogonal_loss * lamda_1 + l2_loss * lamda_2
        ######################################################################

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        elif self.deepspeed:
            # loss gets scaled under gradient_accumulation_steps in deepspeed
            loss = self.deepspeed.backward(loss)
        else:
            loss.backward()

        # # 检查historical_scalings的梯度
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter: {name}, Requires Grad: {param.requires_grad}, Grad Norm: {param.grad.norm() if param.grad is not None else 'No Grad'}")
        
        # print("=== Checking historical_scalings initialization ===")
        # for name, param in model.named_parameters():
        #     if "module.base_model.model.encoder.block.0.layer.0.SelfAttention.q.historical_scalings.default" in name or 'module.base_model.model.encoder.block.0.layer.0.SelfAttention.q.loranew_A' in name:
        #         print(f"Parameter: {name}")
        #         print(f"  Shape: {param.shape}")
        #         print(f"  Value: {param.data}")
        #         print(f"  Requires grad: {param.requires_grad}")
        #         print(f"  Device: {param.device}")
        #         print(f"  Dtype: {param.dtype}")
        # 检查prompt梯度回传
        # if self.state.global_step % 50 == 0 or self.state.global_step < 10:
        #     check_prompt_gradients(model, self.state.global_step, log_details=True)
        return loss.detach()


    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train init deepspeed here
        if args.deepspeed and not self.deepspeed:

            # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
            # from the checkpoint eventually
            deepspeed_engine, _, _ = deepspeed_init(
                self, num_training_steps=0, resume_from_checkpoint=None, # inference=True
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine

        model = self._wrap_model(self.model, training=False)

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = dataloader.batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader.dataset):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = dataloader.dataset

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

            # Update containers on host
            if loss is not None:
                losses = self._nested_gather(loss.repeat(batch_size))
                losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
            if labels is not None:
                labels = self._pad_across_processes(labels)
                labels = self._nested_gather(labels)
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
            if logits is not None:
                logits = self._pad_across_processes(logits)
                logits = self._nested_gather(logits)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, labels_host = None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and hasattr(eval_dataset, "num_examples"):
            num_samples = eval_dataset.num_examples
        else:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            metrics = self.compute_metrics(dataset=eval_dataset, preds=all_preds, save_prefix=metric_key_prefix)
        else:
            metrics = {}

        metrics["global_step"] = self.state.global_step

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = self._gen_kwargs
        gen_kwargs["synced_gpus"] = True if is_deepspeed_zero3_enabled() else False

        if "attention_mask" in inputs:
            gen_kwargs["attention_mask"] = inputs.get("attention_mask", None)

        generation_config = GenerationConfig(**gen_kwargs)

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if hasattr(self.model, "encoder") and self.model.encoder.main_input_name != self.model.main_input_name:
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        # 为HiDe-Prompt在生成时添加dataset信息用于prompt选择
        # 通过临时设置模型属性来传递Dataset信息
        if "Dataset" in inputs:
            self.model._current_datasets = inputs["Dataset"]
            
        generated_tokens = self.model.generate(
            input_ids=generation_inputs, 
            generation_config=generation_config
        )
        
        # 打印生成的token ID和对应的文本（通过环境变量控制）
        # print("=== 生成结果 ===")
        # for i, tokens in enumerate(generated_tokens):
        #     decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        #     print(f"样本 {i}: {decoded_text}")
        #     # 也打印token IDs
        #     print(f"Token IDs: {tokens.tolist()}")
        # print("===============")
        # sys.stdout.flush()

        bs, source_len = inputs['input_ids'].shape
        # in case the batch is shorter than max length, the output should be padded
        if check_model(self.model.config._name_or_path, SUPPORTED_DECODER_MODELS):
            max_length = source_len + gen_kwargs["max_new_tokens"]
        else:
            max_length = gen_kwargs["max_new_tokens"]

        if generated_tokens.shape[-1] < max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, max_length)

        with torch.no_grad():
            if has_labels and not getattr(self.args, "skip_predict_loss", False):
                with self.autocast_smart_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_new_tokens"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_new_tokens"])
        else:
            labels = None

        return (loss, generated_tokens, labels)
