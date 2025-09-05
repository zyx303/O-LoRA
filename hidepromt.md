1. 训练的参数是ψ（lm_head), prompt
2. default是只启用e_prompt（vit的**前5个layer**, 用prefix，train和infer都是)：
   

> 原论文有一个结构是TII，用另一个original model预测task id（prompt id)，这里不实现，假设task id已知

> 关于prompt id：
> 正在收集工作区信息- prompt_id 是“任务身份”的预测结果，用来在前向时为每个样本选择要用的 E‑Prompt。生成流程：
>   - 先用未指导的原始模型分类得到类别 logits；取 argmax 得到类别，再由类别→任务映射表 target_task_map 转成任务 id，即 prompt_id，见 `engines.hide_promtp_wtp_and_tap_engine.train_one_epoch` 中
>     - 原始模型前向
>     - 将类别映射为任务 id
>     - 调用模型时传入 `prompt_id`：`model(input, task_id=task_id, prompt_id=prompt_id, ...)`
>
> - 模型如何使用 prompt_id：
>   - 在前向特征里把 prompt 送入 Transformer 前，调用 E‑Prompt 生成该 batch 的提示；这里会把 `prompt_id` 传给 E‑Prompt，见 `vits.hide_prompt_vision_transformer.VisionTransformer.forward_features` 调用 `peft.prompt.hide_prompt.EPrompt.forward`。
>   - E‑Prompt 内部用 `prompt_idx`（即 prompt_id）或 `prompt_weight/prompt_mask`从提示池中选取对应索引的提示，见 `peft.prompt.hide_prompt.EPrompt.forward`。
>
> - prompt 是否按 task 划分：
>   - 是。提示池按任务连续切分，常用配置是每个任务 `top_k=1` 个提示，索引区间为 `[task_id*top_k, (task_id+1)*top_k)`。训练阶段会显式构造只允许当前任务区间的 `prompt_mask`，从而“强制使用本任务提示”，见 `vits.hide_prompt_vision_transformer.VisionTransformer.forward_features` 中基于 `task_id` 生成的 `prompt_mask`。
>   - 推理/评估时不再给 `prompt_mask`，而是依赖原始模型推断得到的 `prompt_id` 选择对应任务的提示，见上面的前向路径。
>   - 提示池“按任务连续”这一布局也体现在跨任务拷贝/迁移参数的代码里，例如把上个任务学到的提示复制到本任务区间，见 `engines.hide_promtp_wtp_and_tap_engine.train_and_evaluate` 中 “Transfer previous learned prompt params” 段，以及任务结束后的动量融合，见同文件对 `args.prompt_momentum` 的处理。
>
> - 相关配置示例（每任务一个提示，池大小等于任务数）：见 configs/cifar100_hideprompt_5e.py、configs/five_datasets_hideprompt_5e.py 中的 `size`、`top_k`、`num_tasks` 等参数。
>
> 小结
> - prompt_id 用于把样本路由到“该样本所属任务”的提示上。
> - HiDe‑Prompt 将提示池按任务切分；训练用 `prompt_mask` 强约束当前任务区间，推理用原始模型得到的 prompt_id 选择对应区间的提示。

> [!warning]
>
> ## 1. `shared_prompt_key` 参数
>
> 在 HiDe-Prompt 配置中，`shared_prompt_key` 默认为 **`False`**：
>
> ```python
> # 在所有 HiDe-Prompt 配置文件中
> subparsers.add_argument('--shared_prompt_key', default=False, type=bool)
> ```
>
> ## 2. HiDe-Prompt 是否有 `e_prompt.prompt_key`
>
> **HiDe-Prompt 通常没有可学习的 `prompt_key`**，因为：
>
> - `prompt_key` 参数在 HiDe-Prompt 配置中默认为 **`False`**
> - 在 `peft.prompt.hide_prompt.EPrompt.__init__` 中：
>
> ```python
> if prompt_key: # False in HiDe-Prompt
>     # 创建可学习的 prompt_key
>     key_shape = (pool_size, embed_dim)
>     self.prompt_key = nn.Parameter(torch.randn(key_shape))
> else:
>     # 使用提示参数的均值作为键（但在 HiDe-Prompt 中通常不使用）
>     prompt_mean = torch.mean(self.prompt, dim=[0, 2])
>     self.prompt_key = prompt_mean 
> ```
>
> ## 3. HiDe-Prompt 如何选择 prompt
>
> HiDe-Prompt 的提示选择机制**不依赖 prompt_key**，而是使用：
>
> ### 训练阶段
> - 使用 `prompt_mask` 强制选择当前任务的提示：
> ```python
> # 在 VisionTransformer.forward_features 中
> if self.use_prompt_mask and train:
>     start = task_id * self.e_prompt.top_k
>     end = (task_id + 1) * self.e_prompt.top_k
>     prompt_mask = torch.arange(start, end).to(x.device)
> ```
>
> ### 推理阶段
> - 使用原始模型推断得到的 `prompt_id`：
> ```python
> # 在 train_one_epoch/evaluate 中
> output = original_model(input)
> logits = output['logits']
> prompt_id = torch.max(logits, dim=1)[1]  # 取最大值索引
> prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_id], device=device)
> ```
>
> ### 在 EPrompt.forward 中的选择逻辑
> ```python
> # 优先级：prompt_mask > prompt_id > prompt_weight
> if prompt_mask is not None:
>     idx = prompt_mask  # 训练时使用
> else:
>     idx = prompt_idx   # 推理时使用 prompt_id
> ```
>
> ## 4. `reduce_sim` 辅助损失
>
> **HiDe-Prompt 没有 `reduce_sim` 损失**。`reduce_sim` 是 DualPrompt/L2P 中的对比学习损失，出现在 `peft.prompt.dp_prompt.EPrompt.forward` 中：
>
> ```python
> # 在 dp_prompt.py 中（DualPrompt/L2P）
> x_embed_norm = x_embed_norm.unsqueeze(1) # B, 1, C
> sim = batched_key_norm * x_embed_norm # B, top_k, C
> reduce_sim = torch.sum(sim) / x_embed.shape[0] # Scalar
> out['reduce_sim'] = reduce_sim
> ```
>
> 但在 `peft.prompt.hide_prompt.EPrompt.forward` 中没有这个计算。
>
> ## 5. HiDe-Prompt 使用的损失
>
> HiDe-Prompt 主要使用：
>
> 1. **交叉熵损失**：基础分类损失
> 2. **正交损失** (`orth_loss`)：对比正则化
> ```python
> loss = criterion(logits, target)  # CrossEntropyLoss
> loss += orth_loss(output['pre_logits'], target, device, args)  # 正交正则
> ```
>
> ## 总结
>
> - HiDe-Prompt 不使用基于相似度的提示选择（无 `prompt_key`、无 `reduce_sim`）
> - 而是使用**任务身份推断**（TII）+ **提示掩码**的层次化分解策略
> - 这体现了 HiDe-Prompt 与传统基于键值匹配方法（如 L2P/DualPrompt）的根本差异

> prompt_weight不使用