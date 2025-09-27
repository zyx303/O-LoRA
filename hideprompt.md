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

![image-20250905203952371](https://blogstastic-zyx.oss-cn-hangzhou.aliyuncs.com/images/image-20250905203952371.png)

> 正在收集工作区信息下面按算法伪代码（图中第1–16行）逐步对应到仓库中的实现代码与符号。
>
> - 1 初始化 e1、ω、ψ  
>   - 初始化提示与分类头等参数：`vits.hide_prompt_vision_transformer.VisionTransformer.__init__`（构建 e‑prompt/g‑prompt、`head`、可选 `MlpMapping`）。  
>   - e‑Prompt 的权重与可选键向量初始化：`peft.prompt.hide_prompt.EPrompt.__init__`。  
>   - 原始模型用于任务识别的 MLP（ω）初始化：`vits.hide_prompt_vision_transformer.MlpMapping` 由 trainer 在创建 `original_model` 时通过 `mlp_structure` 注入：`trainers.hideprompt_trainer.train`。
>
> - 2 for t = 1 … T（任务循环）  
>   - 任务级训练主循环与保存检查点：`engines.hide_promtp_wtp_and_tap_engine.train_and_evaluate`。
>
> - 3–4 对每个类别获取未指导表示 Ĝc（Uninstructed）  
>   - 直接用冻结的原始模型前向得到未加提示的输出/特征：`engines.hide_promtp_wtp_and_tap_engine.train_one_epoch` 中
>     `output = original_model(input)` → `logits`，随后据此推断 `prompt_id`（对应任务 id）。  
>   - 推断阶段同理见评估：`engines.hide_promtp_wtp_and_tap_engine.evaluate`。
>
> - 5–9 构造当前任务提示 pt  
>   - t>1 时把上一任务已学提示拷贝到新任务位置（e_t ← e_{t−1} 的实现）：[`engines.hide_promtp_wtp_and_tap_engine.train_and_evaluate` “Transfer previous learned prompt params” 段](engines/hide_promtp_wtp_and_tap_engine.py)。  
>   - 线性融合 α·∑e_i + (1−α)·e_t（代码里称为 prompt momentum）：  
>     - 前缀调优情形融合：`peft.prompt.hide_prompt.EPrompt.forward` 中  
>       `batched_prompt_raw = (1 - prompt_momentum) * ... + prompt_momentum * batched_prompt_momentum`；  
>     - 另一实现（DualPrompt 版本）亦在：`peft.prompt.dp_prompt.EPrompt.forward`。  
>   - t=1 时直接使用当前 e_t：同一 `forward` 中未触发融合分支即为 pt = e_t。
>
> - 10–13 训练循环（epoch = 1 … E）  
>   - 11 用 L_WTP 优化 pt 与 ψ（Within‑Task Prediction）  
>     - 单轮训练：`engines.hide_promtp_wtp_and_tap_engine.train_one_epoch`。  
>       关键点：得到 `logits` 后交叉熵 + 正交正则 `orth_loss`（WTP 主损失）。  
>   - 12 用 L_TII 优化 ω（Task‑Identity Inference）  
>     - 第一阶段仅训练任务识别模块（脚本里先运行 `--train_inference_task_only`）：  
>       调用的训练逻辑：`engines.hide_tii_engine.train_one_epoch`，由 `trainers.hideprompt_trainer.train` 依据该开关驱动；  
>       任务掩码与交叉熵见同文件中的 `train_one_epoch/evaluate`。  
>     - 推理阶段利用 `original_model` 预测 `prompt_id` 参与后续 WTP：同第3–4步中的 `train_one_epoch`。  
>   - 13 用 L_TAP 优化 ψ（Task‑Adaptive Prediction）  
>     - 根据已统计的类均值/协方差按高斯采样特征，冻结 backbone，仅训练 `mlp/head`：  
>       `engines.hide_promtp_wtp_and_tap_engine.train_task_adaptive_prediction` 与  
>       `engines.hide_tii_engine.train_task_adaptive_prediction`（`model(inp, fc_only=True)` 仅更新分类头）。
>
> - 14–15 获取指导表示 Gc（Instructed）  
>   - e‑Prompt/g‑Prompt 的生成与注入 Transformer 各层：  
>     - 选择/生成批量提示：`peft.prompt.hide_prompt.EPrompt.forward`。  
>     - 将提示送入块并前向：`vits.hide_prompt_vision_transformer.VisionTransformer.forward_features`。  
>     - 分类头前向得到 logits / features：`vits.hide_prompt_vision_transformer.VisionTransformer.forward_head` 与 `forward`。
>
> - 16 返回（p1…pT, ω, ψ）  
>   - 每任务结束保存当前参数（相当于“返回”本任务学习到的 p_t/ψ，并在两阶段之间加载 ω）：  
>     检查点路径与保存逻辑：[`engines.hide_promtp_wtp_and_tap_engine.train_and_evaluate` 中的 `checkpoint_path`](engines/hide_promtp_wtp_and_tap_engine.py)。  
>   - 第二阶段开始前会加载第一阶段（TII）训练好的原始模型权重：[`trainers.hideprompt_trainer.train` 加载 `--trained_original_model` 检查点](trainers/hideprompt_trainer.py)。
>
> 补充提示
> - 训练脚本体现“两阶段”（先 TII，再 WTP+TAP）的运行顺序与关键超参 α（对应 `--prompt_momentum`）：见 train_cifar100_ibot21k.sh 与 README 用法示例。  
> - 任务/类别掩码以增量设置类别可见性：见 `engines.hide_promtp_wtp_and_tap_engine.train_one_epoch` 中的 `train_mask` 与 `engines.hide_tii_engine.train_one_epoch`。



> ortho loss：
>
> 正在收集工作区信息默认行为概述
> - 位置与调用：在engines/hide_promtp_wtp_and_tap_engine.py中定义为函数`orth_loss`，训练时由`train_one_epoch`每个 batch 调用并加到主损失上：
>   - loss = CE(logits, target) + orth_loss(pre_logits, …)
>
> - 计算方式（温度τ=0.8，固定常数）
>   1) 若已存在类别统计均值 cls_mean（见下文来源），将其与当前 batch 的特征拼在一起：
>      - M = concat([sample_mean, features], dim=0)
>      - sim = (M @ M^T) / 0.8
>      - 目标标签是 0..N-1 的“自监督式”身份标签（与行索引对应），计算交叉熵 CE(sim, arange(N))
>   2) 若 cls_mean 为空，仅用当前 batch 的 features 做同样的自对比 CE。
>   3) 返回值乘以系数 args.reg：return args.reg * loss
>
> - cls_mean 的来源与时机：由同文件的`_compute_mean`在每个任务训练结束后统计并缓存（支持 covariance/variance/multi-centroid 三种方式）。因此：
>   - 训练首个任务时，cls_mean 为空，走“仅 batch 特征”分支；
>   - 之后任务训练时，会利用以往任务的类均值参与对比正则。
>
> - 超参默认值
>   - 系数 args.reg 默认 0.01（见configs/imr_hideprompt_5e.py），脚本中常覆盖为 0.5（如training_scripts/train_imr_vit.sh）。
>   - 温度常数为 0.8（硬编码于`orth_loss`）。
>   - 在评估路径`evaluate`不会调用 orth_loss，仅训练时生效。
>
> - 其它注意
>   - 传入的 targets 参数在`orth_loss`内并未使用；
>   - 设 --reg 0 可等效关闭该正则。



> ![image-20250905205233619](https://blogstastic-zyx.oss-cn-hangzhou.aliyuncs.com/images/image-20250905205233619.png)
> 正在收集工作区信息下面把论文中4个损失和仓库里的实现一一对应，并标出用到的特征来源 pre_logits。
>
> - LCR(pt)（Eq. 9，类间/正交正则）
>   - 实现：`orth_loss`。
>   - 调用位置（训练 WTP 时加到总损失里）：`train_one_epoch` 中
>     loss = criterion(logits, target) + orth_loss(output['pre_logits'], …)。
>   - 细节：温度系数固定 0.8；权重 λ 对应 `args.reg`。使用的特征是模型前向返回的 `pre_logits`。
>
> - LWTP(ψ, pt) = LCE(ψ, pt) + λ·LCR(pt)（Eq. 10，WTP 总损失）
>   - 交叉熵部分：`train_one_epoch` 的
>     loss = criterion(logits, target)。
>   - 正则部分：同上加号右边的 `orth_loss`。
>   - logits 与 pre_logits 的产生：模型前向 `VisionTransformer.forward` / `forward_features` / `forward_head`；前向返回字典里含有 ['logits', 'pre_logits']。
>
> - LTII(ω)（Eq. 11，任务身份推断的分类损失）
>   - 实现与训练位置：`engines.hide_tii_engine.train_one_epoch`。
>   - 形式：标准交叉熵 `criterion(logits, target)`，优化的是原始模型的辅助头/分类头 ω（原始模型创建见 `trainers.hideprompt_trainer.train`）。
>
> - LTAP(ψ)（Eq. 12，任务自适应预测的分类损失）
>   - 实现与训练位置：`train_task_adaptive_prediction`。
>   - 形式：对根据类别统计采样得到的特征做分类的交叉熵 `criterion(logits, tgt)`；仅更新分类头（前向用 `model(inp, fc_only=True)`）。
>
> 补充
> - pre_logits 的来源：模型前向返回，键名为 `'pre_logits'`，由 `VisionTransformer.forward_head` 在分类头之前导出；其值在 WTP 的 `train_one_epoch` 中传入 `orth_loss`。





![image-20250927164622261](https://blogstastic-zyx.oss-cn-hangzhou.aliyuncs.com/images/image-20250927164622261.png)

uie_collator.seq2seq_call

![image-20250927164841814](./hideprompt.assets/image-20250927164841814.png)

![image-20250927164917880](./hideprompt.assets/image-20250927164917880.png)

[B,len]