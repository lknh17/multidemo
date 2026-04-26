# SFT 参数原理与经验设置手册

> **目标**：把 `p03_sft_finetuning/` 目录下涉及的**每一个**超参数讲透——包含原理推导、内部机制、不同场景下的经验值、以及调参时的诊断信号。读完即可把"调 SFT"这件事从玄学变成工程。
>
> **配套代码**：`config.py` / `train.py` / `dataset.py` / `ds_config.json`
>
> **适用范围**：Qwen2.5 / LLaMA 系列 7B 以内的小模型 SFT。大模型（70B+）经验值会在各节注明差异。

---

## 目录

1. [SFT 本质：和预训练到底哪里不一样](#1-sft-本质和预训练到底哪里不一样)
2. [模型与方法选择类参数](#2-模型与方法选择类参数)
3. [数据类参数](#3-数据类参数)
4. [核心训练超参数](#4-核心训练超参数)
5. [学习率调度相关参数](#5-学习率调度相关参数)
6. [正则化与稳定性参数](#6-正则化与稳定性参数)
7. [精度与显存优化参数](#7-精度与显存优化参数)
8. [LoRA / QLoRA / DoRA 专属参数](#8-lora--qlora--dora-专属参数)
9. [DeepSpeed 配置参数](#9-deepspeed-配置参数)
10. [保存、日志、评估参数](#10-保存日志评估参数)
11. [一眼看懂的「参数速查表」](#11-一眼看懂的参数速查表)
12. [四种典型场景的「整套配置」](#12-四种典型场景的整套配置)
13. [调参流程与诊断信号](#13-调参流程与诊断信号)

---

## 1. SFT 本质：和预训练到底哪里不一样

在谈参数之前，必须先理解 SFT 的内部数学，否则每个超参数都只是「一个数字」。

### 1.1 损失函数：Causal LM Loss + Label Mask

SFT 和预训练**都用** next-token prediction：

$$
\mathcal{L} = -\frac{1}{|\mathcal{M}|} \sum_{t \in \mathcal{M}} \log p_\theta(x_t \mid x_{<t})
$$

唯一区别：**$\mathcal{M}$ 的定义**。

| 阶段 | $\mathcal{M}$ 的含义 |
|------|----------------------|
| 预训练 | 所有 token（除 pad） |
| **SFT** | **只包含 assistant 回复的 token** |

对应 `dataset.py::tokenize_with_label_mask`：

```python
labels = torch.full_like(input_ids, -100)   # 全部 mask
labels[start:end] = input_ids[start:end]    # 只把 assistant 部分还原
```

**为什么只对 assistant 算 loss？** 如果对 user 部分也算 loss，模型会学习"如何模仿用户提问"，而不是"如何回答"。实测关闭 label mask 会让效果下降 2–5 个点（BLEU / 人工胜率）。

### 1.2 有效 loss token 的数量决定一切

对于一条 512 长度的样本：

- 假设 prompt 占 200 tokens，回复占 120 tokens，padding 192 tokens
- 实际参与 loss 的 token = **120**，占比仅 ~23%

这直接影响：

- **学习率的有效强度**：更少的有效 token → 每步梯度噪声更大 → 需要更温和的 lr
- **epoch 数**：SFT 每个 epoch 的"真实训练量"只有预训练的 1/3 左右
- **batch size 的意义**：当 batch 内 loss token 数量方差大时，有效 batch 可能远小于设定值

`dataset.py` 会打印统计：

```
[Label Mask] 总 token: 2,340,000, 计算 loss: 680,000 (29.1%)
```

**经验值**：健康的 SFT 数据，loss token 占比应在 **20%–40%**。低于 10% → prompt 过长／回复过短；高于 60% → 几乎退化成预训练，不是真正的指令学习。

### 1.3 SFT 的"过拟合"与预训练不同

- 预训练数据海量 → 几乎不会过拟合
- SFT 数据通常只有几万到几十万条 → **2-3 个 epoch 就会过拟合**

表现：训练 loss 继续下降，但 `eval_loss` 先降后升，生成内容出现重复、格式固化、Distinct-N 下降。这是后面设置 `num_train_epochs` / `weight_decay` / `lora_dropout` 的核心理论依据。

---

## 2. 模型与方法选择类参数

### 2.1 `model_name`（基座模型）

```python
model_name: str = "Qwen/Qwen2.5-0.5B"
```

**原理**：SFT 是在一个**已经具备世界知识**的基座上做"行为塑形"，不是从零学。基座质量直接决定 SFT 上限。

**选择经验**：

| 场景 | 推荐基座 | 理由 |
|------|----------|------|
| 学习/调试 | Qwen2.5-0.5B / 1.5B | 7 分钟跑完一次实验，消融成本低 |
| 单卡 24G 生产 | Qwen2.5-7B-Instruct / Llama3-8B | 效果/成本最甜点 |
| 多语言中文 | Qwen2.5 系列 | 中文 tokenizer 效率远高于 Llama |
| 代码任务 | Qwen2.5-Coder / DeepSeek-Coder | 基座已在代码上大量预训练 |

**陷阱**：

1. **Base vs Instruct**：如果基座已经是 Instruct 版（ChatML-tuned），再做 SFT 需要**匹配它原有的模板**，否则会破坏对齐。Qwen2.5-Base 用 ChatML 没问题；Llama3-Base 不应直接套 ChatML。
2. **词表对齐**：如果你扩词表了（p02 继续预训练常见操作），必须确保 SFT 使用同一个 tokenizer，否则 label mask 里的 `<|im_start|>` token id 会错位。

### 2.2 `method`（微调方法）

```python
method: str = "lora"  # full_finetune / lora / qlora / dora
```

四种方法的数学本质：

| 方法 | 数学形式 | 可训练参数 | 显存（7B 模型估算） |
|------|----------|-----------|-----------------|
| **full_finetune** | $W' = W + \Delta W$，$\Delta W$ 和 $W$ 同形 | 100% | ~80 GB（AdamW）|
| **LoRA** | $W' = W + \frac{\alpha}{r} B A$，$B\in\mathbb{R}^{d\times r}$，$A\in\mathbb{R}^{r\times k}$ | 0.1%–2% | ~18 GB |
| **QLoRA** | LoRA + 基座 4-bit NF4 量化 | 0.1%–2% | ~10 GB |
| **DoRA** | $W' = m \cdot \frac{W + \Delta W}{\|W+\Delta W\|}$，分离方向+幅度 | ~LoRA × 1.1 | ~19 GB |

**决策树**：

```
显存 >= 80G × n 卡？  ——>  full_finetune
    |no
显存 12–24G？
    ├── 追求极限效果 → DoRA
    ├── 平衡通用场景 → LoRA  ← 默认首选
    └── 显存爆红 → QLoRA
显存 < 12G？ ——> QLoRA（唯一选择）
```

**关键经验**：

- **效果排序**（多个 paper 综合）：`full ≥ DoRA > LoRA > QLoRA`，但差距通常 < 2%。
- **QLoRA 的陷阱**：4-bit 量化会带来 $O(10^{-3})$ 量级的数值误差，反向传播时 LoRA 侧被迫用 bf16 补偿，**学习率要比 LoRA 降 10%–20%**，否则训练会发散。
- **DoRA 的代价**：比 LoRA 多一次 norm 计算，训练慢 10%–15%，推理（合并后）0 开销。

---

## 3. 数据类参数

### 3.1 `dataset_format`（数据格式）

```python
dataset_format: str = "alpaca"   # alpaca / sharegpt
```

**两种格式**：

```json
// Alpaca: 单轮
{"instruction": "翻译以下句子", "input": "Hello world", "output": "你好世界"}

// ShareGPT: 多轮
{"conversations": [
  {"from": "human", "value": "讲个笑话"},
  {"from": "gpt",   "value": "为什么..."},
  {"from": "human", "value": "再来一个"},
  {"from": "gpt",   "value": "..."}
]}
```

**经验**：

- 工业场景 **80% 以上是 Alpaca 单轮** 足够——多轮质量远比数量重要。
- 如果用 ShareGPT 多轮，注意 `dataset.py` 会把所有 assistant 段都算 loss，**多轮之间会互相"污染梯度"**。解决方案是只保留最后一轮 assistant 的 loss，或截断成多个单轮样本。

### 3.2 `max_seq_length`（最大序列长度）

```python
max_seq_length: int = 512
```

**原理**：Transformer 显存复杂度 $O(L^2)$ ——序列长度翻倍，显存接近 4×。

**经验值矩阵**：

| 任务类型 | 推荐 | 理由 |
|---------|------|------|
| 短问答 / 分类 | 256–512 | 90% 样本 < 300 token，浪费的 pad 更少 |
| 通用指令 | **512–1024** | Alpaca 类任务够用 |
| 代码 / 长文档 | 2048–4096 | 一个函数经常 > 1K token |
| Agent / 多轮 | 4096–8192 | 历史对话会累积 |

**诊断**：跑 `dataset.py` 时观察截断率：

```
[SFT数据] 处理完成: 9800 条有效, 200 条跳过
```

如果"跳过 + 有效截断"超过 5%，说明 `max_seq_length` 设小了。也可以手动统计：

```python
lens = [len(tokenizer.encode(s)) for s in samples]
print(f"P50={np.percentile(lens,50)}, P95={np.percentile(lens,95)}, P99={np.percentile(lens,99)}")
# 经验：设置 max_seq_length ≈ P99，兼顾显存和覆盖率
```

**与 batch 的交换率**：若显存紧张，优先降 `max_seq_length`（显存平方下降）而非降 `batch_size`（线性下降）。

### 3.3 `max_samples`（最大样本数）

```python
max_samples: int = 50000
```

**经验**：

- **小模型（<1B）**：5k–20k 高质量 > 100k 低质量。Alpaca-52k 原数据就够。
- **7B+**：2 万–10 万条中等质量 + 1 千条高质量精调。
- **领域特化**：3k–5k 领域数据 + 通用数据按 1:5 混合，避免灾难遗忘。

**质量远大于数量** —— 一条精炼的 Chain-of-Thought 样本 ≈ 10 条 Alpaca 样本。业界共识：**SFT 数据质量 > 数据量 > 模型大小**。

### 3.4 `val_ratio`（验证集比例）

```python
val_ratio: float = 0.05
```

**原理**：SFT 的 eval_loss 是最直观的过拟合信号，必须留验证集。

**经验**：

- 总量 < 10k：`val_ratio = 0.1`（保证 1k 验证样本，统计显著）
- 10k–100k：`val_ratio = 0.05`
- > 100k：固定 2000–5000 条，不再按比例
- **验证集一定要和训练集同分布**。不要拿 MMLU 做 SFT 的 eval set。

---

## 4. 核心训练超参数

这一节是 SFT 最关键的 5 个参数。

### 4.1 `learning_rate`（学习率）

```python
# LoRA 系列: 2e-4
# 全参微调: 2e-5
```

**原理**：学习率是 SFT 最敏感的超参。不同方法差 10 倍不是拍脑袋：

- **全参微调**：更新的是 $W \in \mathbb{R}^{d\times d}$ 本身，$W$ 的量级是 $O(1/\sqrt{d})$（Xavier 初始化量级），大学习率会直接打乱预训练分布 → 灾难遗忘。
- **LoRA**：$B$ 初始化为 0，$A$ 用 Kaiming → $BA$ 乘积一开始是 0，相当于从"什么都没学"开始，容忍大学习率；而且 LoRA 只有 0.5%–2% 参数，lr 需要放大补偿。

**经验矩阵**：

| 方法 | 模型规模 | 推荐 lr | 可接受区间 |
|------|---------|---------|------------|
| full_finetune | 0.5B–3B | 5e-5 | 2e-5 ~ 1e-4 |
| full_finetune | 7B+ | 2e-5 | 1e-5 ~ 5e-5 |
| LoRA | 0.5B–3B | 3e-4 | 1e-4 ~ 5e-4 |
| LoRA | 7B+ | 2e-4 | 5e-5 ~ 3e-4 |
| QLoRA | 任意 | 1e-4 ~ 2e-4 | 比 LoRA 低一档 |
| DoRA | 任意 | 2e-4 | 和 LoRA 接近 |

**调参启发式**（不借助扫描）：

1. 先用默认 lr 跑 100 步，观察 `grad_norm` —— 健康值 **0.2–1.0**
2. 若 `grad_norm > 5` 持续 → lr 太大
3. 若 loss 在前 500 步几乎不动（方差 < 5%）→ lr 太小
4. 若 loss 先降到一个值后突然 NaN → lr 过大 + 没有 clip

### 4.2 `per_device_train_batch_size`

```python
per_device_train_batch_size: int = 4
```

**原理**：batch 大 → 梯度方差小 → 可用更大 lr → 训练更稳；但显存线性增长。

**显存估算（7B bf16 + LoRA）**：

```
显存 ≈ 模型参数(14G) + 梯度(0，LoRA 只有adapter) + optimizer(LoRA adapter × 2 ≈ 0.3G)
        + activation(∝ batch × seq²)
        + KV cache(训练时不需要)
```

activation 部分才是 batch 和 seq 的主战场：

$$
\text{Activation} \approx 2 \times b \times s \times h \times L
$$

其中 $b$ 是 batch，$s$ 是 seq_len，$h$ 是 hidden（7B ≈ 4096），$L$ 是层数（7B ≈ 32）。

**经验**：

| GPU | 模型 | seq_len | 推荐 batch |
|-----|------|---------|------------|
| 4090 24G | 0.5B LoRA | 512 | 8–16 |
| 4090 24G | 7B LoRA | 512 | 2–4 |
| 4090 24G | 7B QLoRA | 1024 | 4 |
| A6000 48G | 7B LoRA | 1024 | 8 |
| A100 80G | 7B full | 2048 | 8 |

### 4.3 `gradient_accumulation_steps`

```python
gradient_accumulation_steps: int = 4
# 等效 batch = per_device × accumulation = 4 × 4 = 16
```

**原理**：梯度累积 = 免费扩 batch。不增加显存，只增加时间。

**关键关系**：

$$
\text{有效 batch} = \text{per\_device\_bs} \times \text{accum\_steps} \times \text{num\_gpus}
$$

**经验值**：

- SFT 的**有效 batch 目标是 16–64**
- 小模型（< 1B）：16 足够
- 7B+：推荐 32–64，梯度更稳
- **超过 128 通常收益递减**，反而会让模型"记不住"细节

**与 lr 的缩放关系**：batch 翻倍，lr 理论上要 `×√2`（线性缩放 overshoot，平方根缩放更稳）。

### 4.4 `num_train_epochs`

```python
num_train_epochs: int = 3
```

**原理**：SFT 是 **低 epoch** 训练，远小于预训练。

**经验**：

| 数据量 | 推荐 epoch | 原因 |
|-------|-----------|------|
| < 1k | 5–10 | 需要多轮才能记住 |
| 1k–10k | 3–5 | 经典区间 |
| 10k–100k | 2–3 | 默认选 3 |
| > 100k | 1–2 | 过多会过拟合 |

**诊断过拟合的信号**（运行时观察）：

1. `eval_loss` 在第 N 个 epoch 末尾开始上升 → 把 epoch 降到 N
2. 生成内容开始出现 **prompt 中短语的复读机现象** → 过拟合
3. Distinct-N 下降 → 多样性丢失

**记住**：宁可 early stop，也不要跑满 5 epoch。`save_steps` + `save_total_limit=3` 就是为这个 early stop 做的准备。

### 4.5 `weight_decay`（权重衰减）

```python
weight_decay: float = 0.01
```

**原理**：AdamW 的解耦权重衰减：

$$
w_{t+1} = w_t - \eta \cdot \hat{m}_t - \eta \cdot \lambda \cdot w_t
$$

作用：把权重往 0 拉，防过拟合，等效 L2 正则（但解耦后更稳定）。

**经验**：

- **全参微调**：`0.01` 是标准值
- **LoRA**：可以用 `0.01`，也可以设 `0` —— LoRA 的 $B$ 矩阵本来就初始化为 0，再拉回 0 会阻碍学习
- **数据稀少场景**（< 5k）：`0.05`–`0.1` 防止过拟合
- **数据充足场景**（> 50k）：`0.0`–`0.01`

**常见错误**：把 `weight_decay=0.1` 设给 LoRA 的 full finetune lr → 梯度被疯狂拉回 0，loss 不降。

---

## 5. 学习率调度相关参数

### 5.1 `lr_scheduler_type`

```python
lr_scheduler_type: str = "cosine"
```

**原理**：SFT 期间 lr 如何变化。对比：

```
linear:    lr ──────\_________  (线性衰减到 0)
cosine:    lr ──┐
                └─────╮_______  (余弦衰减，后期平缓)
constant:  lr ───────────────   (不变)
constant_with_warmup: lr 先热身，后不变
```

**经验**：

- **cosine**（默认推荐）：SFT 首选，最后 20% 几乎是小 lr 微调，有利于收敛
- **linear**：极短训练（< 500 步）可用
- **constant**：继续训练（resume）或大规模数据微调
- **constant_with_warmup**：RLHF / DPO 阶段常用，SFT 少用

**Cosine 的数学形式**：

$$
\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t - t_{\text{warm}}}{T - t_{\text{warm}}}\pi\right)\right)
$$

### 5.2 `warmup_ratio`

```python
warmup_ratio: float = 0.05   # 前 5% 步数 lr 从 0 线性升到 peak
```

**原理**：训练初期 Adam 的 $\hat{m}$、$\hat{v}$ 估计不准，大 lr 容易让梯度爆炸。Warmup 给优化器"预热"时间。

**经验**：

| 场景 | warmup_ratio |
|------|-------------|
| SFT 默认 | **0.03–0.05** |
| 小数据（总步数 < 500）| 0.1（绝对值至少 20 步）|
| 从已微调模型 resume | 0（已经预热过）|
| 全参微调大模型 | 0.1 更稳 |

**Warmup 步数 vs 比例**：

- `warmup_steps=100` 是绝对步数
- `warmup_ratio=0.05` 会自动算 `total_steps × 0.05`
- **两者只能二选一**。transformers 里 warmup_ratio 优先。

**诊断**：看 lr 曲线。如果前几个 log 的 `lr` 还没到峰值就开始下降 → warmup 太短或配置错了。

### 5.3 `max_grad_norm`（梯度裁剪）

```python
max_grad_norm: float = 1.0
```

**原理**：梯度爆炸时，把全局梯度 L2 范数裁到 `max_grad_norm`：

$$
g' = g \cdot \min\left(1, \frac{\text{max\_grad\_norm}}{\|g\|_2}\right)
$$

**经验**：

- SFT 默认 `1.0`，99% 场景不需要动
- 若 `grad_norm` 日志长期在 0.1–0.5 → 可以调到 `0.5` 更紧
- 若经常触发裁剪（log 里 `grad_norm` 一直贴着 1.0）→ 说明 lr 可能太大，优先降 lr 而不是放宽 clip

---

## 6. 正则化与稳定性参数

### 6.1 `lora_dropout`

```python
lora_dropout: float = 0.05
```

**原理**：LoRA 的 forward 是 $y = Wx + \frac{\alpha}{r} B A x$，dropout 加在 LoRA 路径上：

$$
y = Wx + \frac{\alpha}{r} B (\text{Dropout}(A x))
$$

**经验**：

- 默认 `0.05`，对应 5% 位置被置零
- 小数据（< 3k）：`0.1` 防过拟合
- 大数据（> 50k）：`0.0`，没必要正则
- **QLoRA 推荐 `0.05`**，抵消量化噪声带来的不稳定

### 6.2 `seed`

```python
seed: int = 42
```

控制：

- 数据 shuffle 顺序（每个 epoch）
- 验证集切分（见 `train.py` 里 `torch.randperm`）
- dropout 的随机 mask
- LoRA 权重 A 的 Kaiming 初始化

**经验**：消融实验时，**必须固定 seed**，否则方差会把实验结论淹没。跑 3 个不同 seed 取平均是论文标配。

---

## 7. 精度与显存优化参数

### 7.1 `bf16`

```python
bf16: bool = True
```

**三种精度对比**：

| 精度 | 指数位 | 尾数位 | 动态范围 | SFT 推荐度 |
|------|-------|-------|---------|-----------|
| fp32 | 8 | 23 | 极大 | ❌ 太慢 |
| fp16 | 5 | 10 | $\pm 65504$ | ⚠️ 容易溢出 |
| **bf16** | 8 | 7 | 和 fp32 同 | ✅ **首选** |

**原理**：bf16 牺牲精度换动态范围。大模型训练中，梯度值跨度可能达到 $10^{-8}$ 到 $10^3$，fp16 会溢出，bf16 不会。

**硬件要求**：

- bf16 需要 **Ampere 及以上架构**（A100/3090/4090/H100）
- V100 / T4 只能用 fp16 + loss scaling
- CPU 训练：只能 fp32

**陷阱**：fp16 必须配 `fp16_backend=amp` 和动态 loss scaling；bf16 不需要 loss scaling，设置 `bf16=True` 即可。

### 7.2 `gradient_checkpointing`

```python
gradient_checkpointing: bool = True
```

**原理**：用**时间换显存**——前向时不保存中间 activation，反向时重新计算。

**节省/代价**：

- 显存：activation 从 $O(L \cdot bs \cdot s \cdot h)$ 降到 $O(\sqrt{L} \cdot bs \cdot s \cdot h)$，**~30%–40% 显存**
- 速度：前向多算一次，整体训练**慢 20%–30%**

**经验**：

| GPU 显存 | 模型 | 开 GC？ |
|---------|------|---------|
| 24G | 7B | **必须开** |
| 48G | 7B LoRA | 可关（更快）|
| 80G | 7B full | 看 batch，够用就关 |

**易错点**：

- 开了 GC 后，必须调用 `model.enable_input_require_grads()`（见 `train.py:130`），否则 LoRA 的反向传播链会断
- `use_cache=True` 和 `gradient_checkpointing` 冲突，会被自动设成 `False`

### 7.3 `attn_implementation`（Flash Attention 2）

```python
attn_implementation = "flash_attention_2" if args.flash_attn else "eager"
```

**原理**：Flash Attention 通过 IO-aware 的 tiling 把 attention 的 $O(s^2)$ 显存变成 $O(s)$，速度提 2–4 倍。

**经验**：

- 只要模型和 GPU 支持，**一律开启**
- 要求：Ampere+ GPU，`flash-attn` 包安装成功
- Qwen2.5、Llama3 全系支持
- **唯一不开的场景**：调试数值问题时（flash attn 和 eager 在极端情况下有 $10^{-4}$ 级别差异）

---

## 8. LoRA / QLoRA / DoRA 专属参数

这一节是 LoRA 系列的核心，也是消融实验 (`ablation_runner.py`) 的主战场。

### 8.1 `lora_r`（LoRA 秩）

```python
lora_r: int = 16
```

**原理**：LoRA 假设 $\Delta W = B A$ 是低秩的，$r$ 控制这个秩。数学上：

$$
W \in \mathbb{R}^{d \times k}, \quad B \in \mathbb{R}^{d \times r}, \quad A \in \mathbb{R}^{r \times k}
$$

可训练参数 $= (d + k) \times r$，而 full fine-tune 是 $d \times k$。

**经验矩阵**：

| 任务难度 | r 推荐 | 说明 |
|---------|-------|------|
| 风格微调（客服、人设）| 4–8 | 任务简单，小秩足够 |
| 通用指令 SFT | **16**（默认）| 甜点区 |
| 代码 / 数学 / 推理 | 32–64 | 复杂任务需要更大容量 |
| 领域大模型（医学/法律）| 64–128 | 需要注入大量新知识 |

**消融结论（`ablation_config.lora_ranks=[8,16,32,64,128]`）**：

- 从 8 → 16：通常有 1%–3% 提升
- 从 16 → 32：0.5%–1% 提升
- 从 32 → 64：边际收益很小，显存/参数却翻倍
- **32 是性价比最高的选择**，但 16 是默认安全区

**重要定律**：`r` 不是越大越好。过大的 r 会让 LoRA 逼近 full-finetune，失去"低秩正则化"的优势，反而容易过拟合。

### 8.2 `lora_alpha`（LoRA 缩放系数）

```python
lora_alpha: int = 32
```

**原理**：LoRA 的实际输出是：

$$
h = Wx + \frac{\alpha}{r} (BA) x
$$

$\frac{\alpha}{r}$ 控制 LoRA 分支的**贡献强度**。

**关键经验法则**：

$$
\boxed{\alpha = 2r}
$$

这是业界默认，对应 `scaling = 2`。为什么是 2？

- $\alpha = r$（scaling=1）：LoRA 贡献较弱，学习慢，欠拟合风险
- $\alpha = 2r$：balanced sweet spot
- $\alpha = 4r$：LoRA 主导，训练初期不稳定，易梯度爆炸
- $\alpha$ 过大 + 学习率大 → NaN

**消融结论（`alpha_ratios=[1.0, 2.0, 4.0]`）**：

- 1.0：欠拟合，效果差
- **2.0：最优**，90% 场景选它
- 4.0：效果和 2.0 接近，但训练不稳（grad_norm 波动大）

**rsLoRA 变体**：$\text{scaling} = \alpha / \sqrt{r}$ 代替 $\alpha / r$，大 r 下更稳定。Hugging Face PEFT 用 `use_rslora=True` 开启。

### 8.3 `target_modules`（注入 LoRA 的层）

```python
target_modules: List[str] = field(default_factory=lambda: [
    "q_proj", "k_proj", "v_proj", "o_proj",      # 注意力
    "gate_proj", "up_proj", "down_proj",          # FFN
])
```

**原理**：LoRA 只在被选中的线性层注入低秩分支。

**四种常见配置**（对应 `ablation_config.target_module_groups`）：

| 配置 | 注入层 | 可训练参数比 | 效果 | 典型用例 |
|------|-------|------------|------|---------|
| `qv_only` | q, v | ~0.1% | 最差（原 LoRA 论文）| 仅演示 |
| `attn_only` | q, k, v, o | ~0.2% | 中等 | 风格微调 |
| `attn_ffn` | 全部 7 个线性 | ~0.7% | **最好**（QLoRA 论文）| 默认首选 |
| `all_linear` | 所有 nn.Linear | ~0.8% | 和 attn_ffn 接近 | PEFT 推荐写法 |

**关键结论**（QLoRA paper 实证）：

> **加 FFN 比加 attention 收益更大**。`gate_proj/up_proj/down_proj` 的参数量占 Transformer 70%+，是知识存储的主战场。

**经验**：默认用 `attn_ffn`。如果显存紧张，可以只用 `attn_only`，损失约 1–2%。**不要用 `qv_only`**，那是 2021 年的古老配置。

### 8.4 QLoRA 专属

```python
load_in_4bit: bool = True                 # 开启 4-bit 量化
bnb_4bit_compute_dtype: str = "bfloat16"  # 反量化后的计算精度
bnb_4bit_quant_type: str = "nf4"          # nf4 / fp4
bnb_4bit_use_double_quant: bool = True    # 双重量化
```

**逐个解释**：

- **`load_in_4bit`**：基座模型权重从 fp16 压到 4-bit 存储。7B 模型从 14GB → 3.8GB。
- **`bnb_4bit_compute_dtype`**：计算时反量化成什么精度。**一律用 bf16**，fp16 在 QLoRA 下有 NaN 风险。
- **`bnb_4bit_quant_type`**：
  - `nf4`（NormalFloat4）：理论最优，基于正态分布的非均匀量化
  - `fp4`：标准 4-bit float
  - **一律用 `nf4`**，精度比 fp4 高约 0.5 PPL
- **`bnb_4bit_use_double_quant`**：
  - 量化常数本身也量化一次，再省 ~0.4 GB
  - 精度损失可忽略，**建议一直开**

**QLoRA 训练注意事项**：

1. 必须 `prepare_model_for_kbit_training(model)`，会：
   - 把 LayerNorm 转回 fp32（防止量化误差累积）
   - 把 lm_head 保持 fp16
   - 启用梯度
2. 学习率比 LoRA 降 10%–20%
3. 不能 merge 回 4-bit 权重 —— 合并前必须先反量化到 bf16

### 8.5 DoRA 专属

```python
use_dora: bool = True
```

**原理**：DoRA 把 $W$ 分解成方向和幅度：

$$
W = m \cdot \frac{V}{\|V\|_c}, \quad V = W_0 + BA
$$

- $m \in \mathbb{R}^{1 \times k}$ 是可学习的幅度（per column）
- $V/\|V\|_c$ 是归一化的方向，方向变化由 LoRA $BA$ 提供
- 参数量 ≈ LoRA + $k$，几乎不增加

**效果**：

- 比 LoRA 平均提升 0.5–2%（多个基准）
- 尤其在 `r` 较小时收益明显（$r=4$ 时 DoRA 接近 LoRA r=16 的效果）

**代价**：

- 训练慢 10%–15%（多一次 norm）
- 推理 0 代价（合并后就是普通线性层）
- 对超参不敏感，基本不用调

**经验**：**有时间就用 DoRA**，超参直接复用 LoRA 的。

---

## 9. DeepSpeed 配置参数

`ds_config.json` 的每一项都有讲究：

### 9.1 `zero_optimization.stage`

```json
"stage": 2
```

**ZeRO 三个 stage**：

| Stage | 切分内容 | 显存节省 | 通信开销 | 适用 |
|-------|---------|---------|---------|------|
| 1 | Optimizer states | ~4× | 低 | 单机多卡 |
| **2** | Stage1 + Gradients | ~8× | 中 | **单机多卡默认** |
| 3 | Stage2 + Parameters | ~16×+ | 高 | 多机训练 70B+ |

**经验**：

- SFT 单卡：**不需要 DeepSpeed**
- 单机 2–8 卡：**Stage 2**
- 显存爆了：Stage 3（但通信慢 30%）
- Stage 3 + offload_optimizer(cpu)：极限省显存，慢 2 倍

### 9.2 `offload_optimizer.device`

```json
"offload_optimizer": {"device": "none"}
```

选项：

- `"none"`：所有东西在 GPU
- `"cpu"`：AdamW 的 m/v 挪到 CPU 内存，省 GPU 显存 60%，慢 30%–50%
- `"nvme"`：挪到 SSD，极限场景（训 70B 在 4×24G 上），慢 3 倍+

### 9.3 `optimizer.betas`

```json
"betas": [0.9, 0.95]
```

**原理**：AdamW 的一阶、二阶动量衰减：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

**经验**：

- **`[0.9, 0.999]`**：标准 BERT/GPT 预训练
- **`[0.9, 0.95]`**：Llama、Qwen 等现代大模型推荐，对 loss 变化更敏感，适合 SFT
- SFT 首选 **`[0.9, 0.95]`**

### 9.4 `gradient_accumulation_steps: "auto"`

- `"auto"`：DeepSpeed 从 `TrainingArguments` 自动读取
- 写死数字：用手动指定的值
- SFT 建议 `"auto"`，避免两边不一致

### 9.5 `allgather_bucket_size / reduce_bucket_size`

```json
"allgather_bucket_size": 2e8,
"reduce_bucket_size": 2e8
```

**原理**：通信时把多个 tensor 打包成一个 bucket 发送，减少通信次数。

**经验**：

- 默认 `2e8`（200MB），适合大多数场景
- 显存紧张：降到 `5e7`（50MB），通信次数多但每次占内存少
- 高速网络（NVLink）：可以升到 `5e8`

---

## 10. 保存、日志、评估参数

### 10.1 `logging_steps`

```python
logging_steps: int = 10
```

- 每 N 步打一次 loss
- **经验**：总步数 / 100 ≈ 最优 logging_steps，既不刷屏也不丢信息
- 调试用：`1`（每步都看）
- 生产长训练：`50`

### 10.2 `save_steps` / `save_total_limit`

```python
save_steps: int = 200
save_total_limit: int = 3
```

**经验**：

- `save_steps` 设到总步数的 1/10 ~ 1/5 之间
- `save_total_limit=3`：只保留最近 3 个 ckpt，防磁盘爆
- 用 `load_best_model_at_end=True` + `metric_for_best_model="eval_loss"` 实现 early stop（训练完自动加载最佳 ckpt）

**LoRA 特殊**：每个 ckpt 只有 adapter（几十 MB），可以 `save_total_limit` 设大一点如 `5`。全参微调每个 ckpt 是 GB 级别，限制 `2–3` 即可。

### 10.3 `eval_steps` / `eval_strategy`

```python
eval_steps: int = 100
eval_strategy: str = "steps"
```

- `"no"`：不评估（不推荐）
- `"steps"`：每 `eval_steps` 步评估一次，**SFT 推荐**
- `"epoch"`：每 epoch 末评估一次，粗粒度

**经验**：`eval_steps` 和 `save_steps` 设同值，方便结合 `load_best_model_at_end`。

---

## 11. 一眼看懂的「参数速查表」

**按优先级排序** —— 调参时从上往下改：

| 优先级 | 参数 | 默认值 | 波动范围 | 敏感度 |
|-------|------|-------|---------|--------|
| 🔴 P0 | `learning_rate` | 2e-4 (LoRA) / 2e-5 (full) | ×0.5–×2 | 极高 |
| 🔴 P0 | `num_train_epochs` | 3 | 1–5 | 极高 |
| 🔴 P0 | `method` | lora | 4 选 1 | 极高 |
| 🟠 P1 | `lora_r` | 16 | 8 / 16 / 32 / 64 | 高 |
| 🟠 P1 | `lora_alpha` | 32 (=2r) | =r / =2r / =4r | 高 |
| 🟠 P1 | `max_seq_length` | 512 | 256 / 1024 / 2048 | 高（显存）|
| 🟠 P1 | `per_device_train_batch_size` | 4 | 1–16 | 高（显存）|
| 🟡 P2 | `gradient_accumulation_steps` | 4 | 1–16 | 中 |
| 🟡 P2 | `target_modules` | attn_ffn | 4 档 | 中 |
| 🟡 P2 | `warmup_ratio` | 0.05 | 0.0–0.1 | 中 |
| 🟡 P2 | `weight_decay` | 0.01 | 0.0–0.1 | 中 |
| 🟢 P3 | `lr_scheduler_type` | cosine | 4 选 1 | 低 |
| 🟢 P3 | `lora_dropout` | 0.05 | 0.0–0.1 | 低 |
| 🟢 P3 | `max_grad_norm` | 1.0 | 0.5–1.0 | 低 |
| 🟢 P3 | `bf16` | True | True/False | 硬件决定 |
| 🟢 P3 | `gradient_checkpointing` | True | True/False | 显存决定 |

---

## 12. 四种典型场景的「整套配置」

### 12.1 场景 A：24G 卡 + Qwen2.5-0.5B + LoRA（学习起步）

```python
cfg = SFTConfig(
    model_name="Qwen/Qwen2.5-0.5B",
    method="lora",
    max_seq_length=512,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,      # 有效 batch = 16
    num_train_epochs=3,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=False,       # 0.5B 小模型不需要
)
lcfg = LoRAConfig(lora_r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"])
```

**预期**：7B token 数据（~10k 样本）约 30–60 分钟，loss 从 ~2.5 降到 ~1.2。

### 12.2 场景 B：24G 卡 + Qwen2.5-7B + QLoRA（工业实战）

```python
cfg = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    method="qlora",
    max_seq_length=1024,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,      # 有效 batch = 32
    num_train_epochs=3,
    learning_rate=1.5e-4,               # 比 LoRA 降一档
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.0,                   # QLoRA 推荐关 WD
    bf16=True,
    gradient_checkpointing=True,        # 7B 必开
)
lcfg = LoRAConfig(
    lora_r=32, lora_alpha=64, lora_dropout=0.05,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj"],
    load_in_4bit=True,
    bnb_4bit_compute_dtype="bfloat16",
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
)
```

**预期**：30k 样本约 6–10 小时，显存占用 ~14GB。

### 12.3 场景 C：8×A100 + 7B + 全参微调（高质量对齐）

```python
cfg = SFTConfig(
    model_name="Qwen/Qwen2.5-7B",
    method="full_finetune",
    max_seq_length=2048,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,      # 有效 batch = 4×2×8 = 64
    num_train_epochs=2,
    learning_rate=2e-5,                 # 全参小 lr
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    bf16=True,
    gradient_checkpointing=True,
    deepspeed_config="ds_config.json",  # Stage 2
)
```

### 12.4 场景 D：小数据（< 3k 条）领域微调

```python
# 关键：少量数据容易过拟合，加强正则
cfg = SFTConfig(
    num_train_epochs=5,                 # 小数据多跑几轮
    learning_rate=1e-4,                 # lr 降一档
    warmup_ratio=0.1,                   # 加长 warmup
    weight_decay=0.05,                  # 加强正则
    eval_steps=20,                      # 高频评估
    save_steps=20,
)
lcfg = LoRAConfig(
    lora_r=8,                           # 小 r 防过拟合
    lora_alpha=16,
    lora_dropout=0.1,                   # dropout 加大
)
# 搭配 TrainingArguments(load_best_model_at_end=True, metric_for_best_model="eval_loss")
```

---

## 13. 调参流程与诊断信号

### 13.1 标准调参流程（5 步法）

```
1. 先跑默认配置 100 步
   └─ 观察 grad_norm、loss 下降曲线、显存占用
      ├─ grad_norm > 5    → lr 减半
      ├─ loss 完全不降     → lr ×2 或检查 label mask
      └─ OOM              → 降 batch 或开 GC
                            
2. 跑完整 1 epoch
   └─ 看 eval_loss 是否下降
      ├─ 下降          → 继续跑完所有 epoch
      └─ 不降或上升    → 降 lr 或减 epoch
                        
3. 跑完默认 epoch（如 3）
   └─ 观察 eval_loss 拐点
      ├─ 在第 2 epoch 末拐点 → 下次设 epoch=2
      └─ 一直下降           → 可以加 epoch 到 4–5
                          
4. 生成质量肉眼评估（inference.py）
   └─ 重复/复读 → 过拟合，降 epoch 或加 dropout
       输出混乱 → 欠拟合，加 lr 或 epoch 或 r
                
5. 消融实验（ablation_runner.py）
   └─ 锁定最优 r / alpha / target_modules
```

### 13.2 日志关键信号解读

**正常训练日志**（健康）：

```
step  100  loss: 2.104  lr: 1.98e-04  grad_norm: 0.523  eval_loss: 2.211
step  200  loss: 1.856  lr: 1.95e-04  grad_norm: 0.487  eval_loss: 1.987
step  300  loss: 1.632  lr: 1.90e-04  grad_norm: 0.451  eval_loss: 1.843
```

- loss 平滑下降 ✓
- grad_norm 稳定在 0.4–0.6 ✓
- eval_loss 始终低于 train_loss ✓

**异常 1：学习率太大**

```
step  100  loss: 2.5   grad_norm: 8.3
step  200  loss: 3.8   grad_norm: 12.5
step  300  loss: NaN   grad_norm: NaN
```

→ 立即降 lr 到一半，重启。

**异常 2：过拟合**

```
epoch 1:  train_loss 1.8  eval_loss 1.9
epoch 2:  train_loss 1.3  eval_loss 1.7
epoch 3:  train_loss 0.8  eval_loss 1.9   ← eval_loss 反弹
```

→ 下次训练 `num_train_epochs=2`，或加 `weight_decay=0.05`。

**异常 3：欠拟合**

```
epoch 1:  train_loss 2.5  eval_loss 2.4
epoch 2:  train_loss 2.3  eval_loss 2.3
epoch 3:  train_loss 2.2  eval_loss 2.2
```

→ 加 lr、加 `lora_r`、或检查数据质量。

**异常 4：label mask 错误**

```
epoch 1 loss: 5.6 → 0.3 (突然掉)
```

→ loss 异常低（< 0.5）通常是 label mask 把 user 部分也放开了，模型在"抄"而不是"学"。检查 `dataset.py` 的统计输出是否 **loss token 占比在 20%–40%**。

### 13.3 过拟合三把尺子（对应 `overfit_diagnosis.py`）

1. **eval_loss 拐点**：最可靠的量化指标
2. **Distinct-N 多样性**：< 0.5 警告，< 0.3 严重过拟合
3. **重复率**（Repetition Rate）：> 0.15 警告，> 0.3 严重过拟合

---

## 附录 A：一张图看懂 SFT 参数依赖关系

```
                    ┌──────────────────┐
                    │  model_name      │
                    │  (决定一切上限)  │
                    └──────────┬───────┘
                               │
                               ▼
       ┌──────────────┬────────────────┬──────────────┐
       │  method      │  max_seq_length │  dataset     │
       │  (参数结构)  │  (显存平方级)   │  (质量>数量) │
       └──────┬───────┴─────┬──────────┴──────┬───────┘
              │             │                  │
              ▼             ▼                  ▼
       ┌──────────────────────────────────────────┐
       │  per_device_bs × gradient_accum          │
       │  = 有效 batch (目标 16–64)               │
       └──────────────────────┬───────────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │  learning_rate (方法决定量级) │ ◄── cosine + warmup
              └───────┬───────────────────────┘
                      │
                      ▼
              ┌───────────────────────────────┐
              │  num_train_epochs + eval_loss │ ◄── early stop 信号
              └───────┬───────────────────────┘
                      │
                      ▼
                  最终模型
                      │
                      ▼
              ┌─────────────────────┐
              │  overfit_diagnosis  │
              │  (Distinct-N, 重复) │
              └─────────────────────┘
```

---

## 附录 B：常见错误 FAQ

| 现象 | 根因 | 修复 |
|------|------|------|
| loss 一直 NaN | fp16 溢出 / lr 过大 | 换 bf16 / lr 减半 |
| loss 卡在 5.x 不降 | label 全是 -100 | 检查 ChatML 模板是否和 tokenizer 匹配 |
| 显存突然爆炸 | 没开 GC 或 batch 太大 | `gradient_checkpointing=True` |
| LoRA 训完效果=基座 | adapter 没加载 | 推理时要 `PeftModel.from_pretrained` |
| eval_loss 比 train_loss 低很多 | 数据泄漏 | 检查 train/eval 切分是否有重叠 |
| 多卡训练速度没提升 | DeepSpeed Stage 3 通信瓶颈 | 降到 Stage 2，或装 NVLink |
| QLoRA 训练异常慢 | `bnb_4bit_compute_dtype` 设错 | 必须是 bf16 |
| DoRA 报错 `use_dora not found` | PEFT 版本太旧 | `pip install peft>=0.9.0` |

---

## 结语

把这份手册当**工作台**用：

1. 训练前照着 [§11 速查表](#11-一眼看懂的参数速查表) 过一遍配置
2. 训练中盯 [§13.2](#132-日志关键信号解读) 的异常信号
3. 训练后用 [§13.3](#133-过拟合三把尺子对应-overfit_diagnosispy) 的三把尺子验收

SFT 不是玄学，每个参数背后都有一道数学题和一个工程约束。理解原理 → 读懂信号 → 快速迭代，就是这份手册的全部。
