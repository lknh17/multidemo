# p03 SFT 指令微调 — 原理详解

> 本文详细介绍 SFT（Supervised Fine-Tuning，监督微调）的核心原理，包括 ChatML 对话模板、Label Masking、LoRA/QLoRA/DoRA 的数学推导、以及过拟合诊断方法。

---

## 1. 什么是 SFT？为什么需要 SFT？

### 预训练 vs SFT 的本质区别

**预训练**（Pre-training）的目标是让模型学习语言本身的规律——给定前面的 token，预测下一个 token。预训练后的模型拥有丰富的知识，但它只会"续写"，不会"对话"。

**SFT**（监督微调）的目标是让模型学会按照人类的指令格式进行回复。通过在"指令-回答"数据对上训练，模型学会了：
- 理解用户的意图
- 按照特定格式组织回答
- 给出有帮助的、准确的回复

一个形象的比喻：预训练像是让学生博览群书，SFT 则是教学生如何回答考试题目。

### SFT 在 LLM 训练流程中的位置

```
预训练 (P) → 继续预训练 (CPT) → SFT → RLHF/DPO
    ↓              ↓                ↓        ↓
 学习语言       注入领域知识      学会对话   对齐人类偏好
```

SFT 是让模型从"知识库"变成"助手"的关键一步。

---

## 2. ChatML 对话模板

### 为什么需要对话模板？

LLM 本质上只是在做 next-token prediction，它无法天然区分"谁在说话"。对话模板（Chat Template）通过特殊的格式标记来标识不同角色的发言边界。

### Qwen2.5 的 ChatML 格式

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{用户问题}<|im_end|>
<|im_start|>assistant
{模型回答}<|im_end|>
```

关键标记：
- `<|im_start|>`: 角色发言开始
- `<|im_end|>`: 角色发言结束
- `system/user/assistant`: 角色标识

这些特殊 token 在 tokenizer 的词表中有专门的 token_id，模型通过学习这些边界标记来理解对话结构。

### 多轮对话扩展

```
<|im_start|>system\n...<|im_end|>
<|im_start|>user\n第一轮问题<|im_end|>
<|im_start|>assistant\n第一轮回答<|im_end|>
<|im_start|>user\n第二轮问题<|im_end|>
<|im_start|>assistant\n第二轮回答<|im_end|>
```

---

## 3. Label Masking — SFT 的核心技巧

### 为什么要 Label Masking？

在预训练中，所有 token 都参与 loss 计算——模型学习预测每一个 token。

但在 SFT 中，我们只希望模型学习**如何回答**，而不是学习"如何提问"或"如何写系统提示"。因此：

- **system + user 部分**：label 设为 -100（不计算 loss）
- **assistant 部分**：label 保留原始 token_id（计算 loss）

### 数学表示

给定序列 x = [x₁, x₂, ..., xₙ]，标准的 CLM loss 为：

$$L = -\frac{1}{N} \sum_{i=1}^{N} \log P(x_i | x_{<i})$$

Label Masking 后的 loss：

$$L_{SFT} = -\frac{1}{|A|} \sum_{i \in A} \log P(x_i | x_{<i})$$

其中 A 是 assistant 回复的 token 位置集合。

### 为什么这样做有效？

1. **避免学习提问模式**：如果模型也学习 user 的提问，它可能在回复中混入提问句式
2. **聚焦学习目标**：有限的训练步数内，让模型集中学习回答的模式
3. **保护基座能力**：system/user 部分的梯度为 0，减少对原始知识的干扰

---

## 4. LoRA — 低秩适配

### LoRA 的核心思想

全参微调需要更新所有 W ∈ ℝ^{d×k} 的权重，显存开销巨大。LoRA（Low-Rank Adaptation）的核心假设是：

> 微调时的权重变化 ΔW 是低秩的，可以分解为两个小矩阵的乘积。

### 数学推导

原始线性层：y = Wx

LoRA 修改后：y = Wx + BAx

其中：
- B ∈ ℝ^{d×r}：下投影矩阵
- A ∈ ℝ^{r×k}：上投影矩阵
- r << min(d, k)：秩（rank），通常 8-128

实际计算中还有 scaling factor：

$$y = Wx + \frac{\alpha}{r} \cdot BAx$$

其中 α 是 lora_alpha 超参数，α/r 控制 LoRA 的更新幅度。

### 参数效率分析

以 Qwen2.5-0.5B 为例，hidden_dim = 896：

| 层 | 原始参数 | LoRA (r=16) 参数 | 节省比例 |
|----|---------|-----------------|---------| 
| q_proj (896→896) | 802,816 | 28,672 | 96.4% |
| k_proj (896→128) | 114,688 | 16,384 | 85.7% |
| v_proj (896→128) | 114,688 | 16,384 | 85.7% |
| 全部注意力层 | ~4M/层 | ~0.15M/层 | 96% |

### LoRA 初始化

- A 矩阵：使用 Kaiming 正态初始化
- B 矩阵：初始化为零矩阵
- 效果：训练开始时 BA = 0，不影响原始模型输出

---

## 5. QLoRA — 量化 + LoRA

### QLoRA 的三重优化

QLoRA 在 LoRA 基础上引入三项创新：

1. **4-bit NormalFloat (NF4) 量化**：将基座模型权重量化为 4-bit
2. **双重量化**：对量化常数再做一次量化，进一步压缩
3. **分页优化器**：利用 NVIDIA 统一内存管理避免 OOM

### NF4 量化原理

NF4（Normal Float 4）是专门为正态分布权重设计的 4-bit 数据类型。

核心思想：神经网络权重近似服从正态分布 N(0, σ²)。NF4 的 16 个量化值选取在正态分布的等概率分位点上：

```
对于标准正态分布，将概率空间[0,1]等分为16份
每个区间的中心概率对应一个量化值
```

这样，出现频率高的权重值（接近 0 的）获得更细的表示精度。

### 显存对比

| 组件 | 全参 bf16 | LoRA bf16 | QLoRA 4-bit |
|------|----------|-----------|-------------|
| 基座参数 | 0.93 GB | 0.93 GB | 0.25 GB |
| LoRA 参数 | — | 0.01 GB | 0.01 GB |
| 优化器状态 | 3.73 GB | 0.04 GB | 0.04 GB |
| 合计 | ~8.4 GB | ~3.7 GB | ~2.8 GB |

---

## 6. DoRA — 权重分解的 LoRA

### DoRA 的创新

DoRA（Weight-Decomposed Low-Rank Adaptation）将权重分解为方向和幅度两部分，分别进行适配。

### 数学推导

将权重矩阵 W 分解为：

$$W = m \cdot \frac{V}{||V||_c}$$

其中：
- m ∈ ℝ^d：幅度向量（magnitude）
- V ∈ ℝ^{d×k}：方向矩阵（direction）
- ||V||_c：按列的 L2 范数

DoRA 的更新方式：
- 幅度 m：直接作为可训练参数
- 方向 V：使用 LoRA 更新，V' = V + BA

$$W' = (m + Δm) \cdot \frac{V + BA}{||V + BA||_c}$$

### 为什么 DoRA 更好？

研究发现全参微调时，权重变化的方向和幅度是解耦的。LoRA 将两者耦合在一起学习，而 DoRA 将它们分开，更接近全参微调的行为模式。

---

## 7. 消融实验设计

### LoRA Rank 的选择

Rank 控制 LoRA 的表达能力：
- **太小**（r=4）：欠拟合，无法学习复杂模式
- **太大**（r=256）：接近全参微调，失去效率优势
- **推荐范围**：SFT 任务通常 r=16-64 效果最佳

### Alpha/Rank 比值

α/r 控制 LoRA 更新的幅度：
- α/r = 1：标准幅度
- α/r = 2：较大幅度（常用）
- α/r = 4：大幅度（可能不稳定）

### Target Modules 选择

| 方案 | 模块 | 参数量 | 效果 |
|------|------|--------|------|
| QV-only | q_proj, v_proj | 最少 | 基础 |
| Attention | q/k/v/o_proj | 中等 | 推荐 |
| Attn+FFN | + gate/up/down | 最多 | 最好 |

---

## 8. 学习率策略

### SFT 的学习率选择

不同微调方法对学习率的敏感度不同：

| 方法 | 推荐学习率 | 原因 |
|------|-----------|------|
| 全参微调 | 1e-5 ~ 5e-5 | 参数多，lr 太大会遗忘 |
| LoRA | 1e-4 ~ 5e-4 | 参数少，需要较大 lr |
| QLoRA | 1e-4 ~ 5e-4 | 同 LoRA |
| DoRA | 1e-4 ~ 3e-4 | 略保守，幅度分量敏感 |

### Warmup 的作用

SFT 初期模型输出与目标差距大，loss 很高。如果直接用大学习率，梯度会很大，导致训练不稳定。Warmup 让学习率从很小的值逐渐增大，帮助模型"热身"。

---

## 9. 过拟合诊断

### SFT 过拟合的表现

1. **训练 loss 持续下降，验证 loss 反弹**
2. **Distinct-N 指标下降**：生成文本越来越重复
3. **重复率上升**：模型陷入重复循环（如"我是一个AI助手我是一个AI助手..."）
4. **输出趋同**：对不同问题给出几乎相同的回答

### Distinct-N 指标

$$\text{Distinct-N} = \frac{|\text{unique N-grams}|}{|\text{total N-grams}|}$$

- Distinct-1 > 0.9：字级别多样性好
- Distinct-2 > 0.7：二元组多样性好
- Distinct-3 > 0.5：三元组多样性好

### 防止过拟合的策略

1. **减少训练轮数**：SFT 通常 1-3 个 epoch 足够
2. **增大 LoRA dropout**：0.05-0.1
3. **早停**：监控验证 loss
4. **数据多样性**：混合多个数据集
5. **降低学习率**：过拟合时尝试减半

---

## 10. LoRA 权重合并

### merge_and_unload() 原理

合并过程非常简单：

```
W_merged = W_base + (alpha/r) × B × A
```

1. 计算 LoRA 增量：ΔW = (α/r) × B × A
2. 加到原始权重上：W_new = W_old + ΔW
3. 移除 LoRA 层，恢复原始网络结构

合并后的模型：
- 与原始架构完全相同
- 不需要 PEFT 库即可推理
- 方便部署和分发

### 合并 vs 不合并

| 特性 | 未合并 | 已合并 |
|------|--------|--------|
| 推理依赖 | 需要 PEFT | 不需要 |
| 推理速度 | 略慢（额外矩阵乘） | 正常 |
| 可切换 | 可以动态切换 adapter | 不行 |
| 文件大小 | 只存 adapter (几MB) | 完整模型 (几GB) |

---

## 11. 实践建议

### 数据质量 > 数据数量

SFT 对数据质量极其敏感。1000 条高质量数据的效果可能超过 10 万条低质量数据。

### 推荐起步配置

```python
# LoRA 微调推荐配置
method = "lora"
lora_r = 16
lora_alpha = 32
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                  "gate_proj", "up_proj", "down_proj"]
learning_rate = 2e-4
num_epochs = 3
```

### 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| Loss 不下降 | 学习率太小 | 增大 lr |
| Loss 变 NaN | 学习率太大 | 减小 lr，检查数据 |
| 生成重复 | 过拟合 | 减少 epoch，增大 dropout |
| 输出为空 | 模板错误 | 检查 ChatML 格式 |
| OOM | 显存不足 | 用 QLoRA 或减小 batch |
