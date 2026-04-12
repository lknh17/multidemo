# p06 原理详解 - MLLM 多模态视觉微调

> 本文档详细讲解多模态大语言模型（MLLM）的理论基础：架构设计、视觉 token 注入、训练策略、数据构造与评估方法。

---

## 1. 多模态大语言模型概述

### 什么是 MLLM

多模态大语言模型（Multimodal Large Language Model, MLLM）是在大语言模型（LLM）基础上，增加视觉感知能力的模型。其核心思想是：**将图像编码为一系列"视觉 token"，与文本 token 一起送入 LLM 处理**。

MLLM 的发展历程：

| 阶段 | 代表模型 | 时间 | 特点 |
|------|---------|------|------|
| 早期探索 | VisualGPT, Flamingo | 2022 | 验证"视觉 token + LLM"的可行性 |
| 方法论建立 | LLaVA, InstructBLIP | 2023 | 两阶段训练范式（预训练 + 指令微调）|
| 工程化落地 | Qwen-VL, GPT-4V | 2023-2024 | 高分辨率、多图、视频理解 |
| 统一架构 | Qwen2.5-VL, InternVL2 | 2024-2025 | 动态分辨率、端到端优化 |

### MLLM vs 传统多模态模型

| 维度 | 传统多模态（如 CLIP） | MLLM（如 Qwen2.5-VL） |
|------|---------------------|----------------------|
| 输出形式 | 嵌入向量（用于检索/分类） | 自然语言（开放式回答） |
| 交互方式 | 固定任务头 | 指令驱动（对话式） |
| 泛化能力 | 局限于训练任务 | 零样本跨任务泛化 |
| 推理能力 | 无 | 可进行链式推理 |
| 知识利用 | 仅视觉-语言对齐 | 可利用 LLM 的世界知识 |

MLLM 的关键突破在于：不再为每个视觉任务设计专用模型，而是用一个统一的语言模型来处理所有视觉理解任务。

---

## 2. 三阶段架构

MLLM 的核心架构由三个模块组成：

### 视觉编码器（Vision Encoder）

负责将图像像素转换为高维特征向量：

$$\mathbf{V} = \text{VisionEncoder}(\mathbf{I}) \in \mathbb{R}^{N_v \times D_v}$$

其中 $N_v$ 是视觉 token 数量，$D_v$ 是视觉特征维度。

常用视觉编码器：

| 编码器 | 参数量 | 特点 | 代表模型 |
|--------|-------|------|---------|
| ViT-L/14 (CLIP) | 300M | CLIP 预训练，对齐能力强 | LLaVA |
| ViT-G (EVA-CLIP) | 1B | 更大容量，更细粒度 | InternVL |
| ViT (自训练) | 600M | 动态分辨率，Native Resolution | Qwen2.5-VL |
| SigLIP | 400M | Sigmoid Loss，更稳定 | PaliGemma |

### 跨模态投影器（Cross-modal Projector）

将视觉特征映射到 LLM 的嵌入空间：

$$\mathbf{H}_v = \text{Projector}(\mathbf{V}) \in \mathbb{R}^{N_v' \times D_t}$$

其中 $D_t$ 是 LLM 的隐藏维度。投影器可以改变 token 数量（$N_v' \neq N_v$）。

常见投影器设计：

| 类型 | 结构 | 特点 |
|------|------|------|
| Linear | 单层线性变换 | 最简单，LLaVA-1.0 使用 |
| MLP | 两层 MLP + GELU | LLaVA-1.5 使用，效果更好 |
| Q-Former | 可学习 Query + Cross-Attention | BLIP-2 使用，可压缩 token 数 |
| Perceiver Resampler | 类似 Q-Former | Flamingo 使用 |
| Spatial Merge | 相邻 token 合并 | Qwen2.5-VL 使用，保留空间信息 |

### 大语言模型（LLM）

核心推理引擎，接收混合的视觉 + 文本 token 序列：

$$\mathbf{Y} = \text{LLM}([\mathbf{H}_v; \mathbf{H}_t])$$

LLM 的优势：
- **世界知识**: 预训练积累的海量知识
- **推理能力**: 链式思维（Chain-of-Thought）
- **指令跟随**: 理解用户意图并格式化输出
- **上下文学习**: Few-shot 能力

---

## 3. Qwen2.5-VL 架构详解

Qwen2.5-VL 是目前开源 MLLM 中性能最强的之一，其架构有几个独特设计：

### 动态分辨率（Native Resolution）

传统做法是将所有图像 resize 到固定尺寸（如 224×224），这会丢失细节信息。Qwen2.5-VL 采用**动态分辨率**策略：

1. 保持图像原始宽高比
2. 将图像调整到合适的分辨率范围（由 `min_pixels` 和 `max_pixels` 控制）
3. 确保宽和高都是 28 的倍数（patch size）
4. 视觉 token 数量 = (H/28) × (W/28)

$$N_{\text{tokens}} = \frac{H}{28} \times \frac{W}{28}$$

**优势**：高分辨率图像保留更多细节（如 OCR 场景），低分辨率图像节省计算。

### Spatial Merge 投影器

Qwen2.5-VL 的投影器使用 Spatial Merge 策略：

1. 将相邻的 2×2 个 patch token 合并为一个
2. 合并后通过 MLP 投影到 LLM 维度
3. Token 数量减少 4 倍，大幅节省 LLM 的计算量

$$N_{\text{merged}} = \frac{N_{\text{tokens}}}{4}$$

### 3D RoPE 位置编码

为了让 LLM 理解视觉 token 的空间位置，Qwen2.5-VL 使用了 3D 旋转位置编码（3D RoPE）：

- **维度 1**: 时间维度（视频帧序号，图像为 0）
- **维度 2**: 高度维度（patch 的行坐标）
- **维度 3**: 宽度维度（patch 的列坐标）

这使得模型能理解"左上角"、"右下方"等空间概念。

---

## 4. 视觉 Token 注入机制

### Token 混合序列

MLLM 的输入是一个混合序列，包含视觉 token 和文本 token：

```
[System Prompt] [<image_start>] [vis_1, vis_2, ..., vis_N] [<image_end>] [User Text] [Assistant Response]
```

其中：
- `<image_start>` 和 `<image_end>` 是特殊分隔符
- `vis_1` 到 `vis_N` 是视觉编码器输出的视觉 token
- 视觉 token 和文本 token 使用相同的 Transformer 层处理

### 注意力掩码设计

在自回归生成中，注意力掩码确保：
1. 所有 token 可以看到之前的所有 token（因果掩码）
2. 视觉 token 之间可以互相看到（双向注意力，可选）
3. 文本 token 可以看到所有视觉 token

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + \text{Mask}\right)V$$

### 视觉 token 数量控制

视觉 token 太多会：
- 增加计算成本（LLM 的复杂度与序列长度平方成正比）
- 挤占文本 token 的位置
- 引入冗余信息

视觉 token 太少会：
- 丢失图像细节
- 影响 OCR 等需要高分辨率的任务

Qwen2.5-VL 通过 `min_pixels` 和 `max_pixels` 参数平衡这一 tradeoff。

---

## 5. 训练策略

### 两阶段训练范式

大多数 MLLM 采用两阶段训练：

**Stage 1: 预训练（Alignment）**

- **目标**: 让视觉特征与 LLM 的文本空间对齐
- **数据**: 大规模图文对（如 LAION、CC3M），约 500M-1B 对
- **策略**: 冻结视觉编码器和 LLM，只训练投影器
- **学习率**: 较大（1e-3 ~ 1e-4）
- **Epoch**: 1

**Stage 2: 指令微调（Instruction Tuning）**

- **目标**: 让模型学会理解指令并生成结构化回答
- **数据**: 高质量指令数据（如 LLaVA-Instruct），约 100K-1M 条
- **策略**: 冻结视觉编码器，微调 LLM + 投影器
- **学习率**: 较小（1e-5 ~ 2e-5）
- **Epoch**: 1-3

### 冻结策略的理论基础

**为什么冻结视觉编码器？**

1. **预训练质量高**: CLIP/SigLIP 等视觉编码器已在海量数据上预训练，特征质量很好
2. **避免灾难性遗忘**: 微调可能破坏视觉编码器的泛化能力
3. **节省显存**: 视觉编码器通常有 300M-1B 参数
4. **加速训练**: 不需要计算视觉编码器的梯度

**什么时候需要解冻？**

1. 视觉任务与预训练差距大（如医学影像）
2. 需要更细粒度的视觉特征
3. 有足够的数据和计算资源

### LoRA 应用于 MLLM

在 MLLM 中使用 LoRA 的最佳实践：

$$W' = W + \Delta W = W + BA$$

其中 $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, $r \ll d$。

- **只对 LLM 部分使用 LoRA**: 不修改视觉编码器和投影器
- **目标模块**: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
- **秩的选择**: r=16-32 通常够用（视觉指令微调不需要学习太多新知识）
- **Alpha 设置**: α = 2r（经验值）

---

## 6. 多模态数据构造

### 指令数据格式

高质量的指令数据是 MLLM 微调的关键。一条数据包含：

```json
{
    "image": "path/to/image.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\n请描述这张图片中的场景。"},
        {"from": "gpt", "value": "图片展示了一个公园的场景。在画面中央，一个小女孩正在荡秋千..."}
    ]
}
```

### 数据类型

| 类型 | 比例 | 示例 | 能力 |
|------|------|------|------|
| 详细描述 | 30% | "描述这张图片的所有细节" | 全面理解 |
| 简短描述 | 20% | "一句话描述这张图" | 摘要能力 |
| VQA | 25% | "图中有几辆车？" | 问答能力 |
| 推理 | 15% | "这是什么季节？为什么？" | 推理能力 |
| OCR/文档 | 10% | "读出招牌上的文字" | 文字识别 |

### 数据质量要素

1. **图像多样性**: 覆盖不同场景、光照、角度
2. **问题多样性**: 不同粒度、不同思维层次的问题
3. **回答质量**: 准确、详细、有结构
4. **格式一致性**: 统一的对话格式
5. **平衡性**: 各类型数据比例均衡

### 负样本构造

为了增强模型的抗幻觉能力，可以构造负样本：

- **不相关图文对**: 随机配对图像和文本
- **错误描述**: 故意修改描述中的物体、颜色、数量
- **拒绝回答**: 当图像无法回答问题时，模型应说"无法确定"

---

## 7. 评估方法

### Benchmark 评估

| Benchmark | 评估能力 | 指标 | 说明 |
|-----------|---------|------|------|
| MMBench | 综合能力 | 准确率 | 20+ 维度评估 |
| SEED-Bench | 视觉理解 | 准确率 | 图像+视频 |
| TextVQA | OCR + 问答 | 准确率 | 图像中的文字 |
| DocVQA | 文档理解 | ANLS | 文档/表格问答 |
| ChartQA | 图表理解 | 准确率 | 图表数据提取 |
| RealWorldQA | 现实世界 | 准确率 | 实际场景理解 |

### 常用评估指标

**准确率（Accuracy）**: 选择题/判断题的正确率

**ANLS（Average Normalized Levenshtein Similarity）**: 衡量 OCR 类任务的答案相似度

$$\text{ANLS} = \frac{1}{N}\sum_{i=1}^{N} \max\left(1 - \frac{\text{NL}(a_i, g_i)}{\max(|a_i|, |g_i|)}, 0\right)$$

**CIDEr**: 衡量描述文本与参考描述的相似度，常用于 captioning 任务

### 定性评估

除了定量指标，定性评估也很重要：

- **幻觉检测**: 模型是否描述了图中不存在的物体
- **细节准确性**: 颜色、数量、位置是否正确
- **推理合理性**: 推理过程是否逻辑自洽
- **指令遵循**: 是否按要求格式输出
