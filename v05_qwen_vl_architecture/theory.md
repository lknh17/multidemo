# Qwen-VL 架构深度解析

## 1. 整体架构

Qwen-VL 由三部分组成：

```
输入图像 → Vision Encoder (ViT) → Visual Resampler → [视觉 tokens]
                                                          ↓
                                               与文本 tokens 拼接
                                                          ↓
                                               Qwen LLM (Decoder)
                                                          ↓
                                                     文本输出
```

### 三大模块
1. **Vision Encoder**: 基于 ViT-bigG (EVA-CLIP)，提取图像特征
2. **Visual Resampler (Perceiver Resampler)**: 将可变数量的视觉 token 压缩为固定数量
3. **Qwen LLM**: 自回归语言模型，处理拼接后的多模态 token 序列

## 2. Vision Encoder

- 基于 **ViT-bigG/14**（约 2B 参数的超大 ViT）
- 输入 448×448 图像 → 1024 个 patch token（每个 14×14）
- 输出维度通常为 1664

关键点：Vision Encoder 在预训练初期通常**冻结**，只在后期微调时部分解冻。

## 3. Visual Resampler (核心创新)

### 问题
ViT 输出 1024 个视觉 token，直接拼接到 LLM 输入会：
- 占用大量上下文窗口
- 计算代价高（O(n²) 注意力）

### 解决方案：Perceiver Resampler

使用 **可学习的 Query token** 通过 Cross-Attention 从 1024 个视觉 token 中提取信息，压缩为固定数量（如 256 个）的视觉表示。

```
可学习 Query [256, D] ──(Q)──→
                                Cross-Attention → 压缩后的视觉表示 [256, D]
视觉 token [1024, D] ──(K,V)──→
```

### 数学表示

$$\text{Resampled} = \text{CrossAttn}(Q=\text{learnable\_queries}, K=V=\text{vision\_features})$$

## 4. 多模态输入拼接

Qwen-VL 将视觉 token 和文本 token 拼接为一个统一序列：

```
[BOS] <img> [vis_1] [vis_2] ... [vis_256] </img> 这张广告图片展示了什么？ [EOS]
```

特殊 token `<img>` 和 `</img>` 标记视觉 token 的边界，让 LLM 知道哪些是图像信息。

## 5. Qwen2-VL 的改进

- **动态分辨率**: 支持任意分辨率输入，按 patch 数量动态调整
- **M-RoPE**: 多模态旋转位置编码，为视觉和文本分配不同的位置 ID
- **Native Resolution**: 不再强制 resize，保留原始图像信息

## 6. 对广告场景的意义

理解 Qwen-VL 架构对于：
- 知道从哪一层提取 embedding（DeepStack 需要）
- 理解 Visual Resampler 对视觉信息的压缩（影响细节保留）
- 设计预训练和微调策略（冻结/解冻哪些模块）
