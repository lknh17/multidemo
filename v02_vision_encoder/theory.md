# Vision Transformer (ViT) 原理

## 核心思想

**"An Image is Worth 16x16 Words"** (Dosovitskiy et al., 2020)

ViT 的革命性想法：**把图像当作"文本"来处理**——将图像切成小块（patch），每块相当于一个"token"，然后直接用标准 Transformer 编码器处理。

## 1. 图像 Patch 化

### 1.1 为什么不直接把像素当 token？

一张 224×224 的图像有 50,176 个像素，自注意力复杂度 O(n²) 会爆炸。

**解决方案**：将图像切成 P×P 的 patch，每个 patch 作为一个 token。

- 224×224 图像 + 16×16 patch → 只有 196 个 token
- 这让标准 Transformer 可以直接处理

### 1.2 Patch Embedding

```
图像 [3, 224, 224]
  → 切成 14×14 = 196 个 patch，每个 [3, 16, 16]
  → 展平为 [768]（3×16×16 = 768）
  → 线性投影到 d_model 维
  → 196 个 d_model 维的 token
```

实现上等价于一个 stride=kernel_size 的卷积：
`Conv2d(3, d_model, kernel_size=16, stride=16)`

## 2. [CLS] Token

在 patch 序列前加一个可学习的特殊 token [CLS]。

经过 Transformer 后，[CLS] 位置的输出被用作整张图像的全局表示，用于分类。

**为什么用 [CLS]？**
- 它没有对应任何具体的 patch，是一个"全局聚合器"
- 通过注意力机制，它可以汇聚所有 patch 的信息
- 类似 BERT 中的 [CLS] token

## 3. 位置编码

ViT 使用**可学习的位置编码**（而非正弦编码）。

这是一个 (num_patches + 1) × d_model 的可训练矩阵。

研究发现 ViT 学到的位置编码会呈现出 2D 空间模式：相近位置的编码相似。

## 4. ViT 完整架构

```
输入图像 [B, 3, H, W]
    │
    ▼
Patch Embedding (Conv2d)  → [B, num_patches, d_model]
    │
    ▼
Prepend [CLS] Token       → [B, num_patches+1, d_model]
    │
    ▼
+ Positional Encoding      → [B, num_patches+1, d_model]
    │
    ▼
Transformer Encoder (×N)   → [B, num_patches+1, d_model]
    │
    ▼
取 [CLS] 输出             → [B, d_model]
    │
    ▼
Classification Head (MLP)  → [B, num_classes]
```

## 5. ViT vs CNN

| 特性 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 强（局部性、平移不变性） | 弱 |
| 数据需求 | 较少 | 较多（需大规模预训练） |
| 长距离依赖 | 需要多层堆叠 | 第一层就能全局交互 |
| 计算效率 | 高（局部计算） | 自注意力 O(n²) |

**关键发现**：ViT 在大规模数据上预训练后，性能超过 CNN！这启发了后续所有视觉大模型的设计。

## 6. 对多模态的意义

ViT 是多模态模型的视觉骨干——Qwen-VL、CLIP、LLaVA 的视觉编码器都是 ViT 变体。掌握 ViT 是理解多模态模型的基础。
