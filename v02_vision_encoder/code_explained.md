# v02 代码解释：ViT 实现详解

## PatchEmbedding 的卷积实现

`Conv2d(3, d_model, kernel_size=P, stride=P)` 等价于：切成 P×P 不重叠 patch → 展平 → 线性投影。卷积实现更高效，因为 GPU 对卷积运算有深度优化。

## [CLS] Token 的作用

`cls_token` 是一个 `nn.Parameter`，在训练中学习。它被拼接在 patch 序列最前面，经过所有 Transformer 层后，其输出作为整图表示送入分类头。

## 位置编码

ViT 使用可学习的 `pos_embed`（非正弦），因为实验表明在图像任务上两者差异不大，而可学习编码更灵活。

## 数据增强

CIFAR-10 训练时使用 RandomCrop + RandomFlip，这是图像分类的标准增强策略，能有效防止过拟合。
