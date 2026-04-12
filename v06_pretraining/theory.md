# 多模态预训练原理

## 1. 预训练目标

多模态预训练通常包含多个互补的学习目标：

### ITC (Image-Text Contrastive)
双塔对比学习，对齐图文在共享空间的表示。同 CLIP 的 InfoNCE Loss。

### ITM (Image-Text Matching)
二分类任务：判断给定的图文对是否匹配。使用融合后的表示（而非独立的双塔表示），能学到更深层的跨模态语义。

### MLM / Captioning
- **MLM**：遮蔽部分文本 token，根据图像和上下文预测被遮蔽的 token
- **Captioning**：给定图像，自回归生成描述文本

### 联合目标
$$\mathcal{L} = \lambda_1 \mathcal{L}_{ITC} + \lambda_2 \mathcal{L}_{ITM} + \lambda_3 \mathcal{L}_{Cap}$$

## 2. 数据配比

预训练数据通常包含多种来源，需要合理配比：
- 图文对数据（互联网爬取）：量大但噪声多
- 高质量标注数据：量少但质量高
- 课程学习（Curriculum Learning）：先易后难，先用简单数据再用复杂数据

## 3. 训练策略

### 阶段式训练
1. **Stage 1**: 冻结视觉编码器和 LLM，只训练 Resampler（对齐模态）
2. **Stage 2**: 解冻部分/全部参数，联合微调

### 学习率设计
- 不同模块用不同学习率（layer-wise LR decay）
- 视觉编码器 LR 通常远小于其他模块（已预训练好，需要保守更新）
