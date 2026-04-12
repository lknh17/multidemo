# v05 代码解释
## Perceiver Resampler
核心是 `nn.Parameter` 的可学习 Query + Cross-Attention。Query 的数量决定压缩比。Qwen-VL 用 256 个 query 将 ~1000 个视觉 token 压缩到 256 个。

## 多模态拼接
视觉 tokens 和文本 tokens 直接拼接后共享同一个 Transformer。因果掩码确保自回归性质。视觉 tokens 不需要预测（只做条件）。
