# v06 代码解释
## 多任务 Loss 组合
三个 Loss (ITC+ITM+Captioning) 通过权重系数加权求和。实践中需要调参平衡各任务，避免某个任务 dominate。

## ITM 负样本构造
50% 概率将图文配对打乱作为负样本，模型需要判断图文是否匹配。

## Zero-shot 评估
预训练后直接用 ITC 的 embedding 做检索，无需额外微调。R@K 反映模型的跨模态对齐质量。
