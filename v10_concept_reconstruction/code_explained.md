# v10 代码解释
## Uncertainty Weighting
每个 Loss 有一个可学习的 log_sigma 参数，Loss 大时 sigma 自动增大（降低该 Loss 权重），避免某个 Loss dominate。

## 概念重构头
多个独立的 MLP 头分别预测行业/品牌/属性/意图。属性用 BCE（多标签），其他用 CE（单标签）。
