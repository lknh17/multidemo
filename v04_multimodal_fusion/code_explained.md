# v04 代码解释

## InfoNCE Loss
batch 中 N 个图文对形成 N×N 的相似度矩阵。对角线是正例，其他是负例。对称 Loss 同时优化 image→text 和 text→image 方向。

## 可学习温度系数
`logit_scale = nn.Parameter(log(1/0.07))`，用 exp 转换。模型可以自适应调整温度，实践中效果优于固定温度。

## L2 归一化
embedding 经过 `F.normalize` 后，内积 = 余弦相似度。这是对比学习的标准做法。
