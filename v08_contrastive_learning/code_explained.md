# v08 代码解释
## InfoNCE vs Circle Loss
InfoNCE 把 batch 看作 N 分类问题；Circle Loss 自适应加权每对正/负样本，对困难样本更敏感。

## Hard Negative Mining
Hardest mining 选最相似的负样本——信息量最大但训练易崩；Semi-hard 是折中。广告场景常用同行业不同商品作 hard negative。
