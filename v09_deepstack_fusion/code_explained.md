# v09 代码解释
## forward_with_intermediates
编码器每层计算后检查是否在提取列表中，若是则保存中间特征。这比 hook 方式更直观。

## 多层 Loss
各层分别做 InfoNCE Loss，强迫每层都学到有区分力的表示。fusion_loss_weight 通常大于 layer_loss_weight。
