# v03 代码解释

## RoPE 实现要点
`precompute_rope_freqs` 预计算频率表，`apply_rope` 对 Q/K 的每对维度做 2D 旋转。KV Cache 场景下需要 offset 来获取正确位置的频率。

## KV Cache 机制
每个 block 缓存 (K, V)。推理时只传入新 token，与缓存的 K/V 拼接后做注意力。这把生成 n 个 token 的复杂度从 O(n³) 降到 O(n²)。

## 权重共享
`tok_embed.weight = lm_head.weight` — embedding 层和输出投影共享权重，这是 GPT-2 以来的标准做法，减少了约 30% 参数量。

## SwiGLU
`SiLU(xW_gate) * (xW_up)` 的门控机制让 FFN 可以学习"让多少信息通过"，效果优于 ReLU/GELU。
