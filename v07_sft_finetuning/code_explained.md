# v07 代码解释
## LoRA 从零实现
`LoRALinear` 包裹原始 `nn.Linear`，冻结原始权重，只训练 A 和 B。初始化时 B=0 保证 ΔW=0。`merge_weights` 将 BA·scaling 加到原始权重上，推理时无额外开销。

## inject_lora
遍历模型所有 Linear 层，匹配名称后替换为 LoRALinear。只需几行代码就能把全参微调改为 LoRA 微调。
