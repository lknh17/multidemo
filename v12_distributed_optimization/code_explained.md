# v12 代码解释

## DeepSpeed 训练脚本
`deepspeed.initialize()` 自动处理模型分片、优化器分片、通信。配置文件 `ds_config.json` 定义 ZeRO stage、混合精度等。

## 梯度累积
每 `gradient_accumulation_steps` 步才执行 `optimizer.step()`，等效于将 batch_size 放大 N 倍，适合显存受限场景。

## 混合精度
`torch.amp.autocast` 将计算转为 FP16，`GradScaler` 防止梯度下溢。BF16 在 A100+ 上更优（不需要 GradScaler）。

## 动态量化
`quantize_dynamic` 将 Linear 层权重量化为 INT8，推理时反量化计算。模型大小减半，CPU 推理速度提升 2-4x。
