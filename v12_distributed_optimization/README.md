# v12: 分布式训练 & 部署优化
## 快速开始
```bash
# 单卡训练 (含梯度累积+混合精度)
python train_deepspeed.py

# DeepSpeed 多卡
deepspeed --num_gpus=4 train_deepspeed.py

# FSDP 多卡
torchrun --nproc_per_node=4 train_fsdp.py

# 推理优化 Benchmark
python inference_optimized.py
```
## 核心知识点
DeepSpeed ZeRO / FSDP / 混合精度 / 梯度累积 / INT8 量化 / ONNX 导出
