# p10 原理详解 - 最佳实践总结

> 本文档汇总大模型训练全流程的最佳参数、常见故障排查方法、硬件选型指南和未来趋势。

---

## 1. 最佳参数速查表

### 各阶段核心参数对比

| 参数 | 继续预训练 | SFT | DPO | RL (GRPO) |
|------|-----------|-----|-----|-----------|
| 学习率 | 2e-5 | 2e-5 | 5e-6 | 1e-6 |
| 学习率策略 | cosine | cosine | cosine | constant |
| Warmup 比例 | 5% | 3% | 3% | 0% |
| 有效 Batch | 16 | 16 | 16 | 16 |
| Epoch | 1-3 | 2-5 | 1-2 | 1 |
| 序列长度 | 512-1024 | 1024-2048 | 1024 | 512 |
| LoRA r | 64 | 16 | 16 | - |
| 梯度裁剪 | 1.0 | 1.0 | 1.0 | 0.5 |
| 权重衰减 | 0.01 | 0.01 | 0.0 | 0.0 |

**核心规律**:
- **学习率递减**: 预训练 → SFT → DPO → RL，学习率越来越小
- **Epoch 递减**: 预训练多轮，RL 通常只 1 轮
- **LoRA rank**: 预训练用大 rank（学知识），SFT/DPO 用小 rank（学格式）

### 0.5B 模型显存参考

| 训练方式 | 24G GPU | 48G GPU |
|----------|---------|---------|
| 全参 + GC | batch=4, seq=512 | batch=8, seq=1024 |
| LoRA + GC | batch=8, seq=1024 | batch=16, seq=2048 |
| QLoRA 4bit | batch=16, seq=2048 | batch=32, seq=4096 |

---

## 2. 环境搭建故障排查

### 问题 2.1: CUDA 版本不匹配

**症状**: `RuntimeError: CUDA error: no kernel image is available`

**诊断**:
```bash
python -c "import torch; print(torch.version.cuda)"
nvidia-smi  # 查看驱动支持的最高 CUDA 版本
```

**解决**: PyTorch 的 CUDA 版本不能高于驱动支持的版本。重新安装匹配版本的 PyTorch。

### 问题 2.2: Flash Attention 安装失败

**症状**: `No module named 'flash_attn'`

**解决**:
```bash
pip install flash-attn --no-build-isolation
# 如果仍失败，确保 CUDA toolkit 版本与 PyTorch 匹配
```

### 问题 2.3: DeepSpeed 编译错误

**症状**: 各种 C++ 编译错误

**解决**: 安装预编译版本 `pip install deepspeed`，或设置 `DS_BUILD_OPS=0`。

### 问题 2.4: Tokenizer 加载失败

**症状**: `OSError: Can't load tokenizer`

**解决**: 设置 `trust_remote_code=True`，或先下载到本地 `huggingface-cli download`。

### 问题 2.5: 权限不足

**症状**: `PermissionError` 写入 cache 目录

**解决**: `export HF_HOME=/path/to/writable/dir`

---

## 3. 继续预训练故障排查

### 问题 3.1: Loss 不下降

**原因**: 学习率太小、数据质量差、labels 设置错误

**排查**:
- 检查 `labels` 是否等于 `input_ids`（CLM 任务）
- 检查 padding token 是否被 mask 为 -100
- 尝试增大学习率 2-5 倍

### 问题 3.2: Loss 变 NaN

**原因**: 学习率太大、数据中有异常值、fp16 溢出

**排查**:
- 降低学习率
- 使用 bf16 代替 fp16
- 增大 `max_grad_norm`（如从 1.0 → 5.0）

### 问题 3.3: 灾难性遗忘

**症状**: 中文提升但英文/代码能力下降

**解决**: 混入 10-20% 通用数据、降低学习率、使用 LoRA

### 问题 3.4: Packing 模式下 Loss 异常偏低

**原因**: EOS token 的 loss 贡献过大（EOS 很容易预测）

**解决**: 正常现象，Packing 模式的 loss 通常比 Padding 低 0.5-1.0

### 问题 3.5: 数据加载速度慢

**解决**: 增大 `num_workers`、使用 `streaming=True`、预处理后缓存

---

## 4. SFT 微调故障排查

### 问题 4.1: 模型只输出重复内容

**原因**: 过拟合、数据格式错误

**解决**: 减少 epoch、增大数据量、检查 Chat Template 是否正确

### 问题 4.2: 模型不遵循指令格式

**原因**: Chat Template 未正确应用

**排查**: 打印 tokenize 后的文本，确认 `<|im_start|>` 等特殊 token 正确

### 问题 4.3: SFT 后基础能力下降

**原因**: 过度微调（alignment tax）

**解决**: 减少 epoch、使用 LoRA、降低学习率

### 问题 4.4: 多轮对话训练效果差

**原因**: 只计算了第一轮的 loss

**解决**: 确保所有 assistant 轮次的 loss 都被计算（非 -100）

### 问题 4.5: NEFTune 导致 Loss 震荡

**原因**: 噪声幅度过大

**解决**: 降低 `neft_alpha`（从 5.0 → 1.0）

---

## 5. DPO/RL 故障排查

### 问题 5.1: DPO Loss 不下降

**原因**: chosen 和 rejected 差异太小

**排查**: 检查偏好数据质量，chosen 和 rejected 应有明显质量差异

### 问题 5.2: DPO 后模型变得啰嗦

**原因**: DPO 过度优化了"长回答"偏好

**解决**: 在数据中加入"简洁优于啰嗦"的偏好对

### 问题 5.3: RL 训练奖励不增长

**原因**: 奖励函数设计问题、学习率太小

**排查**: 检查奖励函数在 chosen/rejected 上的分布是否合理

### 问题 5.4: KL 散度爆炸

**原因**: 模型偏离 reference model 太远

**解决**: 增大 `kl_coef`、降低学习率

### 问题 5.5: RL 后模型 hack 奖励函数

**症状**: 奖励分数很高但实际输出质量下降

**解决**: 设计更鲁棒的奖励函数、增大 KL 惩罚

---

## 6. 量化与部署故障排查

### 问题 6.1: 量化后质量严重下降

**解决**: 尝试 GPTQ 4-bit 替代 GGUF Q4，或使用 AWQ

### 问题 6.2: vLLM 启动 OOM

**解决**: 降低 `gpu_memory_utilization`（如 0.90 → 0.80）

### 问题 6.3: 推理速度慢

**排查**: 确认使用了 Continuous Batching、PagedAttention、FlashAttention

### 问题 6.4: 量化模型生成乱码

**原因**: 量化校准数据不足或不匹配

**解决**: 增大 `calibration_samples`、使用更接近目标任务的校准数据

### 问题 6.5: 多GPU推理通信瓶颈

**解决**: 确保 GPU 间有 NVLink，或减少 tensor_parallel 数

---

## 7. 场景选择指南

| 场景 | 推荐流程 | 说明 |
|------|----------|------|
| 注入领域知识 | 预训练 → SFT | 先学知识再学对话 |
| 提升对话质量 | SFT → DPO | 指令微调+偏好对齐 |
| 提升推理能力 | SFT → RL | 用奖励信号优化推理 |
| 安全对齐 | SFT → DPO → RL | 三阶段完整对齐 |
| 快速上线 | SFT only | 最简流程，效果可接受 |
| 资源受限 | QLoRA SFT | 4-bit 量化+LoRA |

---

## 8. 硬件选型指南

| GPU | 显存 | 适合模型 | 价格参考 |
|-----|------|----------|----------|
| RTX 3090 | 24GB | ≤3B (LoRA) | 低 |
| RTX 4090 | 24GB | ≤3B (LoRA) | 中 |
| A6000 | 48GB | ≤7B (全参/LoRA) | 高 |
| A100-40G | 40GB | ≤13B (LoRA) | 很高 |
| A100-80G | 80GB | ≤30B (LoRA) | 极高 |
| H100 | 80GB | ≤70B (LoRA) | 极高 |

**经验法则**:
- 全参训练显存 ≈ 模型参数量 × 18 bytes (bf16 + 优化器)
- LoRA 训练显存 ≈ 模型参数量 × 2.5 bytes + LoRA 参数 × 18 bytes
- 推理显存 ≈ 模型参数量 × 2 bytes (bf16)

---

## 9. 多 GPU 扩展路线图

### 单卡 → 多卡的进阶路径

1. **单卡 LoRA**: 最简单，24G 即可训练 3B 模型
2. **单卡 ZeRO-2**: 优化器状态分片，省 30% 显存
3. **单卡 ZeRO-3 + Offload**: CPU offload，极限省显存
4. **多卡 DDP**: N 卡 N 倍吞吐，最简单的多卡方案
5. **多卡 FSDP/ZeRO-3**: 参数分片，支持超大模型
6. **多节点训练**: 跨机器训练，需要高速网络

### DeepSpeed vs FSDP 选择

| 维度 | DeepSpeed | FSDP |
|------|-----------|------|
| 生态 | 独立库 | PyTorch 原生 |
| 配置 | JSON 配置文件 | Python API |
| CPU Offload | 成熟 | 有但不如 DS |
| 社区支持 | 广泛 | 正在增长 |
| HF 集成 | 好 | 好 |

---

## 10. 未来趋势

### 训练范式演进

- **更高效的对齐**: DPO → SimPO → ORPO（简化训练流程）
- **更好的 RL**: GRPO → RAFT → 在线 RLHF（更稳定的训练）
- **合成数据**: Self-Instruct → Evol-Instruct → 数据飞轮
- **多模态统一**: 文本/图像/音频/视频统一训练

### 推理优化趋势

- **推测解码**: 小模型辅助大模型，加速 2-3 倍
- **KV Cache 优化**: PagedAttention → RadixAttention → 更高效的缓存管理
- **长上下文**: RoPE 外推 → YaRN → 支持 100K+ token

### 模型架构趋势

- **MoE (混合专家)**: DeepSeek-V2/V3 证明 MoE 的训练效率优势
- **状态空间模型**: Mamba 系列探索非 Transformer 架构
- **线性注意力**: 降低注意力复杂度从 O(n²) 到 O(n)
