# p02 继续预训练

> **目标**: 在 Qwen2.5-0.5B 基座上继续预训练，注入更多中文知识，对比多种训练策略。
>
> **前置条件**: 完成 p01 环境搭建，模型已下载。
>
> **预计耗时**: 2-4 小时（含多组对比实验）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 理解"继续预训练"与"从零预训练"和"SFT"的区别
- 掌握 Packing vs Padding 两种数据处理策略
- 实际运行 DeepSpeed ZeRO-2/ZeRO-3 对比实验
- 对比 cosine/linear 学习率策略的 loss 曲线差异
- 理解 Gradient Checkpointing 的显存-速度 tradeoff
- 对比全参预训练和 LoRA 预训练的效果差异
- 检测灾难性遗忘现象

### 本模块代码文件概览

本模块的代码围绕一条完整的实验链条展开：**下载数据 → 预处理 → 训练（多种策略） → 对比 → 验证效果**。通过运行不同的训练配置和对比脚本，你将亲身体会各种策略的差异。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | 预训练的全局配置（模型、数据路径、超参数） | 理解预训练参数设置 |
| `download_data.py` | 下载 Wikipedia 中文语料并转为 JSONL 格式 | 获取训练数据 |
| `dataset.py` | 实现 Packing 和 Padding 两种数据处理方式 | 掌握数据高效利用策略 |
| `train.py` | 主训练脚本，支持 ZeRO-2/3、不同学习率策略 | 运行核心预训练实验 |
| `train_lora.py` | LoRA 预训练脚本，对比全参与 LoRA 的效果差异 | 理解 LoRA 在预训练中的应用 |
| `ds_config_zero2.json` | DeepSpeed ZeRO-2 配置文件 | 了解分布式训练配置 |
| `ds_config_zero3.json` | DeepSpeed ZeRO-3 + CPU Offload 配置 | 对比不同 ZeRO Stage |
| `compare_strategies.py` | 汇总多组实验的 loss 曲线，生成对比图表 | 分析实验结果 |
| `inference.py` | 对比训练前后模型的中文生成质量 | 验证训练效果 |

> 💡 **建议学习顺序**: `download_data.py` → `train.py`（基础实验）→ 修改参数重复跑 `train.py`（ZeRO/学习率对比）→ `train_lora.py` → `compare_strategies.py` → `inference.py`

### 确认前置条件

```bash
cd p02_continual_pretrain
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B',trust_remote_code=True); print(f'Tokenizer 词表: {t.vocab_size}')"
# 预期输出: Tokenizer 词表: 151936
```

---

## 2. 数据下载与探索

```bash
python download_data.py --max-samples 50000
```

下载 Wikipedia 中文数据（约 50000 条文章），保存为 JSONL 格式。

```bash
# 查看数据统计
python download_data.py --explore-only
```

**预期输出**:
```
  总条数:     47832
  平均长度:   2143 字符
  最短:       101 字符
  最长:       98234 字符
```

### 为什么选 Wikipedia

| 数据源 | 质量 | 规模 | 领域 | 获取难度 |
|--------|------|------|------|---------|
| Wikipedia-CN | ★★★★★ | ~100万篇 | 百科全书 | 简单 |
| SkyPile | ★★★★ | 150GB | 网页/新闻 | 中等 |
| WanJuan | ★★★★ | 多领域 | 多领域混合 | 需申请 |

Wikipedia 是质量最高的中文语料之一，适合快速实验。

---

## 3. 数据预处理 — Tokenize 与 Packing

```python
from dataset import create_pretrain_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)

# Packing 模式（推荐）
dataset_pack = create_pretrain_dataset("data/wiki_zh.jsonl", tokenizer, packing=True)

# Padding 模式（对比）
dataset_pad = create_pretrain_dataset("data/wiki_zh.jsonl", tokenizer, packing=False)
```

**Packing vs Padding 效率对比**:

| 指标 | Packing | Padding |
|------|---------|---------|
| 有效 token 占比 | ~98% | ~60% |
| 训练样本数 | 更多 | 更少 |
| GPU 利用率 | 高 | 低（浪费在 pad 上） |
| 实现复杂度 | 中 | 低 |

---

## 4. DeepSpeed 配置详解

查看 `ds_config_zero2.json` 各字段含义：

| 字段 | 值 | 含义 |
|------|-----|------|
| `bf16.enabled` | true | 使用 bf16 混合精度 |
| `zero_optimization.stage` | 2 | ZeRO Stage 2（优化器+梯度分片）|
| `overlap_comm` | true | 通信与计算重叠（提速） |
| `reduce_scatter` | true | 使用 reduce-scatter 替代 all-reduce |
| `contiguous_gradients` | true | 梯度连续存储（提升效率） |
| `"auto"` 值 | - | 由 HF Trainer 自动从 TrainingArguments 填入 |

---

## 5. 实验一：基础预训练

```bash
# 最简单的训练命令
python train.py
```

**预期输出**:
```
  训练配置:
    数据集大小:     18432
    Batch Size:     4 × 4 = 16
    学习率:         2e-05
    LR Scheduler:   cosine
    预计步数:       ~1152
```

观察要点：
- loss 应在前 100 步快速下降
- 最终 loss 通常在 2.0-3.0 之间
- 如果 loss 变 NaN，降低学习率

---

## 6. 实验二：ZeRO-2 vs ZeRO-3 对比

```bash
# ZeRO-2（推荐单卡使用）
deepspeed --num_gpus=1 train.py --deepspeed ds_config_zero2.json

# ZeRO-3 + CPU Offload
deepspeed --num_gpus=1 train.py --deepspeed ds_config_zero3.json
```

**对比记录**:

| 指标 | ZeRO-2 | ZeRO-3 + Offload |
|------|--------|------------------|
| GPU 显存占用 | ~9 GB | ~5 GB |
| 训练速度 | 基准 | ~0.5-0.7x |
| 最终 Loss | 基准 | 接近 |

> 💡 **结论**: 单卡 0.5B 模型用 ZeRO-2 即可。ZeRO-3 在显存不够时才有意义。

---

## 7. 实验三：学习率策略对比

```bash
python train.py --lr-scheduler cosine --learning-rate 2e-5
python train.py --lr-scheduler linear --learning-rate 2e-5
python train.py --lr-scheduler constant_with_warmup --learning-rate 1e-5
```

不同策略的 loss 曲线形态不同，cosine 通常在后期衰减更平滑。

---

## 8. 实验四：Gradient Checkpointing 开关对比

修改 `config.py` 中的 `gradient_checkpointing` 字段：

```python
# config.py 中切换
gradient_checkpointing: bool = True   # 或 False
```

**对比记录**:

| 指标 | GC 开启 | GC 关闭 |
|------|---------|---------|
| GPU 显存 | ~9 GB | ~12 GB |
| 训练速度 | 基准 | ~1.3x |
| 效果 | 相同 | 相同 |

---

## 9. 实验五：LoRA 预训练 vs 全参预训练

```bash
python train_lora.py
```

**对比记录**:

| 指标 | 全参 | LoRA (r=64) |
|------|------|-------------|
| 可训练参数 | 494M (100%) | ~20M (~4%) |
| GPU 显存 | ~9 GB | ~4 GB |
| 训练速度 | 基准 | ~1.5x |
| 最终 Loss | 更低 | 略高 |

---

## 10. 策略对比汇总分析

```bash
python compare_strategies.py --results-dir outputs/pretrain
```

生成各实验的 loss 曲线对比图和汇总表格。

---

## 11. 训练效果验证

```bash
python inference.py
```

用同一组中文知识 prompt 对比训练前后的生成质量，观察中文知识是否增强。

---

## 12. 灾难性遗忘检测

继续预训练可能导致模型"忘记"原有能力（灾难性遗忘）。

用 p01 的基线评测 prompt 重新测试训练后模型：

```bash
cd ../p01_env_setup
python baseline_eval.py --model ../p02_continual_pretrain/outputs/pretrain/final
```

对比 p01 基线结果，看英文能力、代码能力是否下降。

> ⚠️ **注意**: 如果遗忘严重，可以尝试降低学习率（1e-5 → 5e-6）或混入通用语料。

---

## 小结

- [x] 下载并探索了 Wikipedia 中文数据
- [x] 理解了 Packing vs Padding 的效率差异
- [x] 对比了 ZeRO-2 vs ZeRO-3 的显存和速度
- [x] 测试了不同学习率策略
- [x] 对比了全参和 LoRA 预训练
- [x] 验证了训练效果并检测灾难性遗忘

**下一步**: 进入 [p03 SFT 指令微调](../p03_sft_finetuning/README.md)，让模型学会对话。
