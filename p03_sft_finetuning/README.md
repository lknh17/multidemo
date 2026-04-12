# p03 SFT 指令微调

> **目标**: 在 Qwen2.5-0.5B 上进行 SFT 指令微调，对比 Full/LoRA/QLoRA/DoRA 四种方法，掌握 Label Masking 和 ChatML 模板。
>
> **前置条件**: 完成 p02 继续预训练，模型已下载。
>
> **预计耗时**: 3-5 小时（含消融实验）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 理解 SFT（监督微调）与预训练的本质区别
- 掌握 ChatML 对话模板的构建方式
- 理解 Label Masking 为什么只对 assistant 回复计算 loss
- 实际运行 Full/LoRA/QLoRA/DoRA 四种微调方法
- 完成 LoRA rank/alpha/target_modules 消融实验
- 学会 LoRA 权重合并（merge_and_unload）
- 检测并诊断过拟合问题

### 本模块代码文件概览

本模块代码覆盖了 SFT 的完整流程：**数据下载 → ChatML 格式化 → 四种方法训练 → 消融实验 → LoRA 合并 → 过拟合检测 → 推理对比**。每个脚本对应一个关键技能点。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | SFT 全局配置，包含 Full/LoRA/QLoRA/DoRA 四种方法参数 | 理解不同微调方法的参数差异 |
| `download_data.py` | 下载 Alpaca-Chinese/BELLE/Firefly 三个指令数据集 | 获取 SFT 训练数据 |
| `dataset.py` | ChatML 模板构建 + Label Masking（只对 assistant 回复算 loss） | 掌握 SFT 数据处理核心技巧 |
| `train.py` | 主训练脚本，通过 `--method` 切换 full/lora/qlora/dora | 运行四种微调方法的对比实验 |
| `ablation_runner.py` | 自动批量运行 rank/alpha/target_modules 消融实验 | 理解 LoRA 超参数的影响 |
| `merge_lora.py` | 将 LoRA 适配器合并回基座模型，生成独立部署的完整模型 | 掌握 LoRA 部署技巧 |
| `overfit_diagnosis.py` | 检测 Distinct-N 多样性、重复率、训练/验证 loss 趋势 | 学会诊断过拟合 |
| `inference.py` | 用 ChatML 格式对比基座和各微调版本的对话能力 | 直观看到 SFT 效果 |
| `ds_config.json` | DeepSpeed 配置（全参微调时使用） | 了解 SFT 的分布式配置 |

> 💡 **建议学习顺序**: `download_data.py` → `train.py --method lora`（先跑通）→ 依次尝试其他方法 → `ablation_runner.py` → `merge_lora.py` → `overfit_diagnosis.py` → `inference.py`

### 确认前置条件

```bash
cd p03_sft_finetuning
python -c "from transformers import AutoTokenizer; t=AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B',trust_remote_code=True); print(f'Tokenizer 词表: {t.vocab_size}')"
# 预期输出: Tokenizer 词表: 151936

python -c "from peft import LoraConfig; print('PEFT 可用')"
# 预期输出: PEFT 可用
```

---

## 2. 数据下载与探索

```bash
# 下载全部 3 个数据集
python download_data.py --max-samples 50000

# 只下载 Alpaca 中文
python download_data.py --dataset alpaca --max-samples 10000

# 查看数据统计
python download_data.py --explore-only
```

**预期输出**:
```
  📊 数据集: alpaca
     总条数:       48818
     指令平均长度: 43 字符
     输出平均长度: 215 字符
```

### 三个数据集对比

| 数据集 | 格式 | 规模 | 特点 |
|--------|------|------|------|
| Alpaca-Chinese | instruction/input/output | ~5万 | GPT-4 生成，质量高 |
| BELLE | instruction/output | ~50万 | 百万级，覆盖面广 |
| Firefly | kind/input/target | ~110万 | 多任务类型 |

---

## 3. 数据处理 — ChatML 模板与 Label Masking

### ChatML 格式

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
请解释什么是机器学习<|im_end|>
<|im_start|>assistant
机器学习是人工智能的一个分支...<|im_end|>
```

### Label Masking 关键代码

```python
from dataset import create_sft_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
dataset = create_sft_dataset("data/alpaca.jsonl", tokenizer)

# 检查 label masking
sample = dataset[0]
labels = sample["labels"]
masked = (labels == -100).sum().item()
total = (sample["attention_mask"] == 1).sum().item()
print(f"Masked tokens: {masked}/{total} ({masked/total*100:.1f}%)")
```

---

## 4. 实验一：LoRA 微调（推荐起步）

```bash
python train.py --method lora
```

**预期输出**:
```
  微调方法:       lora
  LoRA rank: 16, alpha: 32
  可训练参数: 3,407,872 / 494,032,896 (0.69%)
  训练配置:
    数据集大小:     46377
    Batch Size:     4 × 4 = 16
    学习率:         0.0002
```

**LoRA 为什么高效？**
- 只训练 0.69% 的参数
- 显存仅需 ~4GB（vs 全参 ~9GB）
- 训练速度更快

---

## 5. 实验二：QLoRA 微调（极致省显存）

```bash
python train.py --method qlora
```

QLoRA 额外优势：
- 模型以 4-bit 量化加载，参数显存降低 4 倍
- 适合 16GB 甚至 12GB 显卡

---

## 6. 实验三：DoRA 微调（效果更好）

```bash
python train.py --method dora
```

DoRA 在 LoRA 基础上添加了权重分解：
- 将权重分为方向（direction）和幅度（magnitude）
- 分别学习，效果通常优于 LoRA

---

## 7. 实验四：全参微调（基准对照）

```bash
python train.py --method full_finetune
```

全参微调作为对比基准，观察 LoRA 系列与全参的效果差距。

---

## 8. 消融实验

### LoRA Rank 消融

```bash
python ablation_runner.py --ablation rank
```

测试 rank = 8, 16, 32, 64, 128 对效果的影响。

### Alpha 消融

```bash
python ablation_runner.py --ablation alpha
```

### Target Modules 消融

```bash
python ablation_runner.py --ablation modules
```

测试仅注意力层 vs 注意力+FFN vs 全部线性层。

---

## 9. LoRA 权重合并

```bash
python merge_lora.py --adapter-path outputs/sft_lora/final
```

合并后生成完整模型，推理时不再需要 PEFT 库。

---

## 10. 过拟合诊断

```bash
python overfit_diagnosis.py --model-path outputs/sft_lora/final --log-dir outputs/sft_lora
```

诊断指标：
- Distinct-N 多样性（>0.7 为好）
- 重复率（<0.1 为好）
- 训练/验证 loss 趋势

---

## 11. 推理对比

```bash
python inference.py
```

用 ChatML 格式测试基座和各微调版本的对话能力。

---

## 12. 四种方法对比汇总

| 指标 | Full | LoRA | QLoRA | DoRA |
|------|------|------|-------|------|
| 可训练参数 | 100% | ~0.7% | ~0.7% | ~0.8% |
| GPU 显存 | ~9GB | ~4GB | ~3GB | ~4.5GB |
| 训练速度 | 基准 | ~1.5x | ~1.2x | ~1.3x |
| 效果 | 最好 | 好 | 稍弱 | 接近全参 |
| 推荐场景 | 充裕算力 | 通用 | 低显存 | 追求效果 |

---

## 13. 小结

- [x] 理解了 SFT 的 ChatML 模板和 Label Masking
- [x] 对比了 Full/LoRA/QLoRA/DoRA 四种微调方法
- [x] 完成了 rank/alpha/modules 消融实验
- [x] 掌握了 LoRA 权重合并
- [x] 学会了过拟合诊断

**下一步**: 进入 [p04 DPO 偏好对齐](../p04_dpo_alignment/README.md)，让模型学会人类偏好。
