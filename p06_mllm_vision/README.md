# p06 MLLM 多模态视觉微调

> **目标**: 在 Qwen2.5-VL-3B 上进行多模态指令微调，让模型学会理解图像并回答视觉相关问题。
>
> **前置条件**: 完成 p01-p05，理解 LLM 微调流程。
>
> **预计耗时**: 3-6 小时（含数据准备和多组对比实验）

---

## 1. 本模块目标与前置条件

### 你将收获什么

- 理解多模态大语言模型（MLLM）的三阶段架构
- 掌握 Qwen2.5-VL 的视觉 token 注入机制
- 实际操作三种冻结策略的对比实验
- 掌握 LoRA 应用于 LLM 部分的多模态微调方法
- 了解多模态指令数据的构造与处理
- 运行图像描述、VQA、OCR 等多种视觉理解任务

### 本模块代码文件概览

本模块是从"纯文本 LLM"到"多模态理解"的跨越：**下载图文数据 → 格式转换 → 冻结策略选择 → 多模态微调 → 视觉任务测试**。代码帮你理解 MLLM 的关键组件如何协作。

| 文件 | 作用 | 对应学习目标 |
|------|------|-------------|
| `config.py` | MLLM 训练配置（冻结策略、LoRA 参数、数据路径） | 理解多模态微调的关键参数 |
| `download_data.py` | 下载 LLaVA-Instruct 图文指令数据集 | 获取多模态训练数据 |
| `dataset.py` | LLaVA 格式 → Qwen2.5-VL 格式转换 + 图像预处理 | 掌握多模态数据 Pipeline |
| `train.py` | 主训练脚本，支持三种冻结策略 + LoRA/全参切换 | 运行多模态微调实验 |
| `inference.py` | 图像描述/VQA/OCR 等多任务推理测试 | 直观验证视觉理解能力 |

> 💡 **建议学习顺序**: `download_data.py` → 阅读 `config.py` 理解冻结策略 → `train.py`（先用默认 freeze_vision + LoRA）→ 尝试其他策略 → `inference.py`

### 确认前置条件

```bash
cd p06_mllm_vision
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-VL-3B-Instruct', trust_remote_code=True)
print(f'Tokenizer 词表: {t.vocab_size}')
print(f'特殊 token 数: {len(t.all_special_tokens)}')
"
# 预期输出: Tokenizer 词表: 151936
```

---

## 2. 数据下载与探索

```bash
# 下载 LLaVA-Instruct-150K 子集（20K 样本）
python download_data.py --max-samples 20000

# 快速测试（仅 1000 条）
python download_data.py --max-samples 1000

# 查看已下载数据的统计
python download_data.py --explore-only
```

**预期输出**:
```
  总条数:     20000
  平均对话轮数: 1.2
  平均回答长度: 487 字符
  包含图像:     20000 (100.0%)
```

### 为什么选 LLaVA-Instruct

| 数据集 | 规模 | 图像源 | 标注方式 | 特点 |
|--------|------|--------|----------|------|
| LLaVA-Instruct-150K | 150K | COCO | GPT-4 生成 | 高质量多轮对话 |
| ShareGPT4V | 100K | 多源 | GPT-4V 标注 | 细粒度描述 |
| ALLaVA | 700K | 多源 | 混合 | 大规模低成本 |

LLaVA-Instruct 使用 GPT-4 基于 COCO 图像生成指令，质量高且格式统一，适合快速实验。

---

## 3. 数据预处理 — 图文格式化

```python
from dataset import create_mllm_dataset, ImageProcessor
from transformers import AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", trust_remote_code=True)

train_dataset, val_dataset = create_mllm_dataset(
    "data/llava_instruct_20k.json",
    processor=processor,
    tokenizer=processor.tokenizer,
    max_seq_length=2048,
)
```

**多模态数据格式（LLaVA → Qwen2.5-VL）**:

```
LLaVA 格式:                          Qwen2.5-VL 格式:
{"from": "human",                     {"role": "user",
 "value": "<image>\n描述图片"}          "content": [{"type": "image"},
                                                    {"type": "text", "text": "描述图片"}]}
{"from": "gpt",                       {"role": "assistant",
 "value": "图片展示了..."}               "content": [{"type": "text", "text": "图片展示了..."}]}
```

---

## 4. 冻结策略详解

查看 `config.py` 中的冻结策略配置：

| 策略 | 冻结范围 | 可训练参数 | 显存需求 | 适用场景 |
|------|----------|-----------|----------|---------|
| `freeze_vision` | 视觉编码器全冻结 | LLM + 投影层 | ~10 GB | 24G GPU（推荐） |
| `partial_unfreeze` | 冻结视觉前 N 层 | LLM + 视觉后几层 | ~18 GB | 48G GPU |
| `full` | 全模型可训练 | 全部参数 | ~30 GB | 80G GPU |

> 💡 **推荐**: 对于大多数场景，`freeze_vision` + LoRA 是最佳性价比选择。

---

## 5. 实验一：基础微调（freeze_vision + LoRA）

```bash
# 最推荐的配置：冻结视觉 + LoRA on LLM
python train.py
```

**预期输出**:
```
  训练配置:
    冻结策略:       freeze_vision
    LoRA:           启用 (r=16)
    数据集大小:     19000
    Batch Size:     2 × 8 = 16
    学习率:         2e-04
    预计步数:       ~3562
```

观察要点：
- loss 应在前 200 步快速下降
- 最终 loss 通常在 1.0-2.0 之间
- 验证 loss 应与训练 loss 接近（否则过拟合）

---

## 6. 实验二：冻结策略对比

```bash
# 策略 1: 冻结视觉 + LoRA（默认）
python train.py --freeze-strategy freeze_vision

# 策略 2: 部分解冻视觉后 4 层
python train.py --freeze-strategy partial_unfreeze --unfreeze-layers 4

# 策略 3: 全参训练（需要大显存）
python train.py --freeze-strategy full --no-lora --learning-rate 1e-5
```

**对比记录**:

| 指标 | freeze_vision + LoRA | partial_unfreeze | full |
|------|---------------------|------------------|------|
| 可训练参数 | ~20M (~1%) | ~200M (~10%) | ~3B (100%) |
| GPU 显存 | ~10 GB | ~18 GB | ~30 GB |
| 训练速度 | 基准 | ~0.7x | ~0.5x |
| 最终 Loss | 基准 | 更低 | 最低 |
| VQA 准确率 | 好 | 更好 | 最好（但可能过拟合）|

---

## 7. 实验三：LoRA vs 全参 LLM 微调

```bash
# LoRA 微调（推荐）
python train.py --freeze-strategy freeze_vision

# 全参 LLM 微调（对比）
python train.py --freeze-strategy freeze_vision --no-lora --learning-rate 1e-5
```

| 指标 | LoRA (r=16) | 全参 LLM |
|------|-------------|---------|
| 可训练参数 | ~20M | ~1.5B |
| GPU 显存 | ~10 GB | ~20 GB |
| 训练速度 | 基准 | ~0.6x |
| 效果 | 好 | 略好 |

---

## 8. 推理测试

```bash
# 使用默认测试图像
python inference.py

# 指定训练后模型
python inference.py --model-path outputs/mllm_vision/final

# 自定义图像和问题
python inference.py --image path/to/photo.jpg --question "描述这张图片"
```

**测试覆盖的能力**:

| 能力 | 测试内容 | 示例问题 |
|------|---------|---------|
| 图像描述 | 生成详细描述 | "请详细描述这张图片" |
| VQA | 回答视觉问题 | "图片中有几个人？" |
| OCR | 识别图中文字 | "读出图片中的文字" |
| 空间推理 | 物体位置关系 | "描述物体的空间关系" |
| 细粒度识别 | 详细物体信息 | "主要物体是什么？" |

---

## 小结

- [x] 理解了 MLLM 的三阶段架构（Vision Encoder + Projector + LLM）
- [x] 下载并处理了 LLaVA-Instruct 多模态数据
- [x] 对比了三种冻结策略的显存和效果
- [x] 掌握了 LoRA 应用于 LLM 部分的多模态微调
- [x] 测试了图像描述、VQA、OCR 等多种视觉任务

**下一步**: 进入 [p07 SFT 高级微调](../p07_sft_advanced/README.md)，深入学习更复杂的微调技术。
