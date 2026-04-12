# p02 代码详解 - 继续预训练

> 逐文件解释继续预训练模块的核心代码实现。

---

## 1. config.py — 24G/48G 两套 Preset

### Preset 设计

```python
def create_config_24g() -> PretrainConfig:
    cfg = PretrainConfig()
    cfg.per_device_train_batch_size = 4
    cfg.gradient_accumulation_steps = 4     # 等效 batch = 16
    cfg.max_seq_length = 512
    cfg.gradient_checkpointing = True       # 24G 必须开启
    return cfg
```

**参数取值依据**:
- `batch_size=4 × grad_accum=4 = 16`: 0.5B 模型推荐的有效 batch size
- `max_seq_length=512`: 平衡训练速度和文本覆盖（Wikipedia 平均长度 ~2000 字符 ≈ ~500 tokens）
- `gradient_checkpointing=True`: 24G 下必须开启，否则可能 OOM

**48G vs 24G 差异**: 48G 可以用更大的 batch（8）和更长的序列（1024），不需要 GC。

---

## 2. download_data.py — 流式加载与缓存

```python
dataset = load_dataset("wikipedia", "20220301.zh", split="train", streaming=True)
```

**streaming=True 的优势**:
- 不需要一次性下载全部数据（Wikipedia 中文 ~2GB）
- 按需获取，内存友好
- 可以随时 `break` 控制数据量

**数据质量过滤**: `len(item["text"]) > 100` 过滤过短的文章（如重定向页面）。

---

## 3. dataset.py — Packing 算法实现

### PackedDataset 核心逻辑

```python
def _pack(self, texts, tokenizer):
    all_token_ids = []
    for text in texts:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_token_ids.extend(tokens)
        all_token_ids.append(eos_token_id)  # 文档分隔

    samples = []
    for i in range(0, len(all_token_ids) - max_seq_length, max_seq_length):
        chunk = all_token_ids[i : i + max_seq_length]
        samples.append({"input_ids": chunk, "labels": chunk})
```

**关键细节**:
1. `add_special_tokens=False`: 不添加 BOS/EOS，我们手动在文档间插入 EOS
2. `labels = input_ids`: CLM 的 labels 等于 input_ids，模型内部会自动做 shift（input_ids[:-1] 预测 labels[1:]）
3. 文档间插入 `eos_token_id` 作为分隔符
4. 滑动窗口不重叠（步长 = max_seq_length）

### Padding 模式的 label masking

```python
labels = input_ids.clone()
labels[attention_mask == 0] = -100  # -100 会被 CrossEntropyLoss 忽略
```

PyTorch 的 `CrossEntropyLoss` 中 `ignore_index=-100` 是默认值，标为 -100 的位置不参与 loss 计算。

---

## 4. ds_config_zero2.json — ZeRO-2 配置详解

| 字段 | 值 | 含义 |
|------|-----|------|
| `stage` | 2 | 梯度+优化器分片 |
| `overlap_comm` | true | 通信与计算重叠，隐藏通信延迟 |
| `reduce_scatter` | true | 比 all-reduce 更高效的梯度同步 |
| `contiguous_gradients` | true | 梯度连续存储，减少内存碎片 |
| `allgather_bucket_size` | 2e8 | 通信桶大小（200MB），平衡延迟和吞吐 |

**"auto" 值**: 标记为 "auto" 的字段由 HF Trainer 自动从 `TrainingArguments` 中读取，不需要手动填写。

---

## 5. ds_config_zero3.json — ZeRO-3 与 CPU Offload

ZeRO-3 比 ZeRO-2 多了两个关键配置：

```json
"offload_optimizer": {"device": "cpu", "pin_memory": true},
"offload_param": {"device": "cpu", "pin_memory": true}
```

- `offload_optimizer`: 将优化器状态放到 CPU 内存（省 GPU 显存）
- `offload_param`: 将不活跃的参数也放到 CPU
- `pin_memory=true`: 使用锁页内存，加速 CPU↔GPU 数据传输

**代价**: CPU↔GPU 传输是训练的瓶颈，速度下降 30-50%。

---

## 6. train.py — Trainer 初始化与训练流程

### 模型加载

```python
model = AutoModelForCausalLM.from_pretrained(
    cfg.model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
```

- `torch_dtype=torch.bfloat16`: 以 bf16 加载，节省显存
- `attn_implementation="flash_attention_2"`: 使用 Flash Attention（需要安装 flash-attn）

### TrainingArguments 关键参数

```python
TrainingArguments(
    per_device_train_batch_size=4,      # 每步每 GPU 的样本数
    gradient_accumulation_steps=4,       # 梯度累积步数
    # 等效 batch = 4 × 4 = 16
    # 梯度累积：先算 4 步的梯度并累加，然后一次性更新参数
    # 效果等同于 batch=16，但显存只需 batch=4 的量
)
```

### 断点续训

```python
trainer.train(resume_from_checkpoint=resume_from)
```

Trainer 自动从 checkpoint 目录恢复模型参数、优化器状态、学习率调度器状态和训练步数。

---

## 7. train_lora.py — LoRA 预训练配置

```python
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=64,                    # LoRA 秩（预训练用较大值）
    lora_alpha=128,          # alpha = 2r（缩放因子）
    target_modules=[...],    # 覆盖所有 linear 层
)
model = get_peft_model(model, peft_config)
```

**预训练 vs SFT 的 LoRA 差异**:
- 预训练用 **r=64**（更大），因为需要学习更多知识
- SFT 通常用 r=8-32 就够了
- 预训练的学习率也更大（1e-4 vs 2e-4）

---

## 8. compare_strategies.py — 实验结果对比

```python
state = json.load(open("trainer_state.json"))
log_history = state["log_history"]
losses = [(e["step"], e["loss"]) for e in log_history if "loss" in e]
```

HF Trainer 自动在 `output_dir` 中保存 `trainer_state.json`，包含完整的训练历史（每 N 步的 loss、学习率等）。脚本读取多个实验目录的日志，绘制对比图。

---

## 9. inference.py — 前后对比推理

```python
# 加载两个模型
base_model = load_model("Qwen/Qwen2.5-0.5B")
trained_model = load_model("outputs/pretrain/final")

# 同一 prompt 对比
for prompt in COMPARE_PROMPTS:
    base_output = generate(base_model, tokenizer, prompt)
    trained_output = generate(trained_model, tokenizer, prompt)
```

**对比维度**:
- 中文知识准确性（训练后应更好）
- 生成流畅度（应保持或提升）
- 英文能力（检测遗忘）
