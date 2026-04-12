# p03 SFT 指令微调 — 代码讲解

> 本文逐一讲解 p03 模块中各脚本的核心代码逻辑，帮助理解 SFT 微调的完整实现。

---

## 1. config.py — 配置管理

### 设计思路

使用 Python `@dataclass` 管理所有超参数，分为三个层次：

```python
@dataclass
class SFTConfig:        # 训练基础配置
@dataclass  
class LoRAConfig:       # LoRA/QLoRA/DoRA 专用配置
@dataclass
class AblationConfig:   # 消融实验配置
```

### 核心配置项

**SFT 方法选择**通过 `method` 字段控制：
- `full_finetune`：学习率设为 2e-5（小 lr 防遗忘）
- `lora/qlora/dora`：学习率设为 2e-4（大 lr 因为参数少）

**GPU Preset** 自动适配不同显存：
- 24G：batch=4, seq_len=512, 开启 GC
- 48G：batch=8, seq_len=1024, 关闭 GC

`create_method_config()` 是快捷方法，一行代码创建对应配置。

---

## 2. dataset.py — 数据处理核心

### ChatML 模板构建

`build_chatml_prompt()` 将 Alpaca 格式数据转换为 Qwen2.5 的 ChatML 格式：

```python
# 输入
instruction = "请解释机器学习"
output = "机器学习是..."

# 输出
"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
"<|im_start|>user\n请解释机器学习<|im_end|>\n"  
"<|im_start|>assistant\n机器学习是...<|im_end|>\n"
```

### Label Masking 实现

`tokenize_with_label_mask()` 是本模块最关键的函数：

1. **编码整个文本**为 token 序列
2. **初始化 labels 全为 -100**（全部 mask）
3. **搜索 assistant_prefix 的 token 序列位置**
4. **在 assistant 回复范围内恢复真实 label**

关键搜索逻辑：
```python
# 将 "<|im_start|>assistant\n" 编码为 token id 序列
assistant_prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)

# 在 input_ids 中滑动搜索该子序列
for j in range(len(input_ids_list)):
    if input_ids_list[j:j+len(prefix_ids)] == prefix_ids:
        # 找到 assistant 回复起始位置
        start = j + len(prefix_ids)
        # ... 搜索结束位置
        labels[start:end] = input_ids[start:end]
```

### 数据质量统计

处理完成后自动打印 label masking 统计：
```
[Label Mask] 总 token: 2,345,678, 计算 loss: 891,234 (38.0%)
```

38% 意味着只有 assistant 回复部分参与训练，这是正确的比例。

---

## 3. train.py — 训练主流程

### 模型加载策略

`load_model_for_sft()` 根据 method 参数选择不同的加载方式：

**全参微调**：直接加载 bf16 模型，所有参数可训练。

**LoRA**：加载 bf16 模型 → `get_peft_model()` 添加 LoRA adapter → 冻结基座参数。

**QLoRA**：`BitsAndBytesConfig` 配置 4-bit 量化 → 加载量化模型 → `prepare_model_for_kbit_training()` → 添加 LoRA。

**DoRA**：同 LoRA，但 `LoraConfig(use_dora=True)` 启用权重分解。

### PEFT 集成

```python
from peft import LoraConfig, get_peft_model, TaskType

peft_config = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", ...],
    task_type=TaskType.CAUSAL_LM,
    use_dora=False,  # DoRA 时设为 True
)
model = get_peft_model(model, peft_config)
```

`get_peft_model()` 做了什么？
1. 遍历模型所有 Linear 层
2. 将 target_modules 中的层替换为 LoRA 版本
3. 冻结原始参数，只有 LoRA 参数可训练

### Gradient Checkpointing 注意事项

LoRA + Gradient Checkpointing 需要额外调用：
```python
model.enable_input_require_grads()
```
否则梯度会断裂，LoRA 参数不会更新。

---

## 4. ablation_runner.py — 消融实验

### 实验流程

每个消融实验独立运行：
1. 创建新的模型实例和 LoRA 配置
2. 训练有限步数（默认 500 步）
3. 收集 train/eval loss
4. 释放显存
5. 保存结果到 JSON

### Rank 消融关键代码

```python
for rank in [8, 16, 32, 64, 128]:
    lcfg.lora_r = rank
    lcfg.lora_alpha = rank * 2  # 保持 alpha/r = 2
    result = run_single_experiment(f"rank_{rank}", ...)
```

保持 alpha/r 比值不变，隔离 rank 的影响。

### 显存管理

每次实验结束后必须释放显存：
```python
del model, trainer
torch.cuda.empty_cache()
```

---

## 5. merge_lora.py — 权重合并

### 合并步骤

```python
# 1. 读取 adapter 配置获取基座模型名
peft_config = PeftConfig.from_pretrained(adapter_path)

# 2. 在 CPU 上加载基座模型（避免 OOM）
model = AutoModelForCausalLM.from_pretrained(..., device_map="cpu")

# 3. 加载 LoRA adapter
model = PeftModel.from_pretrained(model, adapter_path)

# 4. 合并 + 卸载 LoRA 层
model = model.merge_and_unload()

# 5. 保存为 safetensors
model.save_pretrained(output_dir, safe_serialization=True)
```

合并在 CPU 上进行是关键，因为需要同时持有基座参数和 LoRA 参数。

---

## 6. overfit_diagnosis.py — 过拟合检测

### Loss 曲线分析

从 `trainer_state.json` 提取训练日志：
```python
for entry in state["log_history"]:
    if "loss" in entry:      # 训练 loss
    if "eval_loss" in entry: # 验证 loss
```

过拟合判据：最近 3 次 eval_loss 持续上升。

### Distinct-N 计算

```python
def compute_distinct_n(texts, n=2):
    all_ngrams = []
    for text in texts:
        tokens = list(text)  # 字级别
        ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
        all_ngrams.extend(ngrams)
    return len(set(all_ngrams)) / len(all_ngrams)
```

字级别而非词级别，因为中文分词会引入额外噪声。

---

## 7. inference.py — 推理对比

### ChatML 格式推理

```python
chat_prompt = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    f"<|im_start|>user\n{prompt}<|im_end|>\n"
    "<|im_start|>assistant\n"
)
```

注意末尾只到 `assistant\n`，不加 `<|im_end|>`——让模型自己生成回复内容。

### LoRA 模型加载

```python
base_model = AutoModelForCausalLM.from_pretrained(base_name, ...)
model = PeftModel.from_pretrained(base_model, adapter_path)
```

推理时 LoRA 的计算：y = Wx + (α/r)·BAx，额外一次矩阵乘法。

---

## 8. 完整流程串联

```bash
# Step 1: 下载数据
python download_data.py --max-samples 50000

# Step 2: LoRA 微调
python train.py --method lora

# Step 3: 消融实验
python ablation_runner.py --ablation rank

# Step 4: 合并权重
python merge_lora.py --adapter-path outputs/sft_lora/final

# Step 5: 过拟合诊断
python overfit_diagnosis.py --model-path outputs/merged

# Step 6: 推理对比
python inference.py
```

每一步都可以独立运行，也可以按顺序串联完成完整实验。
