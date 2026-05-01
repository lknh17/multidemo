# p06 代码详解 - MLLM 多模态视觉微调

> 逐文件解释多模态大语言模型微调模块的核心代码实现。

---

## 1. config.py — 三层配置与冻结策略

### MLLMConfig 核心参数

```python
@dataclass
class MLLMConfig:
    model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    freeze_strategy: str = "freeze_vision"    # 三种策略
    vision_unfreeze_layers: int = 4           # partial_unfreeze 时解冻的层数
```

**冻结策略设计依据**:
- `freeze_vision`: 视觉编码器已有良好的预训练权重，冻结可避免灾难性遗忘，同时大幅减少显存（视觉编码器通常占 30% 以上参数）
- `partial_unfreeze`: 折中方案，让视觉编码器的高层特征适应下游任务
- `full`: 端到端训练效果最好但显存需求大，适合数据量充足的场景

### LoRAConfig 仅作用于 LLM

```python
@dataclass
class LoRAConfig:
    lora_r: int = 16          # MLLM 微调用 r=16 就够了
    lora_alpha: int = 32      # alpha = 2r
    target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
```

**r=16 而非 r=64 的原因**: 指令微调不需要学习大量新知识（不像继续预训练），r=16 的表达力足以让模型学会视觉对话格式。

### ImageConfig 动态分辨率

```python
@dataclass
class ImageConfig:
    min_pixels: int = 256 * 28 * 28    # ~7168 像素 → ~256 visual tokens
    max_pixels: int = 1280 * 28 * 28   # ~35840 像素 → ~1280 visual tokens
```

`min_pixels` 和 `max_pixels` 控制视觉 token 数量的范围。28 是 ViT 的 patch size，所以 token 数 = 总像素数 / (28×28)。

---

## 2. download_data.py — LLaVA 数据下载与解析

### 数据格式转换

```python
def parse_conversations(sample: dict) -> dict:
    """LLaVA 格式 → Qwen2.5-VL 格式"""
    for conv in conversations:
        if conv["from"] == "human":
            content = []
            if "<image>" in value:
                content.append({"type": "image"})      # 图像占位符
                value = value.replace("<image>", "")
            content.append({"type": "text", "text": value})
```

**为什么要转换格式**: LLaVA 使用 `<image>` 文本标签标记图像位置，而 Qwen2.5-VL 使用结构化的 `{"type": "image"}` 格式。转换后可以直接使用 Qwen 的 `apply_chat_template` 处理。

### 数据质量过滤

```python
if len(sample["conversations"]) >= 2:  # 至少一轮完整对话
    samples.append(sample)
```

过滤掉不完整的对话（如只有问题没有回答），确保每条数据都有完整的输入-输出对。

---

## 3. dataset.py — 多模态数据处理

### ImageProcessor 动态分辨率处理

```python
def preprocess(self, image):
    w, h = image.size
    pixels = w * h
    
    if pixels < self.min_pixels:
        scale = (self.min_pixels / pixels) ** 0.5
    elif pixels > self.max_pixels:
        scale = (self.max_pixels / pixels) ** 0.5
    
    # 确保是 28 的倍数
    w = max(28, (w // 28) * 28)
    h = max(28, (h // 28) * 28)
```

**关键细节**:
1. 等比缩放保持宽高比（不变形）
2. 对齐到 28 的倍数（ViT patch size）
3. 先 resize 再 normalize（顺序不能反）

### Label Masking — 只对 assistant 部分计算 loss

```python
labels = input_ids.clone()
labels[attention_mask == 0] = -100  # padding 忽略
# user 输入部分也标记为 -100，只对 assistant 回答计算 loss
```

**为什么不对 user 输入计算 loss**: 
- user 的输入是固定的（不需要模型去"学习"输入什么）
- 只对 assistant 的回答计算 loss，模型学习"给定图像和问题，如何生成回答"
- `-100` 是 PyTorch CrossEntropyLoss 的 `ignore_index` 默认值

### MLLMDataset 的 prompt 构建

```python
def _build_prompt(self, sample: dict) -> str:
    """将 LLaVA 格式对话转为 Qwen2.5-VL 消息格式"""
    value = value.replace("<image>", "<|image_pad|>")
    messages.append({"role": "user", "content": value})
```

`<|image_pad|>` 是 Qwen2.5-VL 的视觉 token 占位符，在实际编码时会被替换为视觉编码器的输出。

---

## 4. train.py — 冻结策略实现与训练流程

### 冻结策略的参数级实现

```python
def apply_freeze_strategy(model, strategy, unfreeze_layers=4):
    if strategy == "freeze_vision":
        for name, param in model.named_parameters():
            if "visual" in name or "vision" in name:
                param.requires_grad = False
```

**通过参数名匹配**: Qwen2.5-VL 的视觉编码器参数名包含 "visual" 关键字。通过 `requires_grad = False` 冻结参数，不会计算梯度、不会更新权重、不占用优化器状态的显存。

### LoRA 应用后的参数统计

```python
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()
# 输出: trainable params: 19,922,944 || all params: 2,207,843,328 || trainable: 0.90%
```

LoRA 只增加了约 0.9% 的可训练参数（约 20M），但可以达到接近全参微调的效果。

### 学习率选择

```python
lr = lora_cfg.learning_rate if use_lora else cfg.learning_rate
# LoRA: 2e-4（较大，因为参数少、梯度方差大）
# 全参: 1e-5（较小，避免灾难性遗忘）
```

LoRA 微调使用更大的学习率是因为：
1. 可训练参数少，需要更大的步幅来收敛
2. LoRA 参数从零初始化，初始梯度较大
3. 不会修改原始权重，不用担心遗忘

---

## 5. inference.py — 多模态推理流程

### Qwen2.5-VL 推理流水线

```python
def generate_with_image(model, tokenizer, processor, image_path, question):
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": question},
    ]}]
    
    text = processor.apply_chat_template(messages, ...)
    inputs = processor(text=[text], images=[image], ...)
    outputs = model.generate(**inputs, ...)
```

**推理流程**:
1. 构造多模态消息（图像 + 文本）
2. `apply_chat_template` 将消息转为模型的 prompt 格式
3. `processor` 同时处理文本（tokenize）和图像（resize + normalize）
4. `model.generate` 自回归生成回答

### 测试图像生成

```python
def create_test_image(save_path):
    img = Image.new("RGB", (640, 480), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([50, 50, 200, 200], fill=(65, 105, 225))
    draw.text((50, 280), "MLLM 视觉理解测试", ...)
```

当没有真实图像时，自动生成包含几何图形和中文文字的测试图像，用于验证模型的基础视觉理解和 OCR 能力。
