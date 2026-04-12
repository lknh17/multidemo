# p04 DPO 偏好对齐 - 代码讲解

## 1. config.py — 多算法统一配置

```python
@dataclass
class DPOConfig:
    """DPO / SimPO / ORPO / KTO 对齐训练配置"""
    
    # 算法选择
    algorithm: str = "dpo"       # dpo / simpo / orpo / kto
    
    # DPO 核心超参数
    beta: float = 0.1            # KL 散度惩罚系数
    loss_type: str = "sigmoid"   # sigmoid / hinge / ipo
    
    # SimPO 特有参数
    simpo_gamma: float = 0.5     # margin 参数
    
    # ORPO 特有参数
    orpo_alpha: float = 1.0      # odds ratio loss 权重
    
    # KTO 特有参数
    kto_desirable_weight: float = 1.0
    kto_undesirable_weight: float = 1.0
```

**设计要点**：
- 用一个统一的 `DPOConfig` 管理所有 4 种算法的参数
- `algorithm` 字段决定使用哪种算法
- `ALGORITHM_DEFAULTS` 字典为每种算法预设推荐超参数
- `apply_algorithm_defaults()` 自动应用默认值，避免每次手动设置
- `create_config_24g()` / `create_config_48g()` 提供不同 GPU 的优化配置

**beta 消融实验**：
```python
beta_ablation_values: List[float] = field(default_factory=lambda: [
    0.01, 0.05, 0.1, 0.2, 0.5, 1.0
])
```
预设了 6 组 beta 值，方便批量实验。

---

## 2. dataset.py — 偏好数据预处理

```python
class PreferenceDataset:
    """DPO/SimPO/ORPO 的偏好数据集"""
    
    def _process(self, data, tokenizer):
        for item in data:
            # 关键：用 chat template 格式化
            chosen_formatted = format_as_chat(prompt, chosen, tokenizer)
            rejected_formatted = format_as_chat(prompt, rejected, tokenizer)
            
            samples.append({
                "prompt": prompt,
                "chosen": chosen_formatted,    # 完整对话（含 prompt）
                "rejected": rejected_formatted,
            })
```

**为什么需要 chat template？**

DPOTrainer 需要完整的对话格式，不能只传 response：
- `chosen` = `[user] prompt [assistant] chosen_response`
- `rejected` = `[user] prompt [assistant] rejected_response`

不同模型有不同的 chat template（Qwen 用 `<|im_start|>`，LLaMA 用 `[INST]`），
`tokenizer.apply_chat_template()` 自动处理差异。

**KTO 数据集**：
```python
class KTODataset:
    """KTO 不需要配对，拆分为单条数据"""
    
    def _process(self, data, tokenizer):
        for item in data:
            # chosen → label=True
            samples.append({"prompt": ..., "completion": ..., "label": True})
            # rejected → label=False  
            samples.append({"prompt": ..., "completion": ..., "label": False})
```

KTO 的数据格式与其他三种不同：每条数据是 `(prompt, completion, label)` 三元组，
其中 `label=True` 表示好回复，`label=False` 表示坏回复。

---

## 3. train.py — 四种算法统一训练

```python
# 算法分发逻辑
if args.algorithm in ("dpo", "simpo"):
    train_dpo(cfg, args)     # DPO 和 SimPO 共用 DPOTrainer
elif args.algorithm == "orpo":
    train_orpo(cfg, args)    # ORPO 用 ORPOTrainer
elif args.algorithm == "kto":
    train_kto(cfg, args)     # KTO 用 KTOTrainer
```

**DPO 训练核心**：
```python
from trl import DPOConfig as TRLDPOConfig, DPOTrainer

training_args = TRLDPOConfig(
    beta=0.1,                    # KL 惩罚系数
    loss_type="sigmoid",         # Bradley-Terry sigmoid loss
    max_length=1024,             # prompt + response 最大长度
    max_prompt_length=512,       # prompt 最大长度
    ...
)

trainer = DPOTrainer(
    model=model,                 # policy 模型
    ref_model=ref_model,         # 参考模型（SimPO 时为 None）
    args=training_args,
    train_dataset=train_dataset,
    processing_class=tokenizer,
    peft_config=peft_config,     # LoRA 配置
)
```

**SimPO vs DPO 的代码差异**：
```python
if cfg.algorithm == "simpo":
    # SimPO 不需要参考模型
    dpo_kwargs["ref_model"] = None
    # SimPO 额外参数
    dpo_kwargs["simpo_gamma"] = cfg.simpo_gamma  # margin
    dpo_kwargs["cpo_alpha"] = cfg.cpo_alpha      # NLL loss 权重
```

SimPO 在 trl 库中是作为 DPOTrainer 的一个模式实现的，
只需设置 `ref_model=None` 并传入 `simpo_gamma` 即可。

---

## 4. build_preference.py — Reject Sampling 构造偏好数据

```python
def generate_multiple_responses(model, tokenizer, prompt, num_responses=4):
    """对同一 prompt 生成多个不同回复"""
    responses = []
    for _ in range(num_responses):
        outputs = model.generate(
            **inputs,
            temperature=0.8,      # 高 temperature → 多样性
            top_p=0.95,
            do_sample=True,
        )
        responses.append(decode(outputs))
    return responses
```

**为什么用高 temperature？**
低 temperature 生成的回复几乎相同，无法构造有意义的偏好对。
高 temperature 引入随机性，让不同采样产生质量差异。

**评分策略**：
```python
# 策略1: 长度评分（简单基线）
score_responses_by_length(responses)

# 策略2: 规则评分（多维度）
score_responses_by_rules(responses, prompt)
# → 长度适中性 + 不重复 + 完整性 + 相关性

# 策略3: Reward Model 评分（最准确）
score_responses_by_reward_model(responses, prompt, reward_model)
```

**构造偏好对**：
```python
# best_worst: 取最好和最差
chosen = responses[argmax(scores)]
rejected = responses[argmin(scores)]

# all_pairs: 所有两两配对（数据量 ×6）
for i, j in combinations:
    if scores[i] > scores[j] + threshold:
        pairs.append((responses[i], responses[j]))
```

---

## 5. inference.py — 多模型并排对比

```python
# 加载 LoRA adapter
adapter_config = os.path.join(model_path, "adapter_config.json")
is_lora = os.path.exists(adapter_config)

if is_lora:
    # LoRA 模型：先加载基座，再加载 adapter，最后合并
    model = AutoModelForCausalLM.from_pretrained(base_model_name)
    model = PeftModel.from_pretrained(model, model_path)
    model = model.merge_and_unload()  # 合并权重，加速推理
```

**为什么要 merge_and_unload？**
推理时不需要保持 LoRA 的低秩结构，合并后：
- 消除了 adapter 的额外前向传播开销
- 推理速度与全参模型相同

**对比维度设计**：
```python
COMPARE_PROMPTS = [
    "请解释量子计算...",        # 知识问答
    "请用三个要点总结...",      # 指令遵循
    "教我如何入侵...",          # 安全性测试
    "请写一首五言绝句...",      # 创意写作
    "一个房间里有3盏灯...",     # 多步推理
    "用Python实现二分查找...",  # 代码生成
]
```

覆盖了 6 个维度，其中安全性测试最能体现 DPO 对齐的效果。

---

## 6. 完整训练 Pipeline

```bash
# Step 1: 下载偏好数据
python download_data.py --dataset ultrafeedback --max-samples 10000

# Step 2: DPO 训练
python train.py --algorithm dpo --beta 0.1

# Step 3: SimPO 训练（对比）
python train.py --algorithm simpo

# Step 4: ORPO 训练（对比）
python train.py --algorithm orpo

# Step 5: KTO 训练（对比）
python train.py --algorithm kto

# Step 6: Beta 消融
python train.py --algorithm dpo --beta 0.01
python train.py --algorithm dpo --beta 0.5

# Step 7: 对比推理
python inference.py --models dpo simpo orpo kto
```

**关键调参经验**：
1. DPO 学习率 5e-7 是安全起点，SimPO 可用 1e-6，ORPO 可用 5e-6
2. 先用小数据（1K）跑通 pipeline，再扩大数据规模
3. 观察 `rewards/margins` 指标——应该单调递增
4. 如果 chosen loss 和 rejected loss 同时下降，说明 beta 太小
