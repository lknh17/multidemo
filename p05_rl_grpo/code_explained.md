# p05 代码详解 - 强化学习 GRPO

> 逐文件解释 GRPO 模块的核心代码实现。

---

## 1. config.py — GRPO 核心参数

### GRPOConfig 设计

```python
@dataclass
class GRPOConfig:
    group_size: int = 8           # 每个 prompt 采样 G 个响应
    temperature: float = 0.7      # 采样温度
    clip_ratio: float = 0.2       # PPO clip 范围 ε
    kl_coef: float = 0.05         # KL 惩罚系数 β
    learning_rate: float = 5e-7   # RL 学习率（比 SFT 小 100 倍）
```

**参数取值依据**:
- `group_size=8`: 在 24G 显存下的最佳平衡点。太小（4）优势估计不准，太大（16）显存不够
- `temperature=0.7`: 既要多样性（探索新推理路径），又不能太随机（保证质量）
- `clip_ratio=0.2`: PPO 的标准值，限制每次更新步长
- `kl_coef=0.05`: 经验值，防止策略偏离过大
- `learning_rate=5e-7`: RL 梯度方差大，必须用极小的学习率

### 奖励权重设计

```python
reward_correctness_weight: float = 0.6   # 正确性是核心目标
reward_format_weight: float = 0.3        # 格式是辅助信号
reward_length_weight: float = 0.1        # 长度控制是弱约束
```

权重比例 6:3:1 确保模型优先追求正确答案，同时保持推理格式。

### 消融实验配置

```python
ablation_configs = {
    "group_4": create_ablation_group_size(4),
    "group_8": create_ablation_group_size(8),
    "kl_0.01": create_ablation_kl_coef(0.01),
    ...
}
```

每个消融配置自动设置不同的输出目录，便于对比实验。

---

## 2. dataset.py — GSM8K 数据处理

### 答案提取核心逻辑

```python
def extract_answer(answer_text: str) -> Optional[float]:
    if "####" in answer_text:
        ans_str = answer_text.split("####")[-1].strip()
        ans_str = ans_str.replace(",", "").replace("$", "")
        return float(ans_str)
```

GSM8K 的标准答案格式是 `推理过程\n#### 数值`，提取 `####` 后面的数字即可。

### Chain-of-Thought Prompt 构建

```python
COT_TEMPLATE = """请一步一步解决以下数学问题：

{question}

请按照以下格式回答：
步骤 1：...
步骤 2：...
...
#### 最终答案"""
```

**设计要点**:
1. 明确要求"一步一步"→ 引导 CoT 推理
2. 给出格式模板 → 便于后续格式奖励评判
3. 指定答案标记 `####` → 便于答案提取

### 模型答案提取的鲁棒性

```python
def extract_model_answer(response: str) -> Optional[float]:
    # 尝试 #### 格式
    # 尝试 \boxed{} 格式
    # 尝试 "答案是" 格式
    # 最后: 取最后一个数字
```

模型可能用各种格式输出答案，需要多种 pattern 依次尝试，提高提取成功率。

---

## 3. reward.py — 三种奖励函数

### 正确性奖励

```python
def correctness_reward(response, ground_truth, tolerance=1e-5):
    model_answer = extract_model_answer(response)
    if model_answer is None:
        return -1.0                    # 无法提取 → 强惩罚
    if abs(model_answer - ground_truth) < tolerance:
        return 1.0                     # 完全正确 → 满分
    return -0.5                        # 错误 → 半惩罚
```

**设计考虑**:
- 无法提取答案（-1.0）比答案错误（-0.5）惩罚更重，鼓励模型至少给出一个数字
- 使用容差（tolerance）处理浮点精度问题
- 还增加了相对误差检查（< 1%），处理四舍五入差异

### 格式奖励

```python
def format_reward(response):
    step_patterns = [
        r'步骤\s*\d+[：:]',     # 中文格式
        r'[Ss]tep\s*\d+[：:]',  # 英文格式
        r'\d+\.\s+',            # 编号格式
    ]
    # 检查步骤数、答案标记等
```

格式奖励提供了**密集的训练信号**：即使答案错误，只要推理格式好也能获得正向奖励。这对 RL 训练初期特别重要。

### 组合奖励的字典返回

```python
def composite_reward(response, ground_truth):
    return {
        "total": 0.78,
        "correctness": 1.0,
        "format": 0.6,
        "length": 0.0,
        "weights": {"correctness": 0.6, "format": 0.3, "length": 0.1}
    }
```

返回完整的奖励分解，方便调试和分析奖励构成。

---

## 4. train.py — GRPOTrainer 训练

### 奖励函数包装

```python
def create_reward_function(reward_type):
    def reward_fn(completions, ground_truths=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            # 提取文本 → 计算奖励
            rewards.append(r)
        return rewards
    return reward_fn
```

trl 的 GRPOTrainer 需要特定签名的奖励函数。我们用闭包将 ground_truth 传入。

### TRL GRPOConfig 配置

```python
training_config = TRLGRPOConfig(
    num_generations=group_size,         # = G，每个 prompt 采样的响应数
    max_completion_length=512,          # 最大生成长度
    beta=kl_coef,                       # KL 惩罚系数
    # ... 标准 TrainingArguments ...
)
```

`num_generations` 是 GRPO 的核心参数，对应论文中的 Group Size G。

### 数据集格式

```python
train_ds = Dataset.from_dict({
    "prompt": prompts,
    "ground_truth": ground_truths,
})
```

GRPOTrainer 需要包含 `prompt` 列的 HuggingFace Dataset。`ground_truth` 作为额外信息传递给奖励函数。

---

## 5. train_openrlhf.py — OpenRLHF 替代方案

### 数据格式转换

```python
processed.append({
    "prompt": prompt_text,    # Chatml 格式的 prompt
    "metadata": {
        "ground_truth": gt,   # 标准答案
        "question": question, # 原始问题
    }
})
```

OpenRLHF 使用自己的数据格式，需要显式构建 Chatml prompt。

### 奖励函数接口

```python
class GSM8KRewardFunction:
    def __call__(self, prompts, responses, metadata=None):
        # 返回 list[float]
```

OpenRLHF 支持将奖励函数作为独立服务运行，通过 HTTP API 调用。这在多 GPU 场景下更高效。

### CLI 命令生成

```python
openrlhf_args = {
    "num_episodes": group_size,      # = G
    "kl_coef": kl_coef,             # = β
    "generate_max_len": 512,         # 最大生成长度
}
```

当 OpenRLHF 未安装时，脚本生成可执行的 shell 命令。

---

## 6. monitor.py — 训练监控

### 奖励欺骗检测算法

```python
def detect_reward_hacking(log_dir, threshold=0.3):
    # 检查 1: 奖励后期加速上升
    if second_slope > first_slope * 2:
        warnings.append("奖励后半段上升过快")
    
    # 检查 2: KL 过大
    if max_kl > 10.0:
        warnings.append("KL 散度过大")
    
    # 检查 3: 奖励方差急剧减小
    if late_var < early_var * 0.1:
        warnings.append("奖励方差急剧减小，可能收敛到固定模式")
```

三个检查维度：
1. **奖励加速**: 正常训练奖励增速应减缓（边际递减），加速上升是异常信号
2. **KL 爆炸**: KL > 10 说明策略严重偏离参考模型
3. **方差消失**: 如果所有响应的奖励几乎相同，说明模型在"复制"固定模式

### 多实验对比图

```python
def plot_reward_curves(log_dirs, save_path):
    fig, axes = plt.subplots(2, 2)
    # 奖励曲线 | KL 散度
    # 训练 Loss | 监控要点
```

2×2 布局展示核心指标，便于一眼看出实验差异。

---

## 7. inference.py — 多模型对比

### 评估流程

```python
def eval_gsm8k(model, tokenizer, data_path, max_samples=100):
    for s in samples:
        response = generate(model, tokenizer, s["question"])
        model_ans = extract_model_answer(response)
        is_correct = abs(model_ans - gt) < 1e-5
```

自动化的 GSM8K 评估：生成 → 提取答案 → 精确比较。

### 多模型对比框架

```python
model_paths = {
    "Base": "Qwen/Qwen2.5-0.5B-Instruct",
    "SFT": "../p03_sft/outputs/final",
    "DPO": "../p04_dpo/outputs/final",
    "GRPO": "outputs/grpo/final",
}
compare_models(model_paths, test_data)
```

支持一次性对比所有模型（Base → SFT → DPO → GRPO），展示训练各阶段的效果累积。

---

## 8. 代码架构总结

```
p05_rl_grpo/
├── config.py           # 配置中心（GRPO 参数 + 消融实验）
├── download_data.py    # 数据下载（GSM8K）
├── dataset.py          # 数据处理（CoT prompt + 答案提取）
├── reward.py           # 奖励函数（核心组件）
├── train.py            # trl GRPOTrainer 训练
├── train_openrlhf.py   # OpenRLHF 替代方案
├── monitor.py          # 训练监控 + 奖励欺骗检测
├── inference.py        # 推理对比（SFT/DPO/GRPO）
├── requirements.txt
├── README.md           # 教程
├── theory.md           # 原理讲解
├── code_explained.md   # 代码详解（本文件）
├── theory_visual.html  # 原理动画
└── code_visual.html    # 代码动画
```

**核心数据流**:
```
GSM8K 数据 → CoT Prompt → 模型采样 G 个响应 → 奖励评分 → 组内排名 → 策略更新
```

**与 p03/p04 的关系**:
- p03 SFT: 让模型学会对话格式 → 作为 GRPO 的起点模型
- p04 DPO: 让模型学会偏好 → 可以和 GRPO 效果对比
- p05 GRPO: 在可验证任务上进一步优化 → 超越 DPO 的效果上限
