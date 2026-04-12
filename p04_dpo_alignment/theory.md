# p04 DPO 偏好对齐 - 原理详解

## 1. RLHF 回顾：从人类反馈到模型对齐

### 为什么需要对齐？

预训练 + SFT 后的模型虽然能对话，但仍然存在关键问题：

- **有害输出**: 可能生成攻击性、歧视性内容
- **幻觉**: 自信地输出错误信息
- **不遵从指令**: 忽略用户的格式要求或约束条件
- **价值观偏差**: 输出不符合人类期望的内容

"对齐"（Alignment）的目标是让模型的输出符合人类的偏好和价值观。

### RLHF 三阶段流程

经典的 RLHF（Reinforcement Learning from Human Feedback）分三步：

1. **SFT 阶段**: 用指令数据微调，让模型学会遵循指令
2. **Reward Model 训练**: 用人类偏好数据训练奖励模型
3. **PPO 强化学习**: 用 PPO 算法优化 policy，最大化 reward

```
阶段1: SFT
  预训练模型 + 指令数据 → SFT 模型 (π_sft)

阶段2: Reward Model
  偏好数据 (chosen > rejected) → Reward Model (r_φ)
  Loss = -log σ(r_φ(chosen) - r_φ(rejected))    [Bradley-Terry 模型]

阶段3: PPO
  max_π E[r_φ(y|x)] - β · KL(π || π_ref)
  用 PPO 迭代优化 policy，同时用 KL 约束防止偏离参考模型太远
```

### RLHF 的痛点

PPO 训练非常不稳定，需要同时维护 4 个模型：

| 模型 | 作用 | 显存占用 |
|------|------|----------|
| Policy (π) | 被优化的模型 | ★★★ |
| Reference (π_ref) | KL 约束基准 | ★★★ |
| Reward Model (r_φ) | 评分 | ★★ |
| Value Model (V) | PPO 的价值函数 | ★★ |

对于 7B 模型，4 个模型需要 ~120GB 显存，远超单卡容量。

**DPO 的核心贡献：将 RLHF 的三阶段简化为一阶段，直接从偏好数据优化 policy。**

---

## 2. DPO 推导：从 RLHF 目标到闭式解

### RLHF 的优化目标

RLHF 的目标函数是：

$$\max_\pi \mathbb{E}_{x \sim D, y \sim \pi(\cdot|x)} [r(x, y)] - \beta \cdot \text{KL}[\pi(\cdot|x) \| \pi_{\text{ref}}(\cdot|x)]$$

其中 $r(x,y)$ 是奖励函数，$\beta$ 控制 KL 约束的强度。

### 关键推导：闭式最优策略

对上述目标求解，可以得到最优策略的闭式表达：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{ref}}(y|x) \exp\left(\frac{1}{\beta} r(x, y)\right)$$

其中 $Z(x) = \sum_y \pi_{\text{ref}}(y|x) \exp(\frac{1}{\beta} r(x,y))$ 是配分函数。

### 反解 reward

将上式取对数并重排，得到 reward 的表达式：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta \log Z(x)$$

关键洞察：**reward 可以用 policy 和 reference policy 的 log ratio 来表示。**

### DPO Loss

将 reward 表达式代入 Bradley-Terry 偏好模型：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

配分函数 $Z(x)$ 在做差时抵消，最终得到 DPO 的 loss：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x, y_w, y_l)} \left[\log \sigma\left(\beta \left(\log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)\right)\right]$$

**这个 loss 只需要 policy 模型和参考模型，不需要训练 reward model！**

### DPO 的直觉理解

DPO loss 的梯度可以分解为：

$$\nabla_\theta \mathcal{L}_{\text{DPO}} \propto -\underbrace{(1 - P(y_w \succ y_l))}_{\text{权重}} \left[\underbrace{\nabla_\theta \log \pi_\theta(y_w|x)}_{\text{增大 chosen 概率}} - \underbrace{\nabla_\theta \log \pi_\theta(y_l|x)}_{\text{减小 rejected 概率}}\right]$$

- 当模型已经很好地区分 chosen/rejected 时，权重趋于 0（不再优化）
- 当模型分不清时，权重大，梯度信号强
- 同时增大好回复的概率、减小坏回复的概率

---

## 3. SimPO：无需参考模型的简化 DPO

### 动机

DPO 需要一个参考模型 $\pi_{\text{ref}}$，这带来两个问题：
1. **内存翻倍**: 需要同时加载两个模型
2. **参考模型选择**: 参考模型的质量影响训练效果

SimPO（Simple Preference Optimization）提出两个关键改进：

### 改进 1：序列级平均 log-prob

DPO 使用 token-level log-prob 之和，长序列天然有更低的分数。SimPO 改用序列平均：

$$r_{\text{SimPO}}(x, y) = \frac{1}{|y|} \log \pi_\theta(y|x)$$

这消除了长度偏差，使评分更公平。

### 改进 2：目标 margin γ

引入 margin 参数 $\gamma$ 来增强 chosen 和 rejected 的区分度：

$$\mathcal{L}_{\text{SimPO}} = -\mathbb{E} \left[\log \sigma\left(\frac{\beta}{|y_w|} \log \pi_\theta(y_w|x) - \frac{\beta}{|y_l|} \log \pi_\theta(y_l|x) - \gamma\right)\right]$$

$\gamma > 0$ 要求 chosen 的分数不仅要高于 rejected，还要高出一个 margin。

### SimPO 不需要参考模型

由于使用模型自身的 log-prob 作为隐式 reward，SimPO 完全不需要参考模型，**显存减半**。

---

## 4. ORPO：Odds Ratio 偏好优化

### 动机

ORPO 的核心思想是：**将 SFT 和偏好对齐合并为一个训练阶段**。

传统流程：SFT → DPO（两阶段）
ORPO 流程：直接从预训练模型训练（一阶段）

### Odds Ratio 的定义

定义回复 $y$ 的 odds：

$$\text{odds}_\theta(y|x) = \frac{P_\theta(y|x)}{1 - P_\theta(y|x)}$$

ORPO 使用 odds ratio 来度量偏好：

$$\text{OR}_\theta(y_w, y_l | x) = \frac{\text{odds}_\theta(y_w|x)}{\text{odds}_\theta(y_l|x)}$$

### ORPO Loss

$$\mathcal{L}_{\text{ORPO}} = \underbrace{\mathcal{L}_{\text{NLL}}}_{\text{语言建模}} + \alpha \cdot \underbrace{\log \sigma(\log \text{OR}_\theta(y_w, y_l | x))}_{\text{偏好优化}}$$

- $\mathcal{L}_{\text{NLL}}$: 标准的因果语言建模 loss（让模型学会生成文本）
- Odds Ratio loss: 让 chosen 的 odds 大于 rejected 的 odds
- $\alpha$: 权重超参数

### ORPO 的优势

1. **无需参考模型**: 使用 odds ratio 而非 log ratio
2. **一阶段训练**: 不需要先做 SFT
3. **实现简单**: Loss 直接结合了 NLL 和偏好优化

---

## 5. KTO：基于前景理论的对齐

### 动机

DPO/SimPO/ORPO 都需要 **配对数据**（同一个 prompt 的 chosen 和 rejected 必须成对出现）。
实际中，收集配对数据的成本远高于收集单独的"好"/"坏"标签。

KTO（Kahneman-Tversky Optimization）基于行为经济学的**前景理论**设计。

### 前景理论的核心

Kahneman 和 Tversky 发现人类决策有两个关键偏差：

1. **损失厌恶**: 人对损失的敏感度高于等量收益（losing $100 > gaining $100）
2. **参考点依赖**: 人的判断基于相对于参考点的偏差，而非绝对值

### KTO Loss

对于"好"的回复（desirable）：
$$\mathcal{L}_{\text{KTO}}^+ = 1 - \sigma\left(\beta \left(\log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)} - z_{\text{ref}}\right)\right)$$

对于"坏"的回复（undesirable）：
$$\mathcal{L}_{\text{KTO}}^- = 1 - \sigma\left(\beta \left(z_{\text{ref}} - \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}\right)\right)$$

其中 $z_{\text{ref}}$ 是参考点（KL 散度的期望值）。

总 Loss：
$$\mathcal{L}_{\text{KTO}} = \lambda_+ \cdot \mathcal{L}_{\text{KTO}}^+ + \lambda_- \cdot \mathcal{L}_{\text{KTO}}^-$$

$\lambda_+$ 和 $\lambda_-$ 可以不对称，体现"损失厌恶"（$\lambda_- > \lambda_+$）。

### KTO 的数据优势

| 数据类型 | DPO | SimPO | ORPO | KTO |
|----------|-----|-------|------|-----|
| 配对 (chosen, rejected) | ✅ 必需 | ✅ 必需 | ✅ 必需 | ❌ 不需要 |
| 单条 (response, label) | ❌ | ❌ | ❌ | ✅ 支持 |

---

## 6. Beta 参数深度分析

### Beta 的物理意义

在所有 DPO 变体中，$\beta$ 控制"偏离参考模型的惩罚力度"：

$$\text{隐式 reward} = \beta \cdot \log \frac{\pi_\theta(y|x)}{\pi_{\text{ref}}(y|x)}$$

- **$\beta$ 小**: 每单位 log-ratio 变化对应更小的 reward → 模型可以大幅偏离参考
- **$\beta$ 大**: 每单位 log-ratio 变化对应更大的 reward → 模型被"锁定"在参考附近

### Beta 过小的风险

$$\beta \to 0 \implies \text{KL 约束消失} \implies \text{模型过拟合偏好数据}$$

表现：
- 模型输出高度模板化
- 多样性急剧下降
- 可能出现"reward hacking"（找到 loss 的捷径，但实际质量下降）

### Beta 过大的风险

$$\beta \to \infty \implies \text{KL 约束太强} \implies \pi_\theta \approx \pi_{\text{ref}}$$

表现：
- 训练几乎没有效果
- 模型输出和参考模型一样
- 浪费计算资源

### 推荐 Beta 值

| 算法 | 推荐范围 | 默认值 | 说明 |
|------|----------|--------|------|
| DPO | 0.05-0.5 | 0.1 | 标准设置 |
| SimPO | 1.0-5.0 | 2.0 | SimPO 的 beta 含义不同，通常更大 |
| ORPO | — | — | ORPO 不直接使用 beta |
| KTO | 0.05-0.5 | 0.1 | 与 DPO 类似 |

---

## 7. 偏好数据的构造方法

### 方法 1：人类标注

- **流程**: 给标注者展示同一 prompt 的两个回复，让其选择更好的
- **优点**: 最准确，反映真实人类偏好
- **缺点**: 成本极高，标注一致性难保证

### 方法 2：GPT-4 / 强模型标注

- **流程**: 用 GPT-4 对回复打分或排序，构造偏好对
- **优点**: 成本低，规模化容易
- **缺点**: 受限于标注模型自身的偏好

UltraFeedback 就是用 GPT-4 标注的，效果证明了这种方法的有效性。

### 方法 3：Reject Sampling（自举法）

- **流程**: 用 SFT 模型自身生成多个回复 → 评分 → 取最好/最差构造偏好对
- **优点**: 不需要外部模型或人工标注
- **缺点**: 受限于 SFT 模型的能力上限

```python
# Reject Sampling 伪代码
for prompt in prompts:
    responses = model.generate(prompt, num_return=4, temperature=0.8)
    scores = reward_model.score(prompt, responses)
    chosen = responses[argmax(scores)]
    rejected = responses[argmin(scores)]
    save_pair(prompt, chosen, rejected)
```

### 方法 4：AI Feedback（RLAIF）

- **流程**: 用 AI 系统（如 Constitutional AI）自动生成反馈
- **优点**: 可扩展性强
- **缺点**: AI 的偏好可能与人类不一致

---

## 8. 四种算法的数学对比

### Loss 函数对比

| 算法 | Loss | 参考模型 | 配对数据 |
|------|------|----------|----------|
| DPO | $-\log\sigma(\beta(r_w - r_l))$, 其中 $r = \log\frac{\pi}{\pi_{ref}}$ | 需要 | 需要 |
| SimPO | $-\log\sigma(\beta(\bar{r}_w - \bar{r}_l) - \gamma)$, 其中 $\bar{r} = \frac{1}{\|y\|}\log\pi$ | 不需要 | 需要 |
| ORPO | $L_{NLL} + \alpha \cdot \log\sigma(\log OR)$ | 不需要 | 需要 |
| KTO | $(1-\sigma(\beta(r-z_{ref})))$ 或 $(1-\sigma(\beta(z_{ref}-r)))$ | 需要 | 不需要 |

### 梯度方向对比

所有算法的梯度方向都可以解释为：
1. **增大 chosen/desirable 回复的概率**
2. **减小 rejected/undesirable 回复的概率**
3. **通过某种机制防止过拟合**（KL 约束 / Odds Ratio / margin）

### 计算复杂度对比

| 算法 | 前向传播次数 | 显存（相对） | 实现复杂度 |
|------|-------------|-------------|------------|
| DPO | 4 (π_w, π_l, π_ref_w, π_ref_l) | 1.0x | 中 |
| SimPO | 2 (π_w, π_l) | 0.5x | 低 |
| ORPO | 2 (π_w, π_l) | 0.5x | 中 |
| KTO | 2 (π, π_ref) | 0.7x | 高 |

---

## 9. 实践要点与常见问题

### 训练不稳定怎么办？

1. **降低学习率**: DPO 需要非常小的 lr（1e-7 ~ 5e-6）
2. **增大 beta**: 更强的 KL 约束可以稳定训练
3. **使用 label smoothing**: 设置 0.1-0.2 的 label smoothing
4. **检查数据质量**: 确保 chosen 确实比 rejected 好

### Reward Hacking 怎么办？

Reward Hacking 表现为 loss 下降但实际输出质量不升反降。

解决方案：
1. **定期人工评估**: 不能只看 loss
2. **增大 beta**: 限制模型偏离参考模型的程度
3. **数据多样性**: 使用更多样化的偏好数据
4. **Early stopping**: 在验证集上监控

### LoRA vs 全参训练？

| 方面 | LoRA (r=16) | 全参 |
|------|-------------|------|
| 可训练参数 | ~4M (~0.8%) | 494M (100%) |
| 显存 | ~8GB | ~20GB |
| 训练速度 | 快 | 慢 |
| 效果 | 接近全参 | 略好 |
| 推荐场景 | 日常使用 | 追求极致 |

**结论**: 对于 DPO 对齐，LoRA 是最佳性价比选择。

---

## 10. 前沿进展与展望

### DPO 的局限性

1. **离线算法**: DPO 使用固定的偏好数据，不能在线更新
2. **分布偏移**: 训练数据的分布可能与实际使用时不匹配
3. **Bradley-Terry 假设**: 假设人类偏好遵循 BT 模型，但实际更复杂
4. **单轮偏好**: 只考虑单轮对话的偏好，难以处理多轮交互

### 最新进展

- **Online DPO / OAIF**: 在线版本的 DPO，边训练边采样新数据
- **IPO (Identity Preference Optimization)**: 更鲁棒的偏好优化目标
- **SPIN (Self-Play Fine-Tuning)**: 模型自我博弈生成偏好数据
- **DPO + RLHF 混合**: 先 DPO 初始化，再用 PPO 微调
- **Process Reward Model**: 对推理过程（而非最终答案）给予奖励

### 算法演进路线

```
RLHF (2022) → DPO (2023) → SimPO (2024) → ???
     ↓              ↓             ↓
  复杂昂贵      简单有效      更加简化
     ↓              ↓             ↓
  4个模型        2个模型       1个模型
```

对齐技术正朝着**更简单、更高效、更鲁棒**的方向演进。
