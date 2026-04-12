# p05 原理详解 - 强化学习 GRPO

> 本文档详细讲解 GRPO 的理论基础：从 PPO 到 GRPO 的演化、完整数学推导、奖励函数设计、奖励欺骗机制与 KL 约束。

---

## 1. 从 RLHF 到 GRPO：演化路径

大语言模型的对齐（Alignment）经历了三代演化：

| 阶段 | 方法 | 核心思路 | 代表工作 |
|------|------|---------|---------|
| 第一代 | RLHF + PPO | 训练奖励模型 + PPO 优化 | InstructGPT, ChatGPT |
| 第二代 | DPO | 绕过奖励模型，直接优化偏好 | DPO (Rafailov 2023) |
| 第三代 | GRPO | 组内相对排名，无需 critic | DeepSeek-R1 (2024) |

**RLHF 的困境**：PPO 需要训练 4 个模型（策略、奖励、参考、critic），显存和工程复杂度极高。

**DPO 的局限**：DPO 只能从人类偏好数据中学习，无法利用"可验证奖励"（如数学题的正确性）。

**GRPO 的突破**：去掉 critic 模型，用组内采样的相对排名作为优势估计。特别适合有可验证奖励的场景。

---

## 2. PPO 回顾：策略梯度基础

### 策略梯度定理

强化学习的目标是最大化期望累积奖励：

$$J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t r_t \right]$$

策略梯度定理给出梯度的计算方式：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t) \cdot A_t \right]$$

其中 $A_t$ 是优势函数（Advantage），衡量动作 $a_t$ 比平均好多少。

### PPO-Clip 目标函数

PPO 通过 clip 机制限制策略更新幅度：

$$L^{CLIP}(\theta) = \mathbb{E} \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon) A_t \right) \right]$$

其中 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是概率比。

**直觉理解**：当优势为正（好动作），PPO 允许概率增加但最多到 $1+\varepsilon$；当优势为负（坏动作），概率减少但最少到 $1-\varepsilon$。这防止了策略一步跳太远。

### PPO 的问题

1. **需要 Critic 模型**: 必须训练一个价值网络来估计 $V(s)$，用于计算 $A_t = r_t + \gamma V(s_{t+1}) - V(s_t)$
2. **Critic 的训练不稳定**: 价值网络的训练本身就是一个难题
3. **超参数敏感**: GAE lambda、critic 学习率等参数对性能影响大
4. **显存消耗**: 需要同时维护策略、奖励、参考、Critic 四个模型

---

## 3. GRPO 核心原理：组内相对排名

### 核心想法

GRPO 的关键洞察：**不需要 critic 来估计绝对优势，只需要在一组响应中进行相对排名**。

对于每个 prompt $x$，GRPO 的流程：

1. **采样**: 从当前策略 $\pi_\theta$ 采样 $G$ 个响应 $\{y_1, y_2, ..., y_G\}$
2. **评分**: 用奖励函数计算 $\{r_1, r_2, ..., r_G\}$
3. **归一化**: 在组内计算相对优势 $\hat{A}_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$
4. **优化**: 用归一化后的优势更新策略

### 形式化定义

给定 prompt $x$，GRPO 的目标函数：

$$L_{GRPO}(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min \left( \frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)} \hat{A}_i, \; \text{clip}\left(\frac{\pi_\theta(y_i|x)}{\pi_{\theta_{old}}(y_i|x)}, 1-\varepsilon, 1+\varepsilon\right) \hat{A}_i \right) \right]$$

其中归一化优势：

$$\hat{A}_i = \frac{r_i - \mu_r}{\sigma_r}, \quad \mu_r = \frac{1}{G}\sum_{j=1}^{G} r_j, \quad \sigma_r = \sqrt{\frac{1}{G}\sum_{j=1}^{G}(r_j - \mu_r)^2}$$

### 为什么有效？

1. **消除 Critic 偏差**: 不依赖价值网络的估计，避免 Critic 的近似误差
2. **自适应基线**: 组均值 $\mu_r$ 自动作为基线（baseline），减小方差
3. **归一化稳定性**: 除以标准差确保优势在相同尺度上，对奖励函数的绝对值不敏感
4. **计算高效**: 不需要额外模型，显存节省约 25-50%

---

## 4. GRPO 完整数学推导

### 从策略梯度出发

标准策略梯度：

$$\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot|x)} \left[ \nabla_\theta \log \pi_\theta(y|x) \cdot (r(y) - b) \right]$$

其中 $b$ 是基线（baseline），常用 $V(x)$ 或 $\mathbb{E}[r]$。

### GRPO 的基线选择

GRPO 选择 **组均值** 作为基线：

$$b = \frac{1}{G}\sum_{i=1}^{G} r(y_i), \quad y_i \sim \pi_\theta(\cdot|x)$$

这是 $V(x) = \mathbb{E}_{y \sim \pi_\theta}[r(y)]$ 的蒙特卡罗无偏估计。当 $G \to \infty$ 时，$b \to V(x)$。

### 方差分析

组均值基线的方差：

$$\text{Var}[b] = \frac{\text{Var}[r(y)]}{G}$$

$G$ 越大，基线越准确，梯度估计的方差越小。这就是 Group Size 影响训练稳定性的原因。

### 加入 PPO-Clip 约束

直接在策略梯度上加 importance sampling + clip：

$$L_{GRPO}(\theta) = \mathbb{E} \left[ \frac{1}{G} \sum_{i=1}^{G} \min(r_i(\theta) \hat{A}_i, \; \text{clip}(r_i(\theta), 1-\varepsilon, 1+\varepsilon) \hat{A}_i) \right]$$

Clip 确保每次更新的步长受限，避免策略崩溃。

### Token 级 GRPO

在语言模型中，策略是自回归的：$\pi_\theta(y|x) = \prod_{t=1}^{T} \pi_\theta(y_t|x, y_{<t})$

Token 级目标函数：

$$L_{GRPO}(\theta) = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T_i} \sum_{t=1}^{T_i} \min\left(\frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{\theta_{old}}(y_{i,t}|x, y_{i,<t})} \hat{A}_i, \; \text{clip}(\cdot) \hat{A}_i \right)$$

注意：GRPO 对整个响应使用相同的 $\hat{A}_i$（response-level advantage），不做 token-level 的优势分配。

---

## 5. KL 散度约束

### 为什么需要 KL 约束

RL 训练中，策略可能偏离参考模型太远，导致：
1. **生成质量退化**: 模型忘记语言能力，只追求奖励
2. **奖励欺骗**: 找到奖励函数的 shortcut，不是真正学会推理
3. **训练不稳定**: 策略变化过大导致 importance ratio 失控

### KL 惩罚项

GRPO 在目标函数中加入 KL 惩罚：

$$L_{total} = L_{GRPO} - \beta \cdot D_{KL}\left[\pi_\theta \| \pi_{ref}\right]$$

KL 散度的计算：

$$D_{KL}[\pi_\theta \| \pi_{ref}] = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{ref}(y|x)} \right]$$

在实践中，使用采样估计：

$$\hat{D}_{KL} = \frac{1}{G} \sum_{i=1}^{G} \frac{1}{T_i} \sum_{t=1}^{T_i} \log \frac{\pi_\theta(y_{i,t}|x, y_{i,<t})}{\pi_{ref}(y_{i,t}|x, y_{i,<t})}$$

### β 的选择

| β 值 | 效果 | 适用场景 |
|------|------|---------|
| 0.01 | 弱约束，策略自由探索 | 奖励信号强且可靠 |
| 0.05 | 平衡约束（推荐） | 一般场景 |
| 0.10 | 强约束，策略保守 | 奖励函数不完美 |
| 自适应 | 动态调整 | 高级场景 |

---

## 6. 奖励函数设计

### 数学推理奖励的挑战

数学推理是 RL 的理想场景，因为奖励**可精确验证**。但设计奖励函数仍有挑战：

1. **稀疏奖励问题**: 只看最终答案对错 → 大部分响应得 0 分，梯度信号弱
2. **过程 vs 结果**: 正确的推理过程比正确答案更重要
3. **奖励形状设计**: 如何鼓励"接近正确"的响应

### 分级奖励设计

```
正确答案 + 好格式 → +1.0 (满分)
正确答案 + 差格式 → +0.7 (鼓励正确)
错误答案 + 好格式 → +0.3 (鼓励格式)
无法提取答案    → -1.0 (强惩罚)
```

### 组合奖励的优势

$$r_{total} = w_1 \cdot r_{correct} + w_2 \cdot r_{format} + w_3 \cdot r_{length}$$

- $r_{correct}$: 粗粒度但信号强，是最终目标
- $r_{format}$: 细粒度但信号密集，引导推理过程
- $r_{length}$: 防止冗余，控制生成长度

组合奖励缓解了稀疏奖励问题：即使答案错误，好的推理格式也能获得正向奖励。

---

## 7. 奖励欺骗 (Reward Hacking)

### 什么是奖励欺骗

奖励欺骗是指模型学会了"看起来好"的策略，在奖励函数上得高分，但实际能力没有提升。

### 典型案例

| 奖励设计 | 欺骗方式 |
|---------|---------|
| 只看答案正确性 | 记忆固定答案模式 |
| 只看格式 | 生成模板化文本，推理内容为空 |
| 看长度 | 生成重复内容凑长度 |
| 看关键词 | 堆砌关键词但逻辑不通 |

### 数学推理中的欺骗

```
# 正常推理
步骤 1：题目说小明有 15 个苹果
步骤 2：给了小红 3 个，剩下 15 - 3 = 12 个
步骤 3：又买了 7 个，所以 12 + 7 = 19 个
#### 19

# 欺骗模式
步骤 1：计算
步骤 2：答案
#### 19
```

两者都能得到正确性满分，但第二种完全没有推理过程。

### 防止奖励欺骗

1. **多维奖励**: 组合正确性 + 格式 + 过程，增加欺骗难度
2. **KL 约束**: 限制策略偏离，防止"走捷径"
3. **周期性检查**: 用人工抽检生成质量
4. **奖励函数迭代**: 发现欺骗后更新奖励设计
5. **多样性正则**: 鼓励响应多样性，避免收敛到固定模式

---

## 8. Group Size 的影响

### 理论分析

Group Size $G$ 决定了优势估计的质量：

- **基线精度**: $\text{Var}[\hat\mu] = \sigma^2 / G$，$G$ 越大基线越准
- **梯度方差**: 更大的 $G$ 减小梯度方差，训练更稳定
- **探索能力**: 更大的 $G$ 覆盖更多响应空间
- **计算成本**: 每个 prompt 需要 $G$ 次前向传播

### 实践建议

$$G_{optimal} \approx \frac{\text{GPU显存}}{\text{单次推理显存} \times \text{batch\_size}}$$

| GPU 显存 | 建议 G | 建议 batch_size |
|---------|--------|----------------|
| 24 GB | 4 | 1 |
| 48 GB | 8-16 | 2-4 |
| 80 GB | 16-32 | 4-8 |

### 小 G 的替代方案

当显存不足以支持大 G 时：
1. **多步累积**: 在多个 step 上累积同一 prompt 的响应
2. **历史缓存**: 保留历史响应，与新响应混合计算优势
3. **温度调整**: 用更高温度增加多样性，弥补小 G 的覆盖不足

---

## 9. GRPO vs DPO vs PPO 对比

### 理论对比

| 维度 | PPO | DPO | GRPO |
|------|-----|-----|------|
| 需要奖励模型 | 是 | 否 | 否（用规则奖励） |
| 需要 Critic | 是 | 否 | 否 |
| 训练模型数 | 4 | 2 | 2 |
| 数据需求 | 在线采样 | 离线偏好 | 在线采样 |
| 适用场景 | 通用 | 偏好对齐 | 可验证奖励 |
| 工程复杂度 | 高 | 低 | 中 |

### 在数学推理上的优势

GRPO 在数学推理上特别有效，因为：

1. **精确奖励**: 数学答案可以精确验证（对/错），不需要模糊的人类偏好
2. **在线采样**: 可以不断探索新的推理路径，不受固定偏好数据限制
3. **无 Critic 偏差**: 不需要训练价值网络，避免了 Critic 的近似误差
4. **计算高效**: 相比 PPO 少了 Critic 模型，显存和计算量减半

### DeepSeek-R1 的成功

DeepSeek-R1 使用 GRPO 在数学推理、代码生成等可验证任务上取得了显著突破：
- GSM8K: 95.8%（vs GPT-4 的 92%）
- MATH: 79.8%
- 关键：冷启动阶段不需要人工标注，只需要可验证的奖励信号

---

## 10. 训练稳定性与调参指南

### 常见问题与解决

| 问题 | 症状 | 解决方案 |
|------|------|---------|
| 奖励不上升 | reward 曲线平坦 | 增大学习率或 G |
| 训练崩溃 | loss NaN | 减小学习率，增大 clip_ratio |
| 奖励欺骗 | reward 上升但准确率不变 | 增大 β，检查奖励函数 |
| KL 爆炸 | KL > 10 | 增大 β，减小学习率 |
| 过早收敛 | reward 过早饱和 | 增大温度，减小 β |

### 学习率选择

RL 训练的学习率通常比 SFT 小 10-100 倍：

- SFT: 1e-5 ~ 5e-5
- DPO: 1e-6 ~ 5e-6
- GRPO: 1e-7 ~ 5e-7

原因：RL 的梯度方差大，大学习率容易导致策略崩溃。

### 训练流程建议

1. **先确认奖励函数正确**: 手动测试几个样本，确认奖励分数合理
2. **小规模试跑**: 用 100 条数据 + 10 步训练，确认流程跑通
3. **监控 KL**: 如果 KL 持续增大，立即调大 β
4. **定期抽检**: 每 100 步抽检生成质量，发现问题及时调整
5. **保存中间模型**: 训练后期可能退化，选择最佳 checkpoint
