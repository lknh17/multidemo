# v14 RLHF / DPO 偏好对齐 — 原理详解

## 1. 为什么需要偏好对齐？

预训练 LLM 学会了"续写最可能的 token"，但：
- 可能生成有害/不安全内容
- 倾向于冗长啰嗦（最大似然偏好长序列）
- 不一定遵循用户指令

**偏好对齐**的目标：让模型的输出分布对齐人类偏好。

## 2. Reward Model (Bradley-Terry)

### 2.1 Bradley-Terry 偏好模型

给定 prompt x，人类比较两个回复 y_w (preferred) 和 y_l (rejected)：

```
P(y_w ≻ y_l | x) = σ(r(x, y_w) - r(x, y_l))
```

其中 r(x, y) 是奖励函数，σ 是 sigmoid。

### 2.2 Reward Model 训练

损失函数（最大化偏好对数似然）：

```
L_RM = -E[log σ(r(x, y_w) - r(x, y_l))]
```

实现：在 LLM 最后一层加一个线性 head 输出标量奖励。

## 3. RLHF (PPO)

### 3.1 RL 目标

最大化奖励同时限制与参考模型的偏离：

```
max_π E_{x~D, y~π(·|x)} [r(x, y) - β · KL(π(·|x) || π_ref(·|x))]
```

### 3.2 PPO 算法

1. 用当前策略 π_old 生成回复
2. 用 Reward Model 打分
3. 计算优势函数 A_t (GAE)
4. 更新策略（裁剪目标）：

```
L_PPO = min(ratio · A_t, clip(ratio, 1-ε, 1+ε) · A_t)
ratio = π(a|s) / π_old(a|s)
```

### 3.3 PPO 的问题

- 训练不稳定：需要 4 个模型（Policy, Ref, Reward, Value）
- 超参敏感：clip_ratio, kl_coef, gae_lambda
- 计算昂贵：每个 token 都需要 RL 更新

## 4. DPO (Direct Preference Optimization)

### 4.1 核心推导

从 RLHF 的 KL 约束优化目标出发：

```
max_π E[r(x,y)] - β · KL(π || π_ref)
```

最优解（闭式）：

```
π*(y|x) = (1/Z(x)) · π_ref(y|x) · exp(r(x,y) / β)
```

对 r 求解：

```
r(x,y) = β · log(π*(y|x) / π_ref(y|x)) + β · log Z(x)
```

代入 Bradley-Terry 模型（Z(x) 消去）：

```
P(y_w ≻ y_l) = σ(β · [log π*(y_w|x)/π_ref(y_w|x) - log π*(y_l|x)/π_ref(y_l|x)])
```

### 4.2 DPO 损失函数

```
L_DPO = -E[log σ(β · (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]
```

简写：

```
L_DPO = -E[log σ(β · (r_w - r_l))]
其中 r_w = log π_θ(y_w|x) - log π_ref(y_w|x)  (隐式奖励)
```

### 4.3 DPO 的优势

| 维度 | PPO (RLHF) | DPO |
|------|-----------|-----|
| 模型数 | 4 (Policy, Ref, Reward, Value) | 2 (Policy, Ref) |
| 稳定性 | 差（RL 训练不稳定） | 好（等价于分类问题） |
| 计算量 | 高（在线采样+多步更新） | 低（离线偏好数据） |
| 超参 | 多（clip, kl_coef, gae_lambda） | 少（主要就 β） |

## 5. DPO 变体

### 5.1 SimPO (Simple Preference Optimization)

无需参考模型！用平均对数概率作为隐式奖励：

```
r_SimPO(y|x) = (1/|y|) · Σ log π_θ(y_t | x, y_{<t})
L_SimPO = -E[log σ(β · (r_w - r_l - γ))]
```

γ 是奖励边际（reward margin），确保 preferred 比 rejected 高出至少 γ。

### 5.2 ORPO (Odds Ratio Preference Optimization)

用几率比(odds ratio)替代对数概率比：

```
odds(y|x) = P(y|x) / (1 - P(y|x))
L_ORPO = L_SFT + λ · L_odds
L_odds = -log σ(log(odds(y_w|x) / odds(y_l|x)))
```

### 5.3 KTO (Kahneman-Tversky Optimization)

不需要偏好对！只需要标记每个样本是"好"还是"不好"：

```
L_KTO = E_w[w_d · (1 - σ(β · (r_w - z_ref)))]
      + E_l[w_u · (1 - σ(β · (z_ref - r_l)))]
```

z_ref 是参考点（类似 Kahneman 的前景理论）。

## 6. 广告域偏好对齐

### 6.1 偏好数据构造

为广告多模态理解构造偏好对：

- **图片描述偏好**：准确描述广告图片内容 > 遗漏关键信息
- **标签偏好**：精确标签 > 模糊标签 > 错误标签
- **Embedding 偏好**：相关内容相似度高 > 不相关内容相似度高

### 6.2 Embedding 偏好对齐

将 DPO 思想应用到 Embedding 空间：

```
L_emb_dpo = -log σ(β · (sim(q, e_w) - sim(q, e_l)))
```

其中 q 是 query embedding，e_w 是 preferred 的 embedding，e_l 是 rejected 的 embedding。
