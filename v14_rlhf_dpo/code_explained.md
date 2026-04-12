# v14 RLHF / DPO 偏好对齐 — 代码实现详解

## 1. DPO Loss 实现

DPO 的核心只需一个 loss 函数，实现非常简洁：

```python
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             ref_chosen_logps, ref_rejected_logps, beta):
    # 隐式奖励差
    chosen_rewards = policy_chosen_logps - ref_chosen_logps
    rejected_rewards = policy_rejected_logps - ref_rejected_logps
    # Bradley-Terry loss
    logits = beta * (chosen_rewards - rejected_rewards)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

### 关键点：
1. **对数概率计算**：`log_prob = Σ log π(y_t | x, y_{<t})`，对序列中每个 token 的对数概率求和
2. **参考模型冻结**：`ref_model` 始终保持不变，用 `@torch.no_grad()` 计算
3. **β 的作用**：控制偏离参考模型的程度，通常 β ∈ [0.05, 0.5]

## 2. 序列对数概率计算

```python
def get_batch_logps(logits, labels, average_log_prob=False):
    # logits: [B, L, V], labels: [B, L]
    per_token_logps = torch.gather(
        logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
    ).squeeze(2)
    # 忽略 padding
    loss_mask = (labels != -100).float()
    logps = (per_token_logps * loss_mask).sum(-1)
    if average_log_prob:
        logps = logps / loss_mask.sum(-1)
    return logps
```

### average_log_prob 的重要性
- DPO 原始论文使用 **sum**（对序列长度敏感）
- SimPO 使用 **average**（消除长度偏置）
- 实践中 average 通常更稳定

## 3. Reward Model

Reward Model 本质上是在 LLM 基座上加一个线性头：

```python
class RewardModel(nn.Module):
    def __init__(self, base_model, d_model):
        self.base = base_model
        self.reward_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        hidden = self.base.encode(x)
        # 取最后一个 token 的表示作为奖励
        reward = self.reward_head(hidden[:, -1, :])
        return reward.squeeze(-1)
```

## 4. SimPO 实现要点

```python
def simpo_loss(policy_chosen_logps, policy_rejected_logps, beta, gamma):
    # SimPO 不需要参考模型！
    # 使用平均对数概率作为隐式奖励
    logits = beta * (policy_chosen_logps - policy_rejected_logps - gamma)
    loss = -F.logsigmoid(logits).mean()
    return loss
```

关键区别：
- **无需 ref_model**：减少 50% 显存
- **γ (gamma)**：reward margin，确保 chosen 比 rejected 高出阈值
- **使用 average_log_prob**：消除长度偏置

## 5. KTO 实现

```python
def kto_loss(policy_logps, ref_logps, is_desirable, beta):
    logratios = policy_logps - ref_logps
    ref_point = logratios.detach().mean()  # Kahneman 参考点

    desirable_loss = (1 - torch.sigmoid(beta * (logratios - ref_point)))
    undesirable_loss = (1 - torch.sigmoid(beta * (ref_point - logratios)))

    loss = (is_desirable * desirable_loss + ~is_desirable * undesirable_loss).mean()
    return loss
```

KTO 的优势：不需要偏好对，只需要正/负标注。

## 6. 偏好数据集构造

```python
class PreferenceDataset(Dataset):
    # 每条数据包含：
    # - prompt: 输入
    # - chosen: 人类偏好的回复
    # - rejected: 人类不偏好的回复
    def __getitem__(self, idx):
        return {
            "prompt_ids": ...,
            "chosen_ids": ...,     # prompt + chosen_response
            "rejected_ids": ...,   # prompt + rejected_response
        }
```

### 广告域偏好对构造方法：
1. **GPT-4 打分**：用强模型给弱模型的输出打分
2. **人工标注**：专家标注广告描述的好坏
3. **自动构造**：正确标签 vs 随机标签作为偏好对
