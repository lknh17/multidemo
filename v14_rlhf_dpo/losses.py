"""
v14 RLHF / DPO 偏好对齐 - 损失函数集合

实现多种偏好对齐损失函数：
1. DPO (Direct Preference Optimization)
2. SimPO (Simple Preference Optimization)
3. KTO (Kahneman-Tversky Optimization)
4. Reward Model Loss (Bradley-Terry)
5. Embedding DPO (广告域)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 序列对数概率计算 (所有 DPO 变体的基础)
# ============================================================
def get_batch_logps(
    logits: torch.Tensor,       # [B, L, V]
    labels: torch.Tensor,       # [B, L]
    average_log_prob: bool = False,
) -> torch.Tensor:
    """
    计算序列的对数概率。
    
    Args:
        logits: 模型输出的 logits [B, L, V]
        labels: 目标 token ids [B, L], padding 位置为 -100
        average_log_prob: True → 平均 (SimPO), False → 求和 (DPO)
    
    Returns:
        logps: [B] 每个序列的对数概率
    """
    # 获取每个 token 的对数概率
    # log_softmax → gather 取出 label 对应的概率
    per_token_logps = torch.gather(
        logits.log_softmax(dim=-1),  # [B, L, V]
        dim=2,
        index=labels.clamp(min=0).unsqueeze(2)  # [B, L, 1]
    ).squeeze(2)  # [B, L]
    
    # 构造 loss mask: 忽略 padding (-100)
    loss_mask = (labels != -100).float()
    
    # 序列级对数概率
    logps = (per_token_logps * loss_mask).sum(dim=-1)  # [B]
    
    if average_log_prob:
        # SimPO 使用平均，消除长度偏置
        logps = logps / loss_mask.sum(dim=-1).clamp(min=1)
    
    return logps


# ============================================================
# 2. DPO Loss
# ============================================================
def dpo_loss(
    policy_chosen_logps: torch.Tensor,    # [B]
    policy_rejected_logps: torch.Tensor,  # [B]
    ref_chosen_logps: torch.Tensor,       # [B]
    ref_rejected_logps: torch.Tensor,     # [B]
    beta: float = 0.1,
    label_smoothing: float = 0.0,
) -> tuple:
    """
    DPO (Direct Preference Optimization) 损失函数。
    
    推导链路:
    RLHF目标 → KL约束最优解 → 代入Bradley-Terry → 消去Z(x) → DPO Loss
    
    L_DPO = -E[log σ(β · (r_w - r_l))]
    其中 r = log π_θ(y|x) - log π_ref(y|x) 是隐式奖励
    """
    # 计算隐式奖励（log-ratio）
    chosen_rewards = policy_chosen_logps - ref_chosen_logps
    rejected_rewards = policy_rejected_logps - ref_rejected_logps
    
    # 奖励差
    logits = beta * (chosen_rewards - rejected_rewards)
    
    # DPO 损失（带可选的 label smoothing）
    if label_smoothing > 0:
        # Robust DPO: 允许一定比例的标注错误
        loss = (
            -label_smoothing * F.logsigmoid(-logits)
            - (1 - label_smoothing) * F.logsigmoid(logits)
        ).mean()
    else:
        loss = -F.logsigmoid(logits).mean()
    
    # 用于监控的指标
    with torch.no_grad():
        chosen_reward_mean = chosen_rewards.mean()
        rejected_reward_mean = rejected_rewards.mean()
        reward_margin = (chosen_rewards - rejected_rewards).mean()
        accuracy = (logits > 0).float().mean()  # DPO 隐式准确率
    
    return loss, {
        "chosen_reward": chosen_reward_mean.item(),
        "rejected_reward": rejected_reward_mean.item(),
        "reward_margin": reward_margin.item(),
        "accuracy": accuracy.item(),
    }


# ============================================================
# 3. SimPO Loss (无需参考模型)
# ============================================================
def simpo_loss(
    policy_chosen_logps: torch.Tensor,    # [B] (使用 average_log_prob!)
    policy_rejected_logps: torch.Tensor,  # [B]
    beta: float = 2.0,
    gamma: float = 0.5,
) -> tuple:
    """
    SimPO (Simple Preference Optimization) 损失函数。
    
    核心创新：
    1. 不需要参考模型 → 节省 50% 显存
    2. 使用平均对数概率作为隐式奖励 → 消除长度偏置
    3. 引入 reward margin γ → 确保 chosen 比 rejected 明显更好
    
    L_SimPO = -E[log σ(β · (r_w - r_l - γ))]
    """
    logits = beta * (policy_chosen_logps - policy_rejected_logps - gamma)
    loss = -F.logsigmoid(logits).mean()
    
    with torch.no_grad():
        reward_margin = (policy_chosen_logps - policy_rejected_logps).mean()
        accuracy = (logits > 0).float().mean()
    
    return loss, {
        "reward_margin": reward_margin.item(),
        "accuracy": accuracy.item(),
    }


# ============================================================
# 4. KTO Loss (无需偏好对)
# ============================================================
def kto_loss(
    policy_logps: torch.Tensor,       # [B]
    ref_logps: torch.Tensor,          # [B]
    is_desirable: torch.Tensor,       # [B] bool
    beta: float = 0.1,
    desirable_weight: float = 1.0,
    undesirable_weight: float = 1.0,
) -> tuple:
    """
    KTO (Kahneman-Tversky Optimization) 损失函数。
    
    灵感来自 Kahneman 的前景理论(Prospect Theory)：
    人类对"损失"比"收益"更敏感 → 不对称的损失设计。
    
    优势：不需要成对偏好数据，只需要 good/bad 标注。
    
    L_KTO = w_d · E_good[1 - σ(β(r - z))] + w_u · E_bad[1 - σ(β(z - r))]
    """
    # 隐式奖励
    log_ratios = policy_logps - ref_logps
    
    # Kahneman 参考点: 所有样本的平均奖励
    ref_point = log_ratios.detach().mean()
    
    # 分别计算 desirable 和 undesirable 的损失
    desirable_mask = is_desirable.float()
    undesirable_mask = (~is_desirable).float()
    
    # 对 desirable 样本: 希望 r > z_ref
    desirable_losses = (1 - torch.sigmoid(beta * (log_ratios - ref_point)))
    # 对 undesirable 样本: 希望 r < z_ref
    undesirable_losses = (1 - torch.sigmoid(beta * (ref_point - log_ratios)))
    
    loss = (
        desirable_weight * (desirable_losses * desirable_mask).sum()
        + undesirable_weight * (undesirable_losses * undesirable_mask).sum()
    ) / max(desirable_mask.sum() + undesirable_mask.sum(), 1)
    
    with torch.no_grad():
        desirable_reward = (log_ratios * desirable_mask).sum() / desirable_mask.sum().clamp(1)
        undesirable_reward = (log_ratios * undesirable_mask).sum() / undesirable_mask.sum().clamp(1)
    
    return loss, {
        "desirable_reward": desirable_reward.item(),
        "undesirable_reward": undesirable_reward.item(),
        "ref_point": ref_point.item(),
    }


# ============================================================
# 5. Reward Model Loss (Bradley-Terry)
# ============================================================
def reward_model_loss(
    chosen_rewards: torch.Tensor,    # [B]
    rejected_rewards: torch.Tensor,  # [B]
) -> tuple:
    """
    Reward Model 训练损失 (Bradley-Terry 偏好模型)。
    
    L_RM = -E[log σ(r(x, y_w) - r(x, y_l))]
    """
    logits = chosen_rewards - rejected_rewards
    loss = -F.logsigmoid(logits).mean()
    
    with torch.no_grad():
        accuracy = (logits > 0).float().mean()
    
    return loss, {"accuracy": accuracy.item()}


# ============================================================
# 6. Embedding DPO (广告域多模态对齐)
# ============================================================
def embedding_dpo_loss(
    query_emb: torch.Tensor,        # [B, D]
    chosen_emb: torch.Tensor,       # [B, D]
    rejected_emb: torch.Tensor,     # [B, D]
    beta: float = 0.1,
) -> tuple:
    """
    Embedding 空间的 DPO 损失。
    
    将 DPO 思想从生成模型迁移到 Embedding 模型：
    - 生成模型的 "隐式奖励" = log P(y|x) 
    - Embedding 的 "隐式奖励" = cosine_similarity(q, e)
    
    L = -E[log σ(β · (sim(q, e_w) - sim(q, e_l)))]
    """
    # 归一化
    query_emb = F.normalize(query_emb, dim=-1)
    chosen_emb = F.normalize(chosen_emb, dim=-1)
    rejected_emb = F.normalize(rejected_emb, dim=-1)
    
    # 余弦相似度作为"奖励"
    chosen_sim = (query_emb * chosen_emb).sum(dim=-1)
    rejected_sim = (query_emb * rejected_emb).sum(dim=-1)
    
    logits = beta * (chosen_sim - rejected_sim)
    loss = -F.logsigmoid(logits).mean()
    
    with torch.no_grad():
        accuracy = (logits > 0).float().mean()
        margin = (chosen_sim - rejected_sim).mean()
    
    return loss, {
        "accuracy": accuracy.item(),
        "similarity_margin": margin.item(),
    }
