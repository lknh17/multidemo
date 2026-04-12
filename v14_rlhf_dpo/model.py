"""
v14 RLHF / DPO 偏好对齐 - 模型定义

包含：
1. PolicyModel — 策略模型 (Decoder-only LLM)
2. RewardModel — 奖励模型 (LLM + scalar head)
3. DPOTrainer — DPO 训练封装
"""
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import RLHFConfig


# ============================================================
# 基础组件
# ============================================================
class RMSNorm(nn.Module):
    """RMSNorm — 现代 LLM 标准归一化"""
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class CausalSelfAttention(nn.Module):
    """因果自注意力"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.W_qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, L, dk]
        Q, K, V = qkv[0], qkv[1], qkv[2]

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is None:
            mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            mask = mask.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(~mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.W_o(out)


class FFN(nn.Module):
    """SwiGLU FFN"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.down(F.silu(self.gate(x)) * self.up(x)))


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer Block"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = FFN(d_model, d_ff, dropout)

    def forward(self, x, mask=None):
        x = x + self.attn(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================
# 1. PolicyModel — 策略/参考模型
# ============================================================
class PolicyModel(nn.Module):
    """
    Decoder-only 语言模型，同时用作策略模型和参考模型。
    
    用途：
    - 策略模型 (π_θ): 可训练，通过 DPO 优化
    - 参考模型 (π_ref): 冻结的初始 SFT 模型副本
    """

    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # 权重绑定
        self.lm_head.weight = self.embedding.weight
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, mask=None):
        """
        前向传播，返回 logits。
        
        Args:
            input_ids: [B, L] token ids
        Returns:
            logits: [B, L, V]
        """
        x = self.embedding(input_ids) * math.sqrt(self.config.d_model)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits

    def get_logps(self, input_ids: torch.Tensor, labels: torch.Tensor,
                  average: bool = False):
        """
        计算序列的对数概率。
        
        Args:
            input_ids: [B, L] 完整序列 (prompt + response)
            labels: [B, L] 目标序列 (prompt 部分为 -100)
            average: 是否取平均 (SimPO 需要)
        """
        logits = self.forward(input_ids)
        # 移位：用 t 时刻的 logits 预测 t+1 时刻的 token
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        per_token_logps = torch.gather(
            shift_logits.log_softmax(dim=-1),
            dim=2,
            index=shift_labels.clamp(min=0).unsqueeze(2)
        ).squeeze(2)

        loss_mask = (shift_labels != -100).float()
        logps = (per_token_logps * loss_mask).sum(dim=-1)

        if average:
            logps = logps / loss_mask.sum(dim=-1).clamp(min=1)

        return logps


# ============================================================
# 2. RewardModel — 奖励模型
# ============================================================
class RewardModel(nn.Module):
    """
    奖励模型 = LLM 基座 + 标量奖励头。
    
    输入 (prompt, response)，输出一个标量奖励值 r(x, y)。
    用最后一个非 padding token 的隐藏状态做奖励预测。
    """

    def __init__(self, config: RLHFConfig):
        super().__init__()
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.reward_n_layers)
        ])
        self.norm = RMSNorm(config.d_model)

        # 奖励头：hidden → scalar
        self.reward_head = nn.Sequential(
            nn.Linear(config.d_model, config.reward_head_dim),
            nn.GELU(),
            nn.Linear(config.reward_head_dim, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):
        """
        Args:
            input_ids: [B, L]
            attention_mask: [B, L] (1=real token, 0=padding)
        Returns:
            rewards: [B] 标量奖励
        """
        x = self.embedding(input_ids) * math.sqrt(input_ids.size(-1))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)  # [B, L, D]

        # 取最后一个真实 token 的表示
        if attention_mask is not None:
            # 找到每个序列最后一个非 padding 位置
            last_idx = attention_mask.sum(dim=-1) - 1  # [B]
            last_hidden = x[torch.arange(x.size(0)), last_idx]  # [B, D]
        else:
            last_hidden = x[:, -1, :]  # [B, D]

        rewards = self.reward_head(last_hidden).squeeze(-1)  # [B]
        return rewards


# ============================================================
# 3. DPOTrainer — DPO 训练封装
# ============================================================
class DPOTrainer:
    """
    DPO 训练器，封装完整的训练逻辑。
    
    核心流程：
    1. 前向传播策略模型 → 得到 chosen/rejected 的 logps
    2. 前向传播参考模型 → 得到 chosen/rejected 的 logps
    3. 计算 DPO loss
    4. 反向传播 + 更新
    """

    def __init__(self, policy_model: PolicyModel, config: RLHFConfig, device: torch.device):
        self.policy = policy_model.to(device)
        # 参考模型：深拷贝 + 冻结
        self.ref_model = copy.deepcopy(policy_model).to(device)
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.config = config
        self.device = device

    @torch.no_grad()
    def get_ref_logps(self, input_ids, labels, average=False):
        """用冻结的参考模型计算对数概率"""
        return self.ref_model.get_logps(input_ids, labels, average=average)

    def compute_dpo_loss(self, batch):
        """计算一个 batch 的 DPO 损失"""
        from losses import dpo_loss

        chosen_ids = batch["chosen_ids"].to(self.device)
        rejected_ids = batch["rejected_ids"].to(self.device)
        chosen_labels = batch["chosen_labels"].to(self.device)
        rejected_labels = batch["rejected_labels"].to(self.device)

        # 策略模型的对数概率
        policy_chosen_logps = self.policy.get_logps(chosen_ids, chosen_labels)
        policy_rejected_logps = self.policy.get_logps(rejected_ids, rejected_labels)

        # 参考模型的对数概率
        ref_chosen_logps = self.get_ref_logps(chosen_ids, chosen_labels)
        ref_rejected_logps = self.get_ref_logps(rejected_ids, rejected_labels)

        # DPO loss
        loss, metrics = dpo_loss(
            policy_chosen_logps, policy_rejected_logps,
            ref_chosen_logps, ref_rejected_logps,
            beta=self.config.dpo_beta,
            label_smoothing=self.config.label_smoothing,
        )
        return loss, metrics
