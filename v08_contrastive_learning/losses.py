"""
v08 对比学习 Loss 集合

实现多种对比学习损失函数，带详细注释。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss (CLIP 对比学习损失)。
    对称版本: Image→Text + Text→Image。
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, emb_a, emb_b, logit_scale=None):
        """emb_a, emb_b: [N, D], L2 归一化后的 embedding"""
        if logit_scale is None:
            logit_scale = 1.0 / self.temperature
        logits = logit_scale * emb_a @ emb_b.t()  # [N, N]
        N = logits.size(0)
        labels = torch.arange(N, device=logits.device)
        loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2
        return loss


class TripletLoss(nn.Module):
    """
    Triplet Loss: max(0, d(a,p) - d(a,n) + margin)
    
    需要 (anchor, positive, negative) 三元组。
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(self, anchor, positive, negative):
        d_pos = (anchor - positive).pow(2).sum(dim=-1)  # 正样本距离
        d_neg = (anchor - negative).pow(2).sum(dim=-1)  # 负样本距离
        loss = F.relu(d_pos - d_neg + self.margin)
        return loss.mean()


class CircleLoss(nn.Module):
    """
    Circle Loss: 自适应加权正负样本对。
    
    正样本: 越远的正样本获得越大权重（拉近它）
    负样本: 越近的负样本获得越大权重（推远它）
    """
    def __init__(self, margin: float = 0.25, gamma: float = 80):
        super().__init__()
        self.margin = margin
        self.gamma = gamma
        self.O_p = 1 + margin   # 正样本最优相似度
        self.O_n = -margin       # 负样本最优相似度
    
    def forward(self, emb_a, emb_b):
        sim = emb_a @ emb_b.t()  # [N, N]
        N = sim.size(0)
        mask_pos = torch.eye(N, device=sim.device).bool()
        mask_neg = ~mask_pos
        
        s_p = sim[mask_pos]  # 正样本相似度
        s_n = sim[mask_neg].view(N, N-1)  # 负样本相似度
        
        # 自适应权重: 距离最优值越远，权重越大
        alpha_p = F.relu(self.O_p - s_p.detach())
        alpha_n = F.relu(s_n.detach() - self.O_n)
        
        logit_p = -self.gamma * alpha_p * (s_p - self.margin)
        logit_n = self.gamma * alpha_n * (s_n + self.margin)
        
        loss = F.softplus(torch.logsumexp(logit_n, dim=1) + torch.logsumexp(logit_p.unsqueeze(1), dim=1).squeeze())
        return loss.mean()


class HardNegativeMiner:
    """
    Hard Negative Mining 策略。
    """
    @staticmethod
    def random_negatives(embeddings, labels, n_neg=5):
        """随机负样本采样"""
        B = embeddings.size(0)
        neg_indices = []
        for i in range(B):
            candidates = [j for j in range(B) if labels[j] != labels[i]]
            if len(candidates) >= n_neg:
                neg_indices.append(torch.tensor(candidates[:n_neg]))
            else:
                neg_indices.append(torch.tensor(candidates + candidates[:n_neg - len(candidates)]))
        return torch.stack(neg_indices)
    
    @staticmethod
    def hardest_negatives(anchor_emb, all_emb, labels, n_neg=5):
        """最困难负样本: 选与 anchor 最相似的负样本"""
        sims = anchor_emb @ all_emb.t()  # [B, B]
        B = sims.size(0)
        # 将正样本的相似度设为极小值
        for i in range(B):
            for j in range(B):
                if labels[i] == labels[j]:
                    sims[i, j] = -float('inf')
        # 取最相似的负样本
        _, indices = sims.topk(n_neg, dim=1)
        return indices
