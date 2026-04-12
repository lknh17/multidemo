"""
V25 - 端到端广告多模态管线模型
================================
E2EAdPipeline / CTRPredictor / MultiObjectiveRanker / OnlineLearner
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List

from config import FullConfig
from pipeline_modules import AdEncoder, AdMatcher, SafetyFilter, QualityGate


# ── E2EAdPipeline：全管线 ─────────────────────────────────────

class E2EAdPipeline(nn.Module):
    """
    端到端广告管线：encode → retrieve → rerank → filter → serve
    综合 V01-V24 全部知识
    """

    def __init__(self, config: FullConfig):
        super().__init__()
        self.config = config
        D = config.creative.d_model

        # 编码器
        self.encoder = AdEncoder(config.creative)

        # 匹配器
        self.matcher = AdMatcher(config.matching, D)

        # 安全过滤
        self.safety_filter = SafetyFilter(D) if config.pipeline.enable_safety else None

        # 质量门控
        self.quality_gate = QualityGate(D) if config.pipeline.enable_quality else None

        # 对比学习头（训练用）
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def encode(self, images, text_ids, audio_feats=None):
        return self.encoder(images, text_ids, audio_feats)

    def compute_contrastive_loss(self, query_emb: torch.Tensor,
                                  pos_emb: torch.Tensor) -> torch.Tensor:
        """InfoNCE 对比损失（V08）"""
        B = query_emb.shape[0]
        logits = torch.matmul(query_emb, pos_emb.T) / self.temperature.exp()
        labels = torch.arange(B, device=query_emb.device)
        loss_q2p = F.cross_entropy(logits, labels)
        loss_p2q = F.cross_entropy(logits.T, labels)
        return (loss_q2p + loss_p2q) / 2

    def forward(self, query_images, query_texts, ad_images, ad_texts,
                query_audio=None, ad_audio=None):
        # Stage 1: Encode
        query_emb = self.encode(query_images, query_texts, query_audio)
        ad_emb = self.encode(ad_images, ad_texts, ad_audio)

        # Stage 2-3: Retrieve + Rerank
        top_idx, recall_scores, rerank_scores = self.matcher(query_emb, ad_emb)

        # Stage 4: Safety Filter
        safety_mask, safety_scores = (
            self.safety_filter(ad_emb) if self.safety_filter else
            (torch.ones(ad_emb.shape[0], dtype=torch.bool, device=ad_emb.device), None))

        # Stage 5: Quality Gate
        quality_mask, quality_scores = (
            self.quality_gate(ad_emb) if self.quality_gate else
            (torch.ones(ad_emb.shape[0], dtype=torch.bool, device=ad_emb.device), None))

        # 对比损失
        contrastive_loss = self.compute_contrastive_loss(query_emb, ad_emb)

        return {
            'query_emb': query_emb,
            'ad_emb': ad_emb,
            'top_idx': top_idx,
            'recall_scores': recall_scores,
            'rerank_scores': rerank_scores,
            'safety_mask': safety_mask,
            'safety_scores': safety_scores,
            'quality_mask': quality_mask,
            'quality_scores': quality_scores,
            'contrastive_loss': contrastive_loss,
        }


# ── CTRPredictor：DeepFM 风格 CTR 预估 ───────────────────────

class FMLayer(nn.Module):
    """Factorization Machine 二阶交叉层"""

    def __init__(self, in_dim: int, n_factors: int = 16):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.V = nn.Parameter(torch.randn(in_dim, n_factors) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 一阶
        linear_part = self.linear(x)
        # 二阶交叉（利用恒等式高效计算）
        square_of_sum = torch.matmul(x, self.V).pow(2)
        sum_of_square = torch.matmul(x.pow(2), self.V.pow(2))
        fm_part = 0.5 * (square_of_sum - sum_of_square).sum(dim=-1, keepdim=True)
        return linear_part + fm_part


class CTRPredictor(nn.Module):
    """
    DeepFM 风格 CTR 预估模型
    综合 V04(特征交叉) + V11(embedding)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        in_dim = d_model * 3  # user + ad + context
        self.fm = FMLayer(in_dim)
        self.dnn = nn.Sequential(
            nn.Linear(in_dim, d_model * 2), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(d_model * 2, d_model), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model, 1))

    def forward(self, user_emb: torch.Tensor, ad_emb: torch.Tensor,
                context_emb: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([user_emb, ad_emb, context_emb], dim=-1)
        fm_out = self.fm(combined)
        dnn_out = self.dnn(combined)
        ctr = torch.sigmoid(fm_out + dnn_out).squeeze(-1)
        return ctr


# ── MultiObjectiveRanker：多目标排序 ─────────────────────────

class MultiObjectiveRanker(nn.Module):
    """
    平衡 CTR / 相关性 / 多样性 / 新鲜度
    综合 V14(RLHF偏好) + V22(ABTest)
    """

    def __init__(self, d_model: int = 256, config: Optional[FullConfig] = None):
        super().__init__()
        in_dim = d_model * 3  # user + ad + context

        self.ctr_head = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid())
        self.relevance_head = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid())
        self.diversity_head = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid())
        self.freshness_head = nn.Sequential(
            nn.Linear(in_dim, d_model), nn.ReLU(), nn.Linear(d_model, 1), nn.Sigmoid())

        # 可学习权重
        if config:
            self.weights = nn.Parameter(torch.tensor([
                config.ctr_weight, config.relevance_weight,
                config.diversity_weight, config.freshness_weight]))
        else:
            self.weights = nn.Parameter(torch.tensor([0.4, 0.3, 0.2, 0.1]))

    def forward(self, user_emb: torch.Tensor, ad_emb: torch.Tensor,
                context_emb: torch.Tensor) -> Dict[str, torch.Tensor]:
        features = torch.cat([user_emb, ad_emb, context_emb], dim=-1)

        ctr = self.ctr_head(features).squeeze(-1)
        rel = self.relevance_head(features).squeeze(-1)
        div = self.diversity_head(features).squeeze(-1)
        fresh = self.freshness_head(features).squeeze(-1)

        w = F.softmax(self.weights, dim=0)
        scores = torch.stack([ctr, rel, div, fresh], dim=-1)
        final_score = (scores * w).sum(dim=-1)

        return {
            'ctr_score': ctr,
            'relevance_score': rel,
            'diversity_score': div,
            'freshness_score': fresh,
            'final_score': final_score,
            'weights': w,
        }

    def compute_listwise_loss(self, pred_scores: torch.Tensor,
                               true_relevance: torch.Tensor,
                               tau: float = 1.0) -> torch.Tensor:
        """Listwise 排序损失"""
        pred_dist = F.softmax(pred_scores / tau, dim=-1)
        true_dist = F.softmax(true_relevance / tau, dim=-1)
        loss = -(true_dist * torch.log(pred_dist + 1e-8)).sum(dim=-1).mean()
        return loss


# ── OnlineLearner：在线增量学习 ──────────────────────────────

class OnlineLearner:
    """
    在线学习器：增量更新 + EMA 平滑
    综合 V06(训练) + V12(优化)
    """

    def __init__(self, model: nn.Module, lr: float = 1e-4, ema_beta: float = 0.999):
        self.model = model
        self.ema_model = {k: v.clone() for k, v in model.state_dict().items()}
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.ema_beta = ema_beta
        self.step_count = 0
        self.cumulative_loss = 0.0

    def update(self, user_emb: torch.Tensor, ad_emb: torch.Tensor,
               context_emb: torch.Tensor, click_label: torch.Tensor) -> float:
        """单步在线更新"""
        self.model.train()
        pred = self.model(user_emb, ad_emb, context_emb)
        loss = F.binary_cross_entropy(pred, click_label.float())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        # EMA 更新
        self._update_ema()

        self.step_count += 1
        self.cumulative_loss += loss.item()
        return loss.item()

    def _update_ema(self):
        beta = self.ema_beta
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.ema_model:
                    self.ema_model[name] = beta * self.ema_model[name] + (1 - beta) * param.data

    def get_avg_loss(self) -> float:
        return self.cumulative_loss / max(self.step_count, 1)
