"""
V25 - 广告管线核心模块
======================
AdEncoder / AdMatcher / SafetyFilter / QualityGate / FeatureExtractor
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from config import AdCreativeConfig, PipelineConfig, MatchingConfig


# ── 简易 ViT backbone（复用 V01-V05 思路） ──────────────────────

class PatchEmbedding(nn.Module):
    def __init__(self, config: AdCreativeConfig):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.d_model,
                              kernel_size=config.patch_size, stride=config.patch_size)
        n_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        patches = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        return torch.cat([cls, patches], dim=1) + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Linear(d_model * 4, d_model))
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h)[0]
        x = x + self.ff(self.norm2(x))
        return x


# ── AdEncoder：多模态统一编码器 ────────────────────────────────

class AdEncoder(nn.Module):
    """
    视觉 + 文本 + 音频 → 统一 embedding
    综合 V01(ViT) + V03(文本Transformer) + V04(融合) + V17(音频)
    """

    def __init__(self, config: AdCreativeConfig):
        super().__init__()
        self.config = config
        D = config.d_model

        # 视觉编码器（V01-V05）
        self.patch_embed = PatchEmbedding(config)
        self.vis_blocks = nn.ModuleList(
            [TransformerBlock(D, config.n_heads) for _ in range(config.n_layers)])

        # 文本编码器（V03）
        self.token_embed = nn.Embedding(config.vocab_size, D)
        self.text_pos = nn.Parameter(torch.zeros(1, config.text_max_len, D))
        nn.init.trunc_normal_(self.text_pos, std=0.02)
        self.txt_blocks = nn.ModuleList(
            [TransformerBlock(D, config.n_heads) for _ in range(config.n_layers // 2)])

        # 音频编码器（V17）
        self.has_audio = config.has_audio
        if self.has_audio:
            self.aud_proj = nn.Sequential(
                nn.Linear(config.audio_dim, D), nn.GELU(), nn.Linear(D, D))

        # 注意力融合（V04, V09）
        self.fusion_query = nn.Parameter(torch.zeros(1, 1, D))
        nn.init.trunc_normal_(self.fusion_query, std=0.02)
        self.fusion_attn = nn.MultiheadAttention(D, config.n_heads, batch_first=True)
        self.fusion_norm = nn.LayerNorm(D)
        self.out_proj = nn.Linear(D, D)

    def encode_visual(self, images: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(images)
        for blk in self.vis_blocks:
            x = blk(x)
        return x[:, 0]  # CLS token

    def encode_text(self, text_ids: torch.Tensor) -> torch.Tensor:
        B, L = text_ids.shape
        x = self.token_embed(text_ids) + self.text_pos[:, :L]
        for blk in self.txt_blocks:
            x = blk(x)
        return x.mean(dim=1)  # mean pooling

    def encode_audio(self, audio_feats: torch.Tensor) -> torch.Tensor:
        return self.aud_proj(audio_feats)

    def forward(self, images: torch.Tensor, text_ids: torch.Tensor,
                audio_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = images.shape[0]
        vis = self.encode_visual(images).unsqueeze(1)   # [B,1,D]
        txt = self.encode_text(text_ids).unsqueeze(1)   # [B,1,D]

        modalities = [vis, txt]
        if self.has_audio and audio_feats is not None:
            aud = self.encode_audio(audio_feats).unsqueeze(1)
            modalities.append(aud)

        kv = torch.cat(modalities, dim=1)  # [B, M, D]
        query = self.fusion_query.expand(B, -1, -1)
        fused, _ = self.fusion_attn(query, kv, kv)
        fused = self.fusion_norm(fused + query)
        out = self.out_proj(fused.squeeze(1))
        return F.normalize(out, dim=-1)


# ── AdMatcher：检索 + 重排 ─────────────────────────────────────

class AdMatcher(nn.Module):
    """
    ANN 召回 + 交叉注意力精排
    综合 V08(对比学习) + V11(embedding检索) + V13(注意力)
    """

    def __init__(self, config: MatchingConfig, d_model: int = 256):
        super().__init__()
        self.config = config
        # 交叉编码器用于精排
        self.cross_proj_q = nn.Linear(d_model, d_model)
        self.cross_proj_d = nn.Linear(d_model, d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.score_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1))

    def retrieve(self, query_emb: torch.Tensor, index_embs: torch.Tensor,
                 top_k: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """ANN 召回：余弦相似度 Top-K"""
        top_k = top_k or self.config.retrieval_top_k
        scores = torch.matmul(query_emb, index_embs.T)  # [B, N_index]
        k = min(top_k, scores.shape[-1])
        top_scores, top_idx = scores.topk(k, dim=-1)
        return top_idx, top_scores

    def rerank(self, query_emb: torch.Tensor,
               candidate_embs: torch.Tensor) -> torch.Tensor:
        """交叉注意力精排"""
        q = self.cross_proj_q(query_emb).unsqueeze(1)   # [B,1,D]
        d = self.cross_proj_d(candidate_embs)            # [B,K,D]
        cross_out, _ = self.cross_attn(q, d, d)          # [B,1,D]
        scores = self.score_head(cross_out.squeeze(1))   # [B,1]
        return scores.squeeze(-1)

    def forward(self, query_emb: torch.Tensor, index_embs: torch.Tensor):
        top_idx, recall_scores = self.retrieve(query_emb, index_embs)
        B = query_emb.shape[0]
        # 收集候选
        candidates = []
        for b in range(B):
            candidates.append(index_embs[top_idx[b]])
        candidate_embs = torch.stack(candidates)  # [B, K, D]
        rerank_scores = self.rerank(query_emb, candidate_embs)
        return top_idx, recall_scores, rerank_scores


# ── SafetyFilter：级联安全过滤（V24） ─────────────────────────

class SafetyFilter(nn.Module):
    """
    多级安全检查：快速关键词 → 语义分类 → 综合判定
    """

    def __init__(self, d_model: int = 256, num_safety_classes: int = 5):
        super().__init__()
        # 安全分类器
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(d_model // 2, num_safety_classes), nn.Sigmoid())
        self.threshold = 0.5

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (安全掩码, 各维度分数)"""
        scores = self.classifier(embeddings)       # [B, C]
        safe_mask = (scores < self.threshold).all(dim=-1)  # 所有维度低于阈值 = 安全
        return safe_mask, scores


# ── QualityGate：质量门控（V18） ──────────────────────────────

class QualityGate(nn.Module):
    """
    广告创意质量评估：清晰度/相关性/美观度/信息量/可读性
    """

    def __init__(self, d_model: int = 256, num_dims: int = 5):
        super().__init__()
        self.dim_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, num_dims), nn.Sigmoid())
        self.overall_head = nn.Sequential(
            nn.Linear(d_model + num_dims, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1), nn.Sigmoid())
        self.threshold = 0.4

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dim_scores = self.dim_head(embeddings)
        overall_input = torch.cat([embeddings, dim_scores], dim=-1)
        overall = self.overall_head(overall_input).squeeze(-1)
        pass_mask = overall > self.threshold
        return pass_mask, overall


# ── FeatureExtractor：批量特征提取 ────────────────────────────

class FeatureExtractor:
    """批量高效特征提取（推理用）"""

    def __init__(self, encoder: AdEncoder, device: torch.device):
        self.encoder = encoder.to(device).eval()
        self.device = device

    @torch.no_grad()
    def extract_batch(self, images: torch.Tensor, text_ids: torch.Tensor,
                      audio_feats: Optional[torch.Tensor] = None,
                      batch_size: int = 32) -> torch.Tensor:
        all_embs = []
        N = images.shape[0]
        for i in range(0, N, batch_size):
            end = min(i + batch_size, N)
            img_b = images[i:end].to(self.device)
            txt_b = text_ids[i:end].to(self.device)
            aud_b = audio_feats[i:end].to(self.device) if audio_feats is not None else None
            emb = self.encoder(img_b, txt_b, aud_b)
            all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)
