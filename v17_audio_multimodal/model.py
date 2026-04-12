"""
V17 - CLAP & 全模态融合模型
============================
核心模块：
1. MiniTextEncoder：轻量文本编码器
2. CLAPModel：音频-文本对比学习
3. OmniModalQFormer：全模态 Q-Former 融合
4. OmniModalModel：端到端全模态模型

参考：
- CLAP (ICASSP 2023)
- ImageBind (CVPR 2023)
"""
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from config import (AudioMultimodalFullConfig, CLAPConfig,
                    OmniModalConfig, AudioEncoderConfig)
from audio_modules import AudioSpectrogramTransformer


# ============================================================
#  轻量文本编码器
# ============================================================

class MiniTextEncoder(nn.Module):
    """简易 Transformer 文本编码器"""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int = 4,
                 n_layers: int = 2, max_len: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.embed(token_ids) + self.pos_embed(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0]  # [CLS] 位置


# ============================================================
#  简易 Vision Encoder
# ============================================================

class MiniVisionEncoder(nn.Module):
    """ViT 风格视觉编码器"""

    def __init__(self, image_size: int = 224, patch_size: int = 16,
                 in_channels: int = 3, d_model: int = 256,
                 n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, d_model, patch_size, patch_size)
        n_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + n_patches, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4, batch_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B = images.shape[0]
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, :x.shape[1]]
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x  # [B, 1+N, D]


# ============================================================
#  CLAP 模型
# ============================================================

class CLAPModel(nn.Module):
    """
    CLAP: Contrastive Language-Audio Pretraining
    
    音频-文本双塔对比学习模型
    """

    def __init__(self, config: CLAPConfig, audio_config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        # 音频塔
        self.audio_encoder = AudioSpectrogramTransformer(audio_config)
        self.audio_proj = nn.Sequential(
            nn.Linear(audio_config.d_model, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
        )

        # 文本塔
        self.text_encoder = MiniTextEncoder(
            config.vocab_size, config.text_d_model, max_len=config.max_text_len
        )
        self.text_proj = nn.Sequential(
            nn.Linear(config.text_d_model, config.projection_dim),
            nn.LayerNorm(config.projection_dim),
        )

        # 可学习温度参数
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(1.0 / config.temperature)))

    def forward(self, mel_spec: torch.Tensor,
                token_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            mel_spec: [B, n_mels, T'] Mel 频谱
            token_ids: [B, L] 文本 token
        Returns:
            dict with loss, audio_embed, text_embed, similarity
        """
        # 编码
        audio_feat = self.audio_encoder(mel_spec)  # [B, D_a]
        text_feat = self.text_encoder(token_ids)    # [B, D_t]

        # 投影 + 归一化
        audio_embed = F.normalize(self.audio_proj(audio_feat), dim=-1)
        text_embed = F.normalize(self.text_proj(text_feat), dim=-1)

        # 温度缩放的相似度矩阵
        temperature = self.log_temperature.exp().clamp(max=100)
        logits = torch.matmul(audio_embed, text_embed.T) * temperature

        # 对称 InfoNCE Loss
        B = logits.shape[0]
        labels = torch.arange(B, device=logits.device)
        loss_a2t = F.cross_entropy(logits, labels)
        loss_t2a = F.cross_entropy(logits.T, labels)
        loss = (loss_a2t + loss_t2a) / 2

        return {
            'loss': loss,
            'audio_embed': audio_embed,
            'text_embed': text_embed,
            'similarity': logits.detach(),
            'temperature': temperature.detach(),
        }


# ============================================================
#  全模态 Q-Former 融合
# ============================================================

class CrossAttentionLayer(nn.Module):
    """Cross-Attention + FFN"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # Self-Attention among queries
        q = self.norm1(queries)
        q, _ = self.self_attn(q, q, q)
        queries = queries + q

        # Cross-Attention: Q=queries, KV=multimodal
        q = self.norm2(queries)
        q, _ = self.cross_attn(q, kv, kv)
        queries = queries + q

        # FFN
        queries = queries + self.ffn(self.norm3(queries))
        return queries


class OmniModalQFormer(nn.Module):
    """
    全模态 Q-Former
    
    使用可学习 Query Token 聚合图像+文本+音频信息
    支持模态缺失（训练时随机丢弃）
    """

    def __init__(self, config: OmniModalConfig):
        super().__init__()
        self.config = config

        # 可学习 Query
        self.query_tokens = nn.Parameter(
            torch.zeros(1, config.num_query_tokens, config.d_model)
        )
        nn.init.trunc_normal_(self.query_tokens, std=0.02)

        # 模态类型 Embedding
        self.image_type_embed = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.text_type_embed = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.audio_type_embed = nn.Parameter(torch.zeros(1, 1, config.d_model))

        # 模态投影（对齐到统一维度）
        self.image_proj = nn.Linear(config.image_dim, config.d_model)
        self.text_proj = nn.Linear(config.text_dim, config.d_model)
        self.audio_proj = nn.Linear(config.audio_dim, config.d_model)

        # Cross-Attention 层
        self.cross_layers = nn.ModuleList([
            CrossAttentionLayer(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_fusion_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, image_feats: Optional[torch.Tensor] = None,
                text_feats: Optional[torch.Tensor] = None,
                audio_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            image_feats: [B, N_i, D_i] 图像特征
            text_feats: [B, N_t, D_t] 文本特征
            audio_feats: [B, N_a, D_a] 音频特征
        Returns:
            output: [B, D] 融合后的全模态表征
        """
        B = None
        modality_tokens = []

        if image_feats is not None:
            B = image_feats.shape[0]
            img = self.image_proj(image_feats) + self.image_type_embed
            modality_tokens.append(img)

        if text_feats is not None:
            B = text_feats.shape[0]
            txt = self.text_proj(text_feats) + self.text_type_embed
            modality_tokens.append(txt)

        if audio_feats is not None:
            B = audio_feats.shape[0]
            aud = self.audio_proj(audio_feats) + self.audio_type_embed
            modality_tokens.append(aud)

        assert B is not None, "At least one modality must be provided"

        # 拼接所有模态 token
        kv = torch.cat(modality_tokens, dim=1)  # [B, N_total, D]

        # 可学习 Query
        queries = self.query_tokens.expand(B, -1, -1)  # [B, M, D]

        # Cross-Attention 融合
        for layer in self.cross_layers:
            queries = layer(queries, kv)

        queries = self.norm(queries)

        # 池化输出
        output = queries.mean(dim=1)  # [B, D]
        return output


# ============================================================
#  全模态端到端模型
# ============================================================

class ModalityDropout(nn.Module):
    """训练时随机丢弃模态，增强鲁棒性"""

    def __init__(self, p_drop: float = 0.1):
        super().__init__()
        self.p_drop = p_drop

    def forward(self, image_feats, text_feats, audio_feats):
        if self.training and self.p_drop > 0:
            keep_img = random.random() > self.p_drop
            keep_txt = random.random() > self.p_drop
            keep_aud = random.random() > self.p_drop

            # 至少保留一个
            if not (keep_img or keep_txt or keep_aud):
                keep_txt = True

            if not keep_img:
                image_feats = None
            if not keep_txt:
                text_feats = None
            if not keep_aud:
                audio_feats = None

        return image_feats, text_feats, audio_feats


class OmniModalModel(nn.Module):
    """
    全模态模型
    
    支持图像+文本+音频三种模态的联合编码和分类
    训练时随机丢弃模态增强鲁棒性
    """

    def __init__(self, config: OmniModalConfig, audio_config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        # 单模态编码器
        self.vision_encoder = MiniVisionEncoder(
            config.image_size, config.patch_size, config.in_channels, config.image_dim
        )
        self.text_encoder = MiniTextEncoder(
            config.vocab_size, config.text_dim, max_len=config.max_text_len
        )
        self.audio_encoder = AudioSpectrogramTransformer(audio_config)

        # 全模态融合
        self.qformer = OmniModalQFormer(config)
        self.modality_dropout = ModalityDropout(p_drop=0.1)

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config.d_model, config.num_classes),
        )

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.vision_encoder(images)  # [B, 1+N, D]

    def encode_text(self, token_ids: torch.Tensor) -> torch.Tensor:
        # 返回序列特征
        B, L = token_ids.shape
        pos = torch.arange(L, device=token_ids.device).unsqueeze(0)
        x = self.text_encoder.embed(token_ids) + self.text_encoder.pos_embed(pos)
        for block in self.text_encoder.blocks:
            x = block(x)
        return self.text_encoder.norm(x)  # [B, L, D]

    def encode_audio(self, mel_spec: torch.Tensor) -> torch.Tensor:
        return self.audio_encoder.forward_features(mel_spec)  # [B, 1+N, D]

    def forward(self, images: Optional[torch.Tensor] = None,
                token_ids: Optional[torch.Tensor] = None,
                mel_spec: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # 编码各模态
        img_feats = self.encode_image(images) if images is not None else None
        txt_feats = self.encode_text(token_ids) if token_ids is not None else None
        aud_feats = self.encode_audio(mel_spec) if mel_spec is not None else None

        # 模态 Dropout
        img_feats, txt_feats, aud_feats = self.modality_dropout(img_feats, txt_feats, aud_feats)

        # Q-Former 融合
        fused = self.qformer(img_feats, txt_feats, aud_feats)  # [B, D]

        # 分类
        logits = self.classifier(fused)  # [B, C]

        return {
            'logits': logits,
            'fused_repr': fused,
        }
