"""
v05 Qwen-VL 核心模块复现

1. PerceiverResampler: 视觉 token 压缩
2. MultimodalQwen: 简化版 Qwen-VL 架构
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import QwenVLConfig


class PerceiverResampler(nn.Module):
    """
    Perceiver Resampler（Visual Resampler）。
    
    用 num_queries 个可学习 Query 通过 Cross-Attention
    从 N 个视觉 token 中提取并压缩信息。
    
    输入: vision_features [B, N, vision_dim] (如 [B, 196, 256])
    输出: resampled [B, num_queries, llm_dim] (如 [B, 64, 512])
    """
    def __init__(self, cfg: QwenVLConfig):
        super().__init__()
        # 可学习的查询向量
        self.queries = nn.Parameter(torch.randn(1, cfg.num_queries, cfg.llm_dim) * 0.02)
        # 视觉特征投影到 LLM 维度
        self.vision_proj = nn.Linear(cfg.vision_dim, cfg.llm_dim)
        
        # Cross-Attention layers
        self.layers = nn.ModuleList()
        for _ in range(cfg.resampler_layers):
            self.layers.append(nn.ModuleDict({
                "norm_q": nn.LayerNorm(cfg.llm_dim),
                "norm_kv": nn.LayerNorm(cfg.llm_dim),
                "cross_attn": nn.MultiheadAttention(
                    cfg.llm_dim, cfg.resampler_heads, dropout=cfg.dropout, batch_first=True
                ),
                "norm_ff": nn.LayerNorm(cfg.llm_dim),
                "ffn": nn.Sequential(
                    nn.Linear(cfg.llm_dim, cfg.resampler_ff_dim),
                    nn.GELU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.resampler_ff_dim, cfg.llm_dim),
                    nn.Dropout(cfg.dropout),
                ),
            }))
        self.final_norm = nn.LayerNorm(cfg.llm_dim)
    
    def forward(self, vision_features: torch.Tensor) -> torch.Tensor:
        B = vision_features.size(0)
        # 投影视觉特征到 LLM 维度
        kv = self.vision_proj(vision_features)  # [B, N, llm_dim]
        # 扩展 queries 到 batch
        q = self.queries.expand(B, -1, -1)      # [B, num_queries, llm_dim]
        
        for layer in self.layers:
            # Cross-Attention: Q 是可学习查询, K/V 是视觉特征
            q_norm = layer["norm_q"](q)
            kv_norm = layer["norm_kv"](kv)
            attn_out, _ = layer["cross_attn"](q_norm, kv_norm, kv_norm)
            q = q + attn_out
            # FFN
            q = q + layer["ffn"](layer["norm_ff"](q))
        
        return self.final_norm(q)  # [B, num_queries, llm_dim]


class VisionEncoder(nn.Module):
    """简化版 ViT 视觉编码器。"""
    def __init__(self, cfg: QwenVLConfig):
        super().__init__()
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, cfg.vision_dim, cfg.patch_size, cfg.patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, cfg.vision_dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.vision_dim, 4, cfg.vision_dim * 4, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.norm = nn.LayerNorm(cfg.vision_dim)
    
    def forward(self, images):
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = x + self.pos_embed[:, :x.size(1)]
        x = self.norm(self.encoder(x))
        return x  # [B, num_patches, vision_dim]


class MultimodalQwen(nn.Module):
    """
    简化版 Qwen-VL: Vision Encoder + Resampler + LLM Decoder。
    
    展示完整的多模态输入处理流程。
    """
    def __init__(self, cfg: QwenVLConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionEncoder(cfg)
        self.resampler = PerceiverResampler(cfg)
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.llm_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.llm_dim) * 0.02)
        
        decoder_layer = nn.TransformerEncoderLayer(
            cfg.llm_dim, cfg.n_heads, cfg.d_ff, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.llm_dim)
        self.lm_head = nn.Linear(cfg.llm_dim, cfg.vocab_size, bias=False)
    
    def forward(self, images, input_ids):
        """
        images: [B, 3, H, W]
        input_ids: [B, L] 文本 token ids
        """
        B = images.size(0)
        # 1. 视觉编码
        vis_features = self.vision_encoder(images)  # [B, N, vision_dim]
        # 2. Resampler 压缩
        vis_tokens = self.resampler(vis_features)    # [B, num_queries, llm_dim]
        # 3. 文本 embedding
        txt_tokens = self.tok_embed(input_ids)       # [B, L, llm_dim]
        # 4. 拼接: [视觉 tokens, 文本 tokens]
        combined = torch.cat([vis_tokens, txt_tokens], dim=1)  # [B, Q+L, llm_dim]
        seq_len = combined.size(1)
        combined = combined + self.pos_embed[:, :seq_len]
        # 5. 因果掩码
        causal_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=combined.device)
        # 6. LLM Decoder
        hidden = self.decoder(combined, mask=causal_mask, is_causal=True)
        hidden = self.norm(hidden)
        # 7. 只取文本部分的输出做 LM 预测
        text_hidden = hidden[:, vis_tokens.size(1):, :]
        logits = self.lm_head(text_hidden)
        return logits  # [B, L, vocab_size]


if __name__ == "__main__":
    cfg = QwenVLConfig()
    model = MultimodalQwen(cfg)
    imgs = torch.randn(2, 3, 224, 224)
    ids = torch.randint(0, 5000, (2, 32))
    out = model(imgs, ids)
    print(f"Input: imgs={imgs.shape}, ids={ids.shape}")
    print(f"Output logits: {out.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Vision: {sum(p.numel() for p in model.vision_encoder.parameters()):,}")
    print(f"  Resampler: {sum(p.numel() for p in model.resampler.parameters()):,}")
    print(f"  LLM: {sum(p.numel() for p in model.decoder.parameters()):,}")
