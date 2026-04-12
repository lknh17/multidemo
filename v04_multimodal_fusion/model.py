"""
v04 Mini CLIP 实现 + Cross-Attention 融合模块

核心组件：
1. ImageEncoder (ViT) → 图像 embedding
2. TextEncoder (Transformer) → 文本 embedding
3. 投影层 → 共享空间
4. InfoNCE Loss
5. CrossAttentionFusion 模块
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CLIPConfig


class ImageEncoder(nn.Module):
    """图像编码器（简化 ViT）。"""
    def __init__(self, cfg: CLIPConfig):
        super().__init__()
        num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(cfg.in_channels, cfg.d_model, cfg.patch_size, cfg.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, cfg.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False)  # 投影到共享空间
    
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # [CLS] 输出
        return self.proj(x)     # [B, embed_dim]


class TextEncoder(nn.Module):
    """文本编码器（Transformer Encoder）。"""
    def __init__(self, cfg: CLIPConfig):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_text_len, cfg.d_model) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False)
    
    def forward(self, input_ids):
        B, L = input_ids.shape
        x = self.tok_embed(input_ids) + self.pos_embed[:, :L]
        x = self.encoder(x)
        # Mean Pooling（取所有 token 的平均作为序列表示）
        x = self.norm(x.mean(dim=1))
        return self.proj(x)  # [B, embed_dim]


class MiniCLIP(nn.Module):
    """Mini CLIP: 双塔对比学习模型。"""
    def __init__(self, cfg: CLIPConfig):
        super().__init__()
        self.image_encoder = ImageEncoder(cfg)
        self.text_encoder = TextEncoder(cfg)
        # 可学习的温度系数（初始化为 log(1/0.07)）
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / cfg.temperature)))
    
    def forward(self, images, input_ids):
        """返回 (image_embeds, text_embeds, logit_scale)"""
        img_emb = F.normalize(self.image_encoder(images), dim=-1)
        txt_emb = F.normalize(self.text_encoder(input_ids), dim=-1)
        return img_emb, txt_emb, self.logit_scale.exp()


class InfoNCELoss(nn.Module):
    """
    对称 InfoNCE Loss（CLIP 使用的对比学习损失）。
    
    将 batch 中的 N 个图文对视为 N 个正例和 N*(N-1) 个负例。
    """
    def forward(self, img_emb, txt_emb, logit_scale):
        # 相似度矩阵 [N, N]
        logits = logit_scale * img_emb @ txt_emb.t()
        N = logits.size(0)
        labels = torch.arange(N, device=logits.device)
        # 对称 Loss: image→text + text→image
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.t(), labels)
        return (loss_i2t + loss_t2i) / 2


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention 融合模块。
    Query 来自一个模态，Key/Value 来自另一个模态。
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
    
    def forward(self, query_features, context_features):
        """
        query_features: [B, Lq, D] (模态 A)
        context_features: [B, Lc, D] (模态 B)
        """
        h = self.norm1(query_features)
        c = self.norm1(context_features)
        attn_out, _ = self.cross_attn(h, c, c)
        query_features = query_features + attn_out
        query_features = query_features + self.ffn(self.norm2(query_features))
        return query_features


if __name__ == "__main__":
    cfg = CLIPConfig()
    model = MiniCLIP(cfg)
    imgs = torch.randn(4, 3, 32, 32)
    ids = torch.randint(0, 2000, (4, 16))
    img_emb, txt_emb, scale = model(imgs, ids)
    loss_fn = InfoNCELoss()
    loss = loss_fn(img_emb, txt_emb, scale)
    print(f"img_emb: {img_emb.shape}, txt_emb: {txt_emb.shape}, loss: {loss.item():.4f}")
