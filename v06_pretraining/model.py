"""
v06 多模态预训练模型 — 多任务头: ITC + ITM + Captioning
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from config import PretrainConfig


class VisionEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        np = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, cfg.vision_dim, cfg.patch_size, cfg.patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.vision_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, np + 1, cfg.vision_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.vision_dim, 4, cfg.vision_dim*4, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 4)
        self.norm = nn.LayerNorm(cfg.vision_dim)
    
    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        x = x + self.pos_embed[:, :x.size(1)]
        return self.norm(self.encoder(x))


class TextEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.llm_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_text_len, cfg.llm_dim) * 0.02)
        layer = nn.TransformerEncoderLayer(cfg.llm_dim, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.llm_dim)
    
    def forward(self, ids):
        x = self.tok_embed(ids) + self.pos_embed[:, :ids.size(1)]
        return self.norm(self.encoder(x))


class PretrainModel(nn.Module):
    """多任务预训练模型: ITC + ITM + Captioning"""
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        self.cfg = cfg
        self.vision_enc = VisionEncoder(cfg)
        self.text_enc = TextEncoder(cfg)
        # ITC 投影头
        self.img_proj = nn.Linear(cfg.vision_dim, cfg.embed_dim, bias=False)
        self.txt_proj = nn.Linear(cfg.llm_dim, cfg.embed_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/cfg.temperature)))
        # ITM 头: 融合表示 → 二分类
        self.itm_head = nn.Sequential(
            nn.Linear(cfg.vision_dim + cfg.llm_dim, cfg.llm_dim), nn.GELU(),
            nn.Linear(cfg.llm_dim, 2)
        )
        # Captioning 头
        self.cap_head = nn.Linear(cfg.llm_dim, cfg.vocab_size, bias=False)
    
    def forward(self, images, input_ids, itm_labels=None):
        vis = self.vision_enc(images)    # [B, N+1, vision_dim]
        txt = self.text_enc(input_ids)   # [B, L, llm_dim]
        
        # ITC
        img_emb = F.normalize(self.img_proj(vis[:, 0]), dim=-1)  # CLS
        txt_emb = F.normalize(self.txt_proj(txt.mean(1)), dim=-1)
        scale = self.logit_scale.exp().clamp(max=100)
        logits_itc = scale * img_emb @ txt_emb.t()
        
        # ITM
        fused = torch.cat([vis[:, 0], txt.mean(1)], dim=-1)
        logits_itm = self.itm_head(fused)
        
        # Captioning (简化: 用文本编码器 + LM head)
        logits_cap = self.cap_head(txt)
        
        return {"itc_logits": logits_itc, "itm_logits": logits_itm, "cap_logits": logits_cap}


class PretrainLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.itm_criterion = nn.CrossEntropyLoss()
        self.cap_criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    def forward(self, outputs, itm_labels, cap_labels):
        B = outputs["itc_logits"].size(0)
        # ITC Loss
        labels_itc = torch.arange(B, device=outputs["itc_logits"].device)
        itc = (F.cross_entropy(outputs["itc_logits"], labels_itc) + F.cross_entropy(outputs["itc_logits"].t(), labels_itc)) / 2
        # ITM Loss
        itm = self.itm_criterion(outputs["itm_logits"], itm_labels)
        # Captioning Loss
        cap = self.cap_criterion(outputs["cap_logits"].view(-1, self.cfg.vocab_size), cap_labels.view(-1))
        
        total = self.cfg.itc_weight * itc + self.cfg.itm_weight * itm + self.cfg.cap_weight * cap
        return total, {"itc": itc.item(), "itm": itm.item(), "cap": cap.item(), "total": total.item()}
