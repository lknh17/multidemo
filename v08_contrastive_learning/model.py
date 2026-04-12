"""v08 对比学习模型"""
import torch, torch.nn as nn, torch.nn.functional as F
import math
from config import ContrastiveConfig


class DualEncoder(nn.Module):
    """双塔编码器 + 投影头 + 可学习温度"""
    def __init__(self, cfg: ContrastiveConfig):
        super().__init__()
        # 图像编码器
        np = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, cfg.d_model, cfg.patch_size, cfg.patch_size)
        self.img_pos = nn.Parameter(torch.randn(1, np+1, cfg.d_model) * 0.02)
        self.img_cls = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        img_layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.img_enc = nn.TransformerEncoder(img_layer, cfg.n_layers)
        self.img_norm = nn.LayerNorm(cfg.d_model)
        self.img_proj = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False)
        
        # 文本编码器
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.txt_pos = nn.Parameter(torch.randn(1, cfg.max_text_len, cfg.d_model) * 0.02)
        txt_layer = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.dropout, activation="gelu", batch_first=True, norm_first=True)
        self.txt_enc = nn.TransformerEncoder(txt_layer, cfg.n_layers)
        self.txt_norm = nn.LayerNorm(cfg.d_model)
        self.txt_proj = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False)
        
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/cfg.temperature)))
    
    def encode_image(self, images):
        B = images.size(0)
        x = self.patch_embed(images).flatten(2).transpose(1, 2)
        x = torch.cat([self.img_cls.expand(B, -1, -1), x], dim=1)
        x = x + self.img_pos[:, :x.size(1)]
        x = self.img_norm(self.img_enc(x))
        return F.normalize(self.img_proj(x[:, 0]), dim=-1)
    
    def encode_text(self, input_ids):
        x = self.tok_embed(input_ids) + self.txt_pos[:, :input_ids.size(1)]
        x = self.txt_norm(self.txt_enc(x))
        return F.normalize(self.txt_proj(x.mean(1)), dim=-1)
    
    def forward(self, images, input_ids):
        return self.encode_image(images), self.encode_text(input_ids), self.logit_scale.exp()
