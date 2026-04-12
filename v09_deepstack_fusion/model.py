"""v09 DeepStack 模型"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from config import DeepStackConfig
from fusion import get_fusion_module


class MultiLayerEncoder(nn.Module):
    """支持多层特征提取的编码器"""
    def __init__(self, d_model, n_heads, d_ff, n_layers, dropout, input_proj=None):
        super().__init__()
        self.input_proj = input_proj
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
    
    def forward_with_intermediates(self, x, extract_layers):
        """前向传播并返回指定层的中间特征"""
        if self.input_proj: x = self.input_proj(x)
        intermediates = {}
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in extract_layers:
                intermediates[i] = x[:, 0] if x.dim() == 3 else x  # 取 CLS 或 mean
        x = self.norm(x)
        return x, intermediates


class DeepStackModel(nn.Module):
    """DeepStack: 多层特征提取 + 融合 + 多层 Loss"""
    def __init__(self, cfg: DeepStackConfig):
        super().__init__()
        self.cfg = cfg
        np = (cfg.image_size // cfg.patch_size) ** 2
        
        # 图像编码器
        self.patch_embed = nn.Conv2d(3, cfg.d_model, cfg.patch_size, cfg.patch_size)
        self.img_cls = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.img_pos = nn.Parameter(torch.randn(1, np+1, cfg.d_model) * 0.02)
        self.img_encoder = MultiLayerEncoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers, cfg.dropout)
        
        # 文本编码器
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.txt_pos = nn.Parameter(torch.randn(1, cfg.max_text_len, cfg.d_model) * 0.02)
        self.txt_encoder = MultiLayerEncoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers, cfg.dropout)
        
        # 每层的投影头
        num_extract = len(cfg.layer_indices)
        self.layer_img_projs = nn.ModuleList([nn.Linear(cfg.d_model, cfg.embed_dim) for _ in range(num_extract)])
        self.layer_txt_projs = nn.ModuleList([nn.Linear(cfg.d_model, cfg.embed_dim) for _ in range(num_extract)])
        
        # 融合模块
        self.fusion_img = get_fusion_module(cfg.fusion_method, num_extract, cfg.d_model, cfg.embed_dim)
        self.fusion_txt = get_fusion_module(cfg.fusion_method, num_extract, cfg.d_model, cfg.embed_dim)
        
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/cfg.temperature)))
    
    def forward(self, images, input_ids):
        B = images.size(0)
        layers = self.cfg.layer_indices
        
        # 图像
        x_img = self.patch_embed(images).flatten(2).transpose(1, 2)
        x_img = torch.cat([self.img_cls.expand(B, -1, -1), x_img], dim=1)
        x_img = x_img + self.img_pos[:, :x_img.size(1)]
        _, img_intermediates = self.img_encoder.forward_with_intermediates(x_img, set(layers))
        
        # 文本
        x_txt = self.tok_embed(input_ids) + self.txt_pos[:, :input_ids.size(1)]
        _, txt_intermediates = self.txt_encoder.forward_with_intermediates(x_txt, set(layers))
        txt_intermediates = {k: v.mean(1) for k, v in txt_intermediates.items()}
        
        # 各层 embedding
        layer_img_embs, layer_txt_embs = [], []
        img_feats_for_fusion, txt_feats_for_fusion = [], []
        for i, l in enumerate(layers):
            ie = F.normalize(self.layer_img_projs[i](img_intermediates[l]), dim=-1)
            te = F.normalize(self.layer_txt_projs[i](txt_intermediates[l]), dim=-1)
            layer_img_embs.append(ie)
            layer_txt_embs.append(te)
            img_feats_for_fusion.append(img_intermediates[l])
            txt_feats_for_fusion.append(txt_intermediates[l])
        
        # 融合 embedding
        fused_img = F.normalize(self.fusion_img(img_feats_for_fusion), dim=-1)
        fused_txt = F.normalize(self.fusion_txt(txt_feats_for_fusion), dim=-1)
        
        return {
            "layer_img_embs": layer_img_embs, "layer_txt_embs": layer_txt_embs,
            "fused_img": fused_img, "fused_txt": fused_txt,
            "logit_scale": self.logit_scale.exp(),
        }
