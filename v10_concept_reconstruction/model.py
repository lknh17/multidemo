"""
v10 概念重构头 + 联合模型
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from config import ConceptConfig

# 复用 v09 的组件
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
from fusion import get_fusion_module
from model import MultiLayerEncoder


class ConceptReconstructionHead(nn.Module):
    """
    商业概念重构头: 从 embedding 预测多种商业属性。
    """
    def __init__(self, embed_dim, num_industries, num_brands, num_attributes, num_intents):
        super().__init__()
        self.industry_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, num_industries))
        self.brand_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, num_brands))
        self.attr_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, num_attributes))
        self.intent_head = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, num_intents))
    
    def forward(self, embeddings):
        return {
            "industry": self.industry_head(embeddings),
            "brand": self.brand_head(embeddings),
            "attributes": self.attr_head(embeddings),
            "intent": self.intent_head(embeddings),
        }


class ConceptReconLoss(nn.Module):
    """概念重构 Loss: 多标签分类 + 单标签分类"""
    def __init__(self):
        super().__init__()
        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, predictions, targets):
        loss = 0
        details = {}
        # 行业: 单标签分类
        l = self.ce(predictions["industry"], targets["industry"])
        loss += l; details["industry"] = l.item()
        # 品牌: 单标签
        l = self.ce(predictions["brand"], targets["brand"])
        loss += l; details["brand"] = l.item()
        # 属性: 多标签
        l = self.bce(predictions["attributes"], targets["attributes"].float())
        loss += l; details["attributes"] = l.item()
        # 意图: 单标签
        l = self.ce(predictions["intent"], targets["intent"])
        loss += l; details["intent"] = l.item()
        return loss, details


class ConceptModel(nn.Module):
    """完整模型: DeepStack + 对比学习 + 概念重构"""
    def __init__(self, cfg: ConceptConfig):
        super().__init__()
        self.cfg = cfg
        np = (cfg.image_size // cfg.patch_size) ** 2
        
        self.patch_embed = nn.Conv2d(3, cfg.d_model, cfg.patch_size, cfg.patch_size)
        self.img_cls = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.img_pos = nn.Parameter(torch.randn(1, np+1, cfg.d_model) * 0.02)
        self.img_encoder = MultiLayerEncoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers, cfg.dropout)
        
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.txt_pos = nn.Parameter(torch.randn(1, cfg.max_text_len, cfg.d_model) * 0.02)
        self.txt_encoder = MultiLayerEncoder(cfg.d_model, cfg.n_heads, cfg.d_ff, cfg.n_layers, cfg.dropout)
        
        num_extract = len(cfg.layer_indices)
        self.fusion_img = get_fusion_module(cfg.fusion_method, num_extract, cfg.d_model, cfg.embed_dim)
        self.fusion_txt = get_fusion_module(cfg.fusion_method, num_extract, cfg.d_model, cfg.embed_dim)
        
        self.concept_head = ConceptReconstructionHead(cfg.embed_dim, cfg.num_industries, cfg.num_brands, cfg.num_attributes, cfg.num_intents)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1/cfg.temperature)))
        
        # Uncertainty weighting
        self.log_sigma_contrastive = nn.Parameter(torch.tensor(0.0))
        self.log_sigma_recon = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, images, input_ids):
        B = images.size(0)
        layers = set(self.cfg.layer_indices)
        
        x_img = self.patch_embed(images).flatten(2).transpose(1, 2)
        x_img = torch.cat([self.img_cls.expand(B, -1, -1), x_img], dim=1)
        x_img = x_img + self.img_pos[:, :x_img.size(1)]
        _, img_inters = self.img_encoder.forward_with_intermediates(x_img, layers)
        
        x_txt = self.tok_embed(input_ids) + self.txt_pos[:, :input_ids.size(1)]
        _, txt_inters = self.txt_encoder.forward_with_intermediates(x_txt, layers)
        txt_inters = {k: v.mean(1) for k, v in txt_inters.items()}
        
        img_feats = [img_inters[l] for l in self.cfg.layer_indices]
        txt_feats = [txt_inters[l] for l in self.cfg.layer_indices]
        
        fused_img = F.normalize(self.fusion_img(img_feats), dim=-1)
        fused_txt = F.normalize(self.fusion_txt(txt_feats), dim=-1)
        
        # 概念重构 (从图像和文本 embedding 均做预测)
        img_concepts = self.concept_head(fused_img)
        txt_concepts = self.concept_head(fused_txt)
        
        return {
            "fused_img": fused_img, "fused_txt": fused_txt,
            "logit_scale": self.logit_scale.exp(),
            "img_concepts": img_concepts, "txt_concepts": txt_concepts,
            "log_sigma_c": self.log_sigma_contrastive, "log_sigma_r": self.log_sigma_recon,
        }
