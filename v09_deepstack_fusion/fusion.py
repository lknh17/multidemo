"""
v09 融合策略模块: Concat / Weighted Sum / Attention / Gated
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """拼接融合: 直接拼接 + 投影降维"""
    def __init__(self, d_model: int, num_layers: int, embed_dim: int):
        super().__init__()
        self.proj = nn.Linear(d_model * num_layers, embed_dim)
    
    def forward(self, layer_features):
        """layer_features: list of [B, D]"""
        return self.proj(torch.cat(layer_features, dim=-1))


class WeightedSumFusion(nn.Module):
    """可学习权重加权求和"""
    def __init__(self, num_layers: int, d_model: int, embed_dim: int):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.proj = nn.Linear(d_model, embed_dim)
    
    def forward(self, layer_features):
        w = F.softmax(self.weights, dim=0)
        stacked = torch.stack(layer_features, dim=0)  # [num_layers, B, D]
        fused = (stacked * w.view(-1, 1, 1)).sum(0)   # [B, D]
        return self.proj(fused)


class AttentionFusion(nn.Module):
    """注意力融合: 用可学习 query 聚合各层特征"""
    def __init__(self, num_layers: int, d_model: int, embed_dim: int, n_heads: int = 4):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj = nn.Linear(d_model, embed_dim)
    
    def forward(self, layer_features):
        B = layer_features[0].size(0)
        kv = torch.stack(layer_features, dim=1)        # [B, num_layers, D]
        q = self.query.expand(B, -1, -1)               # [B, 1, D]
        out, _ = self.attn(q, kv, kv)                   # [B, 1, D]
        return self.proj(out.squeeze(1))                 # [B, embed_dim]


class GatedFusion(nn.Module):
    """门控融合: 每层特征经过门控后加权"""
    def __init__(self, num_layers: int, d_model: int, embed_dim: int):
        super().__init__()
        self.gates = nn.ModuleList([nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid()) for _ in range(num_layers)])
        self.proj = nn.Linear(d_model, embed_dim)
    
    def forward(self, layer_features):
        gated = [gate(feat) * feat for gate, feat in zip(self.gates, layer_features)]
        fused = sum(gated) / len(gated)
        return self.proj(fused)


def get_fusion_module(method: str, num_layers: int, d_model: int, embed_dim: int):
    if method == "concat": return ConcatFusion(d_model, num_layers, embed_dim)
    elif method == "weighted": return WeightedSumFusion(num_layers, d_model, embed_dim)
    elif method == "attention": return AttentionFusion(num_layers, d_model, embed_dim)
    elif method == "gated": return GatedFusion(num_layers, d_model, embed_dim)
    else: raise ValueError(f"Unknown fusion: {method}")
