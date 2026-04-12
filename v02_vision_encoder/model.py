"""
v02 Vision Encoder - ViT 从零实现

核心组件:
1. PatchEmbedding: 将图像切成 patch 并投影到 d_model 维
2. ViT: 标准 Transformer Encoder + [CLS] Token + 分类头
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ViTConfig


class PatchEmbedding(nn.Module):
    """
    图像 Patch 嵌入层。
    
    将 [B, C, H, W] 的图像切成 P×P 的 patch 并线性投影到 d_model 维。
    实现上用一个 kernel=stride=patch_size 的卷积等价实现。
    """
    def __init__(self, img_size: int, patch_size: int, in_channels: int, d_model: int):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2  # e.g. (32/4)^2 = 64
        
        # 用卷积实现 patch 切分 + 线性投影（高效且等价）
        # kernel_size=stride=patch_size 意味着不重叠地切分图像
        self.projection = nn.Conv2d(
            in_channels, d_model,
            kernel_size=patch_size, stride=patch_size
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, num_patches, d_model]
        """
        # [B, C, H, W] → [B, d_model, H/P, W/P]
        x = self.projection(x)
        # [B, d_model, H/P, W/P] → [B, d_model, num_patches] → [B, num_patches, d_model]
        x = x.flatten(2).transpose(1, 2)
        return x


class ViT(nn.Module):
    """
    Vision Transformer (ViT) 完整实现。
    
    架构: PatchEmbed → [CLS]+PosEmbed → N × TransformerEncoder → [CLS] → ClassHead
    """
    def __init__(self, config: ViTConfig):
        super().__init__()
        self.config = config
        
        # Patch Embedding
        self.patch_embed = PatchEmbedding(
            config.image_size, config.patch_size, config.in_channels, config.d_model
        )
        num_patches = self.patch_embed.num_patches
        
        # [CLS] Token: 可学习的全局聚合 token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        
        # 可学习位置编码: [1, num_patches + 1, d_model]（+1 for CLS）
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, config.d_model) * 0.02
        )
        self.pos_drop = nn.Dropout(config.dropout)
        
        # Transformer Encoder Layers（复用 v01 的设计但简化为标准 Encoder）
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-Norm
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        
        # 最终 LayerNorm
        self.norm = nn.LayerNorm(config.d_model)
        
        # 分类头: d_model → num_classes
        self.head = nn.Linear(config.d_model, config.num_classes)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C, H, W]
        return: [B, num_classes]
        """
        B = x.size(0)
        
        # 1. Patch Embedding: [B, num_patches, d_model]
        x = self.patch_embed(x)
        
        # 2. Prepend [CLS] token: [B, num_patches+1, d_model]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 3. 加位置编码
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # 4. Transformer Encoder
        x = self.encoder(x)
        
        # 5. 取 [CLS] 输出作为全局表示
        x = self.norm(x[:, 0])  # [B, d_model]
        
        # 6. 分类
        logits = self.head(x)  # [B, num_classes]
        return logits
    
    def get_attention_maps(self, x: torch.Tensor):
        """提取各层的 attention map 用于可视化。"""
        B = x.size(0)
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        
        attn_maps = []
        for layer in self.encoder.layers:
            # 利用 PyTorch 的 hook 或手动计算
            x_norm = layer.norm1(x)
            attn_out, attn_weights = layer.self_attn(
                x_norm, x_norm, x_norm, need_weights=True
            )
            attn_maps.append(attn_weights.detach())
            x = x + attn_out
            x = x + layer.linear2(layer.dropout(layer.activation(layer.linear1(layer.norm2(x)))))
        
        return attn_maps


if __name__ == "__main__":
    cfg = ViTConfig()
    model = ViT(cfg)
    x = torch.randn(2, 3, 32, 32)
    logits = model(x)
    print(f"Input: {x.shape} → Output: {logits.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total:,}")
