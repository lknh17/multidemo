"""
v07 LoRA 实现 + SFT 模型

从零实现 LoRA 适配器，展示低秩分解的核心思想。
"""
import math, torch, torch.nn as nn, torch.nn.functional as F
from config import SFTConfig


class LoRALinear(nn.Module):
    """
    LoRA 适配器：冻结原始 Linear，只训练低秩的 A 和 B。
    
    output = x @ W_0^T + x @ A^T @ B^T * (alpha/r)
    """
    def __init__(self, original_linear: nn.Linear, r: int = 16, alpha: int = 32, dropout: float = 0.05):
        super().__init__()
        self.original = original_linear
        self.original.weight.requires_grad_(False)  # 冻结原始权重
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # A: 降维矩阵, B: 升维矩阵
        self.lora_A = nn.Linear(in_features, r, bias=False)
        self.lora_B = nn.Linear(r, out_features, bias=False)
        self.scaling = alpha / r
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # 初始化: A 用 Kaiming, B 初始化为 0 → 初始时 ΔW = 0
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    
    def forward(self, x):
        # 原始输出 + LoRA 增量
        original_out = self.original(x)
        lora_out = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return original_out + lora_out
    
    def merge_weights(self):
        """将 LoRA 权重合并到原始权重中（推理优化）。"""
        with torch.no_grad():
            delta_w = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            self.original.weight.add_(delta_w)


def inject_lora(model: nn.Module, r: int = 16, alpha: int = 32, target_names=("q_proj", "v_proj")):
    """
    在模型中注入 LoRA 适配器。
    
    替换所有名称匹配 target_names 的 Linear 层。
    """
    replaced = 0
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if isinstance(child, nn.Linear) and any(t in child_name for t in target_names):
                lora = LoRALinear(child, r=r, alpha=alpha)
                setattr(module, child_name, lora)
                replaced += 1
    print(f"[LoRA] 注入了 {replaced} 个 LoRA 适配器 (r={r}, alpha={alpha})")
    return model


class SFTModel(nn.Module):
    """SFT 模型: 简化版多模态模型 + LoRA"""
    def __init__(self, cfg: SFTConfig):
        super().__init__()
        self.cfg = cfg
        # 简化版视觉编码器
        np = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, cfg.vision_dim, cfg.patch_size, cfg.patch_size)
        self.vis_proj = nn.Linear(cfg.vision_dim, cfg.llm_dim)
        self.vis_pos = nn.Parameter(torch.randn(1, np, cfg.vision_dim) * 0.02)
        # 文本
        self.tok_embed = nn.Embedding(cfg.vocab_size, cfg.llm_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.llm_dim) * 0.02)
        # Transformer (这些层会被注入 LoRA)
        layer = nn.TransformerEncoderLayer(
            cfg.llm_dim, cfg.n_heads, cfg.d_ff, cfg.dropout,
            activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.llm_dim)
        self.lm_head = nn.Linear(cfg.llm_dim, cfg.vocab_size, bias=False)
    
    def forward(self, images, input_ids):
        B = images.size(0)
        vis = self.patch_embed(images).flatten(2).transpose(1, 2)
        vis = vis + self.vis_pos[:, :vis.size(1)]
        vis = self.vis_proj(vis)
        txt = self.tok_embed(input_ids)
        combined = torch.cat([vis, txt], dim=1)
        L = combined.size(1)
        combined = combined + self.pos_embed[:, :L]
        mask = nn.Transformer.generate_square_subsequent_mask(L, device=combined.device)
        out = self.norm(self.transformer(combined, mask=mask, is_causal=True))
        text_out = out[:, vis.size(1):, :]
        return self.lm_head(text_out)


if __name__ == "__main__":
    cfg = SFTConfig()
    model = SFTModel(cfg)
    total_before = sum(p.numel() for p in model.parameters())
    model = inject_lora(model, r=cfg.lora_r, alpha=cfg.lora_alpha)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_before:,} | 可训练(LoRA): {trainable:,} | 占比: {trainable/total_before*100:.2f}%")
