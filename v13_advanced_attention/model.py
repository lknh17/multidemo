"""
v13 高级注意力机制 - 模型定义

本文件实现基于不同注意力变体的 Transformer 模型：
1. MHA Transformer (标准多头注意力)
2. GQA Transformer (分组查询注意力)
3. MQA Transformer (多查询注意力)
4. Sliding Window Transformer
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import AdvancedAttentionConfig
from attention_variants import (
    GroupedQueryAttention,
    SlidingWindowAttention,
    RotaryPositionalEncoding,
    NTKAwareRoPE,
)


# ============================================================
# 1. SwiGLU FFN (现代 LLM 标准)
# ============================================================
class SwiGLUFFN(nn.Module):
    """
    SwiGLU 前馈网络 - LLaMA/Qwen 使用的 FFN 变体。
    
    SwiGLU(x) = (Swish(xW_gate) ⊙ xW_up) W_down
    
    相比标准 FFN 多了门控机制，表达力更强。
    参数量：3 * d_model * d_ff（比标准 2 * d_model * d_ff 多 50%）
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)
        self.up_proj = nn.Linear(d_model, d_ff, bias=False)
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, L, d_model]"""
        gate = F.silu(self.gate_proj(x))  # Swish = SiLU
        up = self.up_proj(x)
        x = gate * up  # 门控
        x = self.dropout(x)
        x = self.down_proj(x)
        return x


# ============================================================
# 2. RMSNorm (现代 LLM 标准)
# ============================================================
class RMSNorm(nn.Module):
    """
    RMSNorm - 更轻量的 LayerNorm。
    
    去掉了均值中心化步骤，只做基于 RMS 的缩放。
    计算更快，效果几乎不变。LLaMA/Qwen 使用。
    
    RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
    """
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


# ============================================================
# 3. Transformer Block (使用 GQA + SwiGLU + RMSNorm)
# ============================================================
class AdvancedTransformerBlock(nn.Module):
    """
    现代 Transformer Block。
    
    与 v01 的标准 Block 相比：
    - LayerNorm → RMSNorm
    - MHA → GQA (可配置)
    - FFN → SwiGLU
    - 绝对位置编码 → RoPE
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 d_ff: int, dropout: float = 0.1,
                 window_size: int = None):
        super().__init__()
        
        # 选择注意力类型
        if window_size is not None:
            self.attention = SlidingWindowAttention(
                d_model, n_heads, n_kv_heads, window_size, dropout
            )
        else:
            self.attention = GroupedQueryAttention(
                d_model, n_heads, n_kv_heads, dropout
            )
        
        self.ffn = SwiGLUFFN(d_model, d_ff, dropout)
        
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, mask=None, rope=None, kv_cache=None, start_pos=0):
        """
        Pre-Norm 结构:
        x → RMSNorm → GQA → +残差 → RMSNorm → SwiGLU → +残差
        """
        # Self-Attention + 残差
        residual = x
        x = self.norm1(x)
        x, attn_weights = self.attention(x, mask=mask, rope=rope,
                                          kv_cache=kv_cache, start_pos=start_pos)
        x = self.dropout1(x)
        x = residual + x
        
        # FFN + 残差
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x, attn_weights


# ============================================================
# 4. 完整的高级 Transformer 模型
# ============================================================
class AdvancedTransformer(nn.Module):
    """
    高级 Transformer 模型 - 支持 MHA/GQA/MQA + RoPE + SwiGLU。
    
    用于语言建模任务（Decoder-only，类似 GPT/LLaMA）。
    通过 n_kv_heads 参数控制注意力类型。
    """
    
    def __init__(self, cfg: AdvancedAttentionConfig):
        super().__init__()
        self.cfg = cfg
        
        # ---- Embedding ----
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.d_model)
        
        # ---- RoPE ----
        if cfg.target_seq_len > cfg.max_seq_len:
            # 需要长上下文扩展
            self.rope = NTKAwareRoPE(
                cfg.d_model // cfg.n_heads,
                max_len=cfg.target_seq_len,
                theta=cfg.rope_theta,
                train_len=cfg.max_seq_len,
                target_len=cfg.target_seq_len,
            )
        else:
            self.rope = RotaryPositionalEncoding(
                cfg.d_model // cfg.n_heads,
                max_len=cfg.max_seq_len * 2,
                theta=cfg.rope_theta,
            )
        
        # ---- Transformer Layers ----
        window = cfg.window_size if cfg.use_sliding_window else None
        self.layers = nn.ModuleList([
            AdvancedTransformerBlock(
                cfg.d_model, cfg.n_heads, cfg.n_kv_heads,
                cfg.d_ff, cfg.dropout, window_size=window,
            )
            for _ in range(cfg.n_layers)
        ])
        
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        
        # 权重绑定：embedding 和 lm_head 共享权重
        self.lm_head.weight = self.embedding.weight
        
        self._init_weights()
    
    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            input_ids: [B, L] token ids
            mask: [B, 1, L, L] 因果掩码（None 则自动创建）
        
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = input_ids.shape
        
        x = self.embedding(input_ids)  # [B, L, d_model]
        
        # 自动创建因果掩码
        if mask is None:
            mask = torch.tril(torch.ones(L, L, device=x.device, dtype=torch.bool))
            mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
        
        # 逐层前向
        for layer in self.layers:
            x, _ = layer(x, mask=mask, rope=self.rope)
        
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50):
        """自回归生成（带 KV Cache）"""
        B = input_ids.size(0)
        
        # 初始化每层的 KV Cache
        kv_caches = [{} for _ in range(len(self.layers))]
        
        # Prefill 阶段
        generated = input_ids.tolist()[0]
        
        for step in range(max_new_tokens):
            if step == 0:
                x = self.embedding(input_ids)
                start_pos = 0
                L = input_ids.size(1)
            else:
                # 只处理最新一个 token
                last_token = torch.tensor([[generated[-1]]], device=input_ids.device)
                x = self.embedding(last_token)
                start_pos = len(generated) - 1
                L = 1
            
            for i, layer in enumerate(self.layers):
                x, _ = layer(x, mask=None, rope=self.rope,
                            kv_cache=kv_caches[i], start_pos=start_pos)
            
            x = self.final_norm(x)
            logits = self.lm_head(x[:, -1:])  # 只取最后一个位置
            next_token = logits.argmax(dim=-1).item()
            generated.append(next_token)
        
        return generated
    
    @staticmethod
    def count_kv_cache_size(n_layers, n_kv_heads, d_k, seq_len, dtype_bytes=4):
        """计算 KV Cache 的内存大小 (bytes)"""
        # 每层: 2 (K+V) × n_kv_heads × seq_len × d_k × dtype_bytes
        per_layer = 2 * n_kv_heads * seq_len * d_k * dtype_bytes
        total = per_layer * n_layers
        return total


if __name__ == "__main__":
    from config import config
    
    # 对比不同注意力类型
    configs = [
        ("MHA", 8, 8),    # 标准多头
        ("GQA", 8, 2),    # 分组查询 (每4个Q head共享1对KV)
        ("MQA", 8, 1),    # 多查询
    ]
    
    for name, n_heads, n_kv_heads in configs:
        cfg = AdvancedAttentionConfig(n_heads=n_heads, n_kv_heads=n_kv_heads)
        model = AdvancedTransformer(cfg)
        
        total_params = sum(p.numel() for p in model.parameters())
        kv_cache_size = model.count_kv_cache_size(
            cfg.n_layers, n_kv_heads, cfg.d_model // n_heads,
            cfg.max_seq_len
        )
        
        print(f"\n{name} (n_heads={n_heads}, n_kv_heads={n_kv_heads}):")
        print(f"  参数量: {total_params:,}")
        print(f"  KV Cache: {kv_cache_size / 1024:.1f} KB (seq_len={cfg.max_seq_len})")
        
        # 测试前向
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        logits = model(x)
        print(f"  输出形状: {logits.shape}")
