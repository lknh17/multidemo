"""
v03 Mini GPT 实现 — 带 RoPE 和 KV Cache

核心组件:
1. RMSNorm (替代 LayerNorm)
2. RoPE 旋转位置编码
3. 因果自注意力 + KV Cache
4. SwiGLU FFN
5. GPT Decoder-Only 模型
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import GPTConfig


class RMSNorm(nn.Module):
    """RMSNorm: 比 LayerNorm 更轻量，LLaMA/Qwen 都在用。"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


def precompute_rope_freqs(dim: int, max_len: int, base: float = 10000.0):
    """预计算 RoPE 频率。"""
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_len)
    freqs = torch.outer(t, freqs)  # [max_len, dim/2]
    cos = freqs.cos()  # [max_len, dim/2]
    sin = freqs.sin()
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """
    对 Q 或 K 施加旋转位置编码。
    x: [B, H, L, D]
    """
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    cos = cos[:x.shape[2], :].unsqueeze(0).unsqueeze(0)  # [1, 1, L, D/2]
    sin = sin[:x.shape[2], :].unsqueeze(0).unsqueeze(0)
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    return torch.cat([out1, out2], dim=-1)


class CausalSelfAttention(nn.Module):
    """带 RoPE 和 KV Cache 的因果自注意力。"""
    def __init__(self, d_model: int, n_heads: int, max_len: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        
        cos, sin = precompute_rope_freqs(self.d_k, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)
    
    def forward(self, x, kv_cache=None):
        """
        x: [B, L, D]
        kv_cache: (cached_k, cached_v) 或 None
        返回: (output, new_kv_cache)
        """
        B, L, D = x.shape
        qkv = self.W_qkv(x).reshape(B, L, 3, self.n_heads, self.d_k)
        q, k, v = qkv.unbind(2)  # 各 [B, L, H, dk]
        q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)  # [B,H,L,dk]
        
        # 应用 RoPE（只对 Q 和 K）
        if kv_cache is not None:
            # 推理时 offset 需要考虑已缓存的长度
            offset = kv_cache[0].shape[2]
            cos = self.rope_cos[offset:offset+L]
            sin = self.rope_sin[offset:offset+L]
        else:
            cos, sin = self.rope_cos[:L], self.rope_sin[:L]
        
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # KV Cache 拼接
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)
        
        # 因果注意力
        scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.d_k)
        total_len = k.shape[2]
        causal_mask = torch.triu(torch.ones(L, total_len, device=x.device), diagonal=total_len-L+1).bool()
        scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v).transpose(1,2).contiguous().view(B, L, D)
        return self.W_o(out), new_cache


class SwiGLU(nn.Module):
    """SwiGLU FFN: LLaMA/Qwen 使用的门控 FFN。"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, max_len, dropout=0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, max_len, dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
    
    def forward(self, x, kv_cache=None):
        h = self.norm1(x)
        h, new_cache = self.attn(h, kv_cache)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x, new_cache


class MiniGPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([
            GPTBlock(config.d_model, config.n_heads, config.d_ff, config.max_seq_len, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # 权重共享: embedding 和 lm_head 共享权重（减少参数量）
        self.tok_embed.weight = self.lm_head.weight
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, input_ids, kv_caches=None):
        x = self.tok_embed(input_ids)
        new_caches = []
        for i, block in enumerate(self.blocks):
            cache = kv_caches[i] if kv_caches else None
            x, new_cache = block(x, cache)
            new_caches.append(new_cache)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits, new_caches

if __name__ == "__main__":
    cfg = GPTConfig()
    model = MiniGPT(cfg)
    ids = torch.randint(0, cfg.vocab_size, (2, 32))
    logits, _ = model(ids)
    print(f"Input: {ids.shape} → Logits: {logits.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
