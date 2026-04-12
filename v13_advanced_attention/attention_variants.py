"""
v13 高级注意力机制 - 注意力变体集合

本文件实现多种注意力机制变体，带详细注释：
1. GroupedQueryAttention (GQA) - Qwen2/LLaMA2 使用
2. MultiQueryAttention (MQA) - PaLM/Falcon 使用
3. SlidingWindowAttention - Mistral 使用
4. Flash Attention 模拟 (tiling + online softmax)
5. NTK-aware RoPE 长上下文扩展
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. RoPE (Rotary Position Embedding)
#    现代 LLM 的标准位置编码
# ============================================================
class RotaryPositionalEncoding(nn.Module):
    """
    旋转位置编码 (RoPE)。
    
    核心思想：不把位置信息加到 embedding 上，而是对 Q 和 K 施加
    与位置相关的旋转变换，使得 Q·K 的内积自然包含相对位置信息。
    
    公式: f(x, pos) = x * cos(pos * theta) + rotate_half(x) * sin(pos * theta)
    """
    
    def __init__(self, d_model: int, max_len: int = 8192, theta: float = 10000.0):
        super().__init__()
        self.d_model = d_model
        self.theta = theta
        
        # 预计算频率: theta_i = 1 / (theta ^ (2i / d))
        freqs = 1.0 / (theta ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("freqs", freqs)  # [d/2]
        
        # 预计算位置对应的 cos/sin
        self._build_cache(max_len)
    
    def _build_cache(self, max_len: int):
        """预计算 cos/sin 缓存"""
        positions = torch.arange(max_len).float()  # [max_len]
        # outer product: [max_len] x [d/2] → [max_len, d/2]
        angles = torch.outer(positions, self.freqs)
        # 拼接成 [max_len, d]: cos 和 sin 交替
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)
    
    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """将向量的前半和后半交换并取反，实现 2D 旋转"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, start_pos: int = 0):
        """
        对 Q 和 K 应用旋转位置编码。
        
        Args:
            q: [B, n_heads, L, d_k]
            k: [B, n_kv_heads, L, d_k]
            start_pos: 起始位置（用于 KV Cache 推理）
        """
        seq_len = q.size(2)
        
        # 取对应位置的 cos/sin [L, d_k/2]
        cos = self.cos_cache[start_pos:start_pos + seq_len]  # [L, d/2]
        sin = self.sin_cache[start_pos:start_pos + seq_len]  # [L, d/2]
        
        # 扩展维度以匹配 q/k: [1, 1, L, d/2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # 对 q 和 k 分别应用旋转
        # q_rot = q * cos + rotate_half(q) * sin
        q_rot = q * cos + self.rotate_half(q) * sin
        k_rot = k * cos + self.rotate_half(k) * sin
        
        return q_rot, k_rot


class NTKAwareRoPE(RotaryPositionalEncoding):
    """
    NTK-aware RoPE 扩展，支持长上下文外推。
    
    当推理长度超过训练长度时，动态调整频率基数 theta，
    使低频分量（编码全局位置）被压缩，高频分量（编码局部位置）保持不变。
    """
    
    def __init__(self, d_model: int, max_len: int = 8192, theta: float = 10000.0,
                 train_len: int = 512, target_len: int = 2048):
        self.train_len = train_len
        self.target_len = target_len
        super().__init__(d_model, max_len, theta)
    
    def _build_cache(self, max_len: int):
        """使用 NTK-aware 缩放的频率"""
        if self.target_len > self.train_len:
            # NTK 缩放因子
            alpha = (self.target_len / self.train_len) ** (
                self.d_model / (self.d_model - 2)
            )
            # 调整频率基数
            scaled_freqs = 1.0 / (
                (self.theta * alpha) ** (
                    torch.arange(0, self.d_model, 2).float() / self.d_model
                )
            )
            self.register_buffer("freqs", scaled_freqs)
        
        # 使用（可能已缩放的）频率构建缓存
        positions = torch.arange(max_len).float()
        angles = torch.outer(positions, self.freqs)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)


# ============================================================
# 2. Grouped Query Attention (GQA)
#    MHA 和 MQA 的统一实现
# ============================================================
class GroupedQueryAttention(nn.Module):
    """
    分组查询注意力 (GQA) - 统一实现 MHA/GQA/MQA。
    
    - n_kv_heads = n_heads → MHA (标准多头注意力)
    - n_kv_heads = 1       → MQA (多查询注意力)
    - 1 < n_kv_heads < n_heads → GQA (分组查询注意力)
    
    核心：Q 有 n_heads 个头，K/V 只有 n_kv_heads 个头，
    每 (n_heads / n_kv_heads) 个 Q head 共享一对 KV head。
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0, \
            f"n_heads ({n_heads}) 必须能被 n_kv_heads ({n_kv_heads}) 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.d_k = d_model // n_heads
        self.n_groups = n_heads // n_kv_heads  # 每组的 Q head 数
        
        # Q 投影: 完整的 n_heads
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        # K/V 投影: 只有 n_kv_heads（比 MHA 小）
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)
        # 输出投影
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,           # [B, L, d_model]
        mask: torch.Tensor = None,  # [B, 1, L, L] 或 broadcastable
        rope: RotaryPositionalEncoding = None,
        kv_cache: dict = None,      # 用于推理的 KV Cache
        start_pos: int = 0,
    ) -> tuple:
        B, L, _ = x.shape
        
        # ---- 投影 ----
        Q = self.W_q(x).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        # Q: [B, n_heads, L, d_k]
        
        K = self.W_k(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, L, self.n_kv_heads, self.d_k).transpose(1, 2)
        # K, V: [B, n_kv_heads, L, d_k]
        
        # ---- RoPE ----
        if rope is not None:
            Q, K = rope(Q, K, start_pos=start_pos)
        
        # ---- KV Cache (推理时使用) ----
        if kv_cache is not None:
            if "k" in kv_cache:
                K = torch.cat([kv_cache["k"], K], dim=2)
                V = torch.cat([kv_cache["v"], V], dim=2)
            kv_cache["k"] = K
            kv_cache["v"] = V
        
        # ---- 扩展 KV heads 以匹配 Q heads ----
        # 每 n_groups 个 Q head 共享同一对 KV head
        if self.n_groups > 1:
            # K: [B, n_kv_heads, Lk, d_k] → [B, n_kv_heads, 1, Lk, d_k]
            #    → expand → [B, n_kv_heads, n_groups, Lk, d_k]
            #    → reshape → [B, n_heads, Lk, d_k]
            K = K.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
            K = K.reshape(B, self.n_heads, -1, self.d_k)
            V = V.unsqueeze(2).expand(-1, -1, self.n_groups, -1, -1)
            V = V.reshape(B, self.n_heads, -1, self.d_k)
        
        # ---- 注意力计算 ----
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        # scores: [B, n_heads, Lq, Lk]
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        # output: [B, n_heads, Lq, d_k]
        
        # ---- 拼接 + 输出投影 ----
        output = output.transpose(1, 2).contiguous().view(B, L, self.d_model)
        output = self.W_o(output)
        
        return output, attn_weights


# ============================================================
# 3. Sliding Window Attention
# ============================================================
class SlidingWindowAttention(GroupedQueryAttention):
    """
    滑动窗口注意力 - 每个 token 只关注前 window_size 个 token。
    
    通过构造带状因果掩码实现：
    mask[i, j] = True  if  j >= max(0, i - window_size + 1) and j <= i
    """
    
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int,
                 window_size: int = 128, dropout: float = 0.1):
        super().__init__(d_model, n_heads, n_kv_heads, dropout)
        self.window_size = window_size
    
    @staticmethod
    def create_sliding_window_mask(seq_len: int, window_size: int, device: torch.device):
        """
        创建滑动窗口因果掩码。
        
        mask[i, j] = True 表示位置 i 可以注意到位置 j
        条件: j <= i (因果) AND i - j < window_size (窗口)
        """
        # 因果掩码
        causal = torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))
        # 窗口掩码：距离超过 window_size 的位置被遮蔽
        rows = torch.arange(seq_len, device=device).unsqueeze(1)
        cols = torch.arange(seq_len, device=device).unsqueeze(0)
        window = (rows - cols) < window_size
        # 组合
        mask = causal & window
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
    
    def forward(self, x, mask=None, rope=None, kv_cache=None, start_pos=0):
        L = x.size(1)
        if mask is None:
            mask = self.create_sliding_window_mask(L, self.window_size, x.device)
        return super().forward(x, mask=mask, rope=rope, kv_cache=kv_cache, start_pos=start_pos)


# ============================================================
# 4. Flash Attention 模拟 (教学用)
#    真实 Flash Attention 需要 CUDA Kernel，这里用 Python 模拟核心思想
# ============================================================
def flash_attention_simulation(
    Q: torch.Tensor,   # [B, H, N, d]
    K: torch.Tensor,   # [B, H, N, d]
    V: torch.Tensor,   # [B, H, N, d]
    block_size: int = 64,
) -> torch.Tensor:
    """
    Flash Attention 的 Python 模拟实现（教学用途）。
    
    核心思想：
    1. 将 Q/K/V 分成小块（tile）
    2. 在每个块内计算局部注意力
    3. 使用 Online Softmax 逐块更新全局统计量
    
    真实 Flash Attention 在 CUDA 中实现，将 tile 放入 SRAM 以减少 HBM 访问。
    这里用 Python 展示算法逻辑，性能不是目标。
    """
    B, H, N, d = Q.shape
    output = torch.zeros_like(Q)
    
    # 分块数
    n_blocks = math.ceil(N / block_size)
    
    for q_block_idx in range(n_blocks):
        q_start = q_block_idx * block_size
        q_end = min(q_start + block_size, N)
        Q_block = Q[:, :, q_start:q_end]  # [B, H, Bq, d]
        
        # 维护 online softmax 的统计量
        # m_i: 当前已见到的最大 score
        # l_i: 当前 exp(score - m_i) 的累积和
        # O_i: 当前加权 V 的累积
        m_i = torch.full((B, H, q_end - q_start, 1), float("-inf"), device=Q.device)
        l_i = torch.zeros(B, H, q_end - q_start, 1, device=Q.device)
        O_i = torch.zeros(B, H, q_end - q_start, d, device=Q.device)
        
        for k_block_idx in range(n_blocks):
            k_start = k_block_idx * block_size
            k_end = min(k_start + block_size, N)
            K_block = K[:, :, k_start:k_end]  # [B, H, Bk, d]
            V_block = V[:, :, k_start:k_end]  # [B, H, Bk, d]
            
            # 局部 scores
            S_block = torch.matmul(Q_block, K_block.transpose(-2, -1)) / math.sqrt(d)
            # S_block: [B, H, Bq, Bk]
            
            # Online Softmax 更新
            m_new = torch.max(m_i, S_block.max(dim=-1, keepdim=True).values)
            
            # 修正旧的累积值
            correction = torch.exp(m_i - m_new)
            l_i = l_i * correction
            O_i = O_i * correction
            
            # 当前块的贡献
            P_block = torch.exp(S_block - m_new)  # [B, H, Bq, Bk]
            l_i = l_i + P_block.sum(dim=-1, keepdim=True)
            O_i = O_i + torch.matmul(P_block, V_block)
            
            m_i = m_new
        
        # 最终归一化
        output[:, :, q_start:q_end] = O_i / l_i
    
    return output


# ============================================================
# 5. Attention Sink 检测器
# ============================================================
class AttentionSinkDetector:
    """
    检测和利用 Attention Sink 现象。
    
    Attention Sink: 大模型中第一个 token 始终获得异常高的注意力权重，
    这是 Softmax 的数学特性导致的（需要一个"垃圾桶"来放置不需要的注意力概率）。
    
    利用：保留 sink tokens + 最近 window 个 token，实现无限长度推理。
    """
    
    @staticmethod
    def detect_sinks(attn_weights: torch.Tensor, threshold: float = 0.1) -> torch.Tensor:
        """
        检测 attention sink 位置。
        
        Args:
            attn_weights: [B, H, Lq, Lk]
            threshold: 平均注意力权重超过此值的位置被认为是 sink
        
        Returns:
            sink_positions: [Lk] 布尔掩码
        """
        # 对所有 query 位置和所有 head 取平均
        avg_attn = attn_weights.mean(dim=(0, 1, 2))  # [Lk]
        return avg_attn > threshold
    
    @staticmethod
    def streaming_inference_mask(
        seq_len: int,
        sink_size: int = 4,
        window_size: int = 128,
        device: torch.device = None,
    ) -> torch.Tensor:
        """
        构造 StreamingLLM 风格的注意力掩码：
        保留前 sink_size 个 token + 最近 window_size 个 token。
        
        Returns:
            mask: [1, 1, seq_len, seq_len]
        """
        mask = torch.zeros(seq_len, seq_len, dtype=torch.bool, device=device)
        for i in range(seq_len):
            # 始终保留前 sink_size 个 token
            mask[i, :sink_size] = True
            # 保留最近 window_size 个 token
            start = max(sink_size, i - window_size + 1)
            mask[i, start:i + 1] = True
        return mask.unsqueeze(0).unsqueeze(0)
