"""
v01 Transformer 基础 - 从零实现 Mini Transformer

本文件从零实现 Transformer 的所有核心组件：
1. Scaled Dot-Product Attention
2. Multi-Head Attention
3. Position-wise Feed-Forward Network
4. 正弦位置编码
5. Transformer Encoder Block / Decoder Block
6. 完整的 Encoder-Decoder 模型

所有代码不依赖 nn.TransformerEncoder 等高层封装，目的是让你理解每一步的实现细节。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import TransformerConfig


# ============================================================
# 1. Scaled Dot-Product Attention
#    这是 Transformer 最核心的运算
# ============================================================
def scaled_dot_product_attention(
    query: torch.Tensor,      # [batch, heads, seq_len_q, d_k]
    key: torch.Tensor,        # [batch, heads, seq_len_k, d_k]
    value: torch.Tensor,      # [batch, heads, seq_len_k, d_v]
    mask: torch.Tensor = None,  # [batch, 1, seq_len_q, seq_len_k] 或 broadcastable
    dropout: nn.Dropout = None,
) -> tuple:
    """
    计算缩放点积注意力。
    
    公式: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
    
    Args:
        query: 查询向量
        key: 键向量
        value: 值向量
        mask: 注意力掩码，值为 False/0 的位置会被遮蔽（设为 -inf）
        dropout: 可选的 Dropout 层
    
    Returns:
        (attention_output, attention_weights)
    """
    d_k = query.size(-1)
    
    # 步骤 1: Q * K^T / sqrt(d_k)
    # query: [B, H, Lq, dk], key: [B, H, Lk, dk]
    # scores: [B, H, Lq, Lk] — 每个 query 对每个 key 的注意力分数
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    
    # 步骤 2: 应用掩码（如果有）
    # 被遮蔽的位置设为极小值，这样 softmax 后权重趋近于 0
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    
    # 步骤 3: Softmax 归一化 → 得到注意力权重（每行和为 1）
    attention_weights = F.softmax(scores, dim=-1)
    
    # 步骤 4: Dropout（训练时随机丢弃部分注意力连接，防止过拟合）
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    
    # 步骤 5: 加权求和 Value
    # attention_weights: [B, H, Lq, Lk], value: [B, H, Lk, dv]
    # output: [B, H, Lq, dv]
    output = torch.matmul(attention_weights, value)
    
    return output, attention_weights


# ============================================================
# 2. Multi-Head Attention
#    将注意力拆成多个头，每个头学习不同的关注模式
# ============================================================
class MultiHeadAttention(nn.Module):
    """
    多头注意力机制。
    
    将 d_model 维的 Q、K、V 分别拆成 n_heads 个头，
    每个头的维度为 d_k = d_model / n_heads。
    各头独立计算注意力后，拼接并经过输出投影。
    """
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0, \
            f"d_model ({d_model}) 必须能被 n_heads ({n_heads}) 整除"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 每个头的维度
        
        # Q, K, V 的线性投影层
        # 虽然有 n_heads 个头，但我们用一个大矩阵一次性投影，然后 reshape
        # 这比分别为每个头创建投影层更高效
        self.W_q = nn.Linear(d_model, d_model)  # [d_model, d_model]
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        
        # 输出投影层：将多头拼接后的结果映射回 d_model 维
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,    # [batch, seq_len_q, d_model]
        key: torch.Tensor,      # [batch, seq_len_k, d_model]
        value: torch.Tensor,    # [batch, seq_len_k, d_model]
        mask: torch.Tensor = None,
    ) -> tuple:
        batch_size = query.size(0)
        
        # 步骤 1: 线性投影
        Q = self.W_q(query)  # [B, Lq, d_model]
        K = self.W_k(key)    # [B, Lk, d_model]
        V = self.W_v(value)  # [B, Lk, d_model]
        
        # 步骤 2: 拆分成多头
        # [B, L, d_model] → [B, L, n_heads, d_k] → [B, n_heads, L, d_k]
        # 转置是为了让 head 维度在 seq_len 前面，方便批量矩阵乘法
        Q = Q.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # 步骤 3: 计算注意力（所有头并行计算）
        attn_output, attn_weights = scaled_dot_product_attention(
            Q, K, V, mask=mask, dropout=self.dropout
        )
        # attn_output: [B, n_heads, Lq, d_k]
        
        # 步骤 4: 拼接多头
        # [B, n_heads, Lq, d_k] → [B, Lq, n_heads, d_k] → [B, Lq, d_model]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        # 步骤 5: 输出投影
        output = self.W_o(attn_output)  # [B, Lq, d_model]
        
        return output, attn_weights


# ============================================================
# 3. Position-wise Feed-Forward Network
#    对每个位置独立应用相同的两层全连接
# ============================================================
class FeedForward(nn.Module):
    """
    逐位置前馈网络。
    
    结构: Linear(d_model → d_ff) → Activation → Dropout → Linear(d_ff → d_model)
    
    d_ff 通常是 d_model 的 4 倍，起到"扩展-压缩"的作用，
    让模型在高维空间中进行非线性变换。
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        # 使用 GELU 激活函数（比 ReLU 更平滑，现代模型的主流选择）
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        
        注意：这里的操作是逐位置的（position-wise），
        即每个位置的 d_model 向量独立通过相同的两层网络。
        位置之间的信息交互完全由前面的注意力层负责。
        """
        x = self.linear1(x)     # [B, L, d_ff] — 升维
        x = self.activation(x)  # 非线性激活
        x = self.dropout(x)     # 防止过拟合
        x = self.linear2(x)     # [B, L, d_model] — 降维回原始维度
        return x


# ============================================================
# 4. 正弦位置编码
#    为序列中的每个位置生成唯一的编码向量
# ============================================================
class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码（原始 Transformer 论文的方式）。
    
    对于位置 pos 和维度 i:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    
    这种编码的优点：
    1. 不增加可学习参数
    2. 可以处理任意长度的序列（外推能力好）
    3. 不同维度对应不同频率，编码了多尺度的位置信息
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # 预计算位置编码表（不参与训练）
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        
        # position: [max_len, 1] → 每行是一个位置 (0, 1, 2, ...)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # div_term: [d_model/2] → 频率因子
        # 10000^(2i/d_model) = exp(2i * ln(10000) / d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        
        # 偶数维度用 sin，奇数维度用 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # sin(pos / 10000^(2i/d))
        pe[:, 1::2] = torch.cos(position * div_term)  # cos(pos / 10000^(2i/d))
        
        # [max_len, d_model] → [1, max_len, d_model]，方便与 batch 广播
        pe = pe.unsqueeze(0)
        
        # register_buffer: 保存在模型中但不参与梯度更新
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model] — 经过 embedding 的输入
        
        将位置编码加到输入上（加法，不是拼接）。
        加法的合理性：位置信息和语义信息在同一空间中表示，
        模型会学习如何同时利用两者。
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


# ============================================================
# 5. Transformer Encoder Block
# ============================================================
class TransformerEncoderBlock(nn.Module):
    """
    Transformer 编码器块（Pre-Norm 版本）。
    
    结构:
        x → LayerNorm → MultiHeadAttention → + (残差) → LayerNorm → FFN → + (残差) → output
    
    使用 Pre-Norm 而非 Post-Norm，因为：
    - 训练更稳定
    - 是现代大模型（GPT、LLaMA、Qwen）的主流做法
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 两个 LayerNorm（Pre-Norm 风格）
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: [batch, seq_len, d_model]
        mask: [batch, 1, 1, seq_len] padding mask
        """
        # Sub-layer 1: Self-Attention + 残差
        residual = x
        x = self.norm1(x)  # Pre-Norm: 先归一化
        x, _ = self.self_attention(x, x, x, mask=mask)  # Q=K=V=x（自注意力）
        x = self.dropout1(x)
        x = residual + x    # 残差连接
        
        # Sub-layer 2: FFN + 残差
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout2(x)
        x = residual + x
        
        return x


# ============================================================
# 6. Transformer Decoder Block
# ============================================================
class TransformerDecoderBlock(nn.Module):
    """
    Transformer 解码器块（Pre-Norm 版本）。
    
    与编码器相比，解码器多了一个 Cross-Attention 层：
    1. Masked Self-Attention（因果掩码，防止看到未来）
    2. Cross-Attention（Q 来自解码器，K/V 来自编码器输出）
    3. FFN
    
    每个 sub-layer 都有残差连接和 LayerNorm。
    """
    
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        # 三个注意力/FFN 子层
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        # 三个 LayerNorm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,           # 解码器输入 [B, Lq, d_model]
        encoder_output: torch.Tensor,  # 编码器输出 [B, Lk, d_model]
        self_attn_mask: torch.Tensor = None,   # 因果掩码
        cross_attn_mask: torch.Tensor = None,  # padding 掩码
    ) -> torch.Tensor:
        
        # Sub-layer 1: Masked Self-Attention（带因果掩码）
        residual = x
        x = self.norm1(x)
        x, _ = self.self_attention(x, x, x, mask=self_attn_mask)
        x = self.dropout1(x)
        x = residual + x
        
        # Sub-layer 2: Cross-Attention
        # Query 来自解码器（当前层的输出），Key 和 Value 来自编码器
        # 这让解码器在生成时能"参考"编码器处理过的输入信息
        residual = x
        x = self.norm2(x)
        x, _ = self.cross_attention(
            query=x,
            key=encoder_output,
            value=encoder_output,
            mask=cross_attn_mask,
        )
        x = self.dropout2(x)
        x = residual + x
        
        # Sub-layer 3: FFN
        residual = x
        x = self.norm3(x)
        x = self.feed_forward(x)
        x = self.dropout3(x)
        x = residual + x
        
        return x


# ============================================================
# 7. 完整的 Encoder-Decoder Transformer
# ============================================================
class MiniTransformer(nn.Module):
    """
    完整的 Mini Transformer 模型（Encoder-Decoder 架构）。
    
    用于序列排序任务：
    - 输入：乱序的数字序列 [5, 2, 8, 1, 3]
    - 输出：排好序的序列 [1, 2, 3, 5, 8]
    
    模型结构:
    1. 输入 Embedding + 位置编码 → Encoder → 编码器表示
    2. 目标 Embedding + 位置编码 → Decoder (参考编码器表示) → 输出分布
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # ---- Embedding 层 ----
        # 将 token id 映射为 d_model 维的稠密向量
        # 编码器和解码器可以共享 embedding（因为它们处理同一套词表）
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # 位置编码
        self.pos_encoding = SinusoidalPositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        
        # ---- Encoder ----
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_encoder_layers)
        ])
        self.encoder_norm = nn.LayerNorm(config.d_model)  # 最后一层的归一化
        
        # ---- Decoder ----
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderBlock(
                config.d_model, config.n_heads, config.d_ff, config.dropout
            )
            for _ in range(config.n_decoder_layers)
        ])
        self.decoder_norm = nn.LayerNorm(config.d_model)
        
        # ---- 输出层 ----
        # 将 d_model 维的表示映射回词表大小，用于预测下一个 token
        self.output_projection = nn.Linear(config.d_model, config.vocab_size)
        
        # Embedding 权重缩放因子
        # 论文中提到 embedding 乘以 sqrt(d_model)，有助于在加上位置编码前
        # 让 embedding 的值域与位置编码匹配
        self.d_model_sqrt = math.sqrt(config.d_model)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """Xavier 均匀初始化，这是 Transformer 常用的初始化方式。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(
        self,
        src: torch.Tensor,       # 源序列 [B, Ls]
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        编码器前向传播。
        
        将输入序列编码为上下文感知的表示。
        """
        # Embedding + 缩放 + 位置编码
        x = self.embedding(src) * self.d_model_sqrt
        x = self.pos_encoding(x)
        
        # 依次通过 N 个编码器层
        for layer in self.encoder_layers:
            x = layer(x, mask=src_mask)
        
        # 最终归一化（Pre-Norm 架构需要在最后加一层 Norm）
        x = self.encoder_norm(x)
        
        return x  # [B, Ls, d_model]
    
    def decode(
        self,
        tgt: torch.Tensor,           # 目标序列 [B, Lt]
        encoder_output: torch.Tensor,  # 编码器输出 [B, Ls, d_model]
        tgt_mask: torch.Tensor = None,
        src_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        解码器前向传播。
        
        根据编码器输出和已生成的部分目标序列，预测下一个 token。
        """
        x = self.embedding(tgt) * self.d_model_sqrt
        x = self.pos_encoding(x)
        
        for layer in self.decoder_layers:
            x = layer(
                x,
                encoder_output,
                self_attn_mask=tgt_mask,    # 因果掩码
                cross_attn_mask=src_mask,   # padding 掩码
            )
        
        x = self.decoder_norm(x)
        
        return x  # [B, Lt, d_model]
    
    def forward(
        self,
        src: torch.Tensor,    # 源序列 [B, Ls]（乱序数字）
        tgt: torch.Tensor,    # 目标序列 [B, Lt]（排好序的数字）
        src_mask: torch.Tensor = None,
        tgt_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        完整的前向传播。
        
        Args:
            src: 源序列（编码器输入）
            tgt: 目标序列（解码器输入，训练时是 teacher forcing）
            src_mask: 源序列 padding 掩码
            tgt_mask: 目标序列因果掩码
        
        Returns:
            logits: [B, Lt, vocab_size] — 每个位置的 token 概率分布
        """
        # 编码
        encoder_output = self.encode(src, src_mask)
        
        # 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_mask)
        
        # 投影到词表空间
        logits = self.output_projection(decoder_output)
        
        return logits  # [B, Lt, vocab_size]
    
    @staticmethod
    def generate_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
        """
        生成因果掩码（下三角矩阵）。
        
        确保解码器中位置 i 只能注意到位置 0..i，无法看到未来信息。
        这是自回归生成的核心约束。
        
        返回: [1, 1, seq_len, seq_len]，可与 [B, H, L, L] 广播
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device)).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, L, L]
    
    @staticmethod
    def generate_padding_mask(
        seq: torch.Tensor, pad_token_id: int = 0
    ) -> torch.Tensor:
        """
        生成 padding 掩码。
        
        将 pad token 的位置标记为 False（不参与注意力计算）。
        
        Args:
            seq: [B, L] token ids
            pad_token_id: padding token 的 id
        
        Returns:
            [B, 1, 1, L] 掩码，可与 [B, H, Lq, Lk] 广播
        """
        mask = (seq != pad_token_id).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, L]
        return mask


if __name__ == "__main__":
    # 快速测试模型
    cfg = TransformerConfig()
    model = MiniTransformer(cfg)
    
    # 模拟输入
    src = torch.randint(1, 30, (2, 10))  # [batch=2, seq_len=10]
    tgt = torch.randint(1, 30, (2, 10))
    
    # 创建掩码
    tgt_mask = MiniTransformer.generate_causal_mask(10, src.device)
    
    # 前向传播
    logits = model(src, tgt, tgt_mask=tgt_mask)
    print(f"输入形状: src={src.shape}, tgt={tgt.shape}")
    print(f"输出形状: logits={logits.shape}")  # [2, 10, 32]
    
    # 参数统计
    total = sum(p.numel() for p in model.parameters())
    print(f"模型总参数量: {total:,}")
