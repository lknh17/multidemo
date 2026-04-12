"""
V15 - 视频编码器
================
实现三种主流视频编码方案：
1. TimeSformer：分离时空注意力（Divided Space-Time Attention）
2. Video Swin Transformer：3D 窗口注意力
3. Conv3D (R(2+1)D)：时空分解卷积

参考论文：
- TimeSformer: Is Space-Time Attention All You Need for Video Understanding?
- Video Swin Transformer (CVPR 2022)
- A Closer Look at Spatiotemporal Convolutions for Action Recognition (R(2+1)D)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import VideoEncoderConfig


class PatchEmbedding3D(nn.Module):
    """
    3D Patch Embedding：将视频转为 Token 序列
    
    输入：[B, C, T, H, W]
    输出：[B, T * (H//P) * (W//P), D]
    
    temporal_kernel=1 表示不在 patch 层面做时序融合，
    时序建模交给后续的注意力层处理。
    """

    def __init__(self, config: VideoEncoderConfig):
        super().__init__()
        self.config = config
        self.num_patches_h = config.frame_size // config.patch_size
        self.num_patches_w = config.frame_size // config.patch_size
        self.num_spatial_patches = self.num_patches_h * self.num_patches_w

        # 3D 卷积做 patch embedding，时序维度不做下采样
        self.proj = nn.Conv3d(
            config.in_channels, config.d_model,
            kernel_size=(1, config.patch_size, config.patch_size),
            stride=(1, config.patch_size, config.patch_size)
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))

        # 分离的时序和空间位置编码
        self.temporal_pos_embed = nn.Parameter(
            torch.randn(1, config.num_frames, 1, config.d_model)
        )
        self.spatial_pos_embed = nn.Parameter(
            torch.randn(1, 1, self.num_spatial_patches, config.d_model)
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.temporal_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.spatial_pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, T, H, W] 视频张量
        Returns:
            tokens: [B, T*N+1, D]  (包含 CLS token)
            T, N: 帧数和每帧 patch 数
        """
        B, C, T, H, W = x.shape

        # Patch embedding: [B, D, T, H', W']
        tokens = self.proj(x)
        # 重排为 [B, T, N, D]
        tokens = tokens.permute(0, 2, 3, 4, 1)  # [B, T, H', W', D]
        tokens = tokens.reshape(B, T, -1, self.config.d_model)  # [B, T, N, D]

        N = tokens.shape[2]

        # 加位置编码
        tokens = tokens + self.temporal_pos_embed[:, :T, :, :]
        tokens = tokens + self.spatial_pos_embed[:, :, :N, :]

        # 展平为 [B, T*N, D]
        tokens = tokens.reshape(B, T * N, self.config.d_model)

        # 加 CLS token
        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        return tokens, T, N


class TemporalAttention(nn.Module):
    """
    时序注意力：同一空间位置的 T 帧之间做 Self-Attention
    
    输入 reshape：(B*T, N, D) -> (B*N, T, D)
    即每个空间位置独立地在时间维度上做注意力
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, T: int, N: int) -> torch.Tensor:
        """
        Args:
            x: [B, T*N, D] (不含 CLS)
            T: 帧数
            N: 每帧 patch 数
        """
        B = x.shape[0] // 1  # 已经是 batch 维度
        residual = x

        # reshape 为 [B*N, T, D]：同一空间位置的 T 帧组成序列
        x = x.reshape(B, T, N, -1)          # [B, T, N, D]
        x = x.permute(0, 2, 1, 3)           # [B, N, T, D]
        x = x.reshape(B * N, T, -1)         # [B*N, T, D]

        x, _ = self.attn(x, x, x)

        # reshape 回 [B, T*N, D]
        x = x.reshape(B, N, T, -1)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(B, T * N, -1)

        return self.norm(residual + x)


class SpatialAttention(nn.Module):
    """
    空间注意力：每帧内 N 个 patch 之间做 Self-Attention
    
    输入已是 (B*T, N, D) 的形式，直接做注意力即可
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, T: int, N: int) -> torch.Tensor:
        """
        Args:
            x: [B, T*N, D]
        """
        B = x.shape[0]
        residual = x

        # reshape 为 [B*T, N, D]：每帧的 N 个 patch 组成序列
        x = x.reshape(B, T, N, -1)
        x = x.reshape(B * T, N, -1)

        x, _ = self.attn(x, x, x)

        x = x.reshape(B, T, N, -1)
        x = x.reshape(B, T * N, -1)

        return self.norm(residual + x)


class TimeSformerBlock(nn.Module):
    """
    TimeSformer Block = 时序注意力 + 空间注意力 + FFN
    
    Divided Space-Time Attention 的核心：
    不做 (T*N)^2 的全局注意力，而是分解为 T^2 + N^2
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.temporal_attn = TemporalAttention(d_model, n_heads, dropout)
        self.spatial_attn = SpatialAttention(d_model, n_heads, dropout)

        self.ffn = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, T: int, N: int) -> torch.Tensor:
        # CLS token 不参与时空分离注意力
        cls_token = x[:, :1]
        x_patches = x[:, 1:]

        # 时序注意力
        x_patches = self.temporal_attn(x_patches, T, N)

        # 空间注意力
        x_patches = self.spatial_attn(x_patches, T, N)

        # 合并 CLS
        x = torch.cat([cls_token, x_patches], dim=1)

        # FFN
        x = x + self.ffn(self.norm(x))
        return x


class R21DBlock(nn.Module):
    """
    R(2+1)D 卷积块：空间 2D + 时序 1D 分解
    
    参数量从 t*h*w*C^2 降至 h*w*C^2 + t*C^2
    """

    def __init__(self, in_channels: int, out_channels: int,
                 temporal_kernel: int = 3, spatial_kernel: int = 3):
        super().__init__()
        # 中间维度：保持计算量平衡
        mid_channels = (in_channels * out_channels * spatial_kernel ** 2 * temporal_kernel) // \
                       (in_channels * spatial_kernel ** 2 + out_channels * temporal_kernel)
        mid_channels = max(mid_channels, 1)

        # 空间卷积（不改变时序维度）
        self.spatial_conv = nn.Conv3d(
            in_channels, mid_channels,
            kernel_size=(1, spatial_kernel, spatial_kernel),
            padding=(0, spatial_kernel // 2, spatial_kernel // 2),
            bias=False
        )
        self.spatial_bn = nn.BatchNorm3d(mid_channels)

        # 时序卷积（不改变空间维度）
        self.temporal_conv = nn.Conv3d(
            mid_channels, out_channels,
            kernel_size=(temporal_kernel, 1, 1),
            padding=(temporal_kernel // 2, 0, 0),
            bias=False
        )
        self.temporal_bn = nn.BatchNorm3d(out_channels)

        # 残差连接
        self.residual = nn.Identity() if in_channels == out_channels else \
            nn.Conv3d(in_channels, out_channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x)

        x = self.relu(self.spatial_bn(self.spatial_conv(x)))
        x = self.temporal_bn(self.temporal_conv(x))

        return self.relu(x + residual)


class VideoSwinBlock(nn.Module):
    """
    Video Swin Transformer Block（简化版）
    
    3D 窗口注意力 + Shifted Window
    """

    def __init__(self, d_model: int, n_heads: int, window_size: tuple = (2, 7, 7),
                 shift_size: tuple = (0, 0, 0)):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.shift_size = shift_size

        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

        # 相对位置偏置
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1) * (2 * window_size[2] - 1),
                n_heads
            )
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x: torch.Tensor, T: int, H: int, W: int) -> torch.Tensor:
        """简化实现：不做实际窗口划分，用全局注意力模拟"""
        B, L, D = x.shape
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        x = x + self.ffn(self.norm2(x))
        return x


class VideoEncoder(nn.Module):
    """
    统一视频编码器：支持三种编码方案
    
    - timesformer：分离时空注意力，适合中等长度视频
    - video_swin：3D 窗口注意力，适合高分辨率
    - conv3d：R(2+1)D 卷积，适合短视频/实时场景
    
    输出：[B, T', D] 的时序特征（T' 可能经过下采样）
    """

    def __init__(self, config: VideoEncoderConfig):
        super().__init__()
        self.config = config

        if config.temporal_model == "timesformer":
            self.patch_embed = PatchEmbedding3D(config)
            self.blocks = nn.ModuleList([
                TimeSformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ])
            self.pool_mode = "cls"

        elif config.temporal_model == "video_swin":
            self.patch_embed = PatchEmbedding3D(config)
            self.blocks = nn.ModuleList([
                VideoSwinBlock(config.d_model, config.n_heads)
                for _ in range(config.n_layers)
            ])
            self.pool_mode = "mean"

        elif config.temporal_model == "conv3d":
            channels = [config.in_channels, 64, 128, config.d_model]
            self.conv_blocks = nn.ModuleList()
            for i in range(len(channels) - 1):
                self.conv_blocks.append(R21DBlock(
                    channels[i], channels[i + 1],
                    temporal_kernel=config.temporal_kernel_size
                ))
            # 全局平均池化到时序维度
            self.adaptive_pool = nn.AdaptiveAvgPool3d((config.num_frames, 1, 1))
            self.pool_mode = "conv"

        self.output_norm = nn.LayerNorm(config.d_model)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video: [B, C, T, H, W]
        Returns:
            temporal_features: [B, T, D] 时序特征
        """
        if self.config.temporal_model in ("timesformer", "video_swin"):
            tokens, T, N = self.patch_embed(video)

            for block in self.blocks:
                tokens = block(tokens, T, N)

            # 提取时序特征：对每帧的空间 token 做平均
            # tokens: [B, 1+T*N, D]
            patch_tokens = tokens[:, 1:]  # 去掉 CLS
            B, _, D = patch_tokens.shape
            patch_tokens = patch_tokens.reshape(B, T, N, D)
            temporal_features = patch_tokens.mean(dim=2)  # [B, T, D]

        else:  # conv3d
            x = video
            for block in self.conv_blocks:
                x = block(x)  # [B, D, T, H', W']

            x = self.adaptive_pool(x)  # [B, D, T, 1, 1]
            temporal_features = x.squeeze(-1).squeeze(-1).permute(0, 2, 1)  # [B, T, D]

        return self.output_norm(temporal_features)


class TokenMerging(nn.Module):
    """
    Token Merging（ToMe）for Video：合并相似 token 减少计算量
    
    核心思想：
    1. 将 token 分为两组（交替分组）
    2. 计算组间余弦相似度
    3. 找 top-r 最相似对，加权平均合并
    4. 每层减少 r 个 token
    
    参考：Token Merging: Your ViT But Faster (ICLR 2023)
    """

    def __init__(self, r: int = 8):
        super().__init__()
        self.r = r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, N, D]
        Returns:
            merged: [B, N-r, D]
        """
        B, N, D = x.shape
        r = min(self.r, N // 2)

        if r == 0:
            return x

        # 分组：偶数索引 vs 奇数索引
        a = x[:, ::2]   # [B, N//2, D]
        b = x[:, 1::2]  # [B, N//2, D]

        # 计算余弦相似度
        a_norm = F.normalize(a, dim=-1)
        b_norm = F.normalize(b, dim=-1)
        sim = torch.bmm(a_norm, b_norm.transpose(1, 2))  # [B, N//2, N//2]

        # 找每行最大值（每个 a token 的最佳匹配 b token）
        max_sim, max_idx = sim.max(dim=-1)  # [B, N//2]

        # 选 top-r 个最相似对
        _, top_indices = max_sim.topk(r, dim=-1)  # [B, r]

        # 获取对应的 b 索引
        b_indices = torch.gather(max_idx, 1, top_indices)  # [B, r]

        # 合并：简单平均
        merged_tokens = []
        for batch in range(B):
            a_keep_mask = torch.ones(a.shape[1], dtype=torch.bool, device=x.device)
            b_keep_mask = torch.ones(b.shape[1], dtype=torch.bool, device=x.device)

            merge_list = []
            for i in range(r):
                ai = top_indices[batch, i]
                bi = b_indices[batch, i]
                merged = (a[batch, ai] + b[batch, bi]) / 2
                merge_list.append(merged)
                a_keep_mask[ai] = False
                b_keep_mask[bi] = False

            # 保留未合并的 token + 合并后的 token
            kept = torch.cat([a[batch][a_keep_mask], b[batch][b_keep_mask]], dim=0)
            merged = torch.stack(merge_list, dim=0) if merge_list else torch.empty(0, D, device=x.device)
            merged_tokens.append(torch.cat([kept, merged], dim=0))

        # 填充到相同长度（N - r）
        max_len = max(t.shape[0] for t in merged_tokens)
        result = torch.zeros(B, max_len, D, device=x.device)
        for i, t in enumerate(merged_tokens):
            result[i, :t.shape[0]] = t

        return result


def keyframe_sampling(features: torch.Tensor, num_samples: int,
                      method: str = "uniform") -> torch.Tensor:
    """
    关键帧采样策略
    
    Args:
        features: [B, T, D] 帧特征
        num_samples: 采样帧数
        method: uniform / motion / cluster
    Returns:
        sampled: [B, num_samples, D]
    """
    B, T, D = features.shape
    num_samples = min(num_samples, T)

    if method == "uniform":
        indices = torch.linspace(0, T - 1, num_samples).long()
        return features[:, indices]

    elif method == "motion":
        # 基于帧差异的采样：选变化最大的帧
        diffs = (features[:, 1:] - features[:, :-1]).norm(dim=-1)  # [B, T-1]
        _, top_idx = diffs.topk(num_samples, dim=-1)
        top_idx = top_idx.sort(dim=-1).values  # 保持时序顺序
        return torch.gather(features, 1, top_idx.unsqueeze(-1).expand(-1, -1, D))

    elif method == "cluster":
        # 基于 K-Means 的采样（简化版）
        # 选择最能代表不同内容的帧
        selected = [0]  # 从第一帧开始
        for _ in range(num_samples - 1):
            # 贪心：选与已选帧最不相似的帧
            selected_feats = features[:, selected].mean(dim=1, keepdim=True)  # [B, 1, D]
            dists = (features - selected_feats).norm(dim=-1)  # [B, T]
            for idx in selected:
                dists[:, idx] = -1  # 已选帧设为不可选
            next_idx = dists.argmax(dim=-1)  # [B]
            selected.append(next_idx[0].item())

        indices = torch.tensor(sorted(selected), device=features.device)
        return features[:, indices]

    else:
        raise ValueError(f"Unknown sampling method: {method}")
