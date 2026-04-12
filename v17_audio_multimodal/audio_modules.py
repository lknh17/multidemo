"""
V17 - 音频特征提取与 AST 编码器
================================
核心模块：
1. MelSpectrogramExtractor：Mel 频谱图提取
2. AudioPatchEmbed：音频 Patch 化
3. AudioSpectrogramTransformer：AST 编码器
4. AudioEventDetector：音频事件检测

参考：
- AST (Interspeech 2021)
- Wav2Vec 2.0 (NeurIPS 2020)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

from config import MelSpectrogramConfig, AudioEncoderConfig


# ============================================================
#  Mel 频谱图提取
# ============================================================

class MelSpectrogramExtractor(nn.Module):
    """
    简化版 Mel 频谱图提取器
    
    真实场景使用 torchaudio.transforms.MelSpectrogram
    这里用可训练的 Mel 滤波器演示原理
    """

    def __init__(self, config: MelSpectrogramConfig):
        super().__init__()
        self.config = config
        self.n_fft = config.n_fft
        self.hop_length = config.hop_length
        self.n_mels = config.n_mels

        # 注册 Mel 滤波器组（简化版）
        mel_filters = self._create_mel_filters(
            config.n_fft // 2 + 1, config.n_mels,
            config.f_min, config.f_max, config.sample_rate
        )
        self.register_buffer('mel_filters', mel_filters)

    def _hz_to_mel(self, freq: float) -> float:
        return 2595.0 * math.log10(1.0 + freq / 700.0)

    def _mel_to_hz(self, mel: float) -> float:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    def _create_mel_filters(self, n_freqs: int, n_mels: int,
                            f_min: float, f_max: float,
                            sample_rate: int) -> torch.Tensor:
        """创建三角 Mel 滤波器组"""
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
        hz_points = torch.tensor([self._mel_to_hz(m.item()) for m in mel_points])
        bin_points = (hz_points * n_freqs * 2 / sample_rate).long().clamp(0, n_freqs - 1)

        filters = torch.zeros(n_mels, n_freqs)
        for i in range(n_mels):
            left = bin_points[i]
            center = bin_points[i + 1]
            right = bin_points[i + 2]
            # 上升沿
            if center > left:
                filters[i, left:center] = torch.linspace(0, 1, center - left)
            # 下降沿
            if right > center:
                filters[i, center:right] = torch.linspace(1, 0, right - center)

        return filters  # [n_mels, n_freqs]

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Args:
            waveform: [B, T] 原始音频
        Returns:
            log_mel: [B, n_mels, T'] log-Mel 频谱图
        """
        # STFT
        window = torch.hann_window(self.n_fft, device=waveform.device)
        stft = torch.stft(
            waveform, self.n_fft, self.hop_length,
            window=window, return_complex=True
        )  # [B, F, T']

        # 功率谱
        power = stft.abs() ** 2  # [B, F, T']

        # Mel 滤波
        mel = torch.matmul(self.mel_filters, power)  # [B, n_mels, T']

        # 对数
        log_mel = torch.log(mel + 1e-9)

        return log_mel


# ============================================================
#  音频 Patch Embedding
# ============================================================

class AudioPatchEmbed(nn.Module):
    """将 Mel 频谱图切分为 2D patch 并投影"""

    def __init__(self, n_mels: int, time_patch: int, freq_patch: int, d_model: int):
        super().__init__()
        self.proj = nn.Conv2d(
            1, d_model,
            kernel_size=(freq_patch, time_patch),
            stride=(freq_patch, time_patch)
        )
        self.n_freq_patches = n_mels // freq_patch

    def forward(self, mel: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        """
        Args:
            mel: [B, n_mels, T'] Mel 频谱
        Returns:
            patches: [B, N, D]
            n_freq_patches, n_time_patches
        """
        x = mel.unsqueeze(1)  # [B, 1, n_mels, T']
        x = self.proj(x)      # [B, D, n_freq, n_time]
        n_freq = x.shape[2]
        n_time = x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        return x, n_freq, n_time


# ============================================================
#  AST 编码器
# ============================================================

class ASTBlock(nn.Module):
    """AST Transformer Block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x

        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = residual + x
        return x


class AudioSpectrogramTransformer(nn.Module):
    """
    Audio Spectrogram Transformer
    
    将 Mel 频谱图视为 2D "图像"，用 ViT 风格架构编码
    支持可变长度输入
    """

    def __init__(self, config: AudioEncoderConfig):
        super().__init__()
        self.config = config

        # Patch Embedding
        self.patch_embed = AudioPatchEmbed(
            config.n_mels, config.time_patch_size,
            config.freq_patch_size, config.d_model
        )

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # 位置编码（频率维 + 时间维分离）
        n_freq_patches = config.n_mels // config.freq_patch_size
        self.freq_pos_embed = nn.Parameter(torch.zeros(1, n_freq_patches, config.d_model))
        self.time_pos_embed = nn.Parameter(torch.zeros(1, config.max_time_patches, config.d_model))
        self.cls_pos_embed = nn.Parameter(torch.zeros(1, 1, config.d_model))

        nn.init.trunc_normal_(self.freq_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_pos_embed, std=0.02)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            ASTBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: [B, n_mels, T'] Mel 频谱
        Returns:
            audio_repr: [B, D] 音频表征
        """
        B = mel_spec.shape[0]

        # Patch Embedding
        patches, n_freq, n_time = self.patch_embed(mel_spec)  # [B, N, D]

        # 2D 位置编码
        freq_pe = self.freq_pos_embed[:, :n_freq]        # [1, Nf, D]
        time_pe = self.time_pos_embed[:, :n_time]         # [1, Nt, D]
        # 外积拼接
        pos_embed = (freq_pe.unsqueeze(2) + time_pe.unsqueeze(1))  # [1, Nf, Nt, D]
        pos_embed = pos_embed.reshape(1, -1, self.config.d_model)  # [1, N, D]
        pos_embed = pos_embed[:, :patches.shape[1]]

        # CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # [B, 1+N, D]

        # 添加位置编码
        cls_pe = self.cls_pos_embed.expand(B, -1, -1)
        pos_embed = pos_embed.expand(B, -1, -1)
        pe = torch.cat([cls_pe, pos_embed], dim=1)
        x = x + pe

        # Transformer 编码
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        audio_repr = x[:, 0]  # CLS token

        return audio_repr

    def forward_features(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """返回所有 token 的特征（用于帧级任务）"""
        B = mel_spec.shape[0]
        patches, n_freq, n_time = self.patch_embed(mel_spec)

        freq_pe = self.freq_pos_embed[:, :n_freq]
        time_pe = self.time_pos_embed[:, :n_time]
        pos_embed = (freq_pe.unsqueeze(2) + time_pe.unsqueeze(1))
        pos_embed = pos_embed.reshape(1, -1, self.config.d_model)
        pos_embed = pos_embed[:, :patches.shape[1]]

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)

        cls_pe = self.cls_pos_embed.expand(B, -1, -1)
        pos_embed = pos_embed.expand(B, -1, -1)
        pe = torch.cat([cls_pe, pos_embed], dim=1)
        x = x + pe

        for block in self.blocks:
            x = block(x)

        return self.norm(x)  # [B, 1+N, D]


# ============================================================
#  音频事件检测
# ============================================================

class AudioEventDetector(nn.Module):
    """
    音频事件检测器
    
    同时输出帧级和片段级预测
    """

    def __init__(self, d_model: int, num_classes: int):
        super().__init__()
        self.frame_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes),
        )
        self.attn_pool = nn.Linear(d_model, 1)
        self.clip_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, T, D] 帧级特征（不含 CLS）
        Returns:
            frame_logits: [B, T, C]
            clip_logits: [B, C]
        """
        frame_logits = self.frame_head(features)

        # 注意力池化
        attn_w = F.softmax(self.attn_pool(features), dim=1)  # [B, T, 1]
        pooled = (features * attn_w).sum(dim=1)               # [B, D]
        clip_logits = self.clip_head(pooled)

        return frame_logits, clip_logits
