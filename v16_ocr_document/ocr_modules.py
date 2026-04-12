"""
V16 - OCR 核心模块
==================
1. DBNet 风格文字检测器（可微二值化）
2. CRNN + Attention 文字识别器
3. CTC 解码工具

参考：
- DBNet: Real-time Scene Text Detection with Differentiable Binarization (AAAI 2020)
- CRNN: An End-to-End Trainable Neural Network for Image-based Sequence Recognition
- ABINet: Read Like Humans: Autonomous, Bidirectional and Iterative Language Modeling (CVPR 2021)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from config import OCRDetectionConfig, OCRRecognitionConfig


# ============================================================
#  简易 CNN Backbone
# ============================================================

class SimpleCNNBackbone(nn.Module):
    """轻量级 CNN 骨干网络，输出多尺度特征"""

    def __init__(self, in_channels: int = 3, channels: List[int] = None):
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]

        layers = []
        prev_c = in_channels
        for c in channels:
            layers.append(nn.Sequential(
                nn.Conv2d(prev_c, c, 3, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
                nn.Conv2d(c, c, 3, stride=2, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ))
            prev_c = c
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回多尺度特征 [1/2, 1/4, 1/8]"""
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return features


# ============================================================
#  DBNet 文字检测
# ============================================================

class DBNetHead(nn.Module):
    """
    DBNet 检测头
    
    输出三个 Map：
    - Probability Map: 每个像素是否属于文字区域
    - Threshold Map: 每个像素的自适应阈值
    - Binary Map: 可微二值化结果（仅训练使用）
    """

    def __init__(self, in_channels: int, inner_channels: int = 256):
        super().__init__()

        # FPN 融合
        self.reduce_layers = nn.ModuleList([
            nn.Conv2d(c, inner_channels, 1)
            for c in [in_channels] * 3  # 假设3个尺度通道相同
        ])

        # Probability Map Head
        self.prob_head = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, stride=2),
            nn.Conv2d(inner_channels // 4, 1, 1),
            nn.Sigmoid(),
        )

        # Threshold Map Head
        self.thresh_head = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels // 4, 3, padding=1),
            nn.BatchNorm2d(inner_channels // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(inner_channels // 4, inner_channels // 4, 2, stride=2),
            nn.Conv2d(inner_channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: List[torch.Tensor]) -> dict:
        """
        Args:
            features: 多尺度特征列表
        Returns:
            prob_map, thresh_map, binary_map
        """
        # 使用最细粒度特征（简化版）
        x = features[0]

        prob_map = self.prob_head(x)        # [B, 1, H', W']
        thresh_map = self.thresh_head(x)    # [B, 1, H', W']

        # 可微二值化: sigmoid(k * (P - T))
        binary_map = self.differentiable_binarize(prob_map, thresh_map, k=50)

        return {
            'prob_map': prob_map.squeeze(1),
            'thresh_map': thresh_map.squeeze(1),
            'binary_map': binary_map.squeeze(1),
        }

    @staticmethod
    def differentiable_binarize(prob: torch.Tensor, thresh: torch.Tensor,
                                 k: int = 50) -> torch.Tensor:
        """
        可微二值化：DBNet 的核心创新
        
        当 k → ∞ 时等价于 step function，但 k=50 时仍可微分
        梯度：k * sigmoid(k(p-t)) * (1 - sigmoid(k(p-t)))
        """
        return torch.sigmoid(k * (prob - thresh))


class TextDetector(nn.Module):
    """端到端文字检测器 = CNN Backbone + DBNet Head"""

    def __init__(self, config: OCRDetectionConfig):
        super().__init__()
        self.backbone = SimpleCNNBackbone(3, config.backbone_channels)
        self.head = DBNetHead(config.backbone_channels[-1], config.d_model)
        self.det_threshold = config.det_threshold

    def forward(self, images: torch.Tensor) -> dict:
        features = self.backbone(images)
        return self.head(features)

    def compute_loss(self, outputs: dict, gt_prob: torch.Tensor,
                     gt_thresh: torch.Tensor) -> dict:
        losses = {}
        losses['prob_loss'] = F.binary_cross_entropy(
            outputs['prob_map'], gt_prob
        )
        # Threshold loss 只在文字区域内计算
        mask = gt_prob > 0.5
        if mask.any():
            losses['thresh_loss'] = F.l1_loss(
                outputs['thresh_map'][mask], gt_thresh[mask]
            )
        else:
            losses['thresh_loss'] = torch.tensor(0.0, device=gt_prob.device)

        losses['binary_loss'] = F.binary_cross_entropy(
            outputs['binary_map'], gt_prob
        )
        losses['total'] = losses['prob_loss'] + losses['thresh_loss'] + losses['binary_loss']
        return losses


# ============================================================
#  文字识别
# ============================================================

class TextRecognitionCNN(nn.Module):
    """文字识别的 CNN 特征提取器"""

    def __init__(self, in_channels: int = 1, d_model: int = 256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),  # H/4
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, d_model, 3, padding=1),
            nn.BatchNorm2d(d_model),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # 高度压缩为1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W] 文字图像
        Returns:
            features: [B, W', D] 序列特征
        """
        x = self.cnn(x)        # [B, D, 1, W']
        x = x.squeeze(2)       # [B, D, W']
        x = x.permute(0, 2, 1) # [B, W', D]
        return x


class CTCRecognizer(nn.Module):
    """CTC 文字识别器 = CNN + BiLSTM + CTC"""

    def __init__(self, config: OCRRecognitionConfig):
        super().__init__()
        self.cnn = TextRecognitionCNN(1, config.d_model)
        self.rnn = nn.LSTM(
            config.d_model, config.d_model // 2,
            num_layers=2, bidirectional=True, batch_first=True
        )
        # +1 for CTC blank
        self.classifier = nn.Linear(config.d_model, config.vocab_size + 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.cnn(images)            # [B, W, D]
        rnn_out, _ = self.rnn(features)        # [B, W, D]
        logits = self.classifier(rnn_out)      # [B, W, V+1]
        return logits

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                     input_lengths: torch.Tensor,
                     target_lengths: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)
        # CTC 需要 (T, B, C) 格式
        return F.ctc_loss(
            log_probs.permute(1, 0, 2),
            targets, input_lengths, target_lengths,
            blank=0, zero_infinity=True,
        )


class AttentionRecognizer(nn.Module):
    """
    Attention 文字识别器 = CNN + Transformer Decoder
    
    相比 CTC，可以建模字符间依赖关系
    """

    def __init__(self, config: OCRRecognitionConfig):
        super().__init__()
        self.config = config
        self.cnn = TextRecognitionCNN(1, config.d_model)

        self.char_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_text_len, config.d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            config.d_model, config.n_heads, config.d_model * 4,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, config.n_layers)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, images: torch.Tensor,
                targets: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Teacher Forcing 模式（训练）或自回归解码（推理）
        """
        memory = self.cnn(images)  # [B, W, D]
        B = memory.shape[0]

        if targets is not None:
            # Teacher Forcing
            L = targets.shape[1]
            pos = torch.arange(L, device=targets.device).unsqueeze(0)
            tgt = self.char_embed(targets) + self.pos_embed(pos)

            causal_mask = torch.triu(
                torch.ones(L, L, device=targets.device), diagonal=1
            ).bool()

            output = self.decoder(tgt, memory, tgt_mask=causal_mask)
            return self.output_proj(output)

        else:
            # 自回归解码
            tokens = torch.full((B, 1), self.config.bos_token_id,
                               dtype=torch.long, device=memory.device)
            for _ in range(self.config.max_text_len - 1):
                L = tokens.shape[1]
                pos = torch.arange(L, device=memory.device).unsqueeze(0)
                tgt = self.char_embed(tokens) + self.pos_embed(pos)

                causal_mask = torch.triu(
                    torch.ones(L, L, device=memory.device), diagonal=1
                ).bool()

                output = self.decoder(tgt, memory, tgt_mask=causal_mask)
                next_token = self.output_proj(output[:, -1]).argmax(-1, keepdim=True)
                tokens = torch.cat([tokens, next_token], dim=1)

                if (next_token == self.config.eos_token_id).all():
                    break

            return tokens


def ctc_greedy_decode(logits: torch.Tensor, blank_id: int = 0) -> List[List[int]]:
    """
    CTC 贪心解码：取 argmax → 去重复 → 去空白
    """
    preds = logits.argmax(dim=-1)  # [B, T]
    results = []

    for pred in preds:
        decoded = []
        prev = -1
        for p in pred.tolist():
            if p != blank_id and p != prev:
                decoded.append(p)
            prev = p
        results.append(decoded)

    return results
