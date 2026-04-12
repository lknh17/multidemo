"""
V15 - Dense Video Captioning & Temporal Grounding 模型
=====================================================
核心模块：
1. TemporalProposalNetwork：时序提议网络（Event Query 方式）
2. CaptionDecoder：提议条件描述生成器
3. DeformableTemporalAttention：可变形时序注意力
4. DenseVideoCaptioningModel：端到端 Dense Caption 模型
5. TemporalGroundingModel：时序 Grounding（Moment Retrieval）

参考论文：
- PDVC: End-to-End Dense Video Captioning with Parallel Decoding (ICCV 2021)
- Vid2Seq (CVPR 2023)
- Moment-DETR (NeurIPS 2021)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List

from config import VideoDenseCaptionFullConfig
from video_encoder import VideoEncoder


# ============================================================
#  辅助模块
# ============================================================

def temporal_iou(spans_a: torch.Tensor, spans_b: torch.Tensor) -> torch.Tensor:
    """
    计算时序 IoU
    
    Args:
        spans_a: [N, 2] (start, end)
        spans_b: [M, 2] (start, end)
    Returns:
        iou: [N, M]
    """
    # 交集
    inter_start = torch.max(spans_a[:, 0:1], spans_b[:, 0:1].T)
    inter_end = torch.min(spans_a[:, 1:2], spans_b[:, 1:2].T)
    inter = (inter_end - inter_start).clamp(min=0)

    # 并集
    len_a = (spans_a[:, 1] - spans_a[:, 0]).unsqueeze(1)
    len_b = (spans_b[:, 1] - spans_b[:, 0]).unsqueeze(0)
    union = len_a + len_b - inter

    return inter / (union + 1e-8)


def temporal_giou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    时序 GIoU Loss
    
    Args:
        pred: [N, 2] (start, end)
        target: [N, 2] (start, end)
    """
    # IoU
    inter_start = torch.max(pred[:, 0], target[:, 0])
    inter_end = torch.min(pred[:, 1], target[:, 1])
    inter = (inter_end - inter_start).clamp(min=0)

    pred_len = (pred[:, 1] - pred[:, 0]).clamp(min=0)
    target_len = (target[:, 1] - target[:, 0]).clamp(min=0)
    union = pred_len + target_len - inter
    iou = inter / (union + 1e-8)

    # 最小包围区间
    hull_start = torch.min(pred[:, 0], target[:, 0])
    hull_end = torch.max(pred[:, 1], target[:, 1])
    hull = (hull_end - hull_start).clamp(min=0)

    giou = iou - (hull - union) / (hull + 1e-8)

    return 1 - giou


def temporal_nms(spans: torch.Tensor, scores: torch.Tensor,
                 threshold: float = 0.5, use_soft: bool = True,
                 sigma: float = 0.5) -> List[int]:
    """
    时序 NMS / Soft-NMS
    
    Soft-NMS 优势：密集事件场景中避免误删有效提议
    
    Args:
        spans: [N, 2] (start, end)
        scores: [N] 置信度
        threshold: IoU 阈值（Hard NMS）
        use_soft: 是否使用 Soft-NMS
        sigma: Soft-NMS 高斯衰减参数
    Returns:
        keep: 保留的索引列表
    """
    scores = scores.clone()
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        rest = order[1:]
        ious = temporal_iou(spans[i:i + 1], spans[rest]).squeeze(0)

        if use_soft:
            # Soft-NMS：高斯衰减置信度
            decay = torch.exp(-ious ** 2 / sigma)
            scores[rest] *= decay
            # 移除极低分
            valid = scores[rest] > 0.001
            order = rest[valid]
            # 重排序
            order = order[scores[order].argsort(descending=True)]
        else:
            # Hard NMS
            mask = ious < threshold
            order = rest[mask]

    return keep


# ============================================================
#  Deformable 时序注意力
# ============================================================

class DeformableTemporalAttention(nn.Module):
    """
    可变形时序注意力：每个 query 只关注 K 个动态确定的时间点
    
    核心优势：
    - 复杂度 O(T·K) vs 标准注意力 O(T^2)
    - 采样位置可学习，自适应关注重要时间段
    - 支持多尺度特征
    
    Args:
        d_model: 特征维度
        n_heads: 注意力头数
        n_levels: 多尺度层数
        n_points: 每层采样点数
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_levels: int = 3, n_points: int = 4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_levels = n_levels
        self.n_points = n_points
        self.head_dim = d_model // n_heads

        # 预测偏移量：每个 head 在每个 level 上有 n_points 个偏移
        self.offset_net = nn.Linear(d_model, n_heads * n_levels * n_points)
        # 预测注意力权重
        self.attn_weight_net = nn.Linear(d_model, n_heads * n_levels * n_points)
        # Value 投影
        self.value_proj = nn.Linear(d_model, d_model)
        # 输出投影
        self.output_proj = nn.Linear(d_model, d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.constant_(self.offset_net.weight, 0.0)
        nn.init.constant_(self.offset_net.bias, 0.0)
        nn.init.xavier_uniform_(self.attn_weight_net.weight)
        nn.init.constant_(self.attn_weight_net.bias, 0.0)

    def forward(self, query: torch.Tensor, reference_points: torch.Tensor,
                value_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            query: [B, Q, D]
            reference_points: [B, Q] 参考时间点 (0~1 归一化)
            value_list: List of [B, T_l, D] 多尺度特征
        Returns:
            output: [B, Q, D]
        """
        B, Q, D = query.shape
        n_levels = len(value_list)

        # 预测偏移量和注意力权重
        offsets = self.offset_net(query)  # [B, Q, H*L*K]
        offsets = offsets.reshape(B, Q, self.n_heads, n_levels, self.n_points)

        attn_weights = self.attn_weight_net(query)  # [B, Q, H*L*K]
        attn_weights = attn_weights.reshape(B, Q, self.n_heads, n_levels * self.n_points)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = attn_weights.reshape(B, Q, self.n_heads, n_levels, self.n_points)

        # 对每个尺度采样
        sampled_values = []
        for l, value in enumerate(value_list):
            T_l = value.shape[1]
            v = self.value_proj(value)  # [B, T_l, D]
            v = v.reshape(B, T_l, self.n_heads, self.head_dim)

            # 采样位置
            ref = reference_points.unsqueeze(-1).unsqueeze(-1)  # [B, Q, 1, 1]
            offset = offsets[:, :, :, l, :]  # [B, Q, H, K]
            sampling_pos = ref + offset / T_l  # [B, Q, H, K] 归一化坐标
            sampling_pos = sampling_pos.clamp(0, 1)

            # 1D 双线性插值
            sampling_idx = sampling_pos * (T_l - 1)  # 转为绝对坐标
            idx_floor = sampling_idx.long().clamp(0, T_l - 2)
            idx_ceil = (idx_floor + 1).clamp(max=T_l - 1)
            weight_ceil = sampling_idx - idx_floor.float()
            weight_floor = 1 - weight_ceil

            # 采样 [B, Q, H, K, head_dim]
            v_floor = torch.gather(
                v.unsqueeze(1).expand(-1, Q, -1, -1, -1),
                2,
                idx_floor.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
            )
            v_ceil = torch.gather(
                v.unsqueeze(1).expand(-1, Q, -1, -1, -1),
                2,
                idx_ceil.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
            )
            sampled = weight_floor.unsqueeze(-1) * v_floor + weight_ceil.unsqueeze(-1) * v_ceil
            sampled_values.append(sampled)  # [B, Q, H, K, head_dim]

        # 加权求和
        output = torch.zeros(B, Q, self.n_heads, self.head_dim, device=query.device)
        for l, sampled in enumerate(sampled_values):
            w = attn_weights[:, :, :, l, :]  # [B, Q, H, K]
            output += (w.unsqueeze(-1) * sampled).sum(dim=3)

        output = output.reshape(B, Q, D)
        return self.output_proj(output)


# ============================================================
#  时序提议网络
# ============================================================

class TemporalProposalNetwork(nn.Module):
    """
    Event Query 风格的时序提议网络（PDVC/DETR 风格）
    
    使用 Learnable Event Queries 通过 Cross-Attention
    与视频特征交互，每个 Query 预测一个事件的时间段和置信度
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 2, num_proposals: int = 100):
        super().__init__()
        self.event_queries = nn.Embedding(num_proposals, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }))

        # 输出头：时间段 (center, width) + 置信度
        self.span_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 2),
            nn.Sigmoid(),  # 归一化到 [0, 1]
        )
        self.cls_head = nn.Linear(d_model, 1)

    def forward(self, video_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            video_features: [B, T, D]
        Returns:
            spans: [B, N_q, 2] (start, end) 归一化到 [0, 1]
            logits: [B, N_q] 置信度
            query_feats: [B, N_q, D] Query 特征（给 Caption 用）
        """
        B = video_features.shape[0]
        queries = self.event_queries.weight.unsqueeze(0).expand(B, -1, -1)

        for layer in self.layers:
            # Self-Attention
            q = layer['norm1'](queries)
            queries = queries + layer['self_attn'](q, q, q)[0]

            # Cross-Attention with video
            q = layer['norm2'](queries)
            queries = queries + layer['cross_attn'](q, video_features, video_features)[0]

            # FFN
            queries = queries + layer['ffn'](layer['norm3'](queries))

        # 预测 spans: (center, width) -> (start, end)
        cw = self.span_head(queries)  # [B, N_q, 2]
        center, width = cw[..., 0], cw[..., 1]
        start = (center - width / 2).clamp(0, 1)
        end = (center + width / 2).clamp(0, 1)
        spans = torch.stack([start, end], dim=-1)

        logits = self.cls_head(queries).squeeze(-1)

        return spans, logits, queries


# ============================================================
#  描述生成解码器
# ============================================================

class CaptionDecoder(nn.Module):
    """
    Proposal-conditioned Caption Decoder
    
    给定提议特征，自回归生成自然语言描述
    """

    def __init__(self, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 3, vocab_size: int = 5000,
                 max_len: int = 30, pad_token_id: int = 0):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.pad_token_id = pad_token_id

        self.token_embed = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        self.pos_embed = nn.Embedding(max_len, d_model)

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'cross_attn': nn.MultiheadAttention(d_model, n_heads, batch_first=True),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model),
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model),
                'norm3': nn.LayerNorm(d_model),
            }))

        self.output_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, memory: torch.Tensor,
                causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            tokens: [B, L] token ids
            memory: [B, M, D] 提议特征 / 视频特征
        Returns:
            logits: [B, L, vocab_size]
        """
        B, L = tokens.shape
        pos = torch.arange(L, device=tokens.device).unsqueeze(0)

        x = self.token_embed(tokens) + self.pos_embed(pos)

        # 因果 mask
        if causal_mask is None:
            causal_mask = torch.triu(
                torch.ones(L, L, device=tokens.device), diagonal=1
            ).bool()

        for layer in self.layers:
            # Causal Self-Attention
            q = layer['norm1'](x)
            x = x + layer['self_attn'](q, q, q, attn_mask=causal_mask)[0]

            # Cross-Attention with memory
            q = layer['norm2'](x)
            x = x + layer['cross_attn'](q, memory, memory)[0]

            # FFN
            x = x + layer['ffn'](layer['norm3'](x))

        return self.output_proj(x)


# ============================================================
#  时序 RoI 池化
# ============================================================

class TemporalRoIPool(nn.Module):
    """
    时序 RoI 池化：根据提议的 (start, end) 提取对应区间的视频特征
    
    使用双线性插值实现可微分的时序裁剪
    """

    def __init__(self, output_size: int = 8):
        super().__init__()
        self.output_size = output_size

    def forward(self, video_features: torch.Tensor,
                spans: torch.Tensor) -> torch.Tensor:
        """
        Args:
            video_features: [B, T, D]
            spans: [B, N_q, 2] (start, end) 归一化 [0, 1]
        Returns:
            roi_features: [B, N_q, output_size, D]
        """
        B, T, D = video_features.shape
        N_q = spans.shape[1]

        roi_features = []
        for i in range(N_q):
            start = spans[:, i, 0] * (T - 1)  # [B]
            end = spans[:, i, 1] * (T - 1)     # [B]

            # 在 [start, end] 区间内均匀采样 output_size 个点
            steps = torch.linspace(0, 1, self.output_size, device=spans.device)
            sample_points = start.unsqueeze(1) + steps.unsqueeze(0) * \
                           (end - start).unsqueeze(1)  # [B, output_size]

            # 双线性插值
            idx_floor = sample_points.long().clamp(0, T - 2)
            idx_ceil = (idx_floor + 1).clamp(max=T - 1)
            w_ceil = (sample_points - idx_floor.float()).unsqueeze(-1)
            w_floor = 1 - w_ceil

            feat_floor = torch.gather(
                video_features, 1,
                idx_floor.unsqueeze(-1).expand(-1, -1, D)
            )
            feat_ceil = torch.gather(
                video_features, 1,
                idx_ceil.unsqueeze(-1).expand(-1, -1, D)
            )
            feat = w_floor * feat_floor + w_ceil * feat_ceil  # [B, output_size, D]
            roi_features.append(feat)

        return torch.stack(roi_features, dim=1)  # [B, N_q, output_size, D]


# ============================================================
#  时序特征金字塔
# ============================================================

class TemporalFPN(nn.Module):
    """
    时序特征金字塔网络（FPN for temporal features）
    
    构建多尺度时序特征，不同层级检测不同时长的事件
    """

    def __init__(self, d_model: int = 256, n_levels: int = 3):
        super().__init__()
        self.n_levels = n_levels

        # 下采样层
        self.downsamples = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
            ) for _ in range(n_levels - 1)
        ])

        # 上采样融合层
        self.lateral_convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=1)
            for _ in range(n_levels)
        ])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Args:
            x: [B, T, D]
        Returns:
            feature_pyramid: List of [B, T_l, D] 多尺度特征
        """
        x = x.permute(0, 2, 1)  # [B, D, T]

        # 自底向上构建金字塔
        pyramid = [x]
        for downsample in self.downsamples:
            pyramid.append(downsample(pyramid[-1]))

        # Lateral connections
        pyramid = [conv(feat) for conv, feat in zip(self.lateral_convs, pyramid)]

        # 自顶向下融合
        for i in range(self.n_levels - 2, -1, -1):
            upsampled = F.interpolate(pyramid[i + 1], size=pyramid[i].shape[2])
            pyramid[i] = pyramid[i] + upsampled

        # 转回 [B, T_l, D]
        return [feat.permute(0, 2, 1) for feat in pyramid]


# ============================================================
#  端到端 Dense Video Captioning 模型
# ============================================================

class DenseVideoCaptioningModel(nn.Module):
    """
    端到端 Dense Video Captioning 模型
    
    Pipeline:
    1. VideoEncoder: 视频 -> 时序特征
    2. TemporalFPN: 多尺度时序特征金字塔
    3. TemporalProposalNetwork: 检测事件时间段
    4. TemporalRoIPool: 提取事件区间特征
    5. CaptionDecoder: 生成事件描述
    
    训练损失 = 提议分类 + 时间段回归 + 描述生成
    """

    def __init__(self, config: VideoDenseCaptionFullConfig):
        super().__init__()
        self.config = config

        # 视频编码器
        self.video_encoder = VideoEncoder(config.video)

        # 时序特征金字塔
        self.temporal_fpn = TemporalFPN(
            config.video.d_model,
            config.dense_caption.n_deform_levels
        )

        # 时序提议网络
        self.proposal_net = TemporalProposalNetwork(
            d_model=config.dense_caption.proposal_hidden_dim,
            n_heads=config.dense_caption.caption_n_heads,
            num_proposals=config.dense_caption.num_proposals,
        )

        # 时序 RoI 池化
        self.roi_pool = TemporalRoIPool(output_size=8)

        # 描述解码器
        self.caption_decoder = CaptionDecoder(
            d_model=config.dense_caption.caption_d_model,
            n_heads=config.dense_caption.caption_n_heads,
            n_layers=config.dense_caption.caption_n_layers,
            vocab_size=config.dense_caption.caption_vocab_size,
            max_len=config.dense_caption.max_caption_len,
        )

    def forward(self, video: torch.Tensor,
                caption_tokens: Optional[torch.Tensor] = None,
                gt_spans: Optional[torch.Tensor] = None,
                gt_labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            video: [B, C, T, H, W]
            caption_tokens: [B, N_events, max_cap_len] GT caption tokens（训练时）
            gt_spans: [B, N_events, 2] GT 事件时间段
            gt_labels: [B, N_events] GT 事件标签
        Returns:
            outputs dict
        """
        # 1. 视频编码
        temporal_features = self.video_encoder(video)  # [B, T, D]

        # 2. 时序金字塔
        fpn_features = self.temporal_fpn(temporal_features)

        # 3. 时序提议（使用最细粒度特征）
        pred_spans, pred_logits, query_feats = self.proposal_net(fpn_features[0])

        outputs = {
            'pred_spans': pred_spans,    # [B, N_q, 2]
            'pred_logits': pred_logits,  # [B, N_q]
            'query_feats': query_feats,
        }

        # 4. 如果有 GT caption，做 Teacher Forcing 训练
        if caption_tokens is not None and gt_spans is not None:
            # 使用 GT spans 提取 RoI 特征
            roi_feats = self.roi_pool(temporal_features, gt_spans)
            B, N_events, roi_len, D = roi_feats.shape

            # 展平 batch 和事件维度
            roi_feats_flat = roi_feats.reshape(B * N_events, roi_len, D)
            caption_flat = caption_tokens.reshape(B * N_events, -1)

            # 描述解码
            caption_logits = self.caption_decoder(caption_flat, roi_feats_flat)
            outputs['caption_logits'] = caption_logits.reshape(
                B, N_events, -1, self.config.dense_caption.caption_vocab_size
            )

        return outputs

    def compute_loss(self, outputs: Dict, gt_spans: torch.Tensor,
                     gt_labels: torch.Tensor,
                     caption_tokens: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        计算联合训练损失
        """
        cfg = self.config.dense_caption
        losses = {}

        # 提议分类损失（Focal Loss 简化版）
        pred_logits = outputs['pred_logits']
        # 使用匈牙利匹配找最优配对
        # 简化：用最近的预测匹配 GT
        pred_spans = outputs['pred_spans']

        # 分类损失：BCE
        target_cls = torch.zeros_like(pred_logits)
        if gt_labels is not None:
            # 为每个 GT 找最近的预测
            for b in range(pred_spans.shape[0]):
                n_gt = (gt_labels[b] >= 0).sum()
                if n_gt == 0:
                    continue
                ious = temporal_iou(pred_spans[b], gt_spans[b, :n_gt])
                max_iou, max_idx = ious.max(dim=1)
                target_cls[b][max_iou > 0.5] = 1.0

        losses['cls_loss'] = F.binary_cross_entropy_with_logits(
            pred_logits, target_cls
        ) * cfg.proposal_cls_weight

        # 回归损失（对匹配上的提议）
        matched_mask = target_cls > 0.5
        if matched_mask.any():
            matched_pred = pred_spans[matched_mask]
            # 找对应的 GT
            matched_gt = []
            for b in range(pred_spans.shape[0]):
                mask_b = matched_mask[b]
                if mask_b.any():
                    n_gt = (gt_labels[b] >= 0).sum()
                    ious = temporal_iou(pred_spans[b][mask_b], gt_spans[b, :n_gt])
                    _, gt_idx = ious.max(dim=1)
                    matched_gt.append(gt_spans[b, gt_idx])
            if matched_gt:
                matched_gt = torch.cat(matched_gt, dim=0)
                losses['reg_loss'] = (
                    F.l1_loss(matched_pred, matched_gt) +
                    temporal_giou_loss(matched_pred, matched_gt).mean()
                ) * cfg.proposal_reg_weight
            else:
                losses['reg_loss'] = torch.tensor(0.0, device=pred_spans.device)
        else:
            losses['reg_loss'] = torch.tensor(0.0, device=pred_spans.device)

        # Caption 损失
        if 'caption_logits' in outputs and caption_tokens is not None:
            logits = outputs['caption_logits']  # [B, N_events, L, V]
            targets = caption_tokens[:, :, 1:]  # shift right
            logits = logits[:, :, :-1]          # align

            losses['caption_loss'] = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                ignore_index=self.config.dense_caption.pad_token_id,
            ) * cfg.caption_weight

        losses['total'] = sum(losses.values())
        return losses


# ============================================================
#  时序 Grounding 模型
# ============================================================

class TemporalGroundingModel(nn.Module):
    """
    时序 Grounding / Moment Retrieval 模型
    
    给定文本查询，在视频中定位对应的时间段
    
    架构：Moment-DETR 风格
    1. 视频编码 + 文本编码
    2. Cross-Modal Fusion
    3. Moment Query Decoder
    4. Span 回归
    """

    def __init__(self, config: VideoDenseCaptionFullConfig):
        super().__init__()
        cfg_tg = config.temporal_grounding
        cfg_v = config.video

        # 视频编码器（共享）
        self.video_encoder = VideoEncoder(cfg_v)

        # 文本编码器（简单 Transformer）
        self.text_embed = nn.Embedding(cfg_tg.vocab_size, cfg_tg.text_d_model)
        self.text_pos = nn.Embedding(cfg_tg.max_text_len, cfg_tg.text_d_model)
        self.text_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                cfg_tg.text_d_model, cfg_tg.n_heads, cfg_tg.d_model * 4,
                batch_first=True
            ),
            num_layers=2
        )

        # 模态投影对齐
        self.video_proj = nn.Linear(cfg_v.d_model, cfg_tg.d_model)
        self.text_proj = nn.Linear(cfg_tg.text_d_model, cfg_tg.d_model)

        # Cross-Modal Fusion
        self.fusion_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                cfg_tg.d_model, cfg_tg.n_heads, cfg_tg.d_model * 4,
                batch_first=True
            ) for _ in range(cfg_tg.n_layers)
        ])

        # Moment Queries
        self.moment_queries = nn.Embedding(cfg_tg.num_moment_queries, cfg_tg.d_model)

        # 输出头
        self.span_head = nn.Sequential(
            nn.Linear(cfg_tg.d_model, cfg_tg.d_model),
            nn.ReLU(),
            nn.Linear(cfg_tg.d_model, 2),
            nn.Sigmoid(),
        )
        self.cls_head = nn.Linear(cfg_tg.d_model, 1)

    def forward(self, video: torch.Tensor, text_tokens: torch.Tensor,
                text_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            video: [B, C, T, H, W]
            text_tokens: [B, L] 文本 token ids
            text_mask: [B, L] padding mask
        Returns:
            pred_spans: [B, N_q, 2]
            pred_logits: [B, N_q]
        """
        B = video.shape[0]

        # 视频编码
        video_feat = self.video_encoder(video)  # [B, T, D_v]
        video_feat = self.video_proj(video_feat)  # [B, T, D]

        # 文本编码
        L = text_tokens.shape[1]
        pos = torch.arange(L, device=text_tokens.device).unsqueeze(0)
        text_feat = self.text_embed(text_tokens) + self.text_pos(pos)
        text_feat = self.text_encoder(text_feat)  # [B, L, D_t]
        text_feat = self.text_proj(text_feat)  # [B, L, D]

        # 拼接视频+文本作为 memory
        memory = torch.cat([video_feat, text_feat], dim=1)  # [B, T+L, D]

        # Moment Query Decoder
        queries = self.moment_queries.weight.unsqueeze(0).expand(B, -1, -1)
        for layer in self.fusion_layers:
            queries = layer(queries, memory)

        # 输出
        cw = self.span_head(queries)
        center, width = cw[..., 0], cw[..., 1]
        start = (center - width / 2).clamp(0, 1)
        end = (center + width / 2).clamp(0, 1)

        return {
            'pred_spans': torch.stack([start, end], dim=-1),
            'pred_logits': self.cls_head(queries).squeeze(-1),
        }

    def compute_loss(self, outputs: Dict, gt_spans: torch.Tensor,
                     gt_labels: torch.Tensor) -> Dict[str, torch.Tensor]:
        """时序 Grounding 损失"""
        cfg = self.config.temporal_grounding if hasattr(self, 'config') else None

        pred_spans = outputs['pred_spans']
        pred_logits = outputs['pred_logits']

        # 简化匹配：每个 GT span 找最近的预测
        losses = {}
        target_cls = torch.zeros_like(pred_logits)

        for b in range(pred_spans.shape[0]):
            n_gt = (gt_labels[b] >= 0).sum()
            if n_gt == 0:
                continue
            ious = temporal_iou(pred_spans[b], gt_spans[b, :n_gt])
            max_iou, _ = ious.max(dim=1)
            target_cls[b][max_iou > 0.3] = 1.0

        losses['cls_loss'] = F.binary_cross_entropy_with_logits(pred_logits, target_cls)

        # Span 回归
        matched = target_cls > 0.5
        if matched.any():
            matched_pred = pred_spans[matched]
            matched_gt_list = []
            for b in range(pred_spans.shape[0]):
                mask_b = matched[b]
                if mask_b.any():
                    n_gt = (gt_labels[b] >= 0).sum()
                    ious = temporal_iou(pred_spans[b][mask_b], gt_spans[b, :n_gt])
                    _, gt_idx = ious.max(dim=1)
                    matched_gt_list.append(gt_spans[b, gt_idx])

            if matched_gt_list:
                matched_gt = torch.cat(matched_gt_list, dim=0)
                losses['span_loss'] = (
                    F.l1_loss(matched_pred, matched_gt) +
                    temporal_giou_loss(matched_pred, matched_gt).mean()
                )
            else:
                losses['span_loss'] = torch.tensor(0.0, device=pred_spans.device)
        else:
            losses['span_loss'] = torch.tensor(0.0, device=pred_spans.device)

        losses['total'] = sum(losses.values())
        return losses
