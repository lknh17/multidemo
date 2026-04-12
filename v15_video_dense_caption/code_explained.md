# V15 - 视频理解与 Dense Captioning：代码详解

## 1. 视频编码器（video_encoder.py）

### 1.1 TimeSformer 分离时空注意力

核心实现是将标准的全局自注意力拆为两步：

```python
# 时序注意力：每个空间位置独立，在 T 帧之间做注意力
x_t = rearrange(x, '(B T) N D -> (B N) T D', T=T)
x_t = self.temporal_attn(x_t)   # [B*N, T, D]
x_t = rearrange(x_t, '(B N) T D -> (B T) N D', N=N)

# 空间注意力：每帧独立，在 N 个 patch 之间做注意力
x_s = self.spatial_attn(x)      # [B*T, N, D]
```

**关键理解**：`rearrange` 操作的本质是改变"谁和谁做注意力"。时序注意力中，同一空间位置的 T 帧组成一个序列；空间注意力中，同一帧的 N 个 patch 组成一个序列。

### 1.2 帧 Embedding

```python
# 3D Patch Embedding：将 (T, C, H, W) 的视频转为 Token 序列
# 使用 Conv3d(in_channels, d_model, kernel=(1, patch_size, patch_size))
# temporal_kernel=1 表示不在 patch 层面做时序融合
patches = self.patch_embed(video)  # [B, D, T, H//P, W//P]
patches = rearrange(patches, 'B D T H W -> B (T H W) D')

# 加入时序位置编码 + 空间位置编码
patches += self.temporal_pos_embed  # [1, T, 1, D] broadcast
patches += self.spatial_pos_embed   # [1, 1, N, D] broadcast
```

### 1.3 3D 卷积编码器备选方案

```python
# R(2+1)D 分解：先空间卷积再时序卷积
self.spatial_conv = nn.Conv3d(C, C, kernel_size=(1, 3, 3), padding=(0, 1, 1))
self.temporal_conv = nn.Conv3d(C, C, kernel_size=(3, 1, 1), padding=(1, 0, 0))
x = F.relu(self.spatial_conv(x))
x = F.relu(self.temporal_conv(x))
```

## 2. Dense Video Captioning 模型（model.py）

### 2.1 时序提议网络

**Event Query 方式（PDVC 风格）**：

```python
# Learnable Event Queries
self.event_queries = nn.Embedding(num_proposals, d_model)

# 与视频特征做 Cross-Attention
for layer in self.proposal_decoder_layers:
    queries = layer(
        tgt=queries,           # Event Queries
        memory=video_features, # 视频编码特征
    )

# 每个 Query 回归时间段和置信度
spans = self.span_head(queries).sigmoid()  # [B, N_q, 2] -> (center, width)
logits = self.cls_head(queries)            # [B, N_q, 1]

# 从 (center, width) 转换为 (start, end)
start = spans[..., 0] - spans[..., 1] / 2
end = spans[..., 0] + spans[..., 1] / 2
```

### 2.2 匈牙利匹配

```python
def hungarian_match(pred_spans, pred_logits, gt_spans, gt_labels):
    # 计算 cost matrix
    cost_cls = -pred_logits[:, gt_labels == 1]  # 分类代价
    cost_span = torch.cdist(pred_spans, gt_spans, p=1)  # L1 距离
    cost_giou = -temporal_giou(pred_spans, gt_spans)     # GIoU 代价

    C = cost_cls + 5 * cost_span + 2 * cost_giou
    indices = linear_sum_assignment(C.cpu().numpy())  # scipy 求解
    return indices
```

### 2.3 描述生成解码器

```python
# Proposal-conditioned Caption：提取提议对应区间的视频特征
def extract_proposal_features(video_feat, spans):
    # spans: [B, N_q, 2] (start, end)
    # 通过 RoI-like 池化提取每个提议的特征
    proposal_feats = []
    for i in range(spans.shape[1]):
        s, e = spans[:, i, 0], spans[:, i, 1]
        # 双线性插值采样该时间段的特征
        feat = temporal_roi_pool(video_feat, s, e, output_size=8)
        proposal_feats.append(feat)
    return torch.stack(proposal_feats, dim=1)

# 自回归解码
for t in range(max_len):
    logits = caption_decoder(
        tgt=token_embeds[:, :t+1],
        memory=proposal_feat,  # 提议特征作为 memory
    )
    next_token = logits[:, -1].argmax(-1)
```

## 3. Deformable 时序注意力

```python
class DeformableTemporalAttention(nn.Module):
    def forward(self, query, reference_points, value, spatial_shapes):
        # query: [B, Q, D]
        # reference_points: [B, Q, L, 2] -> (中心, 宽度)

        # 预测偏移量和注意力权重
        offsets = self.offset_net(query)     # [B, Q, L*K*1]
        attn_weights = self.attn_net(query)  # [B, Q, L*K]
        attn_weights = F.softmax(attn_weights, dim=-1)

        # 计算采样位置
        sampling_points = reference_points + offsets  # [B, Q, L, K, 1]

        # 在每个尺度上采样特征
        sampled = bilinear_sample_1d(value, sampling_points)  # [B, Q, L*K, D]

        # 加权求和
        output = (attn_weights.unsqueeze(-1) * sampled).sum(dim=2)
        return output
```

**关键点**：偏移量是从 query 预测的，这意味着每个 query 学习"看哪里"。

## 4. 时序 NMS 实现

```python
def temporal_nms(proposals, scores, threshold=0.5, use_soft=True, sigma=0.5):
    """
    proposals: [N, 2] (start, end)
    scores: [N]
    """
    order = scores.argsort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0]
        keep.append(i)
        if order.numel() == 1:
            break

        # 计算与剩余提议的 tIoU
        ious = temporal_iou(proposals[i:i+1], proposals[order[1:]])

        if use_soft:
            # Soft-NMS: 高斯衰减
            decay = torch.exp(-ious ** 2 / sigma)
            scores[order[1:]] *= decay
            # 重新排序
            order = order[1:][scores[order[1:]].argsort(descending=True)]
            order = order[scores[order] > 0.001]  # 过滤极低分
        else:
            # Hard NMS
            mask = ious.squeeze(0) < threshold
            order = order[1:][mask]

    return keep
```

Soft-NMS 的优势：密集事件场景中（如快速切换的动作），避免因 IoU 略高就完全删除有效提议。

## 5. Token Merging 长视频优化

```python
class TokenMerging(nn.Module):
    def __init__(self, r=8):
        self.r = r  # 每层合并 r 对 token

    def forward(self, x, token_counts=None):
        B, N, D = x.shape
        if token_counts is None:
            token_counts = torch.ones(B, N, device=x.device)

        # 将 token 分为两组（交替分组）
        a, b = x[:, ::2], x[:, 1::2]
        ca, cb = token_counts[:, ::2], token_counts[:, 1::2]

        # 计算相似度
        sim = F.cosine_similarity(a.unsqueeze(2), b.unsqueeze(1), dim=-1)

        # 找 top-r 最相似配对
        _, indices = sim.flatten(-2).topk(self.r, dim=-1)
        ai = indices // b.shape[1]
        bi = indices % b.shape[1]

        # 加权合并
        merged = (ca[ai] * a[ai] + cb[bi] * b[bi]) / (ca[ai] + cb[bi])

        # 构建合并后的序列（移除被合并的 token，加入合并结果）
        # ... 详见完整实现
```

## 6. 关键帧采样策略对比

| 策略 | 实现复杂度 | 信息保留 | 适用场景 |
|------|-----------|----------|----------|
| 均匀采样 | O(1) | 一般 | 通用 |
| 运动差异采样 | O(T) | 高（动态） | 动作视频 |
| K-Means 聚类 | O(T·K·D) | 高（多样） | 长视频 |
| 自适应 Score | O(T·D) | 最高 | 端到端训练 |

运动差异采样实现：

```python
def motion_based_sampling(frames, num_samples):
    """
    frames: [T, C, H, W]
    基于帧差异选择变化最大的帧
    """
    # 计算相邻帧差异
    diffs = (frames[1:] - frames[:-1]).abs().mean(dim=[1, 2, 3])

    # 选择差异最大的 num_samples 帧
    _, indices = diffs.topk(min(num_samples, len(diffs)))
    indices = indices.sort().values

    return frames[indices]
```
