"""
V15 - 视频理解与 Dense Captioning 推理与实验
============================================
实验内容：
1. TimeSformer vs Conv3D vs Video Swin 编码器对比
2. Dense Caption 提议可视化 + NMS 效果对比
3. Deformable 时序注意力采样点可视化
4. Token Merging 压缩率 vs 精度分析
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device

from config import VideoDenseCaptionFullConfig
from model import (
    DenseVideoCaptioningModel, TemporalGroundingModel,
    temporal_iou, temporal_nms,
)
from video_encoder import VideoEncoder, TokenMerging, keyframe_sampling


def demo_encoder_comparison():
    """实验 1：三种视频编码器对比"""
    logger = get_logger("EncoderComparison")
    device = get_device()
    config = VideoDenseCaptionFullConfig()
    config.video.frame_size = 64
    config.video.num_frames = 8
    config.video.n_layers = 2

    video = torch.randn(2, 3, 8, 64, 64).to(device)

    results = {}
    for model_type in ["timesformer", "conv3d", "video_swin"]:
        config.video.temporal_model = model_type
        encoder = VideoEncoder(config.video).to(device)

        params = sum(p.numel() for p in encoder.parameters())
        with torch.no_grad():
            out = encoder(video)

        results[model_type] = {
            'params': params,
            'output_shape': tuple(out.shape),
        }
        logger.info(f"{model_type:15s} | params: {params:>8,} | output: {tuple(out.shape)}")

    logger.info("\n编码器选型建议：")
    logger.info("  TimeSformer: 适合中等长度视频，分离注意力高效")
    logger.info("  Video Swin:  适合高分辨率视频，局部窗口注意力")
    logger.info("  Conv3D:      适合短视频/实时场景，计算高效")
    return results


def demo_dense_caption_proposals():
    """实验 2：Dense Caption 提议生成 + NMS 对比"""
    logger = get_logger("DenseCaption")
    device = get_device()

    config = VideoDenseCaptionFullConfig()
    config.video.frame_size = 64
    config.video.num_frames = 8
    config.video.n_layers = 2
    config.dense_caption.num_proposals = 20

    model = DenseVideoCaptioningModel(config).to(device)
    video = torch.randn(1, 3, 8, 64, 64).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(video)

    spans = outputs['pred_spans'][0]
    logits = outputs['pred_logits'][0]
    scores = torch.sigmoid(logits)

    logger.info(f"Raw proposals: {spans.shape[0]}")
    logger.info(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Hard NMS vs Soft NMS
    keep_hard = temporal_nms(spans, scores.clone(), threshold=0.5, use_soft=False)
    keep_soft = temporal_nms(spans, scores.clone(), threshold=0.5, use_soft=True)

    logger.info(f"\nHard NMS: {len(keep_hard)} proposals kept")
    logger.info(f"Soft NMS: {len(keep_soft)} proposals kept")

    # 显示 Top-5 提议
    top_k = min(5, len(keep_soft))
    logger.info(f"\nTop-{top_k} proposals (Soft NMS):")
    for i, idx in enumerate(keep_soft[:top_k]):
        s, e = spans[idx].tolist()
        logger.info(f"  #{i+1}: [{s:.3f}, {e:.3f}] score={scores[idx]:.3f}")


def demo_deformable_attention():
    """实验 3：Deformable 时序注意力采样点分析"""
    logger = get_logger("DeformAttn")
    device = get_device()

    from model import DeformableTemporalAttention

    d_model = 128
    attn = DeformableTemporalAttention(
        d_model=d_model, n_heads=4, n_levels=3, n_points=4
    ).to(device)

    query = torch.randn(1, 10, d_model).to(device)
    ref_points = torch.linspace(0.1, 0.9, 10).unsqueeze(0).to(device)

    # 多尺度特征
    value_list = [
        torch.randn(1, T, d_model).to(device)
        for T in [32, 16, 8]
    ]

    with torch.no_grad():
        output = attn(query, ref_points, value_list)

    logger.info(f"Query shape: {tuple(query.shape)}")
    logger.info(f"Output shape: {tuple(output.shape)}")
    logger.info(f"Multi-scale levels: {[v.shape[1] for v in value_list]}")

    # 分析偏移量分布
    with torch.no_grad():
        offsets = attn.offset_net(query)
    offsets_np = offsets.reshape(-1).cpu()
    logger.info(f"\nOffset statistics:")
    logger.info(f"  mean: {offsets_np.mean():.4f}")
    logger.info(f"  std:  {offsets_np.std():.4f}")
    logger.info(f"  range: [{offsets_np.min():.4f}, {offsets_np.max():.4f}]")
    logger.info("  → 偏移量集中在 0 附近：模型初始关注参考点附近")
    logger.info("  → 训练后偏移量分散：学会关注远距离关键时刻")


def demo_token_merging():
    """实验 4：Token Merging 压缩率与精度分析"""
    logger = get_logger("TokenMerging")
    device = get_device()

    # 模拟视频 token 序列
    B, N, D = 2, 64, 128
    tokens = torch.randn(B, N, D).to(device)

    # 添加一些冗余（相邻 token 相似）
    for i in range(1, N):
        tokens[:, i] = tokens[:, i] * 0.3 + tokens[:, i-1] * 0.7

    logger.info("Token Merging 压缩率分析：")
    logger.info(f"原始序列长度: {N}")

    for r in [4, 8, 16, 24, 32]:
        tome = TokenMerging(r=r).to(device)
        merged = tome(tokens)
        compression = 1 - merged.shape[1] / N

        # 评估信息保留：通过重构误差
        # 找每个原始 token 在合并后最近邻
        sim = F.cosine_similarity(
            tokens.unsqueeze(2), merged.unsqueeze(1), dim=-1
        )
        max_sim = sim.max(dim=2).values.mean()

        logger.info(
            f"  r={r:2d} → 长度: {merged.shape[1]:3d} "
            f"(压缩 {compression:.1%}), "
            f"平均最近邻相似度: {max_sim:.4f}"
        )

    # 关键帧采样对比
    logger.info("\n关键帧采样策略对比：")
    features = torch.randn(1, 32, D).to(device)
    for i in range(1, 32):
        features[:, i] = features[:, i] * 0.5 + features[:, i-1] * 0.5

    for method in ["uniform", "motion", "cluster"]:
        sampled = keyframe_sampling(features, num_samples=8, method=method)
        # 评估覆盖度：采样帧两两距离的平均
        dists = torch.cdist(sampled[0], sampled[0])
        avg_dist = dists[dists > 0].mean()
        logger.info(f"  {method:10s}: avg inter-frame dist = {avg_dist:.4f}")


def main():
    set_seed(42)
    print("=" * 60)
    print("V15 - 视频理解与 Dense Captioning 推理实验")
    print("=" * 60)

    demo_encoder_comparison()
    print()
    demo_dense_caption_proposals()
    print()
    demo_deformable_attention()
    print()
    demo_token_merging()


if __name__ == "__main__":
    main()
