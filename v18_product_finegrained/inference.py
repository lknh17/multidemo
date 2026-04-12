"""
V18 - 商品理解推理与实验
========================
1. 多粒度特征 vs 纯全局特征对比
2. 零件注意力多样性分析
3. ArcFace margin 效果
4. 商品图像质量评分演示
"""
import os, sys, torch, torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device
from config import ProductFullConfig
from fine_grained import MultiGranularityModel, ArcFaceHead, SimpleViTBackbone
from model import ProductAttributeModel, QualityAssessmentModel


def demo_multi_granularity():
    logger = get_logger("MultiGranularity")
    device = get_device()
    config = ProductFullConfig()
    model = MultiGranularityModel(config.fine_grained).to(device)
    model.eval()
    images = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        outputs = model(images)
    logger.info("多粒度特征分析：")
    logger.info(f"  全局 logits shape: {outputs['global_logits'].shape}")
    logger.info(f"  融合 logits shape: {outputs['fused_logits'].shape}")
    logger.info(f"  零件数: {len(outputs['part_logits'])}")
    logger.info(f"  注意力图 shape: {outputs['attn_maps'].shape}")
    logger.info(f"  多样性损失: {outputs['diversity_loss'].item():.4f}")
    for k in range(config.fine_grained.num_parts):
        attn = outputs['attn_maps'][0, k]
        logger.info(f"  Part {k}: max_attn={attn.max():.4f}, entropy={-(attn*torch.log(attn+1e-9)).sum():.4f}")


def demo_part_diversity():
    logger = get_logger("PartDiversity")
    device = get_device()
    config = ProductFullConfig()
    model = MultiGranularityModel(config.fine_grained).to(device)
    model.eval()
    images = torch.randn(4, 3, 224, 224).to(device)
    with torch.no_grad():
        outputs = model(images)
    attn = outputs['attn_maps']  # [B, K, N]
    gram = torch.matmul(attn, attn.transpose(-2, -1))
    logger.info("零件注意力重叠分析 (Gram matrix)：")
    for b in range(min(2, attn.shape[0])):
        logger.info(f"  Sample {b}:")
        for i in range(config.fine_grained.num_parts):
            row = [f"{gram[b, i, j].item():.3f}" for j in range(config.fine_grained.num_parts)]
            logger.info(f"    Part {i}: [{', '.join(row)}]")
    logger.info("  → 对角线应大(自相关=1)，非对角线应小(不同part关注不同区域)")


def demo_arcface_margin():
    logger = get_logger("ArcFace")
    device = get_device()
    config = ProductFullConfig()
    logger.info("ArcFace margin 效果：")
    features = F.normalize(torch.randn(8, config.fine_grained.d_model), dim=-1).to(device)
    labels = torch.arange(8).to(device) % 50
    for margin in [0.0, 0.1, 0.3, 0.5, 0.8]:
        arcface = ArcFaceHead(config.fine_grained.d_model, 50, margin=margin).to(device)
        loss = arcface(features, labels)
        logger.info(f"  margin={margin:.1f}: loss={loss.item():.4f}")
    logger.info("  → margin 越大，Loss 越大，学习难度越高，但类间分离更好")


def demo_quality_assessment():
    logger = get_logger("Quality")
    device = get_device()
    config = ProductFullConfig()
    model = QualityAssessmentModel(config.quality).to(device)
    model.eval()
    images = torch.randn(3, 3, 224, 224).to(device)
    images[1] += torch.randn_like(images[1]) * 0.5
    images[2] *= 0.3
    with torch.no_grad():
        outputs = model(images)
    dims = QualityAssessmentModel.QUALITY_DIMS
    logger.info("商品图像质量评估（随机初始化）：")
    for i in range(3):
        name = ['正常图', '噪声图', '暗图'][i]
        scores = outputs['dim_scores'][i]
        overall = outputs['overall_score'][i]
        logger.info(f"  {name}: overall={overall.item():.3f}")
        for d, s in zip(dims, scores):
            logger.info(f"    {d}: {s.item():.3f}")


def main():
    set_seed(42)
    print("=" * 60)
    print("V18 - 商品理解与细粒度视觉 推理实验")
    print("=" * 60)
    demo_multi_granularity(); print()
    demo_part_diversity(); print()
    demo_arcface_margin(); print()
    demo_quality_assessment()

if __name__ == "__main__":
    main()
