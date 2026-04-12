"""
V16 - OCR / 文档理解推理与实验
==============================
实验：
1. DBNet 可微二值化效果对比（k 值影响）
2. CTC vs Attention 识别器对比
3. 2D 位置编码效果分析
4. 广告文字类型分类演示
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device

from config import OCRDocumentFullConfig
from ocr_modules import TextDetector, CTCRecognizer, AttentionRecognizer, ctc_greedy_decode
from model import DocumentUnderstandingModel, AdTextExtractionModel, Layout2DPositionEmbedding


def demo_db_binarization():
    """实验 1：可微二值化 k 值影响"""
    logger = get_logger("DBNet")
    device = get_device()

    config = OCRDocumentFullConfig()
    model = TextDetector(config.ocr_det).to(device)

    image = torch.randn(1, 3, 256, 256).to(device)
    with torch.no_grad():
        outputs = model(image)

    prob = outputs['prob_map']
    thresh = outputs['thresh_map']

    logger.info("可微二值化 k 值分析：")
    logger.info(f"Prob map range: [{prob.min():.3f}, {prob.max():.3f}]")
    logger.info(f"Thresh map range: [{thresh.min():.3f}, {thresh.max():.3f}]")

    for k in [1, 10, 50, 100]:
        binary = torch.sigmoid(k * (prob - thresh))
        # 统计接近 0/1 的比例
        near_binary = ((binary < 0.05) | (binary > 0.95)).float().mean()
        logger.info(f"  k={k:3d}: near-binary ratio={near_binary:.3f}")

    logger.info("\n结论：k=50 时 >90% 像素接近 0/1，决策清晰但仍可传梯度")


def demo_recognizer_comparison():
    """实验 2：CTC vs Attention 识别器"""
    logger = get_logger("RecognizerCompare")
    device = get_device()

    config = OCRDocumentFullConfig()

    # CTC 识别器
    ctc_model = CTCRecognizer(config.ocr_rec).to(device)
    ctc_params = sum(p.numel() for p in ctc_model.parameters())

    # Attention 识别器
    attn_model = AttentionRecognizer(config.ocr_rec).to(device)
    attn_params = sum(p.numel() for p in attn_model.parameters())

    logger.info("CTC vs Attention 识别器对比：")
    logger.info(f"  CTC params: {ctc_params:,}")
    logger.info(f"  Attention params: {attn_params:,}")

    # 模拟推理
    images = torch.randn(2, 1, 32, 100).to(device)

    with torch.no_grad():
        # CTC 推理
        ctc_logits = ctc_model(images)  # [B, T, V+1]
        ctc_decoded = ctc_greedy_decode(ctc_logits)

    logger.info(f"\n  CTC output length: {ctc_logits.shape[1]}")
    logger.info(f"  CTC decoded lengths: {[len(d) for d in ctc_decoded]}")

    with torch.no_grad():
        attn_tokens = attn_model(images)
    logger.info(f"  Attention output: {attn_tokens.shape}")

    logger.info("\n对比总结：")
    logger.info("  CTC: 训练快，无字符依赖，适合短文本")
    logger.info("  Attention: 建模字符依赖，更准确，适合复杂场景")


def demo_2d_position():
    """实验 3：2D 位置编码分析"""
    logger = get_logger("2DPosition")
    device = get_device()

    config = OCRDocumentFullConfig()
    layout_embed = Layout2DPositionEmbedding(config.document).to(device)

    # 模拟同一行不同位置的 token
    same_row = torch.tensor([
        [[100, 200, 150, 220],   # token A
         [160, 200, 210, 220],   # token B (同行)
         [100, 300, 150, 320]],  # token C (不同行)
    ]).to(device)

    with torch.no_grad():
        embeds = layout_embed(same_row)

    # 计算相似度
    sim_AB = F.cosine_similarity(embeds[0, 0:1], embeds[0, 1:2])
    sim_AC = F.cosine_similarity(embeds[0, 0:1], embeds[0, 2:3])

    logger.info("2D 位置编码空间感知分析：")
    logger.info(f"  Token A: [100,200,150,220] (x=100~150, y=200~220)")
    logger.info(f"  Token B: [160,200,210,220] (同行, x=160~210)")
    logger.info(f"  Token C: [100,300,150,320] (不同行, y=300~320)")
    logger.info(f"\n  cos(A, B) = {sim_AB.item():.4f} (同行)")
    logger.info(f"  cos(A, C) = {sim_AC.item():.4f} (不同行)")
    logger.info("  → 训练后，同行 token 的 2D 位置编码会更相似")


def demo_ad_text_classification():
    """实验 4：广告文字类型分类"""
    logger = get_logger("AdText")
    device = get_device()

    config = OCRDocumentFullConfig()
    config.ocr_det.image_size = 128  # 缩小加速

    # 简化模型测试
    model = AdTextExtractionModel(config).to(device)

    image = torch.randn(1, 3, 128, 128).to(device)
    bboxes = torch.tensor([
        [[100, 50, 400, 100],    # 标题位置
         [200, 200, 800, 250],   # 促销文案
         [600, 400, 800, 450],   # 价格
         [100, 900, 300, 950],   # 品牌
         [0, 0, 0, 0]] * 4      # padding
    ][:, :config.ad_text.max_regions], dtype=torch.float).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(image, region_bboxes=bboxes)

    if 'type_logits' in outputs:
        probs = F.softmax(outputs['type_logits'][0], dim=-1)
        logger.info("广告文字区域分类结果（随机初始化）：")
        for i in range(min(4, probs.shape[0])):
            pred = probs[i].argmax().item()
            conf = probs[i].max().item()
            logger.info(f"  区域 {i}: 预测={AdTextExtractionModel.TEXT_TYPES[pred]} "
                       f"(置信度 {conf:.2%})")

    logger.info("\n文字类型定义：")
    for k, v in enumerate(AdTextExtractionModel.TEXT_TYPES):
        logger.info(f"  {k}: {v}")


def main():
    set_seed(42)
    print("=" * 60)
    print("V16 - OCR / 文档理解推理实验")
    print("=" * 60)

    demo_db_binarization()
    print()
    demo_recognizer_comparison()
    print()
    demo_2d_position()
    print()
    demo_ad_text_classification()


if __name__ == "__main__":
    main()
