"""
V24 - 内容安全推理与实验
========================
1. 安全阈值优化（F1-optimal threshold）
2. 对抗鲁棒性测试（FGSM vs PGD）
3. 水印鲁棒性（压缩/噪声测试）
4. 校准可靠性图（Platt Scaling 前后对比）
"""
import os, sys, torch, torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device
from config import SafetyFullConfig
from safety_modules import ContentClassifier, WatermarkEmbedder, AdversarialAttacker
from model import SafetyGuardModel, CalibratedClassifier


def demo_threshold_optimization():
    """实验 1：安全阈值优化"""
    logger = get_logger("ThresholdOpt")
    device = get_device()
    config = SafetyFullConfig()
    model = ContentClassifier(config.safety_cls).to(device)
    model.eval()

    # 生成模拟数据
    images = torch.randn(16, 3, 224, 224).to(device)
    with torch.no_grad():
        outputs = model(images)
    probs = outputs['probs']

    logger.info("安全分类阈值优化分析：")
    logger.info(f"  输入: {images.shape}")
    logger.info(f"  类别概率 shape: {probs.shape}")

    # 不同阈值的效果
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        preds = (probs > threshold).float()
        flagged = preds[:, :-1].sum(dim=-1) > 0  # 排除 normal 类
        flag_rate = flagged.float().mean().item()
        avg_flags = preds[:, :-1].sum().item() / images.shape[0]
        logger.info(f"  threshold={threshold:.1f}: flag_rate={flag_rate:.2%}, avg_flags_per_img={avg_flags:.2f}")

    # 逐类别分析
    logger.info("\n  逐类别概率统计：")
    for k, name in enumerate(ContentClassifier.CATEGORY_NAMES):
        p = probs[:, k]
        logger.info(f"    {name:>12}: mean={p.mean():.4f}, max={p.max():.4f}, min={p.min():.4f}")


def demo_adversarial_robustness():
    """实验 2：对抗鲁棒性测试"""
    logger = get_logger("AdversarialTest")
    device = get_device()
    config = SafetyFullConfig()
    model = ContentClassifier(config.safety_cls).to(device)
    model.eval()
    attacker = AdversarialAttacker(config.adversarial)

    images = torch.randn(8, 3, 224, 224).clamp(0, 1).to(device)
    labels = torch.zeros(8, dtype=torch.long, device=device)  # 假设标签

    # 干净样本预测
    with torch.no_grad():
        clean_out = model(images)
    clean_max = clean_out['probs'].max(dim=-1).values

    logger.info("对抗鲁棒性测试：")
    logger.info(f"  干净样本 max_prob: mean={clean_max.mean():.4f}")

    # 不同 epsilon 的 FGSM 攻击
    for eps in [2/255, 4/255, 8/255, 16/255]:
        attacker.epsilon = eps
        # 需要梯度
        model.train()
        adv = attacker.fgsm(model, images, labels,
                            loss_fn=nn.CrossEntropyLoss() if False else None)
        model.eval()
        with torch.no_grad():
            adv_out = model(adv)
        adv_max = adv_out['probs'].max(dim=-1).values
        l2_dist = (adv - images).reshape(8, -1).norm(dim=-1).mean()
        logger.info(f"  FGSM ε={eps:.4f}: adv_max_prob={adv_max.mean():.4f}, L2_dist={l2_dist:.4f}")

    # PGD 攻击
    attacker.epsilon = 8/255
    model.train()
    for steps in [5, 10, 20]:
        attacker.attack_steps = steps
        adv = attacker.pgd(model, images, labels)
        model.eval()
        with torch.no_grad():
            adv_out = model(adv)
        adv_max = adv_out['probs'].max(dim=-1).values
        logger.info(f"  PGD steps={steps}: adv_max_prob={adv_max.mean():.4f}")
        model.train()
    model.eval()


def demo_watermark_robustness():
    """实验 3：水印鲁棒性测试"""
    logger = get_logger("WatermarkTest")
    device = get_device()
    config = SafetyFullConfig()
    model = WatermarkEmbedder(config.watermark).to(device)
    model.eval()

    images = torch.randn(4, 3, 224, 224).clamp(0, 1).to(device)
    watermark = torch.sign(torch.randn(4, config.watermark.watermark_bits)).to(device)

    with torch.no_grad():
        outputs = model(images, watermark)

    logger.info("水印嵌入与检测：")
    logger.info(f"  原始比特准确率: {outputs['bit_accuracy'].item():.4f}")
    logger.info(f"  嵌入 MSE: {outputs['embed_mse'].item():.6f}")

    # PSNR
    mse = outputs['embed_mse'].item()
    psnr = 10 * torch.log10(torch.tensor(1.0 / max(mse, 1e-10))).item()
    logger.info(f"  PSNR: {psnr:.2f} dB")

    # 鲁棒性测试：不同程度噪声
    watermarked = outputs['watermarked']
    logger.info("\n  噪声鲁棒性测试：")
    for noise_std in [0.01, 0.05, 0.1, 0.2]:
        noisy = (watermarked + torch.randn_like(watermarked) * noise_std).clamp(0, 1)
        with torch.no_grad():
            detected = model.detect(noisy)
        pred_bits = (detected > 0).float()
        orig_bits = (watermark > 0).float()
        acc = (pred_bits == orig_bits).float().mean().item()
        logger.info(f"    noise_std={noise_std:.2f}: bit_acc={acc:.4f}")

    # 模拟 JPEG 压缩（量化）
    logger.info("\n  量化鲁棒性测试（模拟压缩）：")
    for levels in [256, 64, 32, 16]:
        quantized = (watermarked * (levels - 1)).round() / (levels - 1)
        with torch.no_grad():
            detected = model.detect(quantized)
        pred_bits = (detected > 0).float()
        orig_bits = (watermark > 0).float()
        acc = (pred_bits == orig_bits).float().mean().item()
        logger.info(f"    levels={levels}: bit_acc={acc:.4f}")


def demo_calibration():
    """实验 4：校准可靠性分析"""
    logger = get_logger("Calibration")
    device = get_device()
    config = SafetyFullConfig()
    base_model = ContentClassifier(config.safety_cls).to(device)
    cal_model = CalibratedClassifier(base_model, config.safety_cls.num_categories).to(device)
    cal_model.eval()

    images = torch.randn(32, 3, 224, 224).to(device)
    labels = torch.zeros(32, config.safety_cls.num_categories).to(device)
    # 随机标注
    for i in range(32):
        if torch.rand(1).item() < 0.6:
            labels[i, -1] = 1
        else:
            labels[i, torch.randint(0, config.safety_cls.num_categories - 1, (1,))] = 1

    with torch.no_grad():
        outputs = cal_model(images)

    logger.info("校准可靠性分析：")
    logger.info(f"  Platt a: {cal_model.platt_a.data.tolist()}")
    logger.info(f"  Platt b: {cal_model.platt_b.data.tolist()}")
    logger.info(f"  Temperature: {cal_model.temperature.item():.4f}")

    # ECE 对比
    raw_ece = cal_model.compute_ece(outputs['raw_probs'].flatten(), labels.flatten())
    platt_ece = cal_model.compute_ece(outputs['platt_probs'].flatten(), labels.flatten())
    temp_ece = cal_model.compute_ece(outputs['temp_probs'].flatten(), labels.flatten())
    logger.info(f"\n  ECE (raw):   {raw_ece.item():.4f}")
    logger.info(f"  ECE (platt): {platt_ece.item():.4f}")
    logger.info(f"  ECE (temp):  {temp_ece.item():.4f}")

    # 可靠性图数据（10 个桶）
    logger.info("\n  可靠性图（Raw）：")
    n_bins = 10
    raw_probs = outputs['raw_probs'].flatten()
    flat_labels = labels.flatten()
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (raw_probs > lo) & (raw_probs <= hi)
        if mask.sum() > 0:
            conf = raw_probs[mask].mean().item()
            acc = flat_labels[mask].mean().item()
            logger.info(f"    bin [{lo:.1f}-{hi:.1f}]: conf={conf:.4f}, acc={acc:.4f}, gap={abs(conf-acc):.4f}, n={mask.sum().item()}")


# 需要的 import
import torch.nn as nn


def main():
    set_seed(42)
    print("=" * 60)
    print("V24 - 内容安全与合规 推理实验")
    print("=" * 60)
    demo_threshold_optimization(); print()
    demo_adversarial_robustness(); print()
    demo_watermark_robustness(); print()
    demo_calibration()

if __name__ == "__main__":
    main()
