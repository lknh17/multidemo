"""
V17 - 音频理解 / 全模态推理与实验
===================================
实验：
1. Mel 频谱图特征分析
2. AST 位置编码可视化
3. CLAP 音频-文本相似度矩阵
4. 全模态融合 vs 单模态对比
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device

from config import AudioMultimodalFullConfig
from audio_modules import MelSpectrogramExtractor, AudioSpectrogramTransformer, AudioEventDetector
from model import CLAPModel, OmniModalModel


def demo_mel_features():
    """实验 1：Mel 频谱图特征分析"""
    logger = get_logger("MelSpec")
    device = get_device()

    config = AudioMultimodalFullConfig()
    mel_extractor = MelSpectrogramExtractor(config.mel).to(device)

    # 合成不同类型音频
    sr = config.mel.sample_rate
    duration = 1.0
    t = torch.linspace(0, duration, int(sr * duration)).to(device)

    signals = {
        '440Hz 纯音': torch.sin(2 * 3.14159 * 440 * t),
        '440+880Hz 和弦': torch.sin(2 * 3.14159 * 440 * t) + torch.sin(2 * 3.14159 * 880 * t),
        '频率扫描(200→2000Hz)': torch.sin(2 * 3.14159 * (200 + 1800 * t / duration) * t),
        '白噪声': torch.randn_like(t),
    }

    logger.info("Mel 频谱图特征分析：")
    for name, signal in signals.items():
        mel = mel_extractor(signal.unsqueeze(0))
        logger.info(f"\n  {name}:")
        logger.info(f"    Mel shape: {mel.shape}")
        logger.info(f"    Energy range: [{mel.min():.2f}, {mel.max():.2f}]")
        logger.info(f"    Mean energy: {mel.mean():.2f}")

        # 找能量最集中的频段
        freq_energy = mel[0].mean(dim=1)  # [n_mels]
        top_k = freq_energy.topk(3)
        logger.info(f"    Top-3 Mel bins: {top_k.indices.tolist()}")


def demo_ast_position():
    """实验 2：AST 2D 位置编码分析"""
    logger = get_logger("ASTPosition")
    device = get_device()

    config = AudioMultimodalFullConfig()
    ast = AudioSpectrogramTransformer(config.audio_enc).to(device)

    logger.info("AST 2D 位置编码分析：")

    # 频率维位置编码相似度
    freq_pe = ast.freq_pos_embed[0]  # [Nf, D]
    freq_sim = F.cosine_similarity(freq_pe.unsqueeze(0), freq_pe.unsqueeze(1), dim=-1)
    logger.info(f"  频率位置编码 shape: {freq_pe.shape}")
    logger.info(f"  相邻频率 cos sim: {freq_sim.diagonal(1).mean():.4f}")
    logger.info(f"  间隔4 频率 cos sim: {freq_sim.diagonal(4).mean():.4f}")

    # 时间维位置编码相似度
    time_pe = ast.time_pos_embed[0, :16]  # [Nt, D]
    time_sim = F.cosine_similarity(time_pe.unsqueeze(0), time_pe.unsqueeze(1), dim=-1)
    logger.info(f"\n  时间位置编码 shape: {time_pe.shape}")
    logger.info(f"  相邻时间 cos sim: {time_sim.diagonal(1).mean():.4f}")
    logger.info(f"  间隔4 时间 cos sim: {time_sim.diagonal(4).mean():.4f}")

    # 不同长度输入测试
    logger.info("\n  可变长度输入测试：")
    for T in [128, 256, 512]:
        mel = torch.randn(1, config.audio_enc.n_mels, T).to(device)
        with torch.no_grad():
            feat = ast(mel)
        logger.info(f"    T={T}: output shape={feat.shape}")


def demo_clap_similarity():
    """实验 3：CLAP 音频-文本相似度矩阵"""
    logger = get_logger("CLAP")
    device = get_device()

    config = AudioMultimodalFullConfig()
    model = CLAPModel(config.clap, config.audio_enc).to(device)
    model.eval()

    # 合成 batch
    B = 4
    mel = torch.randn(B, config.audio_enc.n_mels, 256).to(device)
    tokens = torch.randint(3, config.clap.vocab_size, (B, config.clap.max_text_len)).to(device)

    with torch.no_grad():
        outputs = model(mel, tokens)

    sim = outputs['similarity']
    logger.info("CLAP 音频-文本相似度矩阵（随机初始化）：")
    logger.info(f"  Temperature: {outputs['temperature'].item():.2f}")
    logger.info(f"  Similarity matrix ({B}x{B}):")
    for i in range(B):
        row = [f"{sim[i, j].item():6.2f}" for j in range(B)]
        diag = " ← 正例" if True else ""
        logger.info(f"    [{', '.join(row)}]{diag}")

    # 检查对角线 vs 非对角线
    diag_mean = sim.diagonal().mean().item()
    off_diag = sim.masked_fill(torch.eye(B, device=device).bool(), 0).sum() / (B * B - B)
    logger.info(f"\n  对角线均值 (正例): {diag_mean:.2f}")
    logger.info(f"  非对角线均值 (负例): {off_diag.item():.2f}")
    logger.info(f"  Loss: {outputs['loss'].item():.4f}")


def demo_omni_modal_fusion():
    """实验 4：全模态 vs 单模态对比"""
    logger = get_logger("OmniModal")
    device = get_device()

    config = AudioMultimodalFullConfig()
    model = OmniModalModel(config.omni, config.audio_enc).to(device)
    model.eval()

    B = 2
    images = torch.randn(B, 3, config.omni.image_size, config.omni.image_size).to(device)
    tokens = torch.randint(3, config.omni.vocab_size, (B, config.omni.max_text_len)).to(device)
    mel = torch.randn(B, config.audio_enc.n_mels, 256).to(device)

    logger.info("全模态融合 vs 单模态对比：")

    with torch.no_grad():
        # 全模态
        out_all = model(images, tokens, mel)
        repr_all = out_all['fused_repr']

        # 仅图像
        model.modality_dropout.p_drop = 0  # 关闭随机丢弃
        out_img = model(images, None, None)
        repr_img = out_img['fused_repr']

        # 仅文本
        out_txt = model(None, tokens, None)
        repr_txt = out_txt['fused_repr']

        # 仅音频
        out_aud = model(None, None, mel)
        repr_aud = out_aud['fused_repr']

        # 图像+文本
        out_it = model(images, tokens, None)
        repr_it = out_it['fused_repr']

    logger.info(f"  全模态 logits norm: {out_all['logits'].norm(dim=-1).mean():.4f}")
    logger.info(f"  仅图像 logits norm: {out_img['logits'].norm(dim=-1).mean():.4f}")
    logger.info(f"  仅文本 logits norm: {out_txt['logits'].norm(dim=-1).mean():.4f}")
    logger.info(f"  仅音频 logits norm: {out_aud['logits'].norm(dim=-1).mean():.4f}")
    logger.info(f"  图+文 logits norm: {out_it['logits'].norm(dim=-1).mean():.4f}")

    # 表征相似度
    sim_all_img = F.cosine_similarity(repr_all, repr_img, dim=-1).mean()
    sim_all_txt = F.cosine_similarity(repr_all, repr_txt, dim=-1).mean()
    sim_all_aud = F.cosine_similarity(repr_all, repr_aud, dim=-1).mean()
    sim_all_it = F.cosine_similarity(repr_all, repr_it, dim=-1).mean()

    logger.info(f"\n  全模态 vs 仅图像 cos sim: {sim_all_img.item():.4f}")
    logger.info(f"  全模态 vs 仅文本 cos sim: {sim_all_txt.item():.4f}")
    logger.info(f"  全模态 vs 仅音频 cos sim: {sim_all_aud.item():.4f}")
    logger.info(f"  全模态 vs 图+文 cos sim: {sim_all_it.item():.4f}")
    logger.info("  → 训练后，全模态表征应包含更丰富的信息")


def main():
    set_seed(42)
    print("=" * 60)
    print("V17 - 音频理解与全模态模型 推理实验")
    print("=" * 60)

    demo_mel_features()
    print()
    demo_ast_position()
    print()
    demo_clap_similarity()
    print()
    demo_omni_modal_fusion()


if __name__ == "__main__":
    main()
