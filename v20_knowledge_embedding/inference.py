"""
V20 - 知识增强嵌入推理与实验
==============================
实验：
1. TransE vs TransR 嵌入质量对比
2. 实体链接准确率分析
3. KG 增强检索效果提升
4. 知识蒸馏效果分析
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device

from config import KnowledgeEmbeddingFullConfig
from kg_modules import TransEEmbedding, TransREmbedding, EntityLinker, SimpleGNN
from model import KGEnhancedVisualModel, KGAugmentedRetrieval, KnowledgeDistillModel


def demo_transe_vs_transr():
    """实验 1：TransE vs TransR 嵌入质量"""
    logger = get_logger("TransE-vs-TransR")
    device = get_device()
    config = KnowledgeEmbeddingFullConfig()

    transe = TransEEmbedding(config.kg).to(device)
    transr = TransREmbedding(config.kg).to(device)

    logger.info("TransE vs TransR 嵌入质量对比：")
    logger.info(f"  TransE params: {sum(p.numel() for p in transe.parameters()):,}")
    logger.info(f"  TransR params: {sum(p.numel() for p in transr.parameters()):,}")

    B = 16
    h = torch.randint(0, config.kg.num_entities, (B,)).to(device)
    r = torch.randint(0, config.kg.num_relations, (B,)).to(device)
    t = torch.randint(0, config.kg.num_entities, (B,)).to(device)
    neg_t = torch.randint(0, config.kg.num_entities, (B,)).to(device)

    with torch.no_grad():
        e_pos = transe.score(h, r, t)
        e_neg = transe.score(h, r, neg_t)
        r_pos = transr.score(h, r, t)
        r_neg = transr.score(h, r, neg_t)

    logger.info(f"\n  TransE 正样本分数: {e_pos.mean():.4f} ± {e_pos.std():.4f}")
    logger.info(f"  TransE 负样本分数: {e_neg.mean():.4f} ± {e_neg.std():.4f}")
    logger.info(f"  TransR 正样本分数: {r_pos.mean():.4f} ± {r_pos.std():.4f}")
    logger.info(f"  TransR 负样本分数: {r_neg.mean():.4f} ± {r_neg.std():.4f}")

    e_embs = transe.get_entity_embeddings(torch.arange(min(100, config.kg.num_entities)).to(device))
    sim = F.cosine_similarity(e_embs.unsqueeze(0), e_embs.unsqueeze(1), dim=-1)
    logger.info(f"\n  实体嵌入余弦相似度: mean={sim.mean():.4f}, std={sim.std():.4f}")
    logger.info("  → 训练后正负样本分数差距会扩大，TransR 对复杂关系更优")


def demo_entity_linking():
    """实验 2：实体链接准确率"""
    logger = get_logger("EntityLinking")
    device = get_device()
    config = KnowledgeEmbeddingFullConfig()

    linker = EntityLinker(config.entity_link).to(device)
    logger.info(f"EntityLinker params: {sum(p.numel() for p in linker.parameters()):,}")

    token_ids = torch.randint(3, config.entity_link.vocab_size, (2, 64)).to(device)

    linker.eval()
    with torch.no_grad():
        outputs = linker(token_ids)

    logger.info("\n实体链接检测结果（随机初始化）：")
    logger.info(f"  输入序列长度: {token_ids.shape[1]}")
    logger.info(f"  检测到的 span 数: {outputs['span_repr'].shape[1]}")

    start_probs = torch.sigmoid(outputs['start_logits'][0])
    end_probs = torch.sigmoid(outputs['end_logits'][0])
    logger.info(f"  Start 概率范围: [{start_probs.min():.3f}, {start_probs.max():.3f}]")
    logger.info(f"  End 概率范围: [{end_probs.min():.3f}, {end_probs.max():.3f}]")

    n_cand = 5
    candidate_embeds = torch.randn(2, outputs['span_repr'].shape[1], n_cand, config.entity_link.d_model).to(device)
    with torch.no_grad():
        scores = linker.rank_candidates(outputs['span_repr'], candidate_embeds)
    probs = F.softmax(scores, dim=-1)
    logger.info(f"\n  候选排序（{n_cand} 个候选）:")
    for i in range(min(3, probs.shape[1])):
        pred = probs[0, i].argmax().item()
        conf = probs[0, i].max().item()
        logger.info(f"    Span {i}: 最佳候选={pred}, 置信度={conf:.2%}")


def demo_kg_retrieval():
    """实验 3：KG 增强检索效果"""
    logger = get_logger("KG-Retrieval")
    device = get_device()
    config = KnowledgeEmbeddingFullConfig()

    model = KGAugmentedRetrieval(config).to(device)
    logger.info(f"KGAugmentedRetrieval params: {sum(p.numel() for p in model.parameters()):,}")

    B = 4
    images = torch.randn(B, 3, config.kg_embed.image_size, config.kg_embed.image_size).to(device)
    token_ids = torch.randint(3, 5000, (B, 32)).to(device)
    entity_ids = torch.randint(0, config.kg.num_entities, (B, config.kg_embed.max_entities_per_image)).to(device)

    model.eval()
    with torch.no_grad():
        out_with_kg = model(images, token_ids, entity_ids)
        out_no_kg = model(images, token_ids, None)

    logger.info("\nKG 增强检索对比：")
    sim_kg = out_with_kg['sim_v2t']
    sim_no = out_no_kg['sim_v2t']

    diag_kg = sim_kg.diag()
    diag_no = sim_no.diag()
    logger.info(f"  有 KG 时对角线相似度: {diag_kg.mean():.4f} ± {diag_kg.std():.4f}")
    logger.info(f"  无 KG 时对角线相似度: {diag_no.mean():.4f} ± {diag_no.std():.4f}")

    for i in range(B):
        rank_kg = (sim_kg[i] >= sim_kg[i, i]).sum().item()
        rank_no = (sim_no[i] >= sim_no[i, i]).sum().item()
        logger.info(f"  样本 {i}: KG rank={rank_kg}, No-KG rank={rank_no}")

    logger.info("  → 训练后 KG 增强可以提升检索排名")


def demo_knowledge_distill():
    """实验 4：知识蒸馏效果"""
    logger = get_logger("KnowledgeDistill")
    device = get_device()
    config = KnowledgeEmbeddingFullConfig()

    model = KnowledgeDistillModel(config).to(device)
    t_params = sum(p.numel() for p in model.teacher.parameters())
    s_params = sum(p.numel() for p in model.student.parameters())
    logger.info(f"Teacher params: {t_params:,}")
    logger.info(f"Student params: {s_params:,}")
    logger.info(f"压缩比: {t_params/s_params:.2f}x")

    B = 4
    images = torch.randn(B, 3, config.kg_embed.image_size, config.kg_embed.image_size).to(device)
    entity_ids = torch.randint(0, config.kg.num_entities, (B, config.kg_embed.max_entities_per_image)).to(device)
    labels = torch.randint(0, config.kg.num_relations, (B,)).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(images, entity_ids, labels)

    logger.info(f"\n蒸馏损失分析：")
    logger.info(f"  Task loss: {outputs['task_loss'].item():.4f}")
    logger.info(f"  Distill loss: {outputs['distill_loss'].item():.4f}")
    logger.info(f"  Feature loss: {outputs['feat_loss'].item():.4f}")
    logger.info(f"  Total loss: {outputs['loss'].item():.4f}")

    t_probs = F.softmax(outputs['teacher_logits'], dim=-1)
    s_probs = F.softmax(outputs['student_logits'], dim=-1)
    kl = F.kl_div(s_probs.log(), t_probs, reduction='batchmean')
    logger.info(f"\n  Teacher-Student KL 散度: {kl.item():.4f}")

    feat_sim = F.cosine_similarity(outputs['student_features'], outputs['teacher_features'])
    logger.info(f"  特征余弦相似度: {feat_sim.mean():.4f}")
    logger.info("  → 训练后学生模型逼近教师表现，无需 KG 推理开销")


def main():
    set_seed(42)
    print("=" * 60)
    print("V20 - 知识增强多模态嵌入推理实验")
    print("=" * 60)

    demo_transe_vs_transr()
    print()
    demo_entity_linking()
    print()
    demo_kg_retrieval()
    print()
    demo_knowledge_distill()


if __name__ == "__main__":
    main()
