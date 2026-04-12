"""
V19 - 层级标签理解推理与实验
==============================
实验：
1. 层级一致性检查
2. 标签传播效果分析
3. 双曲空间嵌入距离分析
4. 多标签阈值优化
"""
import os
import sys
import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device

from config import LabelHierarchyFullConfig
from taxonomy import (
    TaxonomyTree, HierarchicalSoftmax, LabelPropagationGNN,
    HyperbolicEmbedding, poincare_distance, LabelConsistencyChecker,
)
from model import HierarchicalClassifier, ConstrainedMultiLabelModel, LabelEmbeddingModel


def demo_hierarchy_consistency():
    """实验 1：层级一致性检查"""
    logger = get_logger("HierarchyConsistency")
    device = get_device()

    config = LabelHierarchyFullConfig()
    tree = TaxonomyTree(config.hierarchy.num_labels_per_level)
    checker = LabelConsistencyChecker(tree)

    logger.info("分类学树结构:")
    for lv in range(tree.num_levels):
        labels = tree.get_level_labels(lv)
        logger.info(f"  Level {lv}: {len(labels)} 标签 (ID {labels[0]}~{labels[-1]})")

    # 模拟预测结果
    B, C = 4, tree.total_labels
    # 不一致的预测：子节点概率高但父节点概率低
    bad_pred = torch.rand(B, C) * 0.3
    fine_labels = tree.get_level_labels(tree.num_levels - 1)
    for b in range(B):
        bad_pred[b, fine_labels[b * 10]] = 0.9  # 细粒度高概率

    stats = checker.check_consistency(bad_pred)
    logger.info(f"\n未约束的预测一致性: {stats['consistency_rate']:.2%}")
    logger.info(f"  违反次数: {stats['violations']}/{stats['total_pairs']}")

    # 强制一致性
    fixed_pred = checker.enforce_consistency(bad_pred)
    stats_fixed = checker.check_consistency(fixed_pred)
    logger.info(f"\n强制一致性后: {stats_fixed['consistency_rate']:.2%}")
    logger.info(f"  违反次数: {stats_fixed['violations']}/{stats_fixed['total_pairs']}")

    # 路径示例
    sample_label = fine_labels[5]
    path = tree.get_path(sample_label)
    logger.info(f"\n示例路径 (标签 {sample_label}): {path}")
    siblings = tree.get_siblings(sample_label)
    logger.info(f"  兄弟节点: {siblings[:5]}... (共 {len(siblings)} 个)")


def demo_label_propagation():
    """实验 2：标签传播效果"""
    logger = get_logger("LabelPropagation")
    device = get_device()

    config = LabelHierarchyFullConfig()
    tree = TaxonomyTree(config.hierarchy.num_labels_per_level)
    gnn = LabelPropagationGNN(tree, config.hierarchy.d_model, n_layers=2).to(device)

    # 初始节点特征
    node_features = torch.randn(tree.total_labels, config.hierarchy.d_model).to(device)

    # 传播前的父子相似度
    with torch.no_grad():
        sim_before = []
        for child, parent in list(tree.parent.items())[:20]:
            sim = F.cosine_similarity(
                node_features[child:child+1], node_features[parent:parent+1]
            ).item()
            sim_before.append(sim)

        # GNN 传播
        propagated = gnn(node_features)

        sim_after = []
        for child, parent in list(tree.parent.items())[:20]:
            sim = F.cosine_similarity(
                propagated[child:child+1], propagated[parent:parent+1]
            ).item()
            sim_after.append(sim)

    avg_before = sum(sim_before) / len(sim_before)
    avg_after = sum(sim_after) / len(sim_after)

    logger.info("标签传播 GNN 效果:")
    logger.info(f"  传播前 父子余弦相似度: {avg_before:.4f}")
    logger.info(f"  传播后 父子余弦相似度: {avg_after:.4f}")
    logger.info(f"  变化: {avg_after - avg_before:+.4f}")
    logger.info("  → GNN 传播使相关标签的表示更加接近")


def demo_hyperbolic_embedding():
    """实验 3：双曲空间嵌入距离分析"""
    logger = get_logger("HyperbolicEmbed")
    device = get_device()

    config = LabelHierarchyFullConfig()
    tree = TaxonomyTree(config.hierarchy.num_labels_per_level)
    hyp_embed = HyperbolicEmbedding(config.embedding).to(device)

    with torch.no_grad():
        all_ids = torch.arange(min(tree.total_labels, config.embedding.label_vocab_size)).to(device)
        embeddings = hyp_embed(all_ids)

        # 各层级标签的范数分布
        logger.info("双曲空间嵌入分析 (Poincaré 球):")
        for lv in range(tree.num_levels):
            labels = tree.get_level_labels(lv)
            valid = [l for l in labels if l < len(all_ids)]
            if valid:
                norms = embeddings[valid].norm(dim=-1)
                logger.info(f"  Level {lv}: 平均范数={norms.mean():.4f}, "
                          f"std={norms.std():.4f}, "
                          f"范围=[{norms.min():.4f}, {norms.max():.4f}]")

        # 父子距离 vs 非父子距离
        parent_dists = []
        random_dists = []

        for child, parent in list(tree.parent.items())[:50]:
            if child < len(all_ids) and parent < len(all_ids):
                d = poincare_distance(
                    embeddings[child:child+1], embeddings[parent:parent+1]
                ).item()
                parent_dists.append(d)

        import random as pyrandom
        pyrandom.seed(42)
        for _ in range(50):
            a, b = pyrandom.sample(range(len(all_ids)), 2)
            d = poincare_distance(
                embeddings[a:a+1], embeddings[b:b+1]
            ).item()
            random_dists.append(d)

        logger.info(f"\n  父子间双曲距离: {sum(parent_dists)/len(parent_dists):.4f}")
        logger.info(f"  随机对双曲距离: {sum(random_dists)/len(random_dists):.4f}")
        logger.info("  → 训练后父子距离应显著小于随机对距离")


def demo_threshold_optimization():
    """实验 4：多标签阈值优化"""
    logger = get_logger("ThresholdOpt")
    device = get_device()

    config = LabelHierarchyFullConfig()
    tree = TaxonomyTree(config.hierarchy.num_labels_per_level)
    C = tree.total_labels

    # 模拟预测和真实标签
    B = 100
    torch.manual_seed(42)
    gt_labels = torch.zeros(B, C)
    pred_scores = torch.rand(B, C) * 0.3

    for b in range(B):
        fine_labels = tree.get_level_labels(tree.num_levels - 1)
        chosen = fine_labels[b % len(fine_labels)]
        path = tree.get_path(chosen)
        for node in path:
            gt_labels[b, node] = 1.0
            pred_scores[b, node] += 0.4

    pred_scores = pred_scores.clamp(0, 1)

    logger.info("多标签阈值优化:")
    best_f1 = 0
    best_thresh = 0.5

    for thresh in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        binary = (pred_scores > thresh).float()
        tp = (binary * gt_labels).sum().item()
        fp = (binary * (1 - gt_labels)).sum().item()
        fn = ((1 - binary) * gt_labels).sum().item()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        logger.info(f"  threshold={thresh:.1f}: P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    logger.info(f"\n  最优阈值: {best_thresh:.1f} (F1={best_f1:.3f})")


def main():
    set_seed(42)
    print("=" * 60)
    print("V19 - 层级标签理解推理实验")
    print("=" * 60)

    demo_hierarchy_consistency()
    print()
    demo_label_propagation()
    print()
    demo_hyperbolic_embedding()
    print()
    demo_threshold_optimization()


if __name__ == "__main__":
    main()
