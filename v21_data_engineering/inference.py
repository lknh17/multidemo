"""
V21 - 多模态数据工程 - 推理/实验脚本
======================================
四组实验：
1. 去重压缩率：MinHash / SimHash 在不同阈值下的压缩效果
2. 增强效果：对比无增强 / MixUp / CutMix / RandAugment
3. 课程 vs 随机：课程学习与随机顺序训练的收敛对比
4. 质量评分分布：数据集质量分数的统计分布

运行方式:
    cd v21_data_engineering
    python inference.py
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import FullConfig
from model import (
    SimplePatchClassifier, AugmentedTrainer, CurriculumScheduler,
    MixUp, CutMix, RandAugment,
)
from dataset import (
    SyntheticImageDataset, TextDatasetForDedup, create_dataloaders,
)
from data_ops import MinHashDedup, SimHashDedup, QualityScorer, DataBalancer
from shared.utils import set_seed, get_device, AverageMeter


# ============================================================
# 实验 1: 去重压缩率
# ============================================================
def experiment_dedup():
    """
    测试 MinHash / SimHash 在不同阈值下的去重效果。
    
    预期：阈值越低，越多样本被判为重复，压缩率越高。
    """
    print("=" * 60)
    print("实验 1: 去重压缩率")
    print("=" * 60)
    
    # MinHash 去重
    print("\n--- MinHash 去重 ---")
    text_ds = TextDatasetForDedup(num_samples=200, dup_ratio=0.3)
    texts = [text_ds[i] for i in range(len(text_ds))]
    print(f"原始文本数: {len(texts)}")
    
    for threshold in [0.3, 0.5, 0.7, 0.9]:
        dedup = MinHashDedup(num_perm=128, threshold=threshold, num_bands=16)
        kept, compression = dedup.deduplicate(texts)
        print(f"  阈值={threshold:.1f}: 保留 {len(kept)}/{len(texts)}, 压缩率={compression:.2%}")
    
    # SimHash 去重
    print("\n--- SimHash 去重 ---")
    n_vectors = 200
    dim = 64
    rng = np.random.RandomState(42)
    vectors = rng.randn(n_vectors, dim).astype(np.float32)
    
    # 插入近似重复
    for i in range(0, 50, 2):
        vectors[i + 1] = vectors[i] + rng.randn(dim) * 0.05
    
    print(f"原始向量数: {n_vectors} (含 ~25 对近似重复)")
    
    for threshold in [0.8, 0.9, 0.95, 0.99]:
        dedup = SimHashDedup(input_dim=dim, hash_bits=128, threshold=threshold)
        kept, compression = dedup.deduplicate(vectors)
        print(f"  阈值={threshold:.2f}: 保留 {len(kept)}/{n_vectors}, 压缩率={compression:.2%}")


# ============================================================
# 实验 2: 增强效果
# ============================================================
def experiment_augmentation():
    """
    对比不同增强策略的训练效果。
    
    对比：无增强 / MixUp / CutMix / RandAugment / Mixed
    """
    print("\n" + "=" * 60)
    print("实验 2: 增强效果对比")
    print("=" * 60)
    
    config = FullConfig()
    config.num_epochs = 10  # 快速实验
    config.num_train_samples = 500
    config.num_val_samples = 100
    device = get_device()
    
    train_loader, val_loader = create_dataloaders(config, imbalanced=False)
    
    strategies = ["none", "mixup", "cutmix", "randaug", "mixed"]
    
    for strategy in strategies:
        set_seed(42)
        model = SimplePatchClassifier(config).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate
        )
        trainer = AugmentedTrainer(config)
        
        final_acc = 0.0
        for epoch in range(1, config.num_epochs + 1):
            model.train()
            for batch in train_loader:
                images = batch["image"].to(device)
                labels = batch["label"].to(device)
                
                if strategy == "none":
                    logits = model(images)
                    loss = F.cross_entropy(logits, labels)
                else:
                    aug_images, soft_labels = trainer.apply_augmentation(
                        images, labels, aug_type=strategy
                    )
                    soft_labels = soft_labels.to(device)
                    logits = model(aug_images)
                    log_probs = F.log_softmax(logits, dim=-1)
                    loss = -(soft_labels * log_probs).sum(dim=-1).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    preds = model(images).argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            final_acc = correct / max(total, 1)
        
        print(f"  {strategy:>10s}: Val Acc = {final_acc:.4f}")


# ============================================================
# 实验 3: 课程 vs 随机训练
# ============================================================
def experiment_curriculum():
    """
    对比课程学习和随机顺序训练的收敛速度。
    
    预期：课程学习在前期收敛更快。
    """
    print("\n" + "=" * 60)
    print("实验 3: 课程学习 vs 随机训练")
    print("=" * 60)
    
    config = FullConfig()
    config.num_epochs = 15
    config.num_train_samples = 500
    config.num_val_samples = 100
    device = get_device()
    
    _, val_loader = create_dataloaders(config, imbalanced=False)
    
    # 准备训练数据
    full_dataset = SyntheticImageDataset(config, config.num_train_samples)
    all_images = torch.stack([full_dataset[i]["image"] for i in range(len(full_dataset))])
    all_labels = torch.stack([full_dataset[i]["label"] for i in range(len(full_dataset))])
    
    for mode_name in ["random", "curriculum"]:
        set_seed(42)
        model = SimplePatchClassifier(config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        if mode_name == "curriculum":
            scheduler = CurriculumScheduler(config, total_epochs=config.num_epochs)
            # 计算初始难度
            difficulties = scheduler.compute_difficulty(
                model, all_images, all_labels, device
            )
        
        accs_over_epochs = []
        
        for epoch in range(1, config.num_epochs + 1):
            if mode_name == "curriculum":
                selected = scheduler.select_samples(difficulties, epoch)
                train_images = all_images[selected]
                train_labels = all_labels[selected]
            else:
                train_images = all_images
                train_labels = all_labels
            
            # 训练
            model.train()
            perm = torch.randperm(len(train_images))
            for start in range(0, len(train_images) - config.batch_size, config.batch_size):
                idx = perm[start:start + config.batch_size]
                images = train_images[idx].to(device)
                labels = train_labels[idx].to(device)
                
                logits = model(images)
                loss = criterion(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            # 验证
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for batch in val_loader:
                    images = batch["image"].to(device)
                    labels = batch["label"].to(device)
                    preds = model(images).argmax(dim=-1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            acc = correct / max(total, 1)
            accs_over_epochs.append(acc)
        
        # 输出前几个 epoch 的准确率
        print(f"\n  {mode_name:>12s} 训练曲线:")
        for e, acc in enumerate(accs_over_epochs, 1):
            bar = "█" * int(acc * 30)
            print(f"    Epoch {e:2d}: {acc:.4f} {bar}")


# ============================================================
# 实验 4: 质量评分分布
# ============================================================
def experiment_quality_scores():
    """
    分析合成数据集的质量评分分布。
    
    展示不同质量维度的评分统计。
    """
    print("\n" + "=" * 60)
    print("实验 4: 数据质量评分分布")
    print("=" * 60)
    
    config = FullConfig()
    dataset = SyntheticImageDataset(
        config, 500, imbalanced=False, include_meta=True
    )
    
    scorer = QualityScorer()
    
    all_scores = {"clip": [], "resolution": [], "aspect_ratio": [], "blur": [], "total": []}
    
    for i in range(len(dataset)):
        meta = dataset.meta[i]
        q = scorer.compute_quality(
            height=meta["height"],
            width=meta["width"],
            laplacian_var=meta["laplacian_var"],
            clip_similarity=meta["clip_similarity"],
        )
        for k in all_scores:
            all_scores[k].append(q[k])
    
    print(f"\n  样本数: {len(dataset)}")
    print(f"  {'维度':>14s} | {'均值':>6s} | {'标准差':>6s} | {'最小':>6s} | {'最大':>6s}")
    print(f"  {'-'*14}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}-+-{'-'*6}")
    
    for dim_name, values in all_scores.items():
        arr = np.array(values)
        print(
            f"  {dim_name:>14s} | {arr.mean():>6.3f} | {arr.std():>6.3f} | "
            f"{arr.min():>6.3f} | {arr.max():>6.3f}"
        )
    
    # 过滤效果
    total_arr = np.array(all_scores["total"])
    for threshold in [0.2, 0.3, 0.4, 0.5, 0.6]:
        kept = np.sum(total_arr >= threshold)
        print(f"\n  质量阈值 {threshold:.1f}: 保留 {kept}/{len(dataset)} ({kept/len(dataset):.1%})")
    
    # 质量分布直方图（文本版）
    print("\n  质量评分分布（直方图）:")
    hist, edges = np.histogram(total_arr, bins=10, range=(0, 1))
    max_count = max(hist)
    for i in range(len(hist)):
        bar = "█" * int(hist[i] / max(max_count, 1) * 30)
        print(f"    [{edges[i]:.1f}-{edges[i+1]:.1f}): {hist[i]:>4d} {bar}")


# ============================================================
# 主入口
# ============================================================
def main():
    set_seed(42)
    
    print("╔" + "═" * 58 + "╗")
    print("║      V21 多模态数据工程 — 推理实验            ║")
    print("╚" + "═" * 58 + "╝")
    
    experiment_dedup()
    experiment_augmentation()
    experiment_curriculum()
    experiment_quality_scores()
    
    print("\n" + "=" * 60)
    print("所有实验完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
