"""
V22 - 合成实验日志与指标数据
==============================
"""
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from config import FullConfig


class RetrievalEvalDataset(Dataset):
    """合成检索评估数据集：生成 query-doc 相关性分数"""

    def __init__(self, config: FullConfig, split="train"):
        self.config = config
        self.split = split
        n = config.num_train_samples if split == "train" else config.num_val_samples
        self.num_samples = n
        self.num_candidates = config.offline.num_candidates
        self.relevance_levels = config.offline.relevance_levels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 200000))

        # 生成候选文档的真实相关性（稀疏，大部分不相关）
        relevance = torch.zeros(self.num_candidates)
        num_relevant = rng.randint(5, 30)
        for _ in range(num_relevant):
            pos = rng.randint(0, self.num_candidates - 1)
            relevance[pos] = rng.randint(1, self.relevance_levels - 1)

        # 模型A的预测分数（较好的模型，相关文档得分偏高）
        noise_a = torch.randn(self.num_candidates) * 0.3
        scores_a = relevance * 0.7 + noise_a + torch.randn(self.num_candidates) * 0.1

        # 模型B的预测分数（较差的模型，噪声更大）
        noise_b = torch.randn(self.num_candidates) * 0.6
        scores_b = relevance * 0.4 + noise_b + torch.randn(self.num_candidates) * 0.2

        return {
            'relevance': relevance,
            'scores_a': scores_a,
            'scores_b': scores_b,
            'query_id': torch.tensor(idx),
        }


class ABTestLogDataset(Dataset):
    """合成 A/B 测试实验日志"""

    def __init__(self, config: FullConfig, split="train"):
        self.config = config
        self.split = split
        n = config.num_train_samples if split == "train" else config.num_val_samples
        self.num_samples = n
        self.num_metrics = config.ab_test.num_metrics

        # 生成模拟日志
        rng = np.random.RandomState(42 if split == "train" else 123)

        # 对照组：基准指标
        self.control_metrics = {
            'ctr': rng.normal(0.05, 0.02, n).clip(0, 1),
            'conversion': rng.normal(0.02, 0.01, n).clip(0, 1),
            'revenue': rng.exponential(10.0, n),
            'session_duration': rng.normal(300, 80, n).clip(0),
            'bounce_rate': rng.normal(0.40, 0.10, n).clip(0, 1),
        }

        # 实验组：部分指标有提升
        self.treatment_metrics = {
            'ctr': rng.normal(0.055, 0.02, n).clip(0, 1),           # +10%
            'conversion': rng.normal(0.022, 0.01, n).clip(0, 1),    # +10%
            'revenue': rng.exponential(10.5, n),                      # +5%
            'session_duration': rng.normal(310, 80, n).clip(0),       # +3%
            'bounce_rate': rng.normal(0.38, 0.10, n).clip(0, 1),     # -5%
        }

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        control = {k: torch.tensor(v[idx], dtype=torch.float32)
                   for k, v in self.control_metrics.items()}
        treatment = {k: torch.tensor(v[idx], dtype=torch.float32)
                     for k, v in self.treatment_metrics.items()}
        return {'control': control, 'treatment': treatment, 'user_id': idx}


class BanditSimDataset(Dataset):
    """多臂老虎机模拟数据集"""

    def __init__(self, config: FullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_arms = config.bandit.num_arms
        self.num_rounds = config.bandit.num_rounds

        # 每个臂的真实奖励概率
        rng = np.random.RandomState(42)
        self.true_rates = np.array([0.1, 0.15, 0.3, 0.25, 0.2])[:self.num_arms]

    def __len__(self):
        return self.num_rounds

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 300000))
        rewards = torch.tensor([
            1.0 if rng.random() < self.true_rates[k] else 0.0
            for k in range(self.num_arms)
        ])
        return {
            'rewards': rewards,
            'round_id': torch.tensor(idx),
        }

    def get_true_rates(self) -> np.ndarray:
        return self.true_rates.copy()


class FairnessDataset(Dataset):
    """公平性评估数据集"""

    def __init__(self, config: FullConfig, split="train"):
        self.config = config
        n = config.num_train_samples if split == "train" else config.num_val_samples
        self.num_samples = n

        rng = np.random.RandomState(42 if split == "train" else 99)

        # 敏感属性（0/1 二值）
        self.sensitive = rng.randint(0, 2, n)

        # 真实标签（与敏感属性有一定相关性来模拟偏差）
        base_prob = 0.3
        bias = 0.1  # 组间偏差
        self.labels = np.array([
            1 if rng.random() < (base_prob + bias * s) else 0
            for s in self.sensitive
        ])

        # 模型预测（可能放大偏差）
        self.predictions = np.array([
            1 if rng.random() < (base_prob + bias * 1.5 * s + 0.05 * l) else 0
            for s, l in zip(self.sensitive, self.labels)
        ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return {
            'prediction': torch.tensor(self.predictions[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
            'sensitive': torch.tensor(self.sensitive[idx], dtype=torch.long),
        }


def create_retrieval_dataloaders(config: FullConfig):
    return (
        DataLoader(RetrievalEvalDataset(config, "train"),
                   batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(RetrievalEvalDataset(config, "val"),
                   batch_size=config.batch_size),
    )


def create_abtest_dataloaders(config: FullConfig):
    return (
        DataLoader(ABTestLogDataset(config, "train"),
                   batch_size=config.batch_size, shuffle=False),
        DataLoader(ABTestLogDataset(config, "val"),
                   batch_size=config.batch_size),
    )


def create_bandit_dataset(config: FullConfig):
    return BanditSimDataset(config, "train")


def create_fairness_dataset(config: FullConfig):
    return FairnessDataset(config, "train")
