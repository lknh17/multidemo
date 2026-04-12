"""
V24 - 内容安全数据集
====================
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from config import SafetyFullConfig


class SafetyClassificationDataset(Dataset):
    """合成安全分类数据集（多标签）"""

    def __init__(self, config: SafetyFullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.safety_cls.image_size
        self.num_categories = config.safety_cls.num_categories

    def __len__(self): return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 200000))
        image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5

        # 多标签：每个类别独立采样
        labels = torch.zeros(self.num_categories)
        # 大部分图片是正常的（最后一个类别）
        if rng.random() < 0.6:
            labels[-1] = 1.0  # normal
        else:
            # 随机触发 1~3 个非安全类别
            n_unsafe = rng.randint(1, 3)
            unsafe_cats = rng.sample(range(self.num_categories - 1), min(n_unsafe, self.num_categories - 1))
            for c in unsafe_cats:
                labels[c] = 1.0
                # 给图像注入类别特征
                channel = c % 3
                cx = rng.randint(20, self.img_size - 20)
                cy = rng.randint(20, self.img_size - 20)
                r = rng.randint(8, 20)
                for dy in range(-r, r):
                    for dx in range(-r, r):
                        if dx*dx+dy*dy < r*r and 0<=cy+dy<self.img_size and 0<=cx+dx<self.img_size:
                            image[channel, cy+dy, cx+dx] += 0.3 + 0.1 * c

        return {'image': image.clamp(0, 1), 'labels': labels}


class ToxicityDataset(Dataset):
    """合成毒性检测数据集（文本 + 图像）"""

    def __init__(self, config: SafetyFullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.toxicity.image_size
        self.seq_len = config.toxicity.max_seq_len
        self.vocab_size = config.toxicity.vocab_size

    def __len__(self): return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 210000))
        image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
        token_ids = torch.randint(0, self.vocab_size, (self.seq_len,))

        # 毒性分数（6 维）
        toxicity = rng.random()
        dim_scores = torch.tensor([rng.random() for _ in range(6)])
        if toxicity > 0.5:
            # 高毒性样本：增强某些维度
            dim_scores[rng.randint(0, 4)] = 0.7 + rng.random() * 0.3
        overall = dim_scores[:5].mean().item() * 0.8 + toxicity * 0.2

        return {
            'image': image.clamp(0, 1),
            'token_ids': token_ids,
            'dim_scores': dim_scores.float(),
            'overall_score': torch.tensor(overall).float(),
        }


class WatermarkDataset(Dataset):
    """合成水印数据集"""

    def __init__(self, config: SafetyFullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.watermark.image_size
        self.watermark_bits = config.watermark.watermark_bits

    def __len__(self): return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 220000))
        image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
        # 随机水印比特 ∈ {-1, +1}
        watermark = torch.tensor([rng.choice([-1.0, 1.0]) for _ in range(self.watermark_bits)])
        return {'image': image.clamp(0, 1), 'watermark': watermark}


class AdversarialDataset(Dataset):
    """对抗训练数据集"""

    def __init__(self, config: SafetyFullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.adversarial.image_size
        self.num_classes = config.adversarial.num_classes

    def __len__(self): return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 230000))
        image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
        # 多标签
        labels = torch.zeros(self.num_classes)
        if rng.random() < 0.6:
            labels[-1] = 1.0
        else:
            for c in rng.sample(range(self.num_classes - 1), rng.randint(1, 2)):
                labels[c] = 1.0
        return {'image': image.clamp(0, 1), 'labels': labels}


def create_safety_dataloaders(config):
    return (DataLoader(SafetyClassificationDataset(config, "train"),
                       batch_size=config.batch_size, shuffle=True, drop_last=True),
            DataLoader(SafetyClassificationDataset(config, "val"),
                       batch_size=config.batch_size))

def create_adversarial_dataloaders(config):
    return (DataLoader(AdversarialDataset(config, "train"),
                       batch_size=config.batch_size, shuffle=True, drop_last=True),
            DataLoader(AdversarialDataset(config, "val"),
                       batch_size=config.batch_size))

def create_watermark_dataloaders(config):
    return (DataLoader(WatermarkDataset(config, "train"),
                       batch_size=config.batch_size, shuffle=True, drop_last=True),
            DataLoader(WatermarkDataset(config, "val"),
                       batch_size=config.batch_size))
