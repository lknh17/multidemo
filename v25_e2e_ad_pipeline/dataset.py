"""
V25 - 广告创意合成数据集
========================
生成带曝光/点击信号的多模态广告数据
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict
from config import FullConfig


class AdCreativeDataset(Dataset):
    """合成广告创意数据：图像 + 文本 + 音频 + 行为信号"""

    def __init__(self, config: FullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.creative.image_size
        self.text_len = config.creative.text_max_len
        self.audio_dim = config.creative.audio_dim
        self.has_audio = config.creative.has_audio

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 200000))

        # 广告类别（影响点击率）
        ad_category = rng.randint(0, 9)
        quality = rng.random()

        # 图像：类别相关的纹理
        image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
        cx, cy = rng.randint(30, self.img_size - 30), rng.randint(30, self.img_size - 30)
        r = rng.randint(15, 30)
        for dy in range(-r, r):
            for dx in range(-r, r):
                if dx * dx + dy * dy < r * r:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < self.img_size and 0 <= px < self.img_size:
                        image[ad_category % 3, py, px] += 0.3 * quality
        image = image.clamp(0, 1)

        # 文本：随机 token 序列（类别偏置）
        text_ids = torch.zeros(self.text_len, dtype=torch.long)
        actual_len = rng.randint(5, self.text_len)
        for i in range(actual_len):
            text_ids[i] = rng.randint(1, self.config.creative.vocab_size - 1)
        # 类别相关 token
        text_ids[0] = ad_category * 100 + 1

        # 音频特征
        audio_feats = torch.randn(self.audio_dim) * 0.3 if self.has_audio else torch.zeros(self.audio_dim)

        # 用户特征（模拟）
        user_emb = torch.randn(self.config.creative.d_model) * 0.1
        user_preference = rng.randint(0, 9)
        user_emb[user_preference * 20:(user_preference + 1) * 20] += 0.5

        # 上下文特征
        context_emb = torch.randn(self.config.creative.d_model) * 0.1
        hour = rng.randint(0, 23)
        context_emb[hour * 10:(hour + 1) * 10] += 0.3

        # 行为信号：CTR 与类别匹配度 + 质量相关
        match_score = 1.0 if ad_category == user_preference else 0.2
        base_ctr = 0.05 + 0.15 * match_score + 0.1 * quality
        clicked = 1 if rng.random() < base_ctr else 0

        # 相关性标签
        relevance = match_score * 0.7 + quality * 0.3

        # 新鲜度（广告年龄：天）
        age_days = rng.randint(0, 30)

        return {
            'image': image,
            'text_ids': text_ids,
            'audio_feats': audio_feats,
            'user_emb': user_emb,
            'context_emb': context_emb,
            'clicked': torch.tensor(clicked, dtype=torch.float),
            'relevance': torch.tensor(relevance, dtype=torch.float),
            'quality': torch.tensor(quality, dtype=torch.float),
            'age_days': torch.tensor(age_days, dtype=torch.float),
            'ad_category': ad_category,
        }


class PairwiseAdDataset(Dataset):
    """成对广告数据：用于对比学习 / 排序学习"""

    def __init__(self, config: FullConfig, split="train"):
        self.base = AdCreativeDataset(config, split)
        self.num_samples = len(self.base)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        query = self.base[idx]
        # 正例：同类别的广告
        rng = random.Random(idx + 300000)
        pos_idx = rng.randint(0, self.num_samples - 1)
        positive = self.base[pos_idx]
        return {'query': query, 'positive': positive}


def create_ad_dataloaders(config: FullConfig):
    return (
        DataLoader(AdCreativeDataset(config, "train"),
                   batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(AdCreativeDataset(config, "val"),
                   batch_size=config.batch_size))


def create_pairwise_dataloaders(config: FullConfig):
    return (
        DataLoader(PairwiseAdDataset(config, "train"),
                   batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(PairwiseAdDataset(config, "val"),
                   batch_size=config.batch_size))
