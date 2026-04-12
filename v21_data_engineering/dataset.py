"""
V21 - 多模态数据工程 - 合成数据集
==================================
构造合成数据集，演示数据管道各阶段（过滤、去重、增强、课程学习、平衡）。
"""

import random
from typing import Tuple, List, Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import FullConfig


class SyntheticImageDataset(Dataset):
    """
    合成图像数据集。
    
    生成带有类别标签的随机图像，用于演示数据工程管道。
    
    特点：
    - 不同类别有不同的"特征模式"（通过不同频率的正弦图案区分）
    - 可配置类别不平衡
    - 可附加质量元信息（分辨率、模糊度等）
    """
    
    def __init__(
        self,
        config: FullConfig,
        num_samples: int,
        imbalanced: bool = False,
        imbalance_ratio: float = 10.0,
        include_meta: bool = True,
    ):
        self.config = config
        self.num_samples = num_samples
        self.include_meta = include_meta
        
        # 生成类别标签
        if imbalanced:
            self.labels = self._generate_imbalanced_labels(
                num_samples, config.num_classes, imbalance_ratio
            )
        else:
            self.labels = [
                random.randint(0, config.num_classes - 1)
                for _ in range(num_samples)
            ]
        
        # 生成元信息
        self.meta = self._generate_meta(num_samples) if include_meta else None
    
    @staticmethod
    def _generate_imbalanced_labels(
        num_samples: int, num_classes: int, ratio: float
    ) -> List[int]:
        """
        生成不平衡的类别分布。
        
        类别 0 的样本数最多，类别 c 的样本数 ∝ 1/ratio^(c/(C-1))。
        """
        weights = [1.0 / (ratio ** (c / max(num_classes - 1, 1)))
                    for c in range(num_classes)]
        total_weight = sum(weights)
        probs = [w / total_weight for w in weights]
        
        labels = random.choices(
            range(num_classes), weights=probs, k=num_samples
        )
        return labels
    
    @staticmethod
    def _generate_meta(num_samples: int) -> List[Dict]:
        """
        生成样本元信息（模拟真实场景的数据质量属性）。
        """
        meta = []
        for _ in range(num_samples):
            meta.append({
                "height": random.choice([64, 128, 224, 256, 512]),
                "width": random.choice([64, 128, 224, 256, 512]),
                "laplacian_var": random.uniform(10, 500),
                "clip_similarity": random.uniform(0.1, 0.9),
                "file_size": random.randint(500, 100000),
            })
        return meta
    
    def _generate_image(self, label: int, idx: int) -> torch.Tensor:
        """
        根据类别生成带特征模式的合成图像。
        
        不同类别使用不同的频率和相位，使分类器有东西可学。
        """
        C = self.config.in_channels
        H = self.config.image_size
        W = self.config.image_size
        
        # 基于类别的频率和相位
        rng = np.random.RandomState(label * 1000 + idx)
        freq = (label + 1) * 2.0
        phase = label * 0.5
        
        # 生成正弦图案
        y_coords = np.linspace(0, freq * np.pi, H)
        x_coords = np.linspace(0, freq * np.pi, W)
        
        pattern = np.sin(y_coords[:, None] + phase) * np.cos(x_coords[None, :] + phase)
        pattern = (pattern + 1) / 2  # 归一化到 [0, 1]
        
        # 添加类别特定的颜色偏移
        image = np.stack([
            pattern * (0.5 + 0.5 * np.sin(label * 0.7 + c))
            for c in range(C)
        ])
        
        # 添加随机噪声
        noise = rng.randn(C, H, W) * 0.1
        image = np.clip(image + noise, 0, 1)
        
        return torch.tensor(image, dtype=torch.float32)
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict:
        label = self.labels[idx]
        image = self._generate_image(label, idx)
        
        sample = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
        }
        
        if self.meta is not None:
            sample["meta"] = self.meta[idx]
        
        return sample


class TextDatasetForDedup(Dataset):
    """
    用于测试去重的合成文本数据集。
    
    会故意插入一些近似重复的样本。
    """
    
    def __init__(self, num_samples: int = 500, dup_ratio: float = 0.3):
        self.texts = self._generate_texts(num_samples, dup_ratio)
    
    @staticmethod
    def _generate_texts(num_samples: int, dup_ratio: float) -> List[str]:
        """生成带有近似重复的文本列表。"""
        templates = [
            "一只{adj}的{animal}在{place}{action}",
            "{place}里有一只{adj}的{animal}正在{action}",
            "今天天气{weather}，适合{activity}",
            "{person}正在{place}里{action}",
            "这张照片展示了{adj}的{scene}",
        ]
        
        adjs = ["可爱", "美丽", "活泼", "安静", "巨大", "小巧"]
        animals = ["小猫", "小狗", "兔子", "鹦鹉", "金鱼"]
        places = ["公园", "草地", "房间", "花园", "森林"]
        actions = ["玩耍", "奔跑", "休息", "吃东西", "睡觉"]
        weathers = ["晴朗", "多云", "微风", "温暖"]
        activities = ["散步", "拍照", "野餐", "运动"]
        persons = ["小明", "小红", "老师", "同学"]
        scenes = ["日落", "山峦", "海滩", "城市夜景"]
        
        texts = []
        for i in range(num_samples):
            tmpl = random.choice(templates)
            text = tmpl.format(
                adj=random.choice(adjs),
                animal=random.choice(animals),
                place=random.choice(places),
                action=random.choice(actions),
                weather=random.choice(weathers),
                activity=random.choice(activities),
                person=random.choice(persons),
                scene=random.choice(scenes),
            )
            texts.append(text)
        
        # 插入近似重复
        n_dup = int(num_samples * dup_ratio)
        for _ in range(n_dup):
            src_idx = random.randint(0, len(texts) - 1)
            # 轻微修改（替换一个字）
            src = texts[src_idx]
            if len(src) > 5:
                pos = random.randint(2, len(src) - 3)
                dup = src[:pos] + random.choice(["的", "了", "在", "着"]) + src[pos+1:]
                texts.append(dup)
        
        return texts
    
    def __len__(self) -> int:
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def create_dataloaders(
    config: FullConfig,
    imbalanced: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """创建训练和验证数据加载器。"""
    train_dataset = SyntheticImageDataset(
        config, config.num_train_samples,
        imbalanced=imbalanced, include_meta=True,
    )
    val_dataset = SyntheticImageDataset(
        config, config.num_val_samples,
        imbalanced=False, include_meta=False,
    )
    
    def collate_fn(batch):
        images = torch.stack([b["image"] for b in batch])
        labels = torch.stack([b["label"] for b in batch])
        result = {"image": images, "label": labels}
        if "meta" in batch[0] and batch[0]["meta"] is not None:
            result["meta"] = [b["meta"] for b in batch]
        return result
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
        collate_fn=collate_fn,
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    config = FullConfig()
    
    # 测试图像数据集
    print("=" * 50)
    print("SyntheticImageDataset 测试")
    print("=" * 50)
    
    train_loader, val_loader = create_dataloaders(config, imbalanced=True)
    batch = next(iter(train_loader))
    
    print(f"训练集: {len(train_loader.dataset)} 样本, {len(train_loader)} 批")
    print(f"Batch 形状:")
    print(f"  image: {batch['image'].shape}")
    print(f"  label: {batch['label'].shape}")
    print(f"  label 分布: {torch.bincount(batch['label'], minlength=10).tolist()}")
    
    if "meta" in batch:
        print(f"  meta 示例: {batch['meta'][0]}")
    
    # 测试文本去重数据集
    print("\n" + "=" * 50)
    print("TextDatasetForDedup 测试")
    print("=" * 50)
    
    text_ds = TextDatasetForDedup(num_samples=100, dup_ratio=0.3)
    print(f"文本数量: {len(text_ds)}")
    print(f"示例: {text_ds[0]}")
    print(f"示例: {text_ds[1]}")
