"""
V18 - 商品理解数据集
====================
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple
from config import ProductFullConfig


class FineGrainedDataset(Dataset):
    def __init__(self, config: ProductFullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.fine_grained.image_size

    def __len__(self): return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 100000))
        label = rng.randint(0, self.config.fine_grained.num_classes - 1)
        image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
        cx, cy = rng.randint(30, self.img_size-30), rng.randint(30, self.img_size-30)
        r = rng.randint(10, 25)
        for dy in range(-r, r):
            for dx in range(-r, r):
                if dx*dx+dy*dy < r*r and 0<=cy+dy<self.img_size and 0<=cx+dx<self.img_size:
                    image[label % 3, cy+dy, cx+dx] += 0.4
        return {'image': image.clamp(0,1), 'label': label}


class ProductAttributeDataset(Dataset):
    def __init__(self, config: ProductFullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.product_attr.image_size
        self.attr_cfg = config.product_attr

    def __len__(self): return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 110000))
        image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
        category = rng.randint(0, self.attr_cfg.num_categories - 1)
        brand = rng.randint(0, self.attr_cfg.num_brands - 1)
        color = torch.zeros(self.attr_cfg.num_colors)
        for _ in range(rng.randint(1, 3)):
            color[rng.randint(0, self.attr_cfg.num_colors - 1)] = 1.0
        material = torch.zeros(self.attr_cfg.num_materials)
        material[rng.randint(0, self.attr_cfg.num_materials - 1)] = 1.0
        style = torch.zeros(self.attr_cfg.num_styles)
        style[rng.randint(0, self.attr_cfg.num_styles - 1)] = 1.0
        return {'image': image.clamp(0,1), 'category': category, 'brand': brand,
                'color': color, 'material': material, 'style': style}


class QualityDataset(Dataset):
    def __init__(self, config: ProductFullConfig, split="train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.quality.image_size

    def __len__(self): return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 120000))
        quality = rng.random()
        image = torch.randn(3, self.img_size, self.img_size) * (0.1 + 0.3 * (1 - quality)) + 0.5
        dim_scores = torch.tensor([rng.random() for _ in range(5)])
        overall = dim_scores.mean().item() * 0.8 + quality * 0.2
        return {'image': image.clamp(0,1), 'dim_scores': dim_scores.float(),
                'overall_score': torch.tensor(overall).float()}


def create_fine_grained_dataloaders(config):
    return (DataLoader(FineGrainedDataset(config, "train"), batch_size=config.batch_size, shuffle=True, drop_last=True),
            DataLoader(FineGrainedDataset(config, "val"), batch_size=config.batch_size))

def create_attribute_dataloaders(config):
    return (DataLoader(ProductAttributeDataset(config, "train"), batch_size=config.batch_size, shuffle=True, drop_last=True),
            DataLoader(ProductAttributeDataset(config, "val"), batch_size=config.batch_size))

def create_quality_dataloaders(config):
    return (DataLoader(QualityDataset(config, "train"), batch_size=config.batch_size, shuffle=True, drop_last=True),
            DataLoader(QualityDataset(config, "val"), batch_size=config.batch_size))
