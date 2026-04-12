"""v04 图文对数据集（模拟数据）"""
import os, json, random
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from config import CLIPConfig


class ImageTextDataset(Dataset):
    """模拟的图文对数据集。"""
    def __init__(self, num_samples=2000, img_size=32, vocab_size=2000, max_text_len=16):
        self.num_samples = num_samples
        self.img_size = img_size
        self.vocab_size = vocab_size
        self.max_text_len = max_text_len
        # 生成模拟数据：每个"类别"对应特定的图像模式和文本模式
        self.num_classes = 20
        random.seed(42)
        np.random.seed(42)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        cls = idx % self.num_classes
        # 图像：以类别决定主色调
        img = torch.randn(3, self.img_size, self.img_size) * 0.3
        img[cls % 3] += 0.5  # 类别决定的颜色偏移
        # 文本：以类别决定的 token 模式
        base_tokens = [(cls * 7 + i * 13) % (self.vocab_size - 4) + 4 for i in range(8)]
        text_ids = base_tokens + [0] * (self.max_text_len - len(base_tokens))
        return {"image": img, "input_ids": torch.tensor(text_ids, dtype=torch.long), "label": cls}


def create_dataloaders(config: CLIPConfig):
    train_ds = ImageTextDataset(2000, config.image_size, config.vocab_size, config.max_text_len)
    val_ds = ImageTextDataset(400, config.image_size, config.vocab_size, config.max_text_len)
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )


def save_demo_data(save_dir="demo_data"):
    os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    captions = []
    for i in range(20):
        captions.append({"id": i, "image": f"images/img_{i:04d}.png", "caption": f"sample caption for class {i}"})
    with open(os.path.join(save_dir, "captions.json"), "w") as f:
        json.dump(captions, f, indent=2)
