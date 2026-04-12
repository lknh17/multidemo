"""v05 多模态对话数据集"""
import os, json, torch
from torch.utils.data import Dataset, DataLoader
from config import QwenVLConfig
import numpy as np

class MultimodalDialogDataset(Dataset):
    def __init__(self, num_samples=500, cfg=None):
        self.cfg = cfg or QwenVLConfig()
        self.num_samples = num_samples
        np.random.seed(42)
    
    def __len__(self): return self.num_samples
    
    def __getitem__(self, idx):
        img = torch.randn(3, self.cfg.image_size, self.cfg.image_size)
        text_len = min(16, self.cfg.max_seq_len)
        ids = torch.randint(4, self.cfg.vocab_size, (text_len,))
        labels = torch.cat([ids[1:], torch.tensor([2])])  # shift + EOS
        return {"image": img, "input_ids": ids, "labels": labels}

def create_dataloaders(cfg):
    train = DataLoader(MultimodalDialogDataset(500, cfg), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val = DataLoader(MultimodalDialogDataset(100, cfg), batch_size=cfg.batch_size)
    return train, val

def save_demo_data():
    os.makedirs("demo_data/images", exist_ok=True)
    convs = [{"id": i, "image": f"images/ad_{i}.png", "question": "这张广告展示了什么产品?", "answer": f"示例回答 {i}"} for i in range(10)]
    with open("demo_data/conversations.json", "w", encoding="utf-8") as f:
        json.dump(convs, f, ensure_ascii=False, indent=2)
