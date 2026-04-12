"""v11 检索评估数据集"""
import os, json, torch, random
from torch.utils.data import Dataset, DataLoader
from config import EmbeddingConfig

class RetrievalDataset(Dataset):
    def __init__(self, num_samples, cfg=None):
        self.cfg = cfg or EmbeddingConfig()
        self.n = num_samples
    def __len__(self): return self.n
    def __getitem__(self, idx):
        cls = idx % 20
        img = torch.randn(3, self.cfg.image_size, self.cfg.image_size)
        img[cls % 3] += 0.5
        ids = torch.tensor([(cls * 7 + i * 13) % (self.cfg.vocab_size - 4) + 4 for i in range(self.cfg.max_text_len)])
        return {"image": img, "input_ids": ids, "label": cls}

def create_dataloaders(cfg):
    return (
        DataLoader(RetrievalDataset(2000, cfg), batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        DataLoader(RetrievalDataset(cfg.num_gallery, cfg), batch_size=cfg.batch_size),
        DataLoader(RetrievalDataset(cfg.num_queries, cfg), batch_size=cfg.batch_size),
    )
