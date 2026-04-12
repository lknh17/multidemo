"""v06 预训练数据集"""
import os, json, torch, random
from torch.utils.data import Dataset, DataLoader
from config import PretrainConfig

class PretrainDataset(Dataset):
    def __init__(self, num_samples=2000, cfg=None):
        self.cfg = cfg or PretrainConfig()
        self.n = num_samples
        random.seed(42)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        img = torch.randn(3, self.cfg.image_size, self.cfg.image_size)
        ids = torch.randint(4, self.cfg.vocab_size, (self.cfg.max_text_len,))
        itm_label = 1 if random.random() > 0.5 else 0  # 50% 负样本
        cap_labels = torch.cat([ids[1:], torch.tensor([2])])
        return {"image": img, "input_ids": ids, "itm_label": itm_label, "cap_labels": cap_labels}

def create_dataloaders(cfg):
    train = DataLoader(PretrainDataset(2000, cfg), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val = DataLoader(PretrainDataset(400, cfg), batch_size=cfg.batch_size)
    return train, val
