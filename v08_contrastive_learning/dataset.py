"""v08 广告相关性数据集"""
import os, json, torch, random
from torch.utils.data import Dataset, DataLoader
from config import ContrastiveConfig

class AdRelevanceDataset(Dataset):
    def __init__(self, num_samples=2000, cfg=None):
        self.cfg = cfg or ContrastiveConfig()
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
        DataLoader(AdRelevanceDataset(2000, cfg), batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        DataLoader(AdRelevanceDataset(400, cfg), batch_size=cfg.batch_size),
    )

def save_demo_data():
    os.makedirs("demo_data/images", exist_ok=True)
    data = [{"id": i, "image": f"images/ad_{i}.png", "query": f"query_{i}", "relevance": random.randint(0,4)} for i in range(50)]
    with open("demo_data/relevance_pairs.json", "w") as f: json.dump(data, f, indent=2)
