"""v10 带商业标签的广告数据集"""
import os, json, torch, random
from torch.utils.data import Dataset, DataLoader
from config import ConceptConfig

class ConceptDataset(Dataset):
    def __init__(self, num_samples=2000, cfg=None):
        self.cfg = cfg or ConceptConfig()
        self.n = num_samples
    def __len__(self): return self.n
    def __getitem__(self, idx):
        cls = idx % 20
        img = torch.randn(3, self.cfg.image_size, self.cfg.image_size)
        img[cls % 3] += 0.5
        ids = torch.tensor([(cls * 7 + i * 13) % (self.cfg.vocab_size - 4) + 4 for i in range(self.cfg.max_text_len)])
        return {
            "image": img, "input_ids": ids,
            "industry": cls % self.cfg.num_industries,
            "brand": (cls * 3) % self.cfg.num_brands,
            "attributes": torch.zeros(self.cfg.num_attributes).scatter_(0, torch.tensor([cls % self.cfg.num_attributes, (cls+1) % self.cfg.num_attributes]), 1.0),
            "intent": cls % self.cfg.num_intents,
        }

def create_dataloaders(cfg):
    return (
        DataLoader(ConceptDataset(2000, cfg), batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        DataLoader(ConceptDataset(400, cfg), batch_size=cfg.batch_size),
    )

def save_demo_data():
    os.makedirs("demo_data/images", exist_ok=True)
    data = [{"id": i, "industry": f"industry_{i%10}", "brand": f"brand_{i%50}", "attributes": [f"attr_{i%20}"], "intent": f"intent_{i%5}"} for i in range(50)]
    with open("demo_data/concept_labels.json", "w") as f: json.dump(data, f, indent=2)
