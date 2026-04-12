"""v07 广告场景 SFT 数据集"""
import os, json, torch, random
from torch.utils.data import Dataset, DataLoader
from config import SFTConfig

AD_INSTRUCTIONS = [
    "请描述这张广告图片的内容",
    "提取广告中的商品属性",
    "这张广告推广的是什么产品",
    "根据广告图片生成一段推广文案",
    "判断这张广告与'{query}'的相关性",
]

class SFTDataset(Dataset):
    def __init__(self, num_samples=1000, cfg=None):
        self.cfg = cfg or SFTConfig()
        self.n = num_samples
        random.seed(42)
    def __len__(self): return self.n
    def __getitem__(self, idx):
        img = torch.randn(3, self.cfg.image_size, self.cfg.image_size)
        ids = torch.randint(4, self.cfg.vocab_size, (self.cfg.max_text_len,))
        labels = torch.cat([ids[1:], torch.tensor([2])])
        return {"image": img, "input_ids": ids, "labels": labels}

def create_dataloaders(cfg):
    train = DataLoader(SFTDataset(1000, cfg), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val = DataLoader(SFTDataset(200, cfg), batch_size=cfg.batch_size)
    return train, val

def save_demo_data():
    os.makedirs("demo_data/images", exist_ok=True)
    data = []
    for i in range(20):
        inst = random.choice(AD_INSTRUCTIONS).format(query=f"query_{i}")
        data.append({"id": i, "image": f"images/ad_{i}.png", "instruction": inst, "response": f"这是一张关于产品{i}的广告"})
    with open("demo_data/sft_instructions.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
