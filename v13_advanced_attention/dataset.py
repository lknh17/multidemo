"""v13 高级注意力机制 - 长序列语言建模数据集"""
import os
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader
from config import AdvancedAttentionConfig


class LongSequenceDataset(Dataset):
    """
    长序列语言建模数据集。
    
    生成模拟的 token 序列用于语言建模（next token prediction）。
    支持不同长度的序列以测试长上下文能力。
    """
    
    def __init__(self, num_samples: int = 2000, cfg: AdvancedAttentionConfig = None):
        self.cfg = cfg or AdvancedAttentionConfig()
        self.num_samples = num_samples
        self.data = self._generate_data()
    
    def _generate_data(self):
        """
        生成模拟语言建模数据。
        
        构造具有局部和全局依赖的序列：
        - 局部依赖：相邻 token 有模式 (n-gram)
        - 全局依赖：序列开头的 token 影响序列末尾
        """
        data = []
        for _ in range(self.num_samples):
            seq_len = self.cfg.max_seq_len
            
            # 生成具有结构的序列
            tokens = []
            # 开头种子（全局依赖：开头决定整体模式）
            seed = random.randint(1, 20)
            tokens.append(seed)
            
            for i in range(1, seq_len):
                # 局部依赖：基于前一个 token
                local = (tokens[-1] * 7 + 13) % (self.cfg.vocab_size - 1) + 1
                # 全局依赖：偶尔回溯到种子
                if i % 32 == 0:
                    local = (seed * 11 + i) % (self.cfg.vocab_size - 1) + 1
                # 添加随机噪声
                if random.random() < 0.1:
                    local = random.randint(1, self.cfg.vocab_size - 1)
                tokens.append(local)
            
            data.append(tokens)
        return data
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels}


def create_dataloaders(cfg: AdvancedAttentionConfig):
    """创建训练和验证数据加载器"""
    train_ds = LongSequenceDataset(2000, cfg)
    val_ds = LongSequenceDataset(400, cfg)
    return (
        DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=cfg.batch_size),
    )


def save_demo_data():
    """保存 demo 数据"""
    os.makedirs("demo_data", exist_ok=True)
    samples = []
    for i in range(20):
        tokens = [random.randint(1, 100) for _ in range(64)]
        samples.append({"id": i, "tokens": tokens, "length": len(tokens)})
    with open("demo_data/long_sequences.json", "w") as f:
        json.dump(samples, f, indent=2)
    print(f"[Demo Data] 已生成 {len(samples)} 个长序列样本")
