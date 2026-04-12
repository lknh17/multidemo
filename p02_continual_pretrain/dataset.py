"""
p02 继续预训练 - 数据集处理

实现两种数据处理策略：
1. Padding 模式 — 每条数据独立 tokenize，短的补 pad
2. Packing 模式 — 多条短文本拼接为一条长序列，效率更高

使用方式:
    from dataset import create_pretrain_dataset
"""

import os
import sys
import json
from typing import Optional, Iterator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 数据加载
# ============================================================
def load_jsonl(file_path: str) -> list:
    """加载 JSONL 格式数据"""
    samples = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


# ============================================================
# 2. Packing 数据集（推荐）
# ============================================================
class PackedDataset:
    """
    Packing 数据集：将多条短文本拼接为固定长度的长序列。
    
    为什么用 Packing？
    - Padding 模式中，短文本的 pad 部分不参与计算但占用显存和算力
    - Packing 将多条文本拼接到 max_seq_length，几乎没有浪费
    - 对于平均长度 500 tokens、max_seq_length 512 的场景，Packing 效率接近 100%
    
    实现方式：
    1. 逐条 tokenize 文本
    2. 将所有 token_ids 拼接成一个长序列
    3. 切分为 max_seq_length 长度的块
    4. 每个块就是一个训练样本
    
    注意：Packing 中不同文档的 token 会相邻，
    但 CLM 的因果注意力确保每个 token 只看前面的，
    所以不会产生跨文档的信息泄漏问题。
    """
    
    def __init__(
        self,
        texts: list,
        tokenizer,
        max_seq_length: int = 512,
    ):
        self.max_seq_length = max_seq_length
        self.samples = self._pack(texts, tokenizer)
    
    def _pack(self, texts: list, tokenizer) -> list:
        """将文本 tokenize 并拼接打包"""
        import torch
        
        # 收集所有 token ids
        all_token_ids = []
        eos_token_id = tokenizer.eos_token_id or tokenizer.pad_token_id
        
        for text in texts:
            if isinstance(text, dict):
                text = text.get("text", "")
            
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_token_ids.extend(tokens)
            all_token_ids.append(eos_token_id)  # 文档分隔符
        
        # 切分为固定长度的块
        samples = []
        for i in range(0, len(all_token_ids) - self.max_seq_length, self.max_seq_length):
            chunk = all_token_ids[i : i + self.max_seq_length]
            if len(chunk) == self.max_seq_length:
                samples.append({
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    "labels": torch.tensor(chunk, dtype=torch.long),
                    # CLM: labels = input_ids（移位在模型内部处理）
                })
        
        print(f"  [Packing] {len(texts)} 条文本 → {len(all_token_ids)} tokens → {len(samples)} 个训练样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 3. Padding 数据集（对比用）
# ============================================================
class PaddedDataset:
    """
    Padding 数据集：每条文本独立 tokenize，短的补 pad。
    
    比 Packing 效率低，但实现简单、语义更清晰。
    """
    
    def __init__(self, texts: list, tokenizer, max_seq_length: int = 512):
        self.samples = self._process(texts, tokenizer, max_seq_length)
    
    def _process(self, texts, tokenizer, max_seq_length):
        import torch
        
        samples = []
        for text in texts:
            if isinstance(text, dict):
                text = text.get("text", "")
            
            encoded = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                padding="max_length",
                return_tensors="pt",
            )
            
            input_ids = encoded["input_ids"].squeeze(0)
            attention_mask = encoded["attention_mask"].squeeze(0)
            
            # labels: 将 pad 位置设为 -100（不参与 loss 计算）
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100
            
            samples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            })
        
        print(f"  [Padding] {len(texts)} 条文本 → {len(samples)} 个训练样本")
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# 4. 统一创建函数
# ============================================================
def create_pretrain_dataset(
    data_path: str,
    tokenizer,
    max_seq_length: int = 512,
    packing: bool = True,
    max_samples: Optional[int] = None,
):
    """
    创建预训练数据集。
    
    Args:
        data_path: JSONL 数据文件路径
        tokenizer: HuggingFace tokenizer
        max_seq_length: 最大序列长度
        packing: True=Packing模式, False=Padding模式
        max_samples: 最大使用样本数
    """
    print(f"\n[数据处理] 模式: {'Packing' if packing else 'Padding'}")
    print(f"  数据文件: {data_path}")
    print(f"  序列长度: {max_seq_length}")
    
    texts = load_jsonl(data_path)
    
    if max_samples:
        texts = texts[:max_samples]
    print(f"  原始数据: {len(texts)} 条")
    
    if packing:
        return PackedDataset(texts, tokenizer, max_seq_length)
    else:
        return PaddedDataset(texts, tokenizer, max_seq_length)
