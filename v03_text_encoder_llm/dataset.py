"""v03 简易字符级 Tokenizer + 文本数据集"""
import os, json, re, random
from collections import Counter
from typing import List
import torch
from torch.utils.data import Dataset, DataLoader
from config import GPTConfig


class SimpleTokenizer:
    """简易字符/词级 Tokenizer (用于 demo，真实场景用 BPE)。"""
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.token2id = {"<pad>": 0, "<bos>": 1, "<eos>": 2, "<unk>": 3}
        self.id2token = {v: k for k, v in self.token2id.items()}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = 0, 1, 2, 3
    
    def fit(self, texts: List[str]):
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        for word, _ in counter.most_common(self.vocab_size - len(self.token2id)):
            idx = len(self.token2id)
            self.token2id[word] = idx
            self.id2token[idx] = word
    
    def encode(self, text: str) -> List[int]:
        return [self.token2id.get(w, self.unk_id) for w in text.split()]
    
    def decode(self, ids: List[int]) -> str:
        return " ".join(self.id2token.get(i, "<unk>") for i in ids if i not in (0,1,2))


class TextDataset(Dataset):
    def __init__(self, token_ids: List[int], seq_len: int):
        self.seq_len = seq_len
        self.data = torch.tensor(token_ids, dtype=torch.long)
    
    def __len__(self):
        return max(0, len(self.data) - self.seq_len)
    
    def __getitem__(self, idx):
        chunk = self.data[idx: idx + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}


def generate_demo_texts(save_dir="demo_data", n=500):
    """生成简单的 demo 文本数据。"""
    os.makedirs(save_dir, exist_ok=True)
    templates = [
        "the cat sat on the mat",
        "a dog runs in the park",
        "the quick brown fox jumps over the lazy dog",
        "machine learning is a branch of artificial intelligence",
        "deep learning models can process images and text",
        "transformers have revolutionized natural language processing",
        "attention mechanism allows models to focus on relevant parts",
        "neural networks learn representations from data",
        "the model predicts the next token in the sequence",
        "training requires computing gradients and updating weights",
    ]
    texts = [random.choice(templates) for _ in range(n)]
    with open(os.path.join(save_dir, "texts.json"), "w") as f:
        json.dump(texts, f, indent=2)
    return texts


def create_dataloaders(config: GPTConfig, data_dir="demo_data"):
    texts = generate_demo_texts(data_dir)
    tokenizer = SimpleTokenizer(config.vocab_size)
    tokenizer.fit(texts)
    all_ids = []
    for t in texts:
        all_ids.extend([tokenizer.bos_id] + tokenizer.encode(t) + [tokenizer.eos_id])
    split = int(len(all_ids) * 0.9)
    train_ds = TextDataset(all_ids[:split], config.max_seq_len)
    val_ds = TextDataset(all_ids[split:], config.max_seq_len)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, tokenizer
