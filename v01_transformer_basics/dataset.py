"""
v01 Transformer 基础 - 数据集：数字序列排序

任务描述：
    输入一个乱序的数字序列，如 [5, 2, 8, 1, 3]
    模型需要输出排好序的序列 [1, 2, 3, 5, 8]

这是一个完美的 Transformer 入门任务：
1. 简单直观，容易验证正确性
2. 需要模型理解全局信息（排序需要看到所有元素）
3. 是一个序列到序列（Seq2Seq）任务，可以展示完整的编码器-解码器流程
4. 不需要额外数据，可以无限生成
"""

import os
import json
import random
from typing import Tuple, List

import torch
from torch.utils.data import Dataset, DataLoader

from config import TransformerConfig


class SortingDataset(Dataset):
    """
    数字排序数据集。
    
    每个样本由以下部分组成：
    - src (源序列): 乱序数字序列，如 [5, 2, 8, 1, 3]
    - tgt_input (目标输入): [BOS, 1, 2, 3, 5, 8]  (给解码器的输入，前面加 BOS)
    - tgt_output (目标输出): [1, 2, 3, 5, 8, EOS]  (解码器应该生成的，后面加 EOS)
    
    为什么目标序列要区分 input 和 output？
    - 训练时使用 Teacher Forcing：解码器在每步的输入是真实的前一个 token
    - tgt_input 比 tgt_output 往右移一位
    - 这样模型学的是"给定已知前缀，预测下一个 token"
    """
    
    def __init__(self, config: TransformerConfig, num_samples: int):
        self.config = config
        self.num_samples = num_samples
        self.data = self._generate_data()
    
    def _generate_data(self) -> List[Tuple[List[int], List[int]]]:
        """
        生成排序数据。
        
        每个样本：
        - 从 [1, num_range) 中随机采样 seq_length 个数字（允许重复）作为输入
        - 排序后作为目标
        
        注意：数字从 1 开始（0 是 PAD token）
        """
        data = []
        for _ in range(self.num_samples):
            # 随机生成乱序数字（1 到 num_range-1，避开 0 因为 0 是 PAD）
            numbers = [
                random.randint(1, self.config.num_range - 1)
                for _ in range(self.config.seq_length)
            ]
            sorted_numbers = sorted(numbers)
            data.append((numbers, sorted_numbers))
        return data
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> dict:
        """
        返回一个训练样本。
        
        Returns:
            dict with keys:
                - src: 源序列 (乱序) [seq_length]
                - tgt_input: 解码器输入 [BOS + sorted] [seq_length + 1]
                - tgt_output: 解码器目标 [sorted + EOS] [seq_length + 1]
        """
        src_numbers, tgt_numbers = self.data[idx]
        
        src = torch.tensor(src_numbers, dtype=torch.long)
        
        # Teacher Forcing 的目标：
        # 解码器输入: [BOS, n1, n2, ..., nk]  — 以 BOS 开头
        # 解码器输出: [n1, n2, ..., nk, EOS]  — 以 EOS 结尾
        tgt_input = torch.tensor(
            [self.config.bos_token_id] + tgt_numbers,
            dtype=torch.long
        )
        tgt_output = torch.tensor(
            tgt_numbers + [self.config.eos_token_id],
            dtype=torch.long
        )
        
        return {
            "src": src,
            "tgt_input": tgt_input,
            "tgt_output": tgt_output,
        }


def create_dataloaders(
    config: TransformerConfig,
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证数据加载器。
    
    Returns:
        (train_loader, val_loader)
    """
    train_dataset = SortingDataset(config, config.num_train_samples)
    val_dataset = SortingDataset(config, config.num_val_samples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,       # 训练时打乱数据，避免模型记住数据顺序
        num_workers=0,       # Demo 数据量小，不需要多进程加载
        drop_last=True,      # 丢弃最后不足 batch_size 的数据
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,       # 验证时不打乱
        num_workers=0,
        drop_last=False,
    )
    
    return train_loader, val_loader


def save_demo_data(config: TransformerConfig, save_dir: str = "demo_data"):
    """
    生成并保存 Demo 数据集到文件。
    
    保存格式为 JSON，方便查看和理解数据结构。
    """
    os.makedirs(save_dir, exist_ok=True)
    
    demo_samples = []
    for i in range(20):  # 保存 20 个示例样本
        numbers = [
            random.randint(1, config.num_range - 1)
            for _ in range(config.seq_length)
        ]
        sorted_numbers = sorted(numbers)
        demo_samples.append({
            "id": i,
            "input": numbers,
            "output": sorted_numbers,
            "description": f"将 {numbers} 排序为 {sorted_numbers}",
        })
    
    save_path = os.path.join(save_dir, "sorting_samples.json")
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(demo_samples, f, ensure_ascii=False, indent=2)
    
    print(f"[Demo Data] 已生成 {len(demo_samples)} 个排序示例 → {save_path}")
    print(f"[Demo Data] 示例: {demo_samples[0]['description']}")


if __name__ == "__main__":
    config = TransformerConfig()
    
    # 保存 demo 数据
    save_demo_data(config)
    
    # 测试数据加载
    train_loader, val_loader = create_dataloaders(config)
    batch = next(iter(train_loader))
    
    print(f"\n训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"Batch 形状:")
    print(f"  src:        {batch['src'].shape}")        # [64, 10]
    print(f"  tgt_input:  {batch['tgt_input'].shape}")  # [64, 11]
    print(f"  tgt_output: {batch['tgt_output'].shape}") # [64, 11]
    print(f"\n示例样本:")
    print(f"  源序列 (乱序): {batch['src'][0].tolist()}")
    print(f"  目标输入 (BOS+sorted): {batch['tgt_input'][0].tolist()}")
    print(f"  目标输出 (sorted+EOS): {batch['tgt_output'][0].tolist()}")
