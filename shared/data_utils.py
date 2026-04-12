"""
公共数据处理工具：图像预处理、文本 Tokenize、数据采样等。
所有版本均可复用这些数据处理函数。
"""

import os
import json
import random
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset


# ============================================================
# 1. 图像预处理
# ============================================================

# ImageNet 标准均值和标准差
# 为什么用这组值？因为大部分预训练视觉模型（ResNet, ViT, CLIP）
# 都是在 ImageNet 上训练的，使用相同的归一化可以保持输入分布一致
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def normalize_image(
    image: torch.Tensor,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    对图像 Tensor 进行归一化。
    
    Args:
        image: shape [C, H, W] 或 [B, C, H, W]，值范围 [0, 1]
        mean: 每个通道的均值
        std: 每个通道的标准差
    
    Returns:
        归一化后的 Tensor
    
    公式: normalized = (x - mean) / std
    效果: 将每个通道的分布变为均值≈0、标准差≈1
    """
    mean = torch.tensor(mean, dtype=image.dtype, device=image.device)
    std = torch.tensor(std, dtype=image.dtype, device=image.device)
    
    if image.dim() == 3:  # [C, H, W]
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif image.dim() == 4:  # [B, C, H, W]
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    
    return (image - mean) / std


def denormalize_image(
    image: torch.Tensor,
    mean: List[float] = IMAGENET_MEAN,
    std: List[float] = IMAGENET_STD,
) -> torch.Tensor:
    """
    反归一化，用于可视化。
    公式: original = x * std + mean
    """
    mean = torch.tensor(mean, dtype=image.dtype, device=image.device)
    std = torch.tensor(std, dtype=image.dtype, device=image.device)
    
    if image.dim() == 3:
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
    elif image.dim() == 4:
        mean = mean.view(1, -1, 1, 1)
        std = std.view(1, -1, 1, 1)
    
    return image * std + mean


# ============================================================
# 2. 注意力掩码工具
# ============================================================
def create_attention_mask(seq_len: int, causal: bool = False) -> torch.Tensor:
    """
    创建注意力掩码。
    
    Args:
        seq_len: 序列长度
        causal: 是否为因果掩码（True 用于自回归模型如 GPT）
    
    Returns:
        shape [seq_len, seq_len] 的掩码，True 表示可以注意到
    
    因果掩码的含义：
    - 位置 i 只能注意到位置 0, 1, ..., i（不能看到未来信息）
    - 这是自回归生成的核心约束
    
    示例（seq_len=4 的因果掩码）:
        [[True,  False, False, False],   # token 0 只看自己
         [True,  True,  False, False],   # token 1 看 0, 1
         [True,  True,  True,  False],   # token 2 看 0, 1, 2
         [True,  True,  True,  True ]]   # token 3 看所有
    """
    if causal:
        # 下三角矩阵，包含对角线
        mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
    else:
        # 全 1 矩阵，所有位置互相可见
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool)
    
    return mask


def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """
    根据每个序列的实际长度创建 padding 掩码。
    
    Args:
        lengths: shape [batch_size]，每个序列的实际长度
        max_len: 填充后的最大长度
    
    Returns:
        shape [batch_size, max_len]，True 表示有效 token，False 表示 padding
    
    为什么需要 padding 掩码？
    - batch 内的序列长度不一，需要 pad 到相同长度
    - 但 pad 的位置不应该参与注意力计算
    """
    batch_size = lengths.size(0)
    # arange: [0, 1, ..., max_len-1]
    # 比较：每个位置是否小于对应序列的实际长度
    mask = torch.arange(max_len, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
    return mask


# ============================================================
# 3. 序列填充工具
# ============================================================
def pad_sequence_custom(
    sequences: List[torch.Tensor],
    padding_value: int = 0,
    max_len: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将不等长的序列列表填充为等长 Tensor。
    
    Args:
        sequences: 不等长 Tensor 列表
        padding_value: 填充值（通常 0 或特殊 token id）
        max_len: 指定最大长度（None 则取最长序列长度）
    
    Returns:
        (padded_tensor, lengths_tensor)
        - padded_tensor: shape [batch_size, max_len, ...]
        - lengths_tensor: shape [batch_size]
    """
    lengths = torch.tensor([seq.size(0) for seq in sequences])
    
    if max_len is None:
        max_len = lengths.max().item()
    
    # 创建全是 padding_value 的 Tensor
    batch_size = len(sequences)
    trailing_dims = sequences[0].shape[1:]  # 除第一维外的形状
    padded = torch.full(
        (batch_size, max_len, *trailing_dims),
        fill_value=padding_value,
        dtype=sequences[0].dtype,
    )
    
    # 填入实际数据
    for i, seq in enumerate(sequences):
        end = min(seq.size(0), max_len)
        padded[i, :end] = seq[:end]
    
    return padded, lengths


# ============================================================
# 4. 数据集工具
# ============================================================
def create_train_val_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[Subset, Subset]:
    """
    将数据集按比例划分为训练集和验证集。
    
    Args:
        dataset: 原始数据集
        val_ratio: 验证集比例
        seed: 随机种子
    
    Returns:
        (train_subset, val_subset)
    """
    total = len(dataset)
    val_size = int(total * val_ratio)
    train_size = total - val_size
    
    # 固定种子的随机划分
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(total, generator=generator).tolist()
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


def load_json(file_path: str) -> Any:
    """加载 JSON 文件。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Any, file_path: str, indent: int = 2):
    """保存为 JSON 文件。"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)
    print(f"[Data] 已保存到 {file_path}")


# ============================================================
# 5. Embedding 后处理工具（后续版本会大量使用）
# ============================================================
def l2_normalize(embeddings: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    L2 归一化，将向量映射到单位超球面上。
    
    为什么要 L2 归一化？
    - 归一化后，向量内积 = 余弦相似度
    - 消除了向量模长的影响，只关注方向
    - 对比学习中几乎必须归一化，否则模型会通过增大模长来降低 Loss
    
    Args:
        embeddings: shape [..., dim]
        dim: 在哪个维度做归一化
    
    Returns:
        归一化后的 Tensor，每个向量的 L2 范数为 1
    """
    return torch.nn.functional.normalize(embeddings, p=2, dim=dim)


def compute_cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    计算两组向量之间的余弦相似度矩阵。
    
    Args:
        a: shape [N, D]
        b: shape [M, D]
    
    Returns:
        shape [N, M] 的相似度矩阵
    """
    a_norm = l2_normalize(a, dim=-1)
    b_norm = l2_normalize(b, dim=-1)
    return torch.mm(a_norm, b_norm.t())
