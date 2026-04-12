"""
V19 - 层级标签数据集
====================
合成带层级结构的标签数据，用于演示完整流水线
"""
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List

from config import LabelHierarchyFullConfig
from taxonomy import TaxonomyTree


class HierarchicalLabelDataset(Dataset):
    """
    层级标签分类数据集
    
    生成 (图像, 粗标签, 中标签, 细标签) 样本
    保证标签间满足树结构约束
    """

    def __init__(self, config: LabelHierarchyFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.image_size = config.hierarchy.image_size
        self.in_channels = config.hierarchy.in_channels

        # 构建分类学树
        self.tree = TaxonomyTree(config.hierarchy.num_labels_per_level)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 50000))

        # 先选择最细粒度标签，然后自底向上推导所有层级
        fine_level = self.tree.num_levels - 1
        fine_labels = self.tree.get_level_labels(fine_level)
        fine_id = rng.choice(fine_labels)

        # 获取从根到叶的路径
        path = self.tree.get_path(fine_id)

        # 各层的局部 ID
        level_labels = []
        for lv in range(self.tree.num_levels):
            global_id = path[lv]
            _, local_id = self.tree.global_to_local(global_id)
            level_labels.append(local_id)

        # 合成图像：不同类别有不同的统计特征
        image = torch.randn(self.in_channels, self.image_size, self.image_size) * 0.2

        # 粗粒度决定整体色调
        coarse_id = level_labels[0]
        hue_shift = coarse_id / self.tree.num_labels_per_level[0]
        image[0] += hue_shift * 0.5
        image[1] += (1 - hue_shift) * 0.3

        # 中粒度决定纹理频率
        mid_id = level_labels[1] if len(level_labels) > 1 else 0
        freq = 2 + mid_id % 5
        for c in range(self.in_channels):
            h, w = self.image_size, self.image_size
            y_grid = torch.arange(h).float().unsqueeze(1).expand(h, w)
            x_grid = torch.arange(w).float().unsqueeze(0).expand(h, w)
            image[c] += 0.1 * torch.sin(freq * x_grid / w * 2 * math.pi)

        # 细粒度决定局部特征
        fine_local = level_labels[-1]
        patch_x = (fine_local % 14) * (self.image_size // 14)
        patch_y = (fine_local // 14) * (self.image_size // 14)
        ps = self.image_size // 14
        image[:, patch_y:patch_y+ps, patch_x:patch_x+ps] += 0.3

        image = image.clamp(-1, 1)

        return {
            'image': image,
            'coarse_label': torch.tensor(level_labels[0], dtype=torch.long),
            'mid_label': torch.tensor(level_labels[1] if len(level_labels) > 1 else 0, dtype=torch.long),
            'fine_label': torch.tensor(level_labels[-1], dtype=torch.long),
            'fine_global_id': torch.tensor(fine_id, dtype=torch.long),
            'path': torch.tensor([path[i] for i in range(self.tree.num_levels)], dtype=torch.long),
        }


class MultiLabelHierarchyDataset(Dataset):
    """
    多标签层级数据集
    
    每个样本可以有多个标签，且标签满足层级约束
    """

    def __init__(self, config: LabelHierarchyFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.image_size = config.hierarchy.image_size
        self.in_channels = config.hierarchy.in_channels
        self.max_labels = config.multi_label.max_labels_per_sample

        # 构建分类学树
        self.tree = TaxonomyTree(config.hierarchy.num_labels_per_level)
        self.total_labels = self.tree.total_labels

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 60000))

        # 随机选择 1~max_labels 个细粒度标签
        n_labels = rng.randint(1, min(self.max_labels, 5))
        fine_labels = self.tree.get_level_labels(self.tree.num_levels - 1)
        selected_fine = rng.sample(fine_labels, min(n_labels, len(fine_labels)))

        # 构建多热标签向量（包含所有祖先）
        multi_hot = torch.zeros(self.total_labels)
        for fine_id in selected_fine:
            path = self.tree.get_path(fine_id)
            for node_id in path:
                multi_hot[node_id] = 1.0

        # 合成图像
        image = torch.randn(self.in_channels, self.image_size, self.image_size) * 0.2

        # 每个选中的细粒度标签贡献一个局部特征
        for fine_id in selected_fine:
            _, local_id = self.tree.global_to_local(fine_id)
            px = (local_id % 10) * (self.image_size // 10)
            py = (local_id // 10) * (self.image_size // 10)
            ps = self.image_size // 10
            image[:, py:min(py+ps, self.image_size), px:min(px+ps, self.image_size)] += 0.4

        image = image.clamp(-1, 1)

        return {
            'image': image,
            'labels': multi_hot,
            'n_labels': torch.tensor(n_labels, dtype=torch.long),
        }


class LabelEmbeddingDataset(Dataset):
    """
    标签嵌入数据集
    
    生成 (图像, 正标签, 负标签) 三元组
    """

    def __init__(self, config: LabelHierarchyFullConfig, split: str = "train",
                 n_negatives: int = 5):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.image_size = config.hierarchy.image_size
        self.in_channels = config.hierarchy.in_channels
        self.n_negatives = n_negatives

        # 构建分类学树
        self.tree = TaxonomyTree(config.hierarchy.num_labels_per_level)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 70000))

        # 随机选择一个细粒度标签
        fine_labels = self.tree.get_level_labels(self.tree.num_levels - 1)
        pos_label = rng.choice(fine_labels)

        # 负样本：从不同粗粒度类别中选择
        pos_ancestors = set(self.tree.get_ancestors(pos_label))
        neg_candidates = [l for l in fine_labels
                         if l != pos_label and l not in pos_ancestors]
        neg_labels = rng.sample(neg_candidates, min(self.n_negatives, len(neg_candidates)))
        # 补齐
        while len(neg_labels) < self.n_negatives:
            neg_labels.append(rng.choice(fine_labels))

        # 合成图像
        _, local_id = self.tree.global_to_local(pos_label)
        image = torch.randn(self.in_channels, self.image_size, self.image_size) * 0.2

        # 标签决定图像特征
        coarse_id = self.tree.get_path(pos_label)[0]
        _, coarse_local = self.tree.global_to_local(coarse_id)
        image[0] += coarse_local * 0.1
        px = (local_id % 10) * (self.image_size // 10)
        py = (local_id // 10) * (self.image_size // 10)
        ps = self.image_size // 10
        image[:, py:min(py+ps, self.image_size), px:min(px+ps, self.image_size)] += 0.5

        image = image.clamp(-1, 1)

        return {
            'image': image,
            'pos_label': torch.tensor(pos_label, dtype=torch.long),
            'neg_labels': torch.tensor(neg_labels, dtype=torch.long),
        }


# ============================================================
#  DataLoader 工厂函数
# ============================================================

def create_hierarchical_dataloaders(
    config: LabelHierarchyFullConfig
) -> Tuple[DataLoader, DataLoader]:
    """创建层级分类数据加载器"""
    train_ds = HierarchicalLabelDataset(config, "train")
    val_ds = HierarchicalLabelDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )


def create_multilabel_dataloaders(
    config: LabelHierarchyFullConfig
) -> Tuple[DataLoader, DataLoader]:
    """创建多标签分类数据加载器"""
    train_ds = MultiLabelHierarchyDataset(config, "train")
    val_ds = MultiLabelHierarchyDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )


def create_embedding_dataloaders(
    config: LabelHierarchyFullConfig
) -> Tuple[DataLoader, DataLoader]:
    """创建标签嵌入数据加载器"""
    train_ds = LabelEmbeddingDataset(config, "train")
    val_ds = LabelEmbeddingDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )
