"""
V20 - 知识增强嵌入数据集
========================
合成数据：
1. KG 三元组 (h, r, t) + 负采样
2. 图像-实体对：图像关联到 KG 实体
3. 文本-实体对：文本中标注 mention + 链接实体
"""
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple

from config import KnowledgeEmbeddingFullConfig


class KGTripletDataset(Dataset):
    """KG 三元组数据集（含负采样）"""

    def __init__(self, config: KnowledgeEmbeddingFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_kg_triplets if split == "train" else config.num_kg_triplets // 5
        self.num_entities = config.kg.num_entities
        self.num_relations = config.kg.num_relations
        self.neg_samples = config.kg.negative_samples

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 80000))

        # 正样本三元组
        h = rng.randint(0, self.num_entities - 1)
        r = rng.randint(0, self.num_relations - 1)
        t = rng.randint(0, self.num_entities - 1)

        # 负采样：随机替换头或尾实体
        if rng.random() < 0.5:
            neg_h = rng.randint(0, self.num_entities - 1)
            neg_t = t
        else:
            neg_h = h
            neg_t = rng.randint(0, self.num_entities - 1)

        return {
            'pos_h': torch.tensor(h, dtype=torch.long),
            'pos_r': torch.tensor(r, dtype=torch.long),
            'pos_t': torch.tensor(t, dtype=torch.long),
            'neg_h': torch.tensor(neg_h, dtype=torch.long),
            'neg_r': torch.tensor(r, dtype=torch.long),
            'neg_t': torch.tensor(neg_t, dtype=torch.long),
        }


class ImageEntityDataset(Dataset):
    """图像-实体对数据集：合成图像关联到 KG 实体"""

    def __init__(self, config: KnowledgeEmbeddingFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.image_size = config.kg_embed.image_size
        self.max_entities = config.kg_embed.max_entities_per_image
        self.num_entities = config.kg.num_entities
        self.num_relations = config.kg.num_relations

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 90000))

        # 合成图像
        image = torch.randn(3, self.image_size, self.image_size) * 0.2 + 0.5

        # 关联实体（每张图 1~max_entities 个实体）
        n_entities = rng.randint(1, self.max_entities)
        entity_ids = torch.zeros(self.max_entities, dtype=torch.long)
        entity_mask = torch.zeros(self.max_entities)
        for i in range(n_entities):
            entity_ids[i] = rng.randint(0, self.num_entities - 1)
            entity_mask[i] = 1.0

            # 在图像中添加实体对应的模式
            cx = rng.randint(20, self.image_size - 20)
            cy = rng.randint(20, self.image_size - 20)
            r = rng.randint(5, 15)
            y_lo = max(0, cy - r)
            y_hi = min(self.image_size, cy + r)
            x_lo = max(0, cx - r)
            x_hi = min(self.image_size, cx + r)
            channel = entity_ids[i].item() % 3
            image[channel, y_lo:y_hi, x_lo:x_hi] += 0.4

        # 分类标签（基于主实体的关系）
        label = entity_ids[0].item() % self.num_relations

        return {
            'image': image.clamp(0, 1),
            'entity_ids': entity_ids,
            'entity_mask': entity_mask,
            'label': torch.tensor(label, dtype=torch.long),
        }


class TextEntityDataset(Dataset):
    """文本-实体链接数据集"""

    def __init__(self, config: KnowledgeEmbeddingFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.vocab_size = config.entity_link.vocab_size
        self.max_seq_len = config.entity_link.max_seq_len
        self.num_entities = config.kg.num_entities

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 95000))

        # 合成文本
        seq_len = rng.randint(20, min(64, self.max_seq_len))
        tokens = torch.zeros(self.max_seq_len, dtype=torch.long)
        for i in range(seq_len):
            tokens[i] = rng.randint(3, self.vocab_size - 1)

        # Mention 标注（BIO-style）
        mention_labels = torch.zeros(self.max_seq_len, dtype=torch.long)  # 0=O
        entity_labels = torch.full((self.max_seq_len,), -1, dtype=torch.long)

        n_mentions = rng.randint(1, 3)
        pos = 2
        for _ in range(n_mentions):
            if pos >= seq_len - 3:
                break
            mention_len = rng.randint(1, 3)
            entity_id = rng.randint(0, self.num_entities - 1)
            mention_labels[pos] = 1  # B-Entity
            entity_labels[pos] = entity_id
            for j in range(1, mention_len):
                if pos + j < seq_len:
                    mention_labels[pos + j] = 2  # I-Entity
                    entity_labels[pos + j] = entity_id
            pos += mention_len + rng.randint(3, 8)

        attention_mask = torch.zeros(self.max_seq_len)
        attention_mask[:seq_len] = 1.0

        return {
            'token_ids': tokens,
            'mention_labels': mention_labels,
            'entity_labels': entity_labels,
            'attention_mask': attention_mask,
        }


class ImageTextKGDataset(Dataset):
    """图像-文本-KG 联合数据集（用于检索训练）"""

    def __init__(self, config: KnowledgeEmbeddingFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.image_size = config.kg_embed.image_size
        self.max_entities = config.kg_embed.max_entities_per_image

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 100000))

        # 图像
        image = torch.randn(3, self.image_size, self.image_size) * 0.2 + 0.5

        # 文本
        seq_len = rng.randint(10, 32)
        token_ids = torch.zeros(128, dtype=torch.long)
        for i in range(seq_len):
            token_ids[i] = rng.randint(3, 4999)

        # 实体
        n_ent = rng.randint(1, self.max_entities)
        entity_ids = torch.zeros(self.max_entities, dtype=torch.long)
        for i in range(n_ent):
            entity_ids[i] = rng.randint(0, self.config.kg.num_entities - 1)

        return {
            'image': image.clamp(0, 1),
            'token_ids': token_ids,
            'entity_ids': entity_ids,
        }


# ============================================================
#  DataLoader 工厂
# ============================================================

def create_kg_dataloaders(config: KnowledgeEmbeddingFullConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = KGTripletDataset(config, "train")
    val_ds = KGTripletDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size * 4, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size * 4, shuffle=False),
    )


def create_image_entity_dataloaders(config: KnowledgeEmbeddingFullConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = ImageEntityDataset(config, "train")
    val_ds = ImageEntityDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )


def create_distill_dataloaders(config: KnowledgeEmbeddingFullConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = ImageEntityDataset(config, "train")
    val_ds = ImageEntityDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )
