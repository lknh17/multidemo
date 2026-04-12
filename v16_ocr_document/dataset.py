"""
V16 - OCR / 文档理解 / 广告文字提取数据集
=========================================
合成数据用于演示完整流水线
"""
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple

from config import OCRDocumentFullConfig


class OCRDetectionDataset(Dataset):
    """OCR 文字检测数据集：合成含文字区域的图像"""

    def __init__(self, config: OCRDocumentFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.image_size = config.ocr_det.image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 50000))
        
        # 合成图像
        image = torch.randn(3, self.image_size, self.image_size) * 0.2 + 0.5
        
        # 合成文字区域 GT
        prob_map = torch.zeros(self.image_size // 4, self.image_size // 4)
        thresh_map = torch.full_like(prob_map, 0.3)
        
        n_regions = rng.randint(1, 5)
        for _ in range(n_regions):
            x = rng.randint(2, prob_map.shape[1] - 10)
            y = rng.randint(2, prob_map.shape[0] - 6)
            w = rng.randint(5, min(15, prob_map.shape[1] - x))
            h = rng.randint(2, min(5, prob_map.shape[0] - y))
            prob_map[y:y+h, x:x+w] = 1.0
            # 在图像对应区域添加 pattern
            sx, sy = x * 4, y * 4
            image[:, sy:sy+h*4, sx:sx+w*4] += 0.3

        return {
            'image': image.clamp(0, 1),
            'prob_map': prob_map,
            'thresh_map': thresh_map,
        }


class DocumentDataset(Dataset):
    """文档理解数据集：合成 token + bbox + 图像"""

    def __init__(self, config: OCRDocumentFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.max_text_len = config.document.max_text_len
        self.vocab_size = config.document.vocab_size
        self.image_size = config.document.image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 60000))

        # 合成文本 tokens
        seq_len = rng.randint(20, min(128, self.max_text_len))
        token_ids = torch.tensor([rng.randint(3, self.vocab_size - 1) for _ in range(seq_len)])
        
        # Padding
        padded_tokens = torch.zeros(self.max_text_len, dtype=torch.long)
        padded_tokens[:seq_len] = token_ids

        # 合成 bboxes (模拟文档布局)
        bboxes = torch.zeros(self.max_text_len, 4, dtype=torch.long)
        y_cursor = 10
        x_cursor = 10
        for i in range(seq_len):
            w = rng.randint(20, 60)
            h = rng.randint(15, 25)
            if x_cursor + w > 990:
                x_cursor = 10
                y_cursor += h + 5
            bboxes[i] = torch.tensor([x_cursor, y_cursor, x_cursor + w, y_cursor + h])
            x_cursor += w + 5

        # 合成图像
        image = torch.randn(3, self.image_size, self.image_size) * 0.15 + 0.8

        # 文档分类标签
        label = rng.randint(0, self.config.document.num_labels - 1)

        # 序列标注标签（BIO）
        token_labels = torch.zeros(self.max_text_len, dtype=torch.long)
        for i in range(seq_len):
            token_labels[i] = rng.randint(0, self.config.document.num_labels - 1)

        # Attention mask
        attention_mask = torch.zeros(self.max_text_len)
        attention_mask[:seq_len] = 1.0

        return {
            'token_ids': padded_tokens,
            'bboxes': bboxes,
            'image': image,
            'label': label,
            'token_labels': token_labels,
            'attention_mask': attention_mask,
        }


class AdTextDataset(Dataset):
    """广告文字提取数据集"""

    def __init__(self, config: OCRDocumentFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.max_regions = config.ad_text.max_regions
        self.image_size = config.ocr_det.image_size

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 70000))

        # 合成广告图像
        image = torch.randn(3, self.image_size, self.image_size) * 0.2 + 0.5

        # 合成文字区域
        n_regions = rng.randint(2, self.max_regions)
        bboxes = torch.zeros(self.max_regions, 4)
        types = torch.full((self.max_regions,), -1, dtype=torch.long)
        region_mask = torch.zeros(self.max_regions)

        for i in range(n_regions):
            x0 = rng.randint(10, 800)
            y0 = rng.randint(10, 800)
            w = rng.randint(50, min(200, 990 - x0))
            h = rng.randint(15, min(50, 990 - y0))
            bboxes[i] = torch.tensor([x0, y0, x0 + w, y0 + h], dtype=torch.float)
            types[i] = rng.randint(0, self.config.ad_text.num_text_types - 1)
            region_mask[i] = 1.0

            # 在图像对应位置添加模式
            px0 = int(x0 / 1000 * self.image_size)
            py0 = int(y0 / 1000 * self.image_size)
            pw = max(1, int(w / 1000 * self.image_size))
            ph = max(1, int(h / 1000 * self.image_size))
            px1 = min(px0 + pw, self.image_size)
            py1 = min(py0 + ph, self.image_size)
            image[:, py0:py1, px0:px1] += 0.3

        # 合成区域图像（文字条）
        region_images = torch.randn(self.max_regions, 1, 32, 64) * 0.3

        return {
            'image': image.clamp(0, 1),
            'bboxes': bboxes,
            'types': types,
            'region_mask': region_mask,
            'region_images': region_images,
        }


def create_ocr_dataloaders(config: OCRDocumentFullConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = OCRDetectionDataset(config, "train")
    val_ds = OCRDetectionDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )


def create_document_dataloaders(config: OCRDocumentFullConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = DocumentDataset(config, "train")
    val_ds = DocumentDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )


def create_ad_text_dataloaders(config: OCRDocumentFullConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = AdTextDataset(config, "train")
    val_ds = AdTextDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )
