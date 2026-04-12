"""
V15 - 视频 Dense Captioning & Temporal Grounding 数据集
======================================================
模拟生成视频数据（合成随机帧序列 + 事件标注）

真实场景中使用 ActivityNet Captions / YouCook2 / ViTT 等数据集，
这里用合成数据演示完整的数据流水线。
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional
import random
import math

from config import VideoDenseCaptionFullConfig


class DenseCaptionDataset(Dataset):
    """
    Dense Video Captioning 数据集
    
    模拟生成：
    - 视频帧序列（随机 + 局部一致性模拟场景变化）
    - 多个事件标注 (start, end, caption_tokens)
    - 事件之间可能有重叠
    
    真实场景对应：ActivityNet Captions（~20k 视频，~100k 事件描述）
    """

    def __init__(self, config: VideoDenseCaptionFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_videos if split == "train" else config.num_val_videos

        # 视频参数
        self.num_frames = config.video.num_frames
        self.frame_size = config.video.frame_size
        self.in_channels = config.video.in_channels

        # 事件参数
        self.max_events = 5        # 每个视频最多 5 个事件
        self.max_caption_len = config.dense_caption.max_caption_len
        self.vocab_size = config.dense_caption.caption_vocab_size

        # 预生成元数据
        self.metadata = self._generate_metadata()

    def _generate_metadata(self):
        """预生成所有视频的事件标注"""
        metadata = []
        for idx in range(self.num_samples):
            rng = random.Random(idx + (0 if self.split == "train" else 10000))

            # 随机生成 1~max_events 个事件
            n_events = rng.randint(1, self.max_events)
            events = []

            for _ in range(n_events):
                # 随机起止时间（归一化到 [0, 1]）
                start = rng.uniform(0, 0.7)
                duration = rng.uniform(0.1, min(0.5, 1.0 - start))
                end = start + duration

                # 随机 caption tokens
                cap_len = rng.randint(5, self.max_caption_len - 2)
                caption = [1]  # BOS
                caption += [rng.randint(3, self.vocab_size - 1) for _ in range(cap_len)]
                caption += [2]  # EOS
                # Padding
                caption += [0] * (self.max_caption_len - len(caption))

                events.append({
                    'start': start,
                    'end': end,
                    'caption': caption[:self.max_caption_len],
                })

            metadata.append({
                'n_events': n_events,
                'events': events,
                'seed': idx,
            })

        return metadata

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[idx]
        rng = torch.Generator().manual_seed(meta['seed'])

        # 生成合成视频帧
        # 模拟场景变化：不同事件对应不同的视觉模式
        video = torch.zeros(self.in_channels, self.num_frames, self.frame_size, self.frame_size)

        # 基础噪声
        video += torch.randn_like(video) * 0.1

        # 每个事件区间添加独特的视觉模式
        for i, event in enumerate(meta['events']):
            t_start = int(event['start'] * self.num_frames)
            t_end = min(int(event['end'] * self.num_frames) + 1, self.num_frames)

            # 不同事件用不同频率的正弦波模拟
            freq = (i + 1) * 2
            pattern = torch.sin(torch.linspace(0, freq * math.pi, self.frame_size))
            pattern = pattern.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, W]
            video[:, t_start:t_end] += pattern * 0.5

        # 构建标注
        n_events = meta['n_events']
        max_events = self.max_events

        spans = torch.zeros(max_events, 2)
        captions = torch.zeros(max_events, self.max_caption_len, dtype=torch.long)
        labels = torch.full((max_events,), -1, dtype=torch.long)  # -1 表示无效

        for i, event in enumerate(meta['events']):
            spans[i, 0] = event['start']
            spans[i, 1] = event['end']
            captions[i] = torch.tensor(event['caption'], dtype=torch.long)
            labels[i] = 1  # 有效事件

        return {
            'video': video,         # [C, T, H, W]
            'spans': spans,         # [max_events, 2]
            'captions': captions,   # [max_events, max_caption_len]
            'labels': labels,       # [max_events]
            'n_events': n_events,
        }


class TemporalGroundingDataset(Dataset):
    """
    时序 Grounding 数据集
    
    模拟生成：视频 + 文本查询 + 目标时间段
    
    真实场景对应：Charades-STA / QVHighlights / TACoS
    """

    def __init__(self, config: VideoDenseCaptionFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_videos if split == "train" else config.num_val_videos

        self.num_frames = config.video.num_frames
        self.frame_size = config.video.frame_size
        self.max_text_len = config.temporal_grounding.max_text_len
        self.vocab_size = config.temporal_grounding.vocab_size

        self.metadata = self._generate_metadata()

    def _generate_metadata(self):
        metadata = []
        for idx in range(self.num_samples):
            rng = random.Random(idx + 20000)

            # 目标时间段
            start = rng.uniform(0, 0.6)
            duration = rng.uniform(0.1, min(0.4, 1.0 - start))
            end = start + duration

            # 文本查询 tokens
            query_len = rng.randint(5, self.max_text_len - 2)
            query = [1]  # BOS
            query += [rng.randint(3, self.vocab_size - 1) for _ in range(query_len)]
            query += [2]  # EOS
            query += [0] * (self.max_text_len - len(query))

            metadata.append({
                'start': start,
                'end': end,
                'query': query[:self.max_text_len],
                'seed': idx,
            })

        return metadata

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        meta = self.metadata[idx]

        # 生成合成视频
        video = torch.randn(self.config.video.in_channels, self.num_frames,
                            self.frame_size, self.frame_size) * 0.3

        # 在目标区间添加特殊模式
        t_start = int(meta['start'] * self.num_frames)
        t_end = min(int(meta['end'] * self.num_frames) + 1, self.num_frames)

        pattern = torch.ones(1, 1, self.frame_size, self.frame_size) * 0.8
        video[:, t_start:t_end] += pattern

        # 文本查询
        query = torch.tensor(meta['query'], dtype=torch.long)
        query_mask = (query != 0).float()

        # 目标 span
        span = torch.tensor([meta['start'], meta['end']])

        return {
            'video': video,
            'query': query,
            'query_mask': query_mask,
            'span': span,
        }


def create_dense_caption_dataloaders(
    config: VideoDenseCaptionFullConfig,
) -> Tuple[DataLoader, DataLoader]:
    """创建 Dense Caption 数据加载器"""
    train_ds = DenseCaptionDataset(config, split="train")
    val_ds = DenseCaptionDataset(config, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def create_grounding_dataloaders(
    config: VideoDenseCaptionFullConfig,
) -> Tuple[DataLoader, DataLoader]:
    """创建 Temporal Grounding 数据加载器"""
    train_ds = TemporalGroundingDataset(config, split="train")
    val_ds = TemporalGroundingDataset(config, split="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader
