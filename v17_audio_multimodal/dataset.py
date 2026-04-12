"""
V17 - 音频理解 / 全模态数据集
================================
合成数据用于演示完整流水线
"""
import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple

from config import AudioMultimodalFullConfig


class AudioTextDataset(Dataset):
    """CLAP 训练用音频-文本配对数据"""

    def __init__(self, config: AudioMultimodalFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.n_mels = config.mel.n_mels
        # 合成固定长度的 Mel 频谱
        max_frames = int(config.mel.max_duration * config.mel.sample_rate / config.mel.hop_length)
        self.max_frames = min(max_frames, config.audio_enc.max_time_patches * config.audio_enc.time_patch_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 80000))

        # 合成 Mel 频谱（模拟不同音频类型）
        audio_type = rng.randint(0, 4)
        mel = torch.randn(self.n_mels, self.max_frames) * 0.5

        # 根据类型添加不同频率模式
        if audio_type == 0:  # 语音
            mel[20:60, :] += 1.0  # 低频能量
        elif audio_type == 1:  # 音乐
            for f in [30, 60, 90]:
                mel[f:f+5, :] += 0.8  # 谐波
        elif audio_type == 2:  # 环境音
            mel += torch.randn_like(mel) * 0.3
        elif audio_type == 3:  # 警报声
            t = torch.arange(self.max_frames).float()
            freq = (50 + 30 * torch.sin(t * 0.1)).long().clamp(0, self.n_mels - 1)
            for i, f in enumerate(freq):
                mel[f, i] += 2.0
        else:  # 静音
            mel *= 0.1

        # 合成文本描述
        text_len = rng.randint(5, min(30, self.config.clap.max_text_len))
        token_ids = torch.tensor([rng.randint(3, self.config.clap.vocab_size - 1)
                                  for _ in range(text_len)])
        padded_tokens = torch.zeros(self.config.clap.max_text_len, dtype=torch.long)
        padded_tokens[:text_len] = token_ids

        return {
            'mel_spec': mel,
            'token_ids': padded_tokens,
            'audio_type': audio_type,
        }


class OmniModalDataset(Dataset):
    """全模态训练数据：图像+文本+音频"""

    def __init__(self, config: AudioMultimodalFullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.image_size = config.omni.image_size
        self.n_mels = config.mel.n_mels
        max_frames = int(config.mel.max_duration * config.mel.sample_rate / config.mel.hop_length)
        self.max_frames = min(max_frames, config.audio_enc.max_time_patches * config.audio_enc.time_patch_size)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 90000))

        # 类别标签
        label = rng.randint(0, self.config.omni.num_classes - 1)

        # 合成图像
        image = torch.randn(3, self.image_size, self.image_size) * 0.2 + 0.5
        # 根据标签添加不同图案
        cx = int((label % 5) / 5 * self.image_size)
        cy = int((label // 5) / 4 * self.image_size)
        r = self.image_size // 8
        for c in range(3):
            for dy in range(-r, r):
                for dx in range(-r, r):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < self.image_size and 0 <= nx < self.image_size:
                        if dx*dx + dy*dy < r*r:
                            image[c, ny, nx] += 0.3 * (c == label % 3)

        # 合成文本
        text_len = rng.randint(5, min(30, self.config.omni.max_text_len))
        token_ids = torch.tensor([rng.randint(3, self.config.omni.vocab_size - 1)
                                  for _ in range(text_len)])
        padded_tokens = torch.zeros(self.config.omni.max_text_len, dtype=torch.long)
        padded_tokens[:text_len] = token_ids

        # 合成音频
        mel = torch.randn(self.n_mels, self.max_frames) * 0.5
        mel[label * 5:label * 5 + 10, :] += 1.0  # 类别相关的频率模式

        # 随机模态缺失标记
        has_image = rng.random() > 0.1
        has_text = rng.random() > 0.1
        has_audio = rng.random() > 0.1
        if not (has_image or has_text or has_audio):
            has_text = True

        return {
            'image': image.clamp(0, 1),
            'token_ids': padded_tokens,
            'mel_spec': mel,
            'label': label,
            'has_image': has_image,
            'has_text': has_text,
            'has_audio': has_audio,
        }


def create_audio_text_dataloaders(config: AudioMultimodalFullConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = AudioTextDataset(config, "train")
    val_ds = AudioTextDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )


def create_omni_modal_dataloaders(config: AudioMultimodalFullConfig) -> Tuple[DataLoader, DataLoader]:
    train_ds = OmniModalDataset(config, "train")
    val_ds = OmniModalDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size, shuffle=False),
    )
