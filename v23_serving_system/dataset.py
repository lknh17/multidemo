"""
V23 - 合成请求日志与服务基准数据
==================================
"""
import random
import time
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple
from config import FullConfig


class ServingRequestDataset(Dataset):
    """合成推理请求数据集"""

    def __init__(self, config: FullConfig, split: str = "train"):
        self.config = config
        self.split = split
        self.num_samples = config.num_train_samples if split == "train" else config.num_val_samples
        self.img_size = config.serving.image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + (0 if self.split == "train" else 100000))
        label = rng.randint(0, self.config.serving.num_classes - 1)

        # 合成图像 (不同类别有不同纹理模式)
        image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
        freq = (label % 10 + 1) * 2
        for c in range(3):
            offset = label * 0.1 + c * 0.3
            pattern = torch.sin(torch.linspace(offset, offset + freq, self.img_size)).unsqueeze(0)
            image[c] += pattern * 0.15

        return {'image': image.clamp(0, 1), 'label': label}


class BenchmarkDataset(Dataset):
    """基准测试数据集：不同批大小和输入复杂度"""

    def __init__(self, num_samples: int = 200, image_size: int = 224):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        rng = random.Random(idx + 200000)
        # 不同复杂度的图像
        complexity = rng.random()
        image = torch.randn(3, self.image_size, self.image_size) * (0.1 + 0.3 * complexity) + 0.5
        return {'image': image.clamp(0, 1), 'complexity': torch.tensor(complexity).float()}


class RequestLogGenerator:
    """模拟真实请求日志生成器"""

    def __init__(self, config: FullConfig):
        self.config = config
        self.img_size = config.serving.image_size

    def generate_poisson_requests(self, rate: float = 10.0,
                                  duration_sec: float = 1.0) -> List[Dict]:
        """生成泊松到达的请求序列"""
        requests = []
        t = 0.0
        rng = random.Random(42)
        while t < duration_sec:
            interval = rng.expovariate(rate)
            t += interval
            if t >= duration_sec:
                break
            image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
            request_id = f"req_{len(requests):06d}"
            requests.append({
                'id': request_id,
                'arrival_time': t,
                'image': image.clamp(0, 1),
                'label': rng.randint(0, self.config.serving.num_classes - 1),
            })
        return requests

    def generate_bursty_requests(self, base_rate: float = 5.0,
                                 burst_rate: float = 50.0,
                                 duration_sec: float = 2.0) -> List[Dict]:
        """生成突发流量模式"""
        requests = []
        t = 0.0
        rng = random.Random(123)
        while t < duration_sec:
            # 前半段正常，后半段突发
            rate = burst_rate if t > duration_sec / 2 else base_rate
            interval = rng.expovariate(rate)
            t += interval
            if t >= duration_sec:
                break
            image = torch.randn(3, self.img_size, self.img_size) * 0.2 + 0.5
            requests.append({
                'id': f"req_{len(requests):06d}",
                'arrival_time': t,
                'image': image.clamp(0, 1),
                'label': rng.randint(0, self.config.serving.num_classes - 1),
            })
        return requests

    def generate_skewed_keys(self, n: int = 1000, alpha: float = 1.2) -> List[str]:
        """生成 Zipf 分布的请求 key（用于缓存测试）"""
        rng = random.Random(456)
        num_unique = max(10, n // 5)
        # Zipf 分布
        weights = [1.0 / (i + 1) ** alpha for i in range(num_unique)]
        total = sum(weights)
        weights = [w / total for w in weights]

        keys = []
        for _ in range(n):
            r = rng.random()
            cumsum = 0.0
            for i, w in enumerate(weights):
                cumsum += w
                if r <= cumsum:
                    keys.append(f"item_{i:04d}")
                    break
        return keys


def create_serving_dataloaders(config: FullConfig):
    """创建服务数据加载器"""
    train_ds = ServingRequestDataset(config, "train")
    val_ds = ServingRequestDataset(config, "val")
    return (
        DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=config.batch_size),
    )


def create_benchmark_dataloader(num_samples: int = 200, batch_size: int = 1,
                                image_size: int = 224):
    """创建基准测试加载器"""
    ds = BenchmarkDataset(num_samples, image_size)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
