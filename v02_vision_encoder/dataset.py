"""v02 数据集: CIFAR-10 图像分类"""
import os
import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as T
from config import ViTConfig

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def get_transforms(train: bool = True):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]),
        ])

def create_dataloaders(config: ViTConfig, data_dir: str = "demo_data"):
    os.makedirs(data_dir, exist_ok=True)
    train_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=get_transforms(True)
    )
    val_set = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=get_transforms(False)
    )
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_set, batch_size=config.batch_size, shuffle=False, num_workers=2)
    return train_loader, val_loader
