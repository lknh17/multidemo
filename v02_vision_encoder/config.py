"""v02 配置"""
from dataclasses import dataclass

@dataclass
class ViTConfig:
    image_size: int = 32        # CIFAR-10 图像大小
    patch_size: int = 4         # patch 大小 → 32/4 = 8×8 = 64 个 patch
    in_channels: int = 3        # RGB
    num_classes: int = 10       # CIFAR-10 类别数
    d_model: int = 192          # 隐藏维度
    n_heads: int = 6            # 注意力头数
    n_layers: int = 6           # Transformer 层数
    d_ff: int = 768             # FFN 中间维度
    dropout: float = 0.1
    # 训练
    batch_size: int = 128
    learning_rate: float = 3e-4
    num_epochs: int = 30
    warmup_epochs: int = 5
    weight_decay: float = 0.05
    # 路径
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

config = ViTConfig()
