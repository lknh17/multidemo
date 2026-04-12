"""v09 配置"""
from dataclasses import dataclass
from typing import List

@dataclass
class DeepStackConfig:
    d_model: int = 256
    embed_dim: int = 128
    image_size: int = 64
    patch_size: int = 8
    n_heads: int = 4
    n_layers: int = 8           # 更多层，便于多层提取
    d_ff: int = 1024
    dropout: float = 0.1
    vocab_size: int = 3000
    max_text_len: int = 32
    temperature: float = 0.07
    # DeepStack
    extract_layers: str = "1,3,5,7"  # 提取哪些层的特征
    fusion_method: str = "attention"  # concat / weighted / attention / gated
    layer_loss_weight: float = 0.3    # 各层 Loss 的权重
    fusion_loss_weight: float = 1.0   # 融合 Loss 的权重
    # 训练
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 20
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    @property
    def layer_indices(self) -> List[int]:
        return [int(x) for x in self.extract_layers.split(",")]

config = DeepStackConfig()
