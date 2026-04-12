"""v11 配置"""
from dataclasses import dataclass

@dataclass
class EmbeddingConfig:
    d_model: int = 256
    embed_dim: int = 128
    image_size: int = 64
    patch_size: int = 8
    n_heads: int = 4
    n_layers: int = 8
    d_ff: int = 1024
    dropout: float = 0.1
    vocab_size: int = 3000
    max_text_len: int = 32
    temperature: float = 0.07
    extract_layers: str = "1,3,5,7"
    fusion_method: str = "attention"
    # 检索
    num_gallery: int = 1000
    num_queries: int = 100
    top_k: int = 10
    # Embedding
    pooling: str = "cls"  # cls / mean / weighted
    normalize: bool = True
    # 训练
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 15
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    @property
    def layer_indices(self):
        return [int(x) for x in self.extract_layers.split(",")]

config = EmbeddingConfig()
