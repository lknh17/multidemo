"""v10 配置"""
from dataclasses import dataclass

@dataclass
class ConceptConfig:
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
    # 商业概念维度
    num_industries: int = 10
    num_brands: int = 50
    num_attributes: int = 20
    num_intents: int = 5
    # Loss 权重
    contrastive_weight: float = 1.0
    recon_weight: float = 0.5
    deepstack_weight: float = 0.3
    # DeepStack
    extract_layers: str = "1,3,5,7"
    fusion_method: str = "attention"
    # 训练
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 20
    weight_decay: float = 0.01
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    @property
    def layer_indices(self):
        return [int(x) for x in self.extract_layers.split(",")]

config = ConceptConfig()
