"""
V18 - 商品理解与细粒度视觉配置
================================
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class FineGrainedConfig:
    """细粒度识别配置"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_parts: int = 4              # 局部零件数
    num_classes: int = 200          # 细粒度类别数（如鸟类品种）


@dataclass
class ProductAttributeConfig:
    """商品属性配置"""
    d_model: int = 256
    n_heads: int = 8
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_categories: int = 50       # 商品类目
    num_brands: int = 100          # 品牌数
    num_colors: int = 16           # 颜色
    num_materials: int = 12        # 材质
    num_styles: int = 20           # 风格


@dataclass
class QualityAssessConfig:
    """商品图像质量评估配置"""
    d_model: int = 256
    n_heads: int = 8
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    num_quality_dims: int = 5      # 质量维度（清晰度/曝光/构图/美感/合规）


@dataclass
class ProductFullConfig:
    """完整配置"""
    fine_grained: FineGrainedConfig = field(default_factory=FineGrainedConfig)
    product_attr: ProductAttributeConfig = field(default_factory=ProductAttributeConfig)
    quality: QualityAssessConfig = field(default_factory=QualityAssessConfig)

    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42
    num_train_samples: int = 1000
    num_val_samples: int = 200
