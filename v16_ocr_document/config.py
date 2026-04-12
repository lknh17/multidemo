"""
V16 - OCR 与文档理解配置
========================
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class OCRDetectionConfig:
    """文字检测配置（DBNet 风格）"""
    backbone_channels: List[int] = field(default_factory=lambda: [64, 128, 256])
    d_model: int = 256
    det_threshold: float = 0.3          # 二值化阈值
    min_area: int = 16                  # 最小文字区域
    image_size: int = 256               # 输入图像尺寸


@dataclass
class OCRRecognitionConfig:
    """文字识别配置（CRNN / ABINet 风格）"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 3
    vocab_size: int = 5000              # 字符词表大小（含中英文）
    max_text_len: int = 25              # 最大文字长度
    feature_height: int = 8             # CNN 输出特征高度
    feature_width: int = 32             # CNN 输出特征宽度
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0


@dataclass
class DocumentUnderstandingConfig:
    """文档理解配置（LayoutLM 风格）"""
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 4
    d_ff: int = 1024
    dropout: float = 0.1

    # 文本
    vocab_size: int = 5000
    max_text_len: int = 512

    # 2D 位置编码
    max_position_x: int = 1000          # 归一化 x 坐标范围
    max_position_y: int = 1000          # 归一化 y 坐标范围
    max_width: int = 1000
    max_height: int = 1000

    # 图像
    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3

    # 预训练任务
    mlm_probability: float = 0.15       # MLM 遮蔽比例
    mim_probability: float = 0.40       # MIM 遮蔽比例

    # 下游任务
    num_labels: int = 10                # 文档分类/字段标注类别数


@dataclass
class AdTextExtractionConfig:
    """广告文字提取配置"""
    d_model: int = 256
    n_heads: int = 8
    num_text_types: int = 8             # 文字类型数（标题/促销/价格/品牌/...）
    max_regions: int = 20               # 每张图最多文字区域数
    region_feature_dim: int = 256


@dataclass
class OCRDocumentFullConfig:
    """完整配置"""
    ocr_det: OCRDetectionConfig = field(default_factory=OCRDetectionConfig)
    ocr_rec: OCRRecognitionConfig = field(default_factory=OCRRecognitionConfig)
    document: DocumentUnderstandingConfig = field(default_factory=DocumentUnderstandingConfig)
    ad_text: AdTextExtractionConfig = field(default_factory=AdTextExtractionConfig)

    # 训练参数
    batch_size: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    num_epochs: int = 20
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    seed: int = 42

    # 数据
    num_train_samples: int = 1000
    num_val_samples: int = 200
