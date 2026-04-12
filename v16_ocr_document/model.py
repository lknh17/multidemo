"""
V16 - 文档理解 & 广告文字提取模型
=================================
核心模块：
1. Layout2DPositionEmbedding：LayoutLM 风格 2D 位置编码
2. DocumentUnderstandingModel：文档理解 Transformer
3. AdTextExtractionModel：广告文字提取端到端模型

参考：
- LayoutLMv3 (ACL 2022)
- DocFormer (ICCV 2021)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple

from config import OCRDocumentFullConfig, DocumentUnderstandingConfig, AdTextExtractionConfig
from ocr_modules import TextDetector, AttentionRecognizer


# ============================================================
#  2D 位置编码
# ============================================================

class Layout2DPositionEmbedding(nn.Module):
    """
    LayoutLM 风格 2D 位置编码
    
    将文字框的空间坐标 (x0, y0, x1, y1, w, h) 编码为向量
    6 个独立 Embedding 表，模型自动学习空间关系
    """

    def __init__(self, config: DocumentUnderstandingConfig):
        super().__init__()
        self.x0_embed = nn.Embedding(config.max_position_x, config.d_model)
        self.y0_embed = nn.Embedding(config.max_position_y, config.d_model)
        self.x1_embed = nn.Embedding(config.max_position_x, config.d_model)
        self.y1_embed = nn.Embedding(config.max_position_y, config.d_model)
        self.w_embed = nn.Embedding(config.max_width, config.d_model)
        self.h_embed = nn.Embedding(config.max_height, config.d_model)

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bbox: [B, L, 4] 归一化坐标 (x0, y0, x1, y1)，值域 [0, max_position-1]
        """
        x0 = bbox[..., 0].clamp(0, self.x0_embed.num_embeddings - 1).long()
        y0 = bbox[..., 1].clamp(0, self.y0_embed.num_embeddings - 1).long()
        x1 = bbox[..., 2].clamp(0, self.x1_embed.num_embeddings - 1).long()
        y1 = bbox[..., 3].clamp(0, self.y1_embed.num_embeddings - 1).long()
        w = (x1 - x0).clamp(0, self.w_embed.num_embeddings - 1)
        h = (y1 - y0).clamp(0, self.h_embed.num_embeddings - 1)

        return (self.x0_embed(x0) + self.y0_embed(y0) +
                self.x1_embed(x1) + self.y1_embed(y1) +
                self.w_embed(w) + self.h_embed(h))


# ============================================================
#  空间感知注意力
# ============================================================

class SpatialAwareAttention(nn.Module):
    """
    空间感知注意力：标准注意力 + 空间偏置
    
    Attn(Q,K,V) = softmax(QK^T/√d + B_spatial) V
    """

    def __init__(self, d_model: int, n_heads: int, max_spatial_dist: int = 64):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        # 空间偏置：将 2D 距离映射为注意力偏置
        self.spatial_bias_net = nn.Sequential(
            nn.Linear(2, 32),
            nn.ReLU(),
            nn.Linear(32, n_heads),
        )

    def forward(self, x: torch.Tensor,
                spatial_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, L, D]
            spatial_pos: [B, L, 2] 归一化中心坐标 (cx, cy)
        """
        B, L, D = x.shape

        q = self.q_proj(x).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, L, L]

        # 空间偏置
        if spatial_pos is not None:
            # 计算两两相对距离
            dx = spatial_pos[:, :, 0:1] - spatial_pos[:, :, 0:1].transpose(1, 2)
            dy = spatial_pos[:, :, 1:2] - spatial_pos[:, :, 1:2].transpose(1, 2)
            spatial_dist = torch.stack([dx.squeeze(-1), dy.squeeze(-1)], dim=-1)  # [B, L, L, 2]

            spatial_bias = self.spatial_bias_net(spatial_dist)  # [B, L, L, H]
            spatial_bias = spatial_bias.permute(0, 3, 1, 2)    # [B, H, L, L]
            attn = attn + spatial_bias

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # [B, H, L, d]
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out_proj(out)


# ============================================================
#  文档理解模型
# ============================================================

class DocumentTransformerBlock(nn.Module):
    """带空间感知注意力的 Transformer Block"""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attn = SpatialAwareAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                spatial_pos: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), spatial_pos)
        x = x + self.ffn(self.norm2(x))
        return x


class DocumentUnderstandingModel(nn.Module):
    """
    LayoutLM 风格文档理解模型
    
    输入：token_ids + bboxes + images
    编码：文本Embedding + 2D位置Embedding + 图像Embedding → Transformer
    输出：用于下游任务（分类/序列标注/信息抽取）
    """

    def __init__(self, config: DocumentUnderstandingConfig):
        super().__init__()
        self.config = config

        # 文本 Embedding
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embed = nn.Embedding(config.max_text_len, config.d_model)

        # 2D 位置 Embedding
        self.layout_embed = Layout2DPositionEmbedding(config)

        # 图像 Patch Embedding
        n_patches = (config.image_size // config.patch_size) ** 2
        self.patch_embed = nn.Conv2d(
            config.in_channels, config.d_model,
            kernel_size=config.patch_size, stride=config.patch_size
        )
        self.image_pos_embed = nn.Embedding(n_patches, config.d_model)

        # Transformer
        self.blocks = nn.ModuleList([
            DocumentTransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)

        # 下游任务头
        self.cls_head = nn.Linear(config.d_model, config.num_labels)  # 文档分类
        self.token_cls_head = nn.Linear(config.d_model, config.num_labels)  # 序列标注

        # MLM 预训练头
        self.mlm_head = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, token_ids: torch.Tensor, bboxes: torch.Tensor,
                images: Optional[torch.Tensor] = None,
                spatial_pos: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            token_ids: [B, L] 文本 token
            bboxes: [B, L, 4] 文字框坐标
            images: [B, C, H, W] 文档图像
            spatial_pos: [B, L, 2] 中心坐标（可选，从 bboxes 计算）
        """
        B, L = token_ids.shape

        # 文本编码
        pos_ids = torch.arange(L, device=token_ids.device).unsqueeze(0)
        text_embed = (self.token_embed(token_ids) +
                     self.position_embed(pos_ids) +
                     self.layout_embed(bboxes))

        # 图像编码
        if images is not None:
            img_tokens = self.patch_embed(images)  # [B, D, H', W']
            img_tokens = img_tokens.flatten(2).permute(0, 2, 1)  # [B, N, D]
            N = img_tokens.shape[1]
            img_pos = torch.arange(N, device=images.device).unsqueeze(0)
            img_tokens = img_tokens + self.image_pos_embed(img_pos)

            # 拼接文本和图像
            combined = torch.cat([text_embed, img_tokens], dim=1)
        else:
            combined = text_embed

        # 计算空间位置（从 bboxes）
        if spatial_pos is None and bboxes is not None:
            cx = (bboxes[..., 0] + bboxes[..., 2]) / 2.0 / self.config.max_position_x
            cy = (bboxes[..., 1] + bboxes[..., 3]) / 2.0 / self.config.max_position_y
            text_spatial = torch.stack([cx, cy], dim=-1)
            if images is not None:
                # 图像 patch 的空间位置
                H_p = W_p = self.config.image_size // self.config.patch_size
                gy, gx = torch.meshgrid(
                    torch.linspace(0, 1, H_p, device=images.device),
                    torch.linspace(0, 1, W_p, device=images.device),
                    indexing='ij'
                )
                img_spatial = torch.stack([gx.flatten(), gy.flatten()], dim=-1)
                img_spatial = img_spatial.unsqueeze(0).expand(B, -1, -1)
                spatial_pos = torch.cat([text_spatial, img_spatial], dim=1)
            else:
                spatial_pos = text_spatial

        # Transformer
        x = combined
        for block in self.blocks:
            x = block(x, spatial_pos)
        x = self.norm(x)

        # 分离文本和图像输出
        text_output = x[:, :L]

        return {
            'text_output': text_output,       # [B, L, D]
            'cls_logits': self.cls_head(text_output[:, 0]),  # [B, num_labels]
            'token_logits': self.token_cls_head(text_output),  # [B, L, num_labels]
            'mlm_logits': self.mlm_head(text_output),         # [B, L, vocab_size]
        }


# ============================================================
#  广告文字提取模型
# ============================================================

class AdTextExtractionModel(nn.Module):
    """
    广告文字提取模型
    
    端到端流水线：
    1. 文字检测（DBNet）→ 文字区域
    2. 文字识别（Attention OCR）→ 文字内容
    3. 区域分类 → 文字类型（标题/价格/促销/...）
    4. 结构化抽取 → 关键字段
    """

    TEXT_TYPES = ['标题', '促销文案', '价格', '品牌名', '产品描述', 'CTA', '免责声明', '其他']

    def __init__(self, config: OCRDocumentFullConfig):
        super().__init__()
        self.config = config

        # 检测器
        self.detector = TextDetector(config.ocr_det)

        # 识别器（简化：共享 CNN 特征）
        self.recognizer = AttentionRecognizer(config.ocr_rec)

        # 区域特征提取
        self.region_proj = nn.Linear(
            config.ocr_rec.d_model + config.ad_text.region_feature_dim,
            config.ad_text.d_model
        )

        # 空间关系编码
        self.spatial_encoder = nn.Sequential(
            nn.Linear(8, 64),  # 相对坐标 + 大小 + 距离
            nn.ReLU(),
            nn.Linear(64, config.ad_text.d_model),
        )

        # 文字类型分类头
        self.type_classifier = nn.Sequential(
            nn.Linear(config.ad_text.d_model * 2, config.ad_text.d_model),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config.ad_text.d_model, config.ad_text.num_text_types),
        )

        # 区域间关系建模（Transformer）
        self.relation_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                config.ad_text.d_model, config.ad_text.n_heads,
                config.ad_text.d_model * 4, batch_first=True
            ),
            num_layers=2,
        )

    def forward(self, images: torch.Tensor,
                region_bboxes: Optional[torch.Tensor] = None,
                region_images: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        简化版前向：假设检测结果已给定
        
        Args:
            images: [B, C, H, W] 原图
            region_bboxes: [B, N_reg, 4] 文字区域框
            region_images: [B, N_reg, 1, 32, W'] 文字区域图像
        """
        outputs = {}

        # 1. 检测（全图）
        det_outputs = self.detector(images)
        outputs['det'] = det_outputs

        # 2. 如果给定了区域，做区域级别的分类
        if region_bboxes is not None:
            B, N_reg = region_bboxes.shape[:2]

            # 区域特征
            if region_images is not None:
                # 识别 CNN 提取特征
                flat_imgs = region_images.reshape(B * N_reg, *region_images.shape[2:])
                from ocr_modules import TextRecognitionCNN
                rec_cnn = TextRecognitionCNN(1, self.config.ocr_rec.d_model)
                rec_cnn = rec_cnn.to(flat_imgs.device)
                rec_feats = rec_cnn(flat_imgs)  # [B*N_reg, W', D]
                rec_feats = rec_feats.mean(dim=1)  # [B*N_reg, D]
                rec_feats = rec_feats.reshape(B, N_reg, -1)
            else:
                rec_feats = torch.randn(B, N_reg, self.config.ocr_rec.d_model,
                                       device=images.device)

            # 空间特征
            spatial_feats = self._compute_spatial_features(region_bboxes)
            spatial_encoded = self.spatial_encoder(spatial_feats)  # [B, N_reg, D]

            # 融合
            combined = torch.cat([rec_feats, spatial_encoded], dim=-1)
            region_feats = self.region_proj(combined)  # [B, N_reg, D]

            # 区域间关系建模
            region_feats = self.relation_layers(region_feats)

            # 分类
            cls_input = torch.cat([region_feats, spatial_encoded], dim=-1)
            type_logits = self.type_classifier(cls_input)  # [B, N_reg, num_types]
            outputs['type_logits'] = type_logits

        return outputs

    def _compute_spatial_features(self, bboxes: torch.Tensor) -> torch.Tensor:
        """计算区域空间特征"""
        x0, y0, x1, y1 = bboxes.unbind(-1)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        w = x1 - x0
        h = y1 - y0
        area = w * h
        aspect = w / (h + 1e-6)
        # 归一化相对位置
        rel_x = cx / 1000.0
        rel_y = cy / 1000.0

        return torch.stack([cx, cy, w, h, area, aspect, rel_x, rel_y], dim=-1).float()
