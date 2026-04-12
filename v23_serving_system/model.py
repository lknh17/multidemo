"""
V23 - ServingPipeline / ONNX导出 / INT8量化 / 模型集成
=======================================================
"""
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from config import FullConfig, ServingConfig, ExportConfig


# ============================================================
# 简化 ViT Backbone (用于 Serving 演示)
# ============================================================
class PatchEmbedding(nn.Module):
    def __init__(self, config: ServingConfig):
        super().__init__()
        self.proj = nn.Conv2d(config.in_channels, config.d_model,
                              kernel_size=config.patch_size, stride=config.patch_size)
        num_patches = (config.image_size // config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.proj(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return x + self.pos_embed


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ffn(self.norm2(x))
        return x


class SimpleViTBackbone(nn.Module):
    def __init__(self, config: ServingConfig):
        super().__init__()
        self.patch_embed = PatchEmbedding(config)
        self.blocks = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads)
            for _ in range(config.n_layers)
        ])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


# ============================================================
# 1. ServingPipeline: 预处理 → 推理 → 后处理
# ============================================================
class ServingPipeline(nn.Module):
    """端到端推理流水线"""

    def __init__(self, config: ServingConfig):
        super().__init__()
        self.config = config
        self.encoder = SimpleViTBackbone(config)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

        # 预处理参数
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """图像预处理：归一化"""
        return (images - self.mean) / self.std

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征"""
        features = self.encoder(x)
        return features[:, 0]  # CLS token

    def postprocess(self, logits: torch.Tensor, top_k: int = 5) -> Dict[str, torch.Tensor]:
        """后处理：Softmax + Top-K"""
        probs = F.softmax(logits, dim=-1)
        k = min(top_k, logits.size(-1))
        top_probs, top_ids = probs.topk(k, dim=-1)
        return {'probs': probs, 'top_k_probs': top_probs, 'top_k_ids': top_ids}

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.preprocess(images)
        features = self.encode(x)
        logits = self.classifier(features)
        result = self.postprocess(logits)
        result['logits'] = logits
        result['features'] = features
        return result


# ============================================================
# 2. ONNXExporter: torch.onnx.export 封装
# ============================================================
class ONNXExporter:
    """ONNX 模型导出器"""

    def __init__(self, config: ExportConfig):
        self.config = config

    def export(self, model: nn.Module, dummy_input: torch.Tensor,
               save_path: str) -> str:
        """导出模型为 ONNX 格式"""
        model.eval()
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)

        dynamic_axes = None
        if self.config.dynamic_axes:
            dynamic_axes = {
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'},
            }

        # 模拟 ONNX 导出（不实际调用 torch.onnx.export，避免依赖 onnx 包）
        # 在实际项目中使用:
        # torch.onnx.export(
        #     model, dummy_input, save_path,
        #     opset_version=self.config.opset_version,
        #     input_names=['input'],
        #     output_names=['output'],
        #     dynamic_axes=dynamic_axes,
        # )

        # 模拟导出：保存 TorchScript 作为替代
        traced = torch.jit.trace(model, dummy_input)
        traced.save(save_path.replace('.onnx', '.pt'))

        export_info = {
            'format': self.config.export_format,
            'opset_version': self.config.opset_version,
            'dynamic_axes': self.config.dynamic_axes,
            'input_shape': list(dummy_input.shape),
            'save_path': save_path,
        }
        return save_path

    def benchmark_export(self, model: nn.Module, dummy_input: torch.Tensor,
                         num_runs: int = 50) -> Dict[str, float]:
        """对比原始 PyTorch 与 TorchScript 推理速度"""
        import time

        model.eval()
        traced = torch.jit.trace(model, dummy_input)

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                model(dummy_input)
                traced(dummy_input)

        # PyTorch
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                model(dummy_input)
        pytorch_time = (time.time() - start) / num_runs * 1000

        # TorchScript
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                traced(dummy_input)
        traced_time = (time.time() - start) / num_runs * 1000

        return {
            'pytorch_ms': pytorch_time,
            'torchscript_ms': traced_time,
            'speedup': pytorch_time / max(traced_time, 1e-6),
        }


# ============================================================
# 3. QuantizedModel: INT8 动态量化
# ============================================================
class QuantizedModel:
    """INT8 量化封装"""

    @staticmethod
    def quantize_dynamic(model: nn.Module) -> nn.Module:
        """动态量化 Linear 层"""
        model_cpu = model.cpu().eval()
        quantized = torch.quantization.quantize_dynamic(
            model_cpu, {nn.Linear}, dtype=torch.qint8
        )
        return quantized

    @staticmethod
    def compare_outputs(fp32_model: nn.Module, int8_model: nn.Module,
                        test_input: torch.Tensor) -> Dict[str, float]:
        """对比 FP32 与 INT8 输出"""
        fp32_model.eval()
        int8_model.eval()
        test_input_cpu = test_input.cpu()

        with torch.no_grad():
            fp32_out = fp32_model.cpu()(test_input_cpu)
            int8_out = int8_model(test_input_cpu)

        # 提取 logits
        fp32_logits = fp32_out['logits'] if isinstance(fp32_out, dict) else fp32_out
        int8_logits = int8_out['logits'] if isinstance(int8_out, dict) else int8_out

        mse = F.mse_loss(fp32_logits, int8_logits).item()
        cos_sim = F.cosine_similarity(fp32_logits, int8_logits, dim=-1).mean().item()

        # Top-1 一致率
        fp32_pred = fp32_logits.argmax(dim=-1)
        int8_pred = int8_logits.argmax(dim=-1)
        agreement = (fp32_pred == int8_pred).float().mean().item()

        return {'mse': mse, 'cosine_sim': cos_sim, 'top1_agreement': agreement}

    @staticmethod
    def benchmark_speed(fp32_model: nn.Module, int8_model: nn.Module,
                        test_input: torch.Tensor, num_runs: int = 50) -> Dict[str, float]:
        """对比推理速度"""
        import time
        fp32_model.cpu().eval()
        int8_model.eval()
        test_input_cpu = test_input.cpu()

        # Warmup
        for _ in range(5):
            with torch.no_grad():
                fp32_model(test_input_cpu)
                int8_model(test_input_cpu)

        # FP32
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                fp32_model(test_input_cpu)
        fp32_ms = (time.time() - start) / num_runs * 1000

        # INT8
        start = time.time()
        for _ in range(num_runs):
            with torch.no_grad():
                int8_model(test_input_cpu)
        int8_ms = (time.time() - start) / num_runs * 1000

        return {
            'fp32_ms': fp32_ms,
            'int8_ms': int8_ms,
            'speedup': fp32_ms / max(int8_ms, 1e-6),
        }


# ============================================================
# 4. ModelEnsemble: 加权模型集成
# ============================================================
class ModelEnsemble(nn.Module):
    """多模型加权集成"""

    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        assert len(weights) == len(models)
        self.weights = weights

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        all_logits = []
        all_features = []
        for model, weight in zip(self.models, self.weights):
            with torch.no_grad():
                out = model(images)
            logits = out['logits'] if isinstance(out, dict) else out
            all_logits.append(logits * weight)
            if isinstance(out, dict) and 'features' in out:
                all_features.append(out['features'])

        ensemble_logits = sum(all_logits)
        probs = F.softmax(ensemble_logits, dim=-1)
        top_probs, top_ids = probs.topk(min(5, probs.size(-1)), dim=-1)

        result = {
            'logits': ensemble_logits,
            'probs': probs,
            'top_k_probs': top_probs,
            'top_k_ids': top_ids,
        }
        if all_features:
            result['features'] = torch.stack(all_features).mean(dim=0)
        return result
