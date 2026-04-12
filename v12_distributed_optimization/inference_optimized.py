"""
v12 优化推理: 量化 + ONNX 导出 + 性能 Benchmark
"""
import os, sys, time, torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v09_deepstack_fusion"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "v11_multimodal_embedding"))

from config import config
from model import EmbeddingModel as BaseModel
from shared.utils import get_device


def benchmark_inference(model, device, name, batch_size=32, n_runs=50):
    """性能基准测试"""
    model.eval()
    img = torch.randn(batch_size, 3, config.image_size, config.image_size).to(device)
    ids = torch.randint(4, config.vocab_size, (batch_size, config.max_text_len)).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(5):
            model.encode_image(img)
    
    # Benchmark
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            model.encode_image(img)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0
    
    throughput = n_runs * batch_size / elapsed
    latency = elapsed / n_runs * 1000
    print(f"[{name}] Throughput: {throughput:.0f} samples/s | Latency: {latency:.1f}ms/batch")
    return throughput


def quantize_model(model):
    """动态量化 (INT8)"""
    quantized = torch.quantization.quantize_dynamic(
        model.cpu(), {torch.nn.Linear}, dtype=torch.qint8
    )
    # 统计模型大小
    orig_size = sum(p.nelement() * p.element_size() for p in model.parameters()) / 1e6
    quant_size = sum(p.nelement() * p.element_size() for p in quantized.parameters()) / 1e6
    print(f"原始模型: {orig_size:.1f}MB → 量化后: {quant_size:.1f}MB ({quant_size/orig_size*100:.0f}%)")
    return quantized


def export_onnx(model, device):
    """导出 ONNX 模型"""
    model.eval()
    img = torch.randn(1, 3, config.image_size, config.image_size).to(device)
    ids = torch.randint(4, config.vocab_size, (1, config.max_text_len)).to(device)
    
    save_path = os.path.join(config.checkpoint_dir, "model.onnx")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        torch.onnx.export(
            model, (img, ids), save_path,
            input_names=["images", "input_ids"],
            output_names=["img_emb", "txt_emb", "scale"],
            dynamic_axes={"images": {0: "batch"}, "input_ids": {0: "batch"}},
            opset_version=14,
        )
        print(f"ONNX 模型已导出到: {save_path}")
        file_size = os.path.getsize(save_path) / 1e6
        print(f"ONNX 文件大小: {file_size:.1f}MB")
    except Exception as e:
        print(f"ONNX 导出失败: {e}")


def main():
    device = get_device()
    model_cfg = type("Cfg", (), {k: v for k, v in vars(config).items()})()
    model = BaseModel(model_cfg).to(device)
    
    print("=" * 60)
    print("v12 推理优化 Benchmark")
    print("=" * 60)
    
    # 1. 原始精度
    print("\n--- FP32 推理 ---")
    benchmark_inference(model, device, "FP32")
    
    # 2. FP16 推理 (如果有 GPU)
    if torch.cuda.is_available():
        print("\n--- FP16 推理 ---")
        model_fp16 = model.half()
        benchmark_inference(model_fp16, device, "FP16")
        model = model.float()
    
    # 3. 动态量化 (CPU)
    print("\n--- INT8 动态量化 (CPU) ---")
    model_q = quantize_model(model)
    benchmark_inference(model_q, torch.device("cpu"), "INT8-CPU")
    
    # 4. ONNX 导出
    print("\n--- ONNX 导出 ---")
    export_onnx(model.to(device), device)


if __name__ == "__main__":
    main()
