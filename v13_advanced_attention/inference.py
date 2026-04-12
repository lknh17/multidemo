"""v13 高级注意力机制 - 推理: KV Cache 速度/显存 Benchmark"""
import os
import sys
import time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import AdvancedAttentionConfig, config
from model import AdvancedTransformer
from attention_variants import flash_attention_simulation, AttentionSinkDetector
from shared.utils import get_device, load_checkpoint


@torch.no_grad()
def benchmark_kv_cache(device):
    """对比 MHA/GQA/MQA 的 KV Cache 大小和推理速度"""
    print("=" * 60)
    print("KV Cache 大小对比")
    print("=" * 60)
    
    variants = [
        ("MHA", 8, 8),
        ("GQA-4", 8, 4),
        ("GQA-2", 8, 2),
        ("MQA", 8, 1),
    ]
    
    seq_lens = [128, 256, 512, 1024]
    
    for name, n_heads, n_kv_heads in variants:
        d_k = config.d_model // n_heads
        print(f"\n  {name} (n_kv_heads={n_kv_heads}):")
        for sl in seq_lens:
            size = AdvancedTransformer.count_kv_cache_size(
                config.n_layers, n_kv_heads, d_k, sl
            )
            print(f"    seq_len={sl:>5}: {size / 1024:>8.1f} KB")


@torch.no_grad()
def benchmark_inference_speed(device):
    """对比不同注意力变体的生成速度"""
    print("\n" + "=" * 60)
    print("推理速度对比 (生成 50 tokens)")
    print("=" * 60)
    
    variants = [
        ("MHA", 8, 8),
        ("GQA", 8, 2),
        ("MQA", 8, 1),
    ]
    
    prompt = torch.randint(1, config.vocab_size, (1, 16), device=device)
    
    for name, n_heads, n_kv_heads in variants:
        cfg = AdvancedAttentionConfig(
            n_heads=n_heads, n_kv_heads=n_kv_heads,
            d_model=config.d_model, n_layers=config.n_layers,
            d_ff=config.d_ff, vocab_size=config.vocab_size,
            max_seq_len=config.max_seq_len,
        )
        model = AdvancedTransformer(cfg).to(device)
        model.eval()
        
        # Warmup
        _ = model(prompt)
        
        # Benchmark
        start = time.time()
        n_runs = 5
        for _ in range(n_runs):
            output = model.generate(prompt, max_new_tokens=50)
        elapsed = (time.time() - start) / n_runs
        
        tokens_per_sec = 50 / elapsed
        print(f"  {name}: {elapsed:.3f}s ({tokens_per_sec:.0f} tok/s)")


@torch.no_grad()
def demo_flash_attention():
    """演示 Flash Attention 模拟"""
    print("\n" + "=" * 60)
    print("Flash Attention 模拟对比")
    print("=" * 60)
    
    B, H, N, d = 2, 4, 128, 32
    Q = torch.randn(B, H, N, d)
    K = torch.randn(B, H, N, d)
    V = torch.randn(B, H, N, d)
    
    # 标准注意力
    scores = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    attn = torch.softmax(scores, dim=-1)
    standard_output = torch.matmul(attn, V)
    
    # Flash Attention 模拟
    flash_output = flash_attention_simulation(Q, K, V, block_size=32)
    
    # 验证精度
    max_diff = (standard_output - flash_output).abs().max().item()
    print(f"  标准 vs Flash 最大差异: {max_diff:.2e}")
    print(f"  结论: {'精确匹配 ✅' if max_diff < 1e-5 else '有误差 ⚠️'}")
    print(f"  Flash Attention 不是近似！是精确的，只是改变了计算顺序")


@torch.no_grad()
def demo_attention_sink(device):
    """演示 Attention Sink 现象"""
    print("\n" + "=" * 60)
    print("Attention Sink 现象检测")
    print("=" * 60)
    
    cfg = AdvancedAttentionConfig(n_layers=2)
    model = AdvancedTransformer(cfg).to(device)
    model.eval()
    
    # 生成一段序列并提取注意力权重
    input_ids = torch.randint(1, cfg.vocab_size, (1, 64), device=device)
    x = model.embedding(input_ids)
    
    mask = torch.tril(torch.ones(64, 64, device=device, dtype=torch.bool))
    mask = mask.unsqueeze(0).unsqueeze(0)
    
    x, attn_weights = model.layers[0](x, mask=mask, rope=model.rope)
    
    # 检测 sink
    avg_attn = attn_weights.mean(dim=(0, 1, 2))  # [L]
    print(f"  前 5 个位置的平均注意力: {avg_attn[:5].tolist()}")
    print(f"  中间 5 个位置的平均注意力: {avg_attn[30:35].tolist()}")
    print(f"  位置 0 的注意力是位置 32 的 {avg_attn[0] / (avg_attn[32] + 1e-8):.1f}x")
    
    # StreamingLLM 掩码
    streaming_mask = AttentionSinkDetector.streaming_inference_mask(
        64, sink_size=4, window_size=16, device=device
    )
    visible_count = streaming_mask[0, 0, -1].sum().item()
    print(f"\n  StreamingLLM 掩码: 最后一个 token 可见 {visible_count} 个位置")
    print(f"  (4 sink tokens + 16 recent tokens = 20)")


def main():
    device = get_device()
    
    print("=" * 60)
    print("v13 高级注意力机制 - 推理与 Benchmark")
    print("=" * 60)
    
    benchmark_kv_cache(device)
    benchmark_inference_speed(device)
    demo_flash_attention()
    demo_attention_sink(device)
    
    # 尝试加载训练好的模型
    for name in ["MHA", "GQA", "MQA"]:
        ckpt = os.path.join(config.checkpoint_dir, f"{name}_best.pt")
        if os.path.exists(ckpt):
            print(f"\n--- 加载 {name} 模型生成文本 ---")
            n_kv = config.n_heads if name == "MHA" else (2 if name == "GQA" else 1)
            cfg = AdvancedAttentionConfig(n_heads=config.n_heads, n_kv_heads=n_kv)
            model = AdvancedTransformer(cfg).to(device)
            load_checkpoint(model, ckpt, device=str(device))
            
            prompt = torch.randint(1, cfg.vocab_size, (1, 8), device=device)
            output = model.generate(prompt, max_new_tokens=20)
            print(f"  Prompt: {prompt[0].tolist()}")
            print(f"  Generated: {output}")


if __name__ == "__main__":
    main()
