"""
V23 - 在线推理服务 推理与实验
================================
1. Batch Size vs 吞吐量 / 延迟
2. ONNX(TorchScript) vs PyTorch 速度对比
3. 缓存命中率分析
4. 量化精度-速度 Tradeoff
"""
import os, sys, time, torch, torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device
from config import FullConfig
from model import ServingPipeline, ONNXExporter, QuantizedModel, ModelEnsemble
from serving_modules import DynamicBatcher, ModelCache, FeatureStore, LatencyProfiler, LoadBalancer
from dataset import RequestLogGenerator


def demo_batch_throughput():
    """实验1: 批大小 vs 吞吐量/延迟"""
    logger = get_logger("BatchThroughput")
    device = get_device()
    config = FullConfig()
    model = ServingPipeline(config.serving).to(device)
    model.eval()

    logger.info("批大小 vs 吞吐量/延迟 分析：")
    logger.info(f"{'BS':>4} {'Latency(ms)':>12} {'P95(ms)':>10} {'Throughput':>12}")
    logger.info("-" * 42)

    profiler = LatencyProfiler()
    for bs in [1, 2, 4, 8, 16, 32]:
        images = torch.randn(bs, 3, config.serving.image_size,
                             config.serving.image_size).to(device)
        # Warmup
        for _ in range(5):
            with torch.no_grad():
                model(images)

        profiler.reset()
        for _ in range(30):
            profiler.start('infer')
            with torch.no_grad():
                model(images)
            profiler.stop('infer')

        stats = profiler.summary()['infer']
        throughput = bs / (stats['mean_ms'] / 1000)
        logger.info(f"{bs:>4} {stats['mean_ms']:>12.2f} {stats['p95_ms']:>10.2f} {throughput:>10.1f} img/s")

    logger.info("→ 吞吐量随 BS 增大而增长（GPU并行），但单请求延迟也增加")


def demo_export_comparison():
    """实验2: PyTorch vs TorchScript 推理速度"""
    logger = get_logger("ExportCompare")
    device = get_device()
    config = FullConfig()
    model = ServingPipeline(config.serving).to(device)
    model.eval()

    exporter = ONNXExporter(config.export)

    logger.info("PyTorch vs TorchScript 推理对比：")
    for bs in [1, 4, 16]:
        dummy = torch.randn(bs, 3, config.serving.image_size,
                            config.serving.image_size).to(device)
        result = exporter.benchmark_export(model, dummy, num_runs=30)
        logger.info(f"  BS={bs}: PyTorch={result['pytorch_ms']:.2f}ms, "
                    f"TorchScript={result['torchscript_ms']:.2f}ms, "
                    f"Speedup={result['speedup']:.2f}x")

    logger.info("→ TorchScript 通过图优化（算子融合、常量折叠）通常有 10-30% 加速")


def demo_cache_analysis():
    """实验3: 缓存命中率分析（不同 Zipf 分布参数）"""
    logger = get_logger("CacheAnalysis")
    config = FullConfig()
    gen = RequestLogGenerator(config)

    logger.info("缓存命中率分析 (LRU, capacity=1024)：")
    logger.info(f"{'Zipf α':>8} {'Hit Rate':>10} {'Hits':>8} {'Misses':>8} {'Evictions':>10}")
    logger.info("-" * 50)

    for alpha in [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]:
        cache = ModelCache(capacity=config.cache.cache_size, ttl_seconds=config.cache.ttl_seconds)
        keys = gen.generate_skewed_keys(n=5000, alpha=alpha)

        for key in keys:
            result = cache.get(key)
            if result is None:
                cache.put(key, f"val_{key}")

        stats = cache.get_stats()
        logger.info(f"{alpha:>8.1f} {stats['hit_rate']:>10.4f} "
                    f"{stats['hits']:>8} {stats['misses']:>8} {stats['evictions']:>10}")

    logger.info("→ α 越大，请求越集中在少量热门 key，缓存效果越好")

    # TTL 敏感性
    logger.info("\nTTL 敏感性分析 (α=1.2)：")
    for ttl in [1.0, 10.0, 60.0, 300.0, 3600.0]:
        cache = ModelCache(capacity=config.cache.cache_size, ttl_seconds=ttl)
        keys = gen.generate_skewed_keys(n=3000, alpha=1.2)
        for key in keys:
            result = cache.get(key)
            if result is None:
                cache.put(key, f"val_{key}")
        logger.info(f"  TTL={ttl:>7.0f}s: hit_rate={cache.hit_rate():.4f}")


def demo_quantization_tradeoff():
    """实验4: 量化精度-速度 Tradeoff"""
    logger = get_logger("Quantization")
    config = FullConfig()

    logger.info("INT8 量化 精度-速度 Tradeoff：")

    model = ServingPipeline(config.serving)
    model.eval()
    int8_model = QuantizedModel.quantize_dynamic(model)

    # 不同输入分析
    logger.info("\n不同批大小下的量化效果：")
    for bs in [1, 4, 8, 16]:
        test_input = torch.randn(bs, 3, config.serving.image_size, config.serving.image_size)
        compare = QuantizedModel.compare_outputs(model, int8_model, test_input)
        speed = QuantizedModel.benchmark_speed(model, int8_model, test_input, num_runs=20)

        logger.info(f"  BS={bs}: MSE={compare['mse']:.6f}, "
                    f"CosSim={compare['cosine_sim']:.4f}, "
                    f"Top1Agree={compare['top1_agreement']:.4f}, "
                    f"Speedup={speed['speedup']:.2f}x")

    # 模型大小对比
    fp32_params = sum(p.numel() * 4 for p in model.parameters())  # 4 bytes per float32
    logger.info(f"\n  FP32 模型大小: {fp32_params / 1024 / 1024:.2f} MB")
    logger.info(f"  INT8 估计大小: ~{fp32_params / 4 / 1024 / 1024:.2f} MB (4x 压缩)")

    # 向量索引检索
    logger.info("\n向量索引对比 (Flat vs IVF)：")
    store = FeatureStore(config.index)
    n_vecs = 1000
    for i in range(n_vecs):
        v = torch.randn(config.index.embedding_dim)
        store.add(f"v_{i:04d}", v / v.norm())
    store.build_ivf_index()

    profiler = LatencyProfiler()
    num_queries = 50
    recalls = []
    for q in range(num_queries):
        query = torch.randn(config.index.embedding_dim)
        query = query / query.norm()

        profiler.start('flat')
        flat_res = store.search_flat(query, top_k=10)
        profiler.stop('flat')

        profiler.start('ivf')
        ivf_res = store.search_ivf(query, top_k=10)
        profiler.stop('ivf')

        flat_keys = set(k for _, k in flat_res)
        ivf_keys = set(k for _, k in ivf_res)
        recalls.append(len(flat_keys & ivf_keys) / max(len(flat_keys), 1))

    summary = profiler.summary()
    avg_recall = sum(recalls) / len(recalls)
    logger.info(f"  Flat: mean={summary['flat']['mean_ms']:.3f}ms, p95={summary['flat']['p95_ms']:.3f}ms")
    logger.info(f"  IVF:  mean={summary['ivf']['mean_ms']:.3f}ms, p95={summary['ivf']['p95_ms']:.3f}ms")
    logger.info(f"  IVF recall@10: {avg_recall:.4f}")
    logger.info(f"  IVF speedup:   {summary['flat']['mean_ms'] / max(summary['ivf']['mean_ms'], 0.001):.2f}x")


def main():
    set_seed(42)
    print("=" * 60)
    print("V23 - 在线推理服务系统 推理实验")
    print("=" * 60)
    demo_batch_throughput(); print()
    demo_export_comparison(); print()
    demo_cache_analysis(); print()
    demo_quantization_tradeoff()


if __name__ == "__main__":
    main()
