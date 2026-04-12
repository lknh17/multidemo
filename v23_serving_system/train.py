"""
V23 - 在线推理服务训练 & 优化脚本
====================================
python train.py --mode export
python train.py --mode benchmark
python train.py --mode optimize
"""
import os, sys, argparse, time
import torch, torch.nn as nn, torch.optim as optim
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, save_checkpoint, get_device, AverageMeter
from config import FullConfig
from model import ServingPipeline, ONNXExporter, QuantizedModel, ModelEnsemble
from serving_modules import DynamicBatcher, ModelCache, FeatureStore, LatencyProfiler
from dataset import create_serving_dataloaders, create_benchmark_dataloader


def train_base_model(config, logger):
    """训练基础模型（用于后续导出/量化）"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Training Base ServingPipeline Model")
    logger.info("=" * 60)

    train_loader, val_loader = create_serving_dataloaders(config)
    model = ServingPipeline(config.serving).to(device)
    logger.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate,
                            weight_decay=config.weight_decay)

    for epoch in range(min(config.num_epochs, 5)):
        model.train()
        loss_meter = AverageMeter()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs['logits'], labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            loss_meter.update(loss.item())
        logger.info(f"Epoch {epoch+1}: loss={loss_meter.avg:.4f}")

    return model


def mode_export(config, logger):
    """模式1: 模型导出（TorchScript / ONNX 模拟）"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Mode: Model Export")
    logger.info("=" * 60)

    model = ServingPipeline(config.serving).to(device)
    model.eval()

    dummy_input = torch.randn(1, 3, config.serving.image_size,
                              config.serving.image_size).to(device)

    # TorchScript 导出
    logger.info("Exporting to TorchScript...")
    exporter = ONNXExporter(config.export)
    save_path = os.path.join(os.path.dirname(__file__), "exported_model.onnx")

    try:
        traced = torch.jit.trace(model, dummy_input)
        ts_path = save_path.replace('.onnx', '.pt')
        traced.save(ts_path)
        logger.info(f"TorchScript saved to: {ts_path}")

        # 验证导出正确性
        with torch.no_grad():
            orig_out = model(dummy_input)
            traced_out = traced(dummy_input)
        if isinstance(orig_out, dict):
            mse = nn.functional.mse_loss(orig_out['logits'], traced_out['logits']).item()
        else:
            mse = nn.functional.mse_loss(orig_out, traced_out).item()
        logger.info(f"Export validation MSE: {mse:.8f}")
    except Exception as e:
        logger.warning(f"Export failed: {e}")

    # 速度对比
    logger.info("\nBenchmarking PyTorch vs TorchScript...")
    bench_result = exporter.benchmark_export(model, dummy_input, num_runs=30)
    logger.info(f"  PyTorch:     {bench_result['pytorch_ms']:.2f} ms")
    logger.info(f"  TorchScript: {bench_result['torchscript_ms']:.2f} ms")
    logger.info(f"  Speedup:     {bench_result['speedup']:.2f}x")

    # INT8 量化
    logger.info("\nQuantizing to INT8...")
    int8_model = QuantizedModel.quantize_dynamic(model)
    test_input = torch.randn(4, 3, config.serving.image_size,
                             config.serving.image_size)
    compare = QuantizedModel.compare_outputs(model, int8_model, test_input)
    logger.info(f"  MSE:            {compare['mse']:.6f}")
    logger.info(f"  Cosine Sim:     {compare['cosine_sim']:.6f}")
    logger.info(f"  Top-1 Agreement:{compare['top1_agreement']:.4f}")

    speed = QuantizedModel.benchmark_speed(model, int8_model, test_input, num_runs=30)
    logger.info(f"  FP32:  {speed['fp32_ms']:.2f} ms")
    logger.info(f"  INT8:  {speed['int8_ms']:.2f} ms")
    logger.info(f"  Speedup: {speed['speedup']:.2f}x")


def mode_benchmark(config, logger):
    """模式2: 推理基准测试"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Mode: Inference Benchmark")
    logger.info("=" * 60)

    model = ServingPipeline(config.serving).to(device)
    model.eval()

    profiler = LatencyProfiler()

    # 不同批大小测试
    logger.info("\n--- Batch Size vs Throughput/Latency ---")
    for batch_size in [1, 2, 4, 8, 16, 32]:
        images = torch.randn(batch_size, 3, config.serving.image_size,
                             config.serving.image_size).to(device)
        # Warmup
        for _ in range(3):
            with torch.no_grad():
                model(images)

        num_runs = 20
        profiler.reset()
        for _ in range(num_runs):
            profiler.start('inference')
            with torch.no_grad():
                model(images)
            profiler.stop('inference')

        stats = profiler.summary()['inference']
        throughput = batch_size / (stats['mean_ms'] / 1000)
        logger.info(f"  BS={batch_size:>2}: latency={stats['mean_ms']:.2f}ms  "
                    f"p95={stats['p95_ms']:.2f}ms  throughput={throughput:.1f} img/s")

    # 动态批处理模拟
    logger.info("\n--- Dynamic Batching Simulation ---")
    batcher = DynamicBatcher(max_batch_size=config.serving.max_batch_size,
                             max_wait_ms=config.serving.max_latency_ms)
    for i in range(100):
        req = torch.randn(3, config.serving.image_size, config.serving.image_size)
        batcher.submit(req)
        batch = batcher.try_form_batch()
        if batch is not None:
            with torch.no_grad():
                batch_device = batch.to(device)
                model(batch_device)

    # 处理剩余
    import time as _time
    _time.sleep(config.serving.max_latency_ms / 1000 + 0.01)
    batch = batcher.try_form_batch()
    if batch is not None:
        with torch.no_grad():
            model(batch.to(device))

    stats = batcher.get_stats()
    logger.info(f"  Total requests: {stats['total_requests']}")
    logger.info(f"  Total batches:  {stats['total_batches']}")
    logger.info(f"  Avg batch size: {stats['avg_batch_size']:.1f}")
    logger.info(f"  Avg wait (ms):  {stats['avg_wait_ms']:.2f}")


def mode_optimize(config, logger):
    """模式3: 全链路优化"""
    device = get_device()
    logger.info("=" * 60)
    logger.info("Mode: Full Pipeline Optimization")
    logger.info("=" * 60)

    model = ServingPipeline(config.serving).to(device)
    model.eval()

    # 1. 缓存效果
    logger.info("\n--- Cache Hit Rate Analysis ---")
    cache = ModelCache(capacity=config.cache.cache_size, ttl_seconds=config.cache.ttl_seconds)
    from dataset import RequestLogGenerator
    gen = RequestLogGenerator(config)
    keys = gen.generate_skewed_keys(n=2000, alpha=1.2)

    for key in keys:
        result = cache.get(key)
        if result is None:
            # 模拟计算
            cache.put(key, f"result_for_{key}")
    cache_stats = cache.get_stats()
    logger.info(f"  Cache size:     {cache_stats['size']}")
    logger.info(f"  Hits:           {cache_stats['hits']}")
    logger.info(f"  Misses:         {cache_stats['misses']}")
    logger.info(f"  Hit rate:       {cache_stats['hit_rate']:.4f}")
    logger.info(f"  Evictions:      {cache_stats['evictions']}")

    # 2. 向量索引
    logger.info("\n--- Vector Index Benchmark ---")
    store = FeatureStore(config.index)
    n_vectors = 500
    for i in range(n_vectors):
        vec = torch.randn(config.index.embedding_dim)
        vec = vec / vec.norm()
        store.add(f"vec_{i:04d}", vec)

    store.build_ivf_index()

    query = torch.randn(config.index.embedding_dim)
    query = query / query.norm()

    profiler = LatencyProfiler()

    # Flat search
    profiler.start('flat')
    flat_results = store.search_flat(query, top_k=10)
    profiler.stop('flat')

    # IVF search
    profiler.start('ivf')
    ivf_results = store.search_ivf(query, top_k=10)
    profiler.stop('ivf')

    summary = profiler.summary()
    logger.info(f"  Flat search: {summary['flat']['mean_ms']:.3f} ms")
    logger.info(f"  IVF search:  {summary['ivf']['mean_ms']:.3f} ms")

    # 召回率
    flat_keys = set(k for _, k in flat_results)
    ivf_keys = set(k for _, k in ivf_results)
    recall = len(flat_keys & ivf_keys) / max(len(flat_keys), 1)
    logger.info(f"  IVF recall@10: {recall:.4f}")

    # 3. 负载均衡
    logger.info("\n--- Load Balancing ---")
    from serving_modules import LoadBalancer
    for strategy in ['round_robin', 'least_connection', 'consistent_hash']:
        lb = LoadBalancer(num_nodes=4, strategy=strategy)
        keys_sample = [f"request_{i}" for i in range(100)]
        dist = lb.get_distribution(keys_sample)
        balance = min(dist.values()) / max(max(dist.values()), 1)
        logger.info(f"  {strategy:>20}: distribution={dict(dist)}, balance={balance:.3f}")

    # 4. 模型集成
    logger.info("\n--- Model Ensemble ---")
    models = [ServingPipeline(config.serving).to(device) for _ in range(3)]
    ensemble = ModelEnsemble(models, weights=[0.5, 0.3, 0.2]).to(device)
    ensemble.eval()

    test_images = torch.randn(4, 3, config.serving.image_size,
                              config.serving.image_size).to(device)
    with torch.no_grad():
        single_out = models[0](test_images)
        ensemble_out = ensemble(test_images)
    logger.info(f"  Single model entropy:   {-(single_out['probs'] * torch.log(single_out['probs'] + 1e-9)).sum(-1).mean():.4f}")
    logger.info(f"  Ensemble entropy:       {-(ensemble_out['probs'] * torch.log(ensemble_out['probs'] + 1e-9)).sum(-1).mean():.4f}")
    logger.info("  → 集成通常熵更低（更确信），准确率更高")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="export",
                        choices=["export", "benchmark", "optimize"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = FullConfig()
    if args.epochs:
        config.num_epochs = args.epochs

    set_seed(config.seed)
    logger = get_logger("V23-Serving")

    {"export": mode_export, "benchmark": mode_benchmark, "optimize": mode_optimize}[args.mode](config, logger)


if __name__ == "__main__":
    main()
