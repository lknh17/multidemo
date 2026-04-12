# V23 - 在线推理服务系统

## 核心知识点
- 模型服务架构（Serving Pipeline）
- ONNX / TensorRT 模型导出与优化
- 动态批处理策略（Dynamic Batching）
- 模型缓存与特征存储（Feature Store）
- 向量索引（FAISS / IVF）
- 延迟优化与性能剖析
- 负载均衡策略（Round-Robin / Least-Connection / Consistent Hash）
- INT8 量化与模型集成（Ensemble）

## 运行方式

```bash
python train.py --mode export
python train.py --mode benchmark
python train.py --mode optimize
python inference.py
```

## 文件说明
| 文件 | 说明 |
|------|------|
| config.py | 服务系统配置（Serving / Export / Cache / Index） |
| serving_modules.py | 动态批处理器 / 模型缓存 / 特征存储 / 延迟剖析 / 负载均衡 |
| model.py | ServingPipeline / ONNX导出 / INT8量化 / 模型集成 |
| dataset.py | 合成请求日志与服务基准数据 |
| train.py | 导出 / 基准测试 / 优化三种模式 |
| inference.py | 推理实验（批大小 vs 吞吐、ONNX对比、缓存命中率、量化精度） |
| theory.md | 数学原理 |
| code_explained.md | 代码详解 |
| theory_visual.html | 原理动画 |
| code_visual.html | 代码动画 |
