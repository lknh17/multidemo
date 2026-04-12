# V23 - 在线推理服务系统：代码详解

## 1. 动态批处理器（serving_modules.py）

### 1.1 DynamicBatcher

```python
class DynamicBatcher:
    def __init__(self, max_batch_size=32, max_wait_ms=10.0):
        self.queue = []           # 待处理请求队列
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms  # 最长等待时间

    def submit(self, request):
        # 1. 将请求加入队列，附带时间戳
        self.queue.append((time.time(), request))

    def try_form_batch(self):
        # 2. 批次形成条件（二选一）：
        #    a. 队列达到 max_batch_size
        #    b. 最早请求等待超过 max_wait_ms
        if len(self.queue) >= self.max_batch_size:
            return self._drain_batch()
        if self.queue and self._wait_time_ms() > self.max_wait_ms:
            return self._drain_batch()
        return None
```

**要点**：超时机制确保低负载时请求不会无限等待，高负载时凑满整批提升 GPU 利用率。

### 1.2 ModelCache (LRU + TTL)

```python
class ModelCache:
    def __init__(self, capacity=1024, ttl=300.0):
        self.cache = OrderedDict()  # LRU 顺序
        self.timestamps = {}        # 插入时间
        self.capacity = capacity
        self.ttl = ttl

    def get(self, key):
        if key in self.cache:
            # 检查 TTL 是否过期
            if time.time() - self.timestamps[key] > self.ttl:
                self._evict(key)
                return None  # Miss
            # 命中：移到末尾 (最近访问)
            self.cache.move_to_end(key)
            return self.cache[key]  # Hit
        return None

    def put(self, key, value):
        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)  # LRU 淘汰最旧
        self.cache[key] = value
        self.timestamps[key] = time.time()
```

## 2. 向量索引与特征存储

### 2.1 FeatureStore

```python
class FeatureStore:
    def __init__(self, dim, capacity):
        self.memory = {}           # 内存存储 (热数据)
        self.disk_path = "./store" # 磁盘存储 (冷数据)

    def add(self, key, embedding):
        self.memory[key] = embedding
        if len(self.memory) > self.capacity:
            self._flush_to_disk()  # 冷数据落盘

    def search(self, query, top_k=10):
        # IVF 近似搜索
        distances = []
        for key, emb in self.memory.items():
            d = torch.norm(query - emb)
            distances.append((d, key))
        return sorted(distances)[:top_k]
```

## 3. ServingPipeline（model.py）

```python
class ServingPipeline(nn.Module):
    def forward(self, images):
        # 三阶段流水线
        # 1. 预处理：归一化、Resize
        x = self.preprocess(images)

        # 2. 推理：ViT 编码 + 分类头
        features = self.encoder(x)
        logits = self.classifier(features[:, 0])

        # 3. 后处理：Softmax、Top-K
        probs = F.softmax(logits, dim=-1)
        top_k_probs, top_k_ids = probs.topk(5)
        return {'logits': logits, 'probs': probs,
                'top_k_probs': top_k_probs, 'top_k_ids': top_k_ids}
```

## 4. ONNX 导出器

```python
class ONNXExporter:
    def export(self, model, dummy_input, path):
        torch.onnx.export(
            model, dummy_input, path,
            opset_version=17,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
        )
        # 导出后可用 onnxruntime 加载推理
        # sess = ort.InferenceSession(path)
        # result = sess.run(None, {'input': data})
```

## 5. INT8 量化

```python
class QuantizedModel:
    def quantize(self, model, calibration_data):
        # 1. 动态量化：不需要校准数据
        q_model = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        # 2. 量化后模型大小减少 ~4x
        # 3. 推理速度提升 ~2-4x (CPU)
        return q_model

    def compare_accuracy(self, fp32_model, int8_model, data):
        # 对比 FP32 与 INT8 的输出差异
        fp32_out = fp32_model(data)
        int8_out = int8_model(data)
        mse = F.mse_loss(fp32_out, int8_out)
        # 典型 MSE < 0.01
```

## 6. 模型集成

```python
class ModelEnsemble(nn.Module):
    def forward(self, images):
        # 加权集成多个模型
        all_logits = []
        for model, weight in zip(self.models, self.weights):
            logits = model(images)['logits']
            all_logits.append(logits * weight)

        ensemble_logits = sum(all_logits)
        # 集成通常提升 1-3% 准确率
        return ensemble_logits
```

## 7. 负载均衡器

```python
class LoadBalancer:
    def route(self, request):
        if self.strategy == 'round_robin':
            node = self.nodes[self.counter % len(self.nodes)]
            self.counter += 1
        elif self.strategy == 'least_connection':
            node = min(self.nodes, key=lambda n: n.active_count)
        elif self.strategy == 'consistent_hash':
            node = self._hash_ring.get_node(request.key)
        return node
```
