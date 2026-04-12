"""
V23 - 在线推理服务核心模块
===========================
DynamicBatcher / ModelCache / FeatureStore / LatencyProfiler / LoadBalancer
"""
import time
import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np

from config import ServingConfig, CacheConfig, IndexConfig


# ============================================================
# 1. 动态批处理器
# ============================================================
class DynamicBatcher:
    """
    基于超时的动态批处理器。
    - 当队列达到 max_batch_size 时立即形成批次
    - 当最早请求等待超过 max_wait_ms 时强制形成批次
    """

    def __init__(self, max_batch_size: int = 32, max_wait_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.queue: List[Tuple[float, torch.Tensor]] = []
        self.lock = threading.Lock()
        self.stats = {'total_requests': 0, 'total_batches': 0, 'total_wait_ms': 0.0}

    def submit(self, request: torch.Tensor) -> int:
        """提交请求到队列，返回队列位置"""
        with self.lock:
            self.queue.append((time.time(), request))
            self.stats['total_requests'] += 1
            return len(self.queue) - 1

    def try_form_batch(self) -> Optional[torch.Tensor]:
        """尝试形成批次"""
        with self.lock:
            if not self.queue:
                return None

            # 条件1: 队列满
            if len(self.queue) >= self.max_batch_size:
                return self._drain_batch(self.max_batch_size)

            # 条件2: 超时
            oldest_time = self.queue[0][0]
            wait_ms = (time.time() - oldest_time) * 1000
            if wait_ms >= self.max_wait_ms:
                return self._drain_batch(len(self.queue))

            return None

    def _drain_batch(self, n: int) -> torch.Tensor:
        """从队列取出 n 个请求组成批次"""
        items = self.queue[:n]
        self.queue = self.queue[n:]

        wait_times = [(time.time() - t) * 1000 for t, _ in items]
        self.stats['total_batches'] += 1
        self.stats['total_wait_ms'] += sum(wait_times)

        tensors = [req for _, req in items]
        return torch.stack(tensors, dim=0)

    def get_stats(self) -> Dict[str, float]:
        """获取统计信息"""
        s = self.stats
        avg_wait = s['total_wait_ms'] / max(s['total_requests'], 1)
        avg_batch = s['total_requests'] / max(s['total_batches'], 1)
        return {
            'total_requests': s['total_requests'],
            'total_batches': s['total_batches'],
            'avg_wait_ms': avg_wait,
            'avg_batch_size': avg_batch,
        }


# ============================================================
# 2. 模型缓存 (LRU + TTL)
# ============================================================
class ModelCache:
    """
    LRU 缓存，支持 TTL 过期淘汰。
    - 命中时移到队尾（最近访问）
    - 容量满时淘汰队首（最久未访问）
    - 条目超过 TTL 视为失效
    """

    def __init__(self, capacity: int = 1024, ttl_seconds: float = 300.0):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, float] = {}
        self.stats = {'hits': 0, 'misses': 0, 'evictions': 0}

    def get(self, key: str) -> Optional[Any]:
        """查询缓存"""
        if key in self.cache:
            # 检查 TTL
            if time.time() - self.timestamps[key] > self.ttl:
                self._evict(key)
                self.stats['misses'] += 1
                return None
            # 命中 → 移到末尾
            self.cache.move_to_end(key)
            self.stats['hits'] += 1
            return self.cache[key]
        self.stats['misses'] += 1
        return None

    def put(self, key: str, value: Any):
        """写入缓存"""
        if key in self.cache:
            self.cache.move_to_end(key)
            self.cache[key] = value
            self.timestamps[key] = time.time()
            return
        if len(self.cache) >= self.capacity:
            oldest_key, _ = self.cache.popitem(last=False)
            del self.timestamps[oldest_key]
            self.stats['evictions'] += 1
        self.cache[key] = value
        self.timestamps[key] = time.time()

    def _evict(self, key: str):
        """淘汰指定条目"""
        del self.cache[key]
        del self.timestamps[key]
        self.stats['evictions'] += 1

    def hit_rate(self) -> float:
        total = self.stats['hits'] + self.stats['misses']
        return self.stats['hits'] / max(total, 1)

    def get_stats(self) -> Dict[str, Any]:
        return {**self.stats, 'hit_rate': self.hit_rate(), 'size': len(self.cache)}


# ============================================================
# 3. 特征存储 (内存 + 磁盘)
# ============================================================
class FeatureStore:
    """
    特征存储 & 向量近邻搜索。
    - 内存存储热数据，支持阈值后落盘
    - 支持 Flat / IVF 近似最近邻搜索
    """

    def __init__(self, config: IndexConfig):
        self.config = config
        self.dim = config.embedding_dim
        self.vectors: Dict[str, torch.Tensor] = {}
        self.centroids: Optional[torch.Tensor] = None
        self.inverted_lists: Optional[Dict[int, List[str]]] = None

    def add(self, key: str, embedding: torch.Tensor):
        """添加向量"""
        assert embedding.shape[-1] == self.dim, f"期望维度 {self.dim}, 得到 {embedding.shape[-1]}"
        self.vectors[key] = embedding.detach().cpu()

    def add_batch(self, keys: List[str], embeddings: torch.Tensor):
        """批量添加"""
        for i, key in enumerate(keys):
            self.vectors[key] = embeddings[i].detach().cpu()

    def build_ivf_index(self):
        """构建 IVF 索引"""
        if len(self.vectors) < self.config.nlist:
            return
        all_vecs = torch.stack(list(self.vectors.values()))
        all_keys = list(self.vectors.keys())

        # K-Means 聚类
        indices = torch.randperm(len(all_vecs))[:self.config.nlist]
        self.centroids = all_vecs[indices].clone()

        for _ in range(10):  # K-Means 迭代
            dists = torch.cdist(all_vecs, self.centroids)
            assignments = dists.argmin(dim=1)
            for c in range(self.config.nlist):
                mask = assignments == c
                if mask.any():
                    self.centroids[c] = all_vecs[mask].mean(dim=0)

        # 构建倒排列表
        dists = torch.cdist(all_vecs, self.centroids)
        assignments = dists.argmin(dim=1)
        self.inverted_lists = {}
        for i, cluster_id in enumerate(assignments.tolist()):
            if cluster_id not in self.inverted_lists:
                self.inverted_lists[cluster_id] = []
            self.inverted_lists[cluster_id].append(all_keys[i])

    def search_flat(self, query: torch.Tensor, top_k: int = 10) -> List[Tuple[float, str]]:
        """暴力精确搜索"""
        if not self.vectors:
            return []
        all_keys = list(self.vectors.keys())
        all_vecs = torch.stack(list(self.vectors.values()))
        dists = torch.norm(all_vecs - query.unsqueeze(0), dim=-1)
        k = min(top_k, len(all_keys))
        top_dists, top_ids = dists.topk(k, largest=False)
        return [(top_dists[i].item(), all_keys[top_ids[i]]) for i in range(k)]

    def search_ivf(self, query: torch.Tensor, top_k: int = 10) -> List[Tuple[float, str]]:
        """IVF 近似搜索"""
        if self.centroids is None or self.inverted_lists is None:
            return self.search_flat(query, top_k)

        # 找最近的 nprobe 个聚类中心
        centroid_dists = torch.norm(self.centroids - query.unsqueeze(0), dim=-1)
        nprobe = min(self.config.nprobe, len(self.centroids))
        _, probe_ids = centroid_dists.topk(nprobe, largest=False)

        # 在这些簇中搜索
        candidates = []
        for cluster_id in probe_ids.tolist():
            if cluster_id in self.inverted_lists:
                for key in self.inverted_lists[cluster_id]:
                    dist = torch.norm(query - self.vectors[key]).item()
                    candidates.append((dist, key))

        candidates.sort(key=lambda x: x[0])
        return candidates[:top_k]

    def search(self, query: torch.Tensor, top_k: int = None) -> List[Tuple[float, str]]:
        """统一搜索接口"""
        k = top_k or self.config.top_k
        if self.config.index_type == "ivf" and self.centroids is not None:
            return self.search_ivf(query, k)
        return self.search_flat(query, k)


# ============================================================
# 4. 延迟剖析器
# ============================================================
class LatencyProfiler:
    """分阶段延迟剖析，支持多次采样求统计"""

    def __init__(self):
        self.records: Dict[str, List[float]] = {}
        self._start_times: Dict[str, float] = {}

    def start(self, stage: str):
        """开始计时"""
        self._start_times[stage] = time.time()

    def stop(self, stage: str) -> float:
        """停止计时，返回耗时 (ms)"""
        if stage not in self._start_times:
            return 0.0
        elapsed_ms = (time.time() - self._start_times[stage]) * 1000
        if stage not in self.records:
            self.records[stage] = []
        self.records[stage].append(elapsed_ms)
        del self._start_times[stage]
        return elapsed_ms

    def summary(self) -> Dict[str, Dict[str, float]]:
        """统计各阶段延迟"""
        result = {}
        for stage, times in self.records.items():
            arr = np.array(times)
            result[stage] = {
                'mean_ms': float(arr.mean()),
                'p50_ms': float(np.percentile(arr, 50)),
                'p95_ms': float(np.percentile(arr, 95)),
                'p99_ms': float(np.percentile(arr, 99)),
                'count': len(times),
            }
        return result

    def reset(self):
        self.records.clear()
        self._start_times.clear()


# ============================================================
# 5. 负载均衡器
# ============================================================
class LoadBalancer:
    """
    支持 Round-Robin / Least-Connection / Consistent Hash 三种策略。
    """

    def __init__(self, num_nodes: int = 4, strategy: str = "round_robin"):
        self.num_nodes = num_nodes
        self.strategy = strategy
        self.counter = 0
        self.active_connections = [0] * num_nodes

        # Consistent Hash Ring
        self.ring: List[Tuple[int, int]] = []  # (hash_value, node_id)
        if strategy == "consistent_hash":
            self._build_hash_ring(virtual_nodes=150)

    def _build_hash_ring(self, virtual_nodes: int = 150):
        """构建一致性哈希环"""
        self.ring = []
        for node_id in range(self.num_nodes):
            for vn in range(virtual_nodes):
                key = f"node_{node_id}_vn_{vn}"
                h = int(hashlib.md5(key.encode()).hexdigest(), 16) % (2 ** 32)
                self.ring.append((h, node_id))
        self.ring.sort(key=lambda x: x[0])

    def _consistent_hash_route(self, request_key: str) -> int:
        """一致性哈希路由"""
        h = int(hashlib.md5(request_key.encode()).hexdigest(), 16) % (2 ** 32)
        for ring_hash, node_id in self.ring:
            if ring_hash >= h:
                return node_id
        return self.ring[0][1] if self.ring else 0

    def route(self, request_key: str = "") -> int:
        """路由请求到某个节点"""
        if self.strategy == "round_robin":
            node = self.counter % self.num_nodes
            self.counter += 1
            return node
        elif self.strategy == "least_connection":
            node = int(np.argmin(self.active_connections))
            self.active_connections[node] += 1
            return node
        elif self.strategy == "consistent_hash":
            return self._consistent_hash_route(request_key)
        else:
            return 0

    def release(self, node_id: int):
        """释放连接（用于 least_connection）"""
        if 0 <= node_id < self.num_nodes:
            self.active_connections[node_id] = max(0, self.active_connections[node_id] - 1)

    def get_distribution(self, keys: List[str]) -> Dict[int, int]:
        """分析请求分布"""
        dist = {i: 0 for i in range(self.num_nodes)}
        for key in keys:
            node = self.route(key)
            dist[node] += 1
        return dist
