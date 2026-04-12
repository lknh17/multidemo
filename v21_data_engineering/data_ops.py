"""
V21 - 多模态数据工程 - 数据操作模块
====================================
实现数据去重（MinHash / SimHash）、质量评分、数据平衡等核心操作。
"""

import math
import random
import hashlib
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn


# ============================================================
# 1. MinHash 去重
#    用随机哈希函数估计集合间的 Jaccard 相似度
# ============================================================
class MinHashDedup:
    """
    MinHash 去重器。
    
    原理：
    - 将每个数据样本转化为 token 集合（shingling）
    - 使用 num_perm 个哈希函数，对每个集合取最小哈希值作为签名
    - 签名相同的比例 ≈ Jaccard 相似度
    - 使用 LSH（Banded MinHash）加速候选对查找
    """
    
    def __init__(self, num_perm: int = 128, threshold: float = 0.8,
                 shingle_size: int = 3, num_bands: int = 16):
        self.num_perm = num_perm
        self.threshold = threshold
        self.shingle_size = shingle_size
        self.num_bands = num_bands
        self.rows_per_band = num_perm // num_bands
        
        # 生成随机哈希参数：h(x) = (a*x + b) % p
        self._max_hash = (1 << 32) - 1
        self._prime = 4294967311  # 一个大素数
        rng = np.random.RandomState(42)
        self._a = rng.randint(1, self._prime, size=num_perm).astype(np.int64)
        self._b = rng.randint(0, self._prime, size=num_perm).astype(np.int64)
    
    def _shingle(self, text: str) -> set:
        """将文本转化为 n-gram 集合（shingling）。"""
        shingles = set()
        for i in range(len(text) - self.shingle_size + 1):
            shingle = text[i:i + self.shingle_size]
            # 将 shingle 转化为整数
            shingles.add(int(hashlib.md5(shingle.encode()).hexdigest()[:8], 16))
        return shingles
    
    def _compute_signature(self, shingles: set) -> np.ndarray:
        """
        计算 MinHash 签名。
        
        对于每个哈希函数 h_i，签名的第 i 个位置 = min{h_i(s) : s ∈ shingles}
        
        数学保证：Pr[sig_i(A) = sig_i(B)] = J(A, B)
        """
        sig = np.full(self.num_perm, np.inf)
        
        for s in shingles:
            # 并行计算所有哈希函数的值
            hashes = (self._a * s + self._b) % self._prime
            sig = np.minimum(sig, hashes)
        
        return sig.astype(np.int64)
    
    def _lsh_buckets(self, signature: np.ndarray) -> List[str]:
        """
        将签名分成 num_bands 个 band，每个 band 计算一个桶 ID。
        
        如果两个签名在任何一个 band 中落入同一个桶，就成为候选对。
        候选对的概率：P = 1 - (1 - J^r)^b
        """
        buckets = []
        for band_idx in range(self.num_bands):
            start = band_idx * self.rows_per_band
            end = start + self.rows_per_band
            band = signature[start:end]
            bucket_id = f"{band_idx}_{hashlib.md5(band.tobytes()).hexdigest()}"
            buckets.append(bucket_id)
        return buckets
    
    def deduplicate(self, texts: List[str]) -> Tuple[List[int], float]:
        """
        对文本列表进行 MinHash 去重。
        
        Returns:
            (保留的索引列表, 压缩率)
        """
        n = len(texts)
        signatures = []
        
        # 步骤 1: 计算所有文档的 MinHash 签名
        for text in texts:
            shingles = self._shingle(text)
            if not shingles:
                signatures.append(np.zeros(self.num_perm, dtype=np.int64))
            else:
                signatures.append(self._compute_signature(shingles))
        
        # 步骤 2: LSH 分桶，找候选对
        bucket_map = defaultdict(list)  # bucket_id -> [doc_indices]
        for idx, sig in enumerate(signatures):
            for bucket_id in self._lsh_buckets(sig):
                bucket_map[bucket_id].append(idx)
        
        # 步骤 3: 对候选对精确计算 Jaccard 估计
        duplicates = set()
        seen_pairs = set()
        
        for bucket_id, indices in bucket_map.items():
            for i in range(len(indices)):
                for j in range(i + 1, len(indices)):
                    pair = (min(indices[i], indices[j]), max(indices[i], indices[j]))
                    if pair in seen_pairs:
                        continue
                    seen_pairs.add(pair)
                    
                    # 估计 Jaccard：签名中相等元素的比例
                    jaccard_est = np.mean(
                        signatures[pair[0]] == signatures[pair[1]]
                    )
                    
                    if jaccard_est >= self.threshold:
                        duplicates.add(pair[1])  # 保留较早的文档
        
        kept_indices = [i for i in range(n) if i not in duplicates]
        compression = 1.0 - len(kept_indices) / n
        
        return kept_indices, compression


# ============================================================
# 2. SimHash 去重
#    将向量映射为二进制串，用 Hamming 距离近似余弦相似度
# ============================================================
class SimHashDedup:
    """
    SimHash 去重器。
    
    原理：
    - 使用随机超平面将向量映射为 bit 串
    - bit_i = 1 if v · r_i >= 0 else 0
    - Hamming 距离 / bits ≈ arccos(cos_sim) / π
    """
    
    def __init__(self, input_dim: int, hash_bits: int = 128, threshold: float = 0.8):
        self.hash_bits = hash_bits
        self.threshold = threshold
        # 随机超平面法向量
        rng = np.random.RandomState(42)
        self.planes = rng.randn(hash_bits, input_dim).astype(np.float32)
    
    def _compute_hash(self, vector: np.ndarray) -> np.ndarray:
        """
        计算 SimHash。
        
        对于每个随机超平面 r_i：
        bit_i = 1 if v · r_i >= 0 else 0
        """
        projections = self.planes @ vector  # [hash_bits]
        return (projections >= 0).astype(np.uint8)
    
    def _hamming_distance(self, h1: np.ndarray, h2: np.ndarray) -> int:
        """计算两个哈希的 Hamming 距离。"""
        return np.sum(h1 != h2)
    
    def _hamming_to_cosine(self, hamming_dist: int) -> float:
        """
        将 Hamming 距离转换为近似余弦相似度。
        
        d_H / bits ≈ θ/π  →  cos(θ) ≈ cos(π · d_H / bits)
        """
        theta = math.pi * hamming_dist / self.hash_bits
        return math.cos(theta)
    
    def deduplicate(self, vectors: np.ndarray) -> Tuple[List[int], float]:
        """
        对向量集合进行 SimHash 去重。
        
        Args:
            vectors: [N, dim] 特征向量
        
        Returns:
            (保留的索引列表, 压缩率)
        """
        n = len(vectors)
        hashes = [self._compute_hash(vectors[i]) for i in range(n)]
        
        duplicates = set()
        for i in range(n):
            if i in duplicates:
                continue
            for j in range(i + 1, n):
                if j in duplicates:
                    continue
                hamming = self._hamming_distance(hashes[i], hashes[j])
                cosine_sim = self._hamming_to_cosine(hamming)
                if cosine_sim >= self.threshold:
                    duplicates.add(j)
        
        kept_indices = [i for i in range(n) if i not in duplicates]
        compression = 1.0 - len(kept_indices) / n
        
        return kept_indices, compression


# ============================================================
# 3. 数据质量评分器
#    基于 CLIP 质量分、分辨率、宽高比等多维度评估
# ============================================================
class QualityScorer:
    """
    多维度数据质量评分器。
    
    综合评分：Q(x) = w_clip * S_clip + w_res * S_res + w_ar * S_ar + w_blur * S_blur
    
    各维度评分归一化到 [0, 1]。
    """
    
    def __init__(
        self,
        clip_weight: float = 0.4,
        resolution_weight: float = 0.3,
        aspect_ratio_weight: float = 0.2,
        blur_weight: float = 0.1,
        target_resolution: int = 224,
    ):
        self.weights = {
            "clip": clip_weight,
            "resolution": resolution_weight,
            "aspect_ratio": aspect_ratio_weight,
            "blur": blur_weight,
        }
        self.target_resolution = target_resolution
    
    def score_resolution(self, height: int, width: int) -> float:
        """
        分辨率评分：S_res = min(1, sqrt(H*W) / R_target)
        """
        geometric_mean = math.sqrt(height * width)
        return min(1.0, geometric_mean / self.target_resolution)
    
    def score_aspect_ratio(self, height: int, width: int) -> float:
        """
        宽高比评分：S_ar = exp(-(log(W/H))^2 / (2σ^2))
        
        偏好接近正方形的图片（σ 控制容忍度）。
        """
        sigma = 0.5
        log_ratio = math.log(max(width, 1) / max(height, 1))
        return math.exp(-(log_ratio ** 2) / (2 * sigma ** 2))
    
    def score_blur(self, laplacian_var: float, threshold: float = 100.0) -> float:
        """
        模糊度评分：基于 Laplacian 方差。
        方差越高越清晰。
        """
        return min(1.0, laplacian_var / threshold)
    
    def score_clip(self, clip_similarity: float) -> float:
        """
        CLIP 质量评分：图文匹配的余弦相似度（假设已计算）。
        """
        return max(0.0, min(1.0, clip_similarity))
    
    def compute_quality(
        self,
        height: int,
        width: int,
        laplacian_var: float = 100.0,
        clip_similarity: float = 0.5,
    ) -> Dict[str, float]:
        """
        计算综合质量评分。
        
        Returns:
            包含各维度评分和综合分的字典
        """
        scores = {
            "clip": self.score_clip(clip_similarity),
            "resolution": self.score_resolution(height, width),
            "aspect_ratio": self.score_aspect_ratio(height, width),
            "blur": self.score_blur(laplacian_var),
        }
        
        total = sum(self.weights[k] * scores[k] for k in scores)
        scores["total"] = total
        
        return scores
    
    def filter_by_quality(
        self,
        samples: List[Dict],
        threshold: float = 0.3,
    ) -> Tuple[List[Dict], List[float]]:
        """
        按质量阈值过滤样本。
        
        Args:
            samples: 样本列表，每个样本需包含 height, width 等字段
            threshold: 质量阈值
        
        Returns:
            (过滤后的样本列表, 对应的质量分列表)
        """
        filtered = []
        quality_scores = []
        
        for sample in samples:
            q = self.compute_quality(
                height=sample.get("height", 224),
                width=sample.get("width", 224),
                laplacian_var=sample.get("laplacian_var", 100.0),
                clip_similarity=sample.get("clip_similarity", 0.5),
            )
            if q["total"] >= threshold:
                filtered.append(sample)
                quality_scores.append(q["total"])
        
        return filtered, quality_scores


# ============================================================
# 4. 数据平衡器
#    类别权重计算与过采样策略
# ============================================================
class DataBalancer:
    """
    数据平衡器，处理类别不平衡问题。
    
    支持：
    - 逆频率权重：w_c = N / (C * n_c)
    - 有效样本数权重：w_c = (1 - β) / (1 - β^n_c)
    - 过采样：复制少数类样本
    """
    
    def __init__(self, num_classes: int, beta: float = 0.9999):
        self.num_classes = num_classes
        self.beta = beta
    
    def compute_class_counts(self, labels: List[int]) -> Dict[int, int]:
        """统计每个类别的样本数。"""
        counts = defaultdict(int)
        for label in labels:
            counts[label] += 1
        return dict(counts)
    
    def inverse_frequency_weights(self, labels: List[int]) -> torch.Tensor:
        """
        逆频率权重：w_c = N / (C * n_c)
        
        直观理解：少数类样本获得更高的权重。
        """
        counts = self.compute_class_counts(labels)
        n = len(labels)
        weights = torch.zeros(self.num_classes)
        
        for c in range(self.num_classes):
            nc = counts.get(c, 1)
            weights[c] = n / (self.num_classes * nc)
        
        return weights
    
    def effective_number_weights(self, labels: List[int]) -> torch.Tensor:
        """
        有效样本数权重：w_c = (1 - β) / (1 - β^n_c)
        
        β → 0 时退化为等权重，β → 1 时退化为逆频率权重。
        """
        counts = self.compute_class_counts(labels)
        weights = torch.zeros(self.num_classes)
        
        for c in range(self.num_classes):
            nc = counts.get(c, 1)
            effective_num = 1.0 - self.beta ** nc
            weights[c] = (1.0 - self.beta) / max(effective_num, 1e-8)
        
        # 归一化
        weights = weights / weights.sum() * self.num_classes
        
        return weights
    
    def oversample_indices(self, labels: List[int]) -> List[int]:
        """
        过采样：让每个类别的样本数接近最大类别。
        
        少数类样本通过重复来增加数量。
        结合数据增强可以减少过拟合风险。
        """
        counts = self.compute_class_counts(labels)
        max_count = max(counts.values())
        
        # 按类别组织索引
        class_indices = defaultdict(list)
        for idx, label in enumerate(labels):
            class_indices[label].append(idx)
        
        oversampled = []
        for c in range(self.num_classes):
            indices = class_indices.get(c, [])
            if not indices:
                continue
            
            # 重复采样直到达到 max_count
            n_repeat = max_count // len(indices)
            n_remain = max_count % len(indices)
            
            oversampled.extend(indices * n_repeat)
            oversampled.extend(random.sample(indices, min(n_remain, len(indices))))
        
        random.shuffle(oversampled)
        return oversampled
    
    def importance_sampling_weights(
        self, labels: List[int], gamma: float = 0.5
    ) -> torch.Tensor:
        """
        重要性采样权重：P(x_i) ∝ 1 / n_{c(x_i)}^γ
        
        γ=0: 均匀采样（不处理不平衡）
        γ=1: 完全平衡
        γ=0.5: 折中方案
        """
        counts = self.compute_class_counts(labels)
        weights = torch.zeros(len(labels))
        
        for idx, label in enumerate(labels):
            nc = counts.get(label, 1)
            weights[idx] = 1.0 / (nc ** gamma)
        
        # 归一化为概率分布
        weights = weights / weights.sum()
        
        return weights


if __name__ == "__main__":
    # ---- 测试 MinHash ----
    print("=" * 50)
    print("MinHash 去重测试")
    print("=" * 50)
    
    dedup = MinHashDedup(num_perm=128, threshold=0.5, num_bands=16)
    texts = [
        "这是一只可爱的小猫在草地上玩耍",
        "这是一只可爱的小猫在草地上奔跑",  # 类似
        "天空中飘着几朵白云非常美丽",
        "天空中飘着几朵白云十分美丽",        # 类似
        "一个完全不同的文本用于测试去重效果",
    ]
    kept, compression = dedup.deduplicate(texts)
    print(f"原始: {len(texts)} 条, 保留: {len(kept)} 条, 压缩率: {compression:.2%}")
    
    # ---- 测试 SimHash ----
    print("\n" + "=" * 50)
    print("SimHash 去重测试")
    print("=" * 50)
    
    vectors = np.random.randn(10, 64).astype(np.float32)
    vectors[3] = vectors[0] + np.random.randn(64) * 0.01  # 近似重复
    vectors[7] = vectors[2] + np.random.randn(64) * 0.01  # 近似重复
    
    sim_dedup = SimHashDedup(input_dim=64, hash_bits=128, threshold=0.95)
    kept_sim, comp_sim = sim_dedup.deduplicate(vectors)
    print(f"原始: 10 条, 保留: {len(kept_sim)} 条, 压缩率: {comp_sim:.2%}")
    
    # ---- 测试质量评分 ----
    print("\n" + "=" * 50)
    print("质量评分测试")
    print("=" * 50)
    
    scorer = QualityScorer()
    q = scorer.compute_quality(height=224, width=224, laplacian_var=150, clip_similarity=0.7)
    print(f"质量评分: {q}")
    
    # ---- 测试数据平衡 ----
    print("\n" + "=" * 50)
    print("数据平衡测试")
    print("=" * 50)
    
    labels = [0]*100 + [1]*20 + [2]*5  # 严重不平衡
    balancer = DataBalancer(num_classes=3)
    
    inv_w = balancer.inverse_frequency_weights(labels)
    print(f"逆频率权重: {inv_w}")
    
    eff_w = balancer.effective_number_weights(labels)
    print(f"有效样本数权重: {eff_w}")
    
    oversampled = balancer.oversample_indices(labels)
    print(f"过采样后样本数: {len(oversampled)} (原始: {len(labels)})")
