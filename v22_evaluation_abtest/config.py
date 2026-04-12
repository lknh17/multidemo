"""
V22 - 评估体系与 A/B 测试配置
================================
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class OfflineMetricsConfig:
    """离线评估指标配置"""
    k_values: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20])
    num_classes: int = 100           # 分类类别数
    num_queries: int = 500           # 查询数量
    num_candidates: int = 1000       # 候选集大小
    relevance_levels: int = 4        # 相关性等级（0-3）


@dataclass
class ABTestConfig:
    """A/B 测试配置"""
    traffic_split: float = 0.5       # 实验组流量比例
    confidence_level: float = 0.95   # 置信度水平
    min_sample_size: int = 1000      # 最小样本量
    max_duration_days: int = 14      # 最大实验天数
    num_metrics: int = 5             # 监控指标数
    bootstrap_iterations: int = 10000  # Bootstrap 采样次数
    correction_method: str = "bonferroni"  # 多重检验校正


@dataclass
class BanditConfig:
    """多臂老虎机配置"""
    num_arms: int = 5                # 臂数量（模型/策略数）
    epsilon: float = 0.1             # Epsilon-Greedy 探索率
    ucb_c: float = 2.0              # UCB1 探索参数
    thompson_alpha: float = 1.0      # Thompson Sampling Beta 先验 α
    thompson_beta: float = 1.0       # Thompson Sampling Beta 先验 β
    num_rounds: int = 5000           # 模拟轮数
    decay_epsilon: bool = True       # 是否衰减探索率


@dataclass
class InterleavingConfig:
    """交错实验配置"""
    list_length: int = 10            # 混合排序列表长度
    num_queries: int = 2000          # 查询次数
    click_model: str = "position_biased"  # 点击模型


@dataclass
class FullConfig:
    """完整配置"""
    offline: OfflineMetricsConfig = field(default_factory=OfflineMetricsConfig)
    ab_test: ABTestConfig = field(default_factory=ABTestConfig)
    bandit: BanditConfig = field(default_factory=BanditConfig)
    interleaving: InterleavingConfig = field(default_factory=InterleavingConfig)

    batch_size: int = 16
    learning_rate: float = 1e-3
    num_epochs: int = 20
    seed: int = 42
    num_train_samples: int = 2000
    num_val_samples: int = 500
