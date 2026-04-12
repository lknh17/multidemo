"""
V22 - A/B 测试框架 / Bandit 选择器 / 交错实验
================================================
"""
import hashlib
import math
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats as scipy_stats

from config import FullConfig, ABTestConfig, BanditConfig, InterleavingConfig


class ABTestFramework:
    """A/B 测试框架：流量分割、指标收集、显著性检验"""

    def __init__(self, config: ABTestConfig):
        self.config = config
        self.traffic_split = config.traffic_split
        self.confidence_level = config.confidence_level
        self.salt = "v22_ab_salt_2024"
        self.control_data: Dict[str, List[float]] = {}
        self.treatment_data: Dict[str, List[float]] = {}

    def assign_group(self, user_id: int) -> str:
        """确定性哈希分组，保证同一用户始终在同一组"""
        hash_input = f"{user_id}_{self.salt}"
        hash_val = int(hashlib.md5(hash_input.encode()).hexdigest(), 16) % 10000
        return 'treatment' if hash_val < self.traffic_split * 10000 else 'control'

    def record_metric(self, group: str, metric_name: str, value: float):
        """记录实验指标"""
        target = self.treatment_data if group == 'treatment' else self.control_data
        if metric_name not in target:
            target[metric_name] = []
        target[metric_name].append(value)

    def welch_t_test(self, control: np.ndarray, treatment: np.ndarray) -> Dict[str, float]:
        """Welch's t-test（不等方差 t 检验）"""
        n_c, n_t = len(control), len(treatment)
        mean_c, mean_t = control.mean(), treatment.mean()
        var_c, var_t = control.var(ddof=1), treatment.var(ddof=1)

        se = math.sqrt(var_c / n_c + var_t / n_t)
        t_stat = (mean_t - mean_c) / max(se, 1e-10)

        # Welch-Satterthwaite 自由度
        numerator = (var_c / n_c + var_t / n_t) ** 2
        denominator = (var_c / n_c) ** 2 / (n_c - 1) + (var_t / n_t) ** 2 / (n_t - 1)
        df = numerator / max(denominator, 1e-10)

        p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), df=df))

        return {
            't_stat': t_stat,
            'p_value': p_value,
            'df': df,
            'mean_control': mean_c,
            'mean_treatment': mean_t,
            'lift': (mean_t - mean_c) / max(abs(mean_c), 1e-10),
            'significant': p_value < (1 - self.confidence_level),
        }

    def bootstrap_ci(self, control: np.ndarray, treatment: np.ndarray,
                     n_iterations: int = None) -> Dict[str, float]:
        """Bootstrap 置信区间"""
        n_iterations = n_iterations or self.config.bootstrap_iterations
        deltas = np.zeros(n_iterations)

        for i in range(n_iterations):
            c_sample = np.random.choice(control, size=len(control), replace=True)
            t_sample = np.random.choice(treatment, size=len(treatment), replace=True)
            deltas[i] = t_sample.mean() - c_sample.mean()

        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(deltas, 100 * alpha / 2)
        ci_upper = np.percentile(deltas, 100 * (1 - alpha / 2))

        return {
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'mean_delta': deltas.mean(),
            'std_delta': deltas.std(),
            'significant': ci_lower > 0 or ci_upper < 0,
        }

    def sample_size_estimation(self, baseline_rate: float, mde: float,
                               alpha: float = 0.05, power: float = 0.8) -> int:
        """
        最小样本量估计
        mde: minimum detectable effect (绝对差值)
        """
        z_alpha = scipy_stats.norm.ppf(1 - alpha / 2)
        z_beta = scipy_stats.norm.ppf(power)
        sigma2 = baseline_rate * (1 - baseline_rate)
        n = (z_alpha + z_beta) ** 2 * 2 * sigma2 / max(mde ** 2, 1e-10)
        return int(math.ceil(n))

    def run_analysis(self, metric_name: str) -> Dict:
        """对指定指标运行完整分析"""
        control = np.array(self.control_data.get(metric_name, []))
        treatment = np.array(self.treatment_data.get(metric_name, []))

        if len(control) < 2 or len(treatment) < 2:
            return {'error': 'insufficient data'}

        t_test = self.welch_t_test(control, treatment)
        bootstrap = self.bootstrap_ci(control, treatment, n_iterations=5000)

        return {
            'metric': metric_name,
            'n_control': len(control),
            'n_treatment': len(treatment),
            't_test': t_test,
            'bootstrap': bootstrap,
        }


class BanditSelector:
    """多臂老虎机选择器：Epsilon-Greedy / UCB1 / Thompson Sampling"""

    def __init__(self, config: BanditConfig):
        self.config = config
        self.num_arms = config.num_arms
        self.epsilon = config.epsilon
        self.ucb_c = config.ucb_c
        self.reset()

    def reset(self):
        """重置所有臂的统计量"""
        self.counts = np.zeros(self.num_arms)
        self.total_rewards = np.zeros(self.num_arms)
        self.alpha = np.ones(self.num_arms) * self.config.thompson_alpha
        self.beta = np.ones(self.num_arms) * self.config.thompson_beta
        self.history: List[Tuple[int, float]] = []

    def select_epsilon_greedy(self, t: int) -> int:
        """Epsilon-Greedy 策略"""
        eps = self.epsilon / math.sqrt(t + 1) if self.config.decay_epsilon else self.epsilon

        if np.random.random() < eps:
            return np.random.randint(self.num_arms)
        else:
            mean_rewards = np.where(
                self.counts > 0,
                self.total_rewards / self.counts,
                0.0
            )
            return int(np.argmax(mean_rewards))

    def select_ucb1(self, t: int) -> int:
        """UCB1 策略"""
        for k in range(self.num_arms):
            if self.counts[k] == 0:
                return k

        mean_rewards = self.total_rewards / self.counts
        bonus = self.ucb_c * np.sqrt(np.log(t + 1) / self.counts)
        ucb_values = mean_rewards + bonus
        return int(np.argmax(ucb_values))

    def select_thompson(self) -> int:
        """Thompson Sampling（Beta-Bernoulli）"""
        samples = np.array([
            np.random.beta(self.alpha[k], self.beta[k])
            for k in range(self.num_arms)
        ])
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float):
        """更新臂的统计量"""
        self.counts[arm] += 1
        self.total_rewards[arm] += reward
        # Thompson Sampling Beta 后验更新
        self.alpha[arm] += reward
        self.beta[arm] += (1 - reward)
        self.history.append((arm, reward))

    def get_stats(self) -> Dict[str, np.ndarray]:
        """获取当前统计"""
        mean_rewards = np.where(
            self.counts > 0,
            self.total_rewards / self.counts,
            0.0
        )
        return {
            'counts': self.counts.copy(),
            'mean_rewards': mean_rewards,
            'total_rewards': self.total_rewards.copy(),
            'alpha': self.alpha.copy(),
            'beta': self.beta.copy(),
        }

    def cumulative_regret(self, true_rates: np.ndarray) -> np.ndarray:
        """计算累积遗憾"""
        best_rate = true_rates.max()
        regret = np.zeros(len(self.history))
        cum = 0.0
        for i, (arm, _) in enumerate(self.history):
            cum += best_rate - true_rates[arm]
            regret[i] = cum
        return regret


class InterleavingExperiment:
    """交错实验：Team Draft Interleaving"""

    def __init__(self, config: InterleavingConfig):
        self.config = config
        self.list_length = config.list_length
        self.wins_a = 0
        self.wins_b = 0
        self.ties = 0

    def team_draft(self, list_a: List[int], list_b: List[int],
                   k: int = None) -> Tuple[List[int], set, set]:
        """
        Team Draft 交错算法
        交替从 A、B 列表中选取文档，生成混合排序
        """
        k = k or self.list_length
        merged = []
        team_a, team_b = set(), set()
        seen = set()
        ptr_a, ptr_b = 0, 0

        for i in range(k):
            if len(team_a) <= len(team_b):
                # A 队选取
                while ptr_a < len(list_a) and list_a[ptr_a] in seen:
                    ptr_a += 1
                if ptr_a < len(list_a):
                    doc = list_a[ptr_a]
                    merged.append(doc)
                    team_a.add(doc)
                    seen.add(doc)
                    ptr_a += 1
            else:
                # B 队选取
                while ptr_b < len(list_b) and list_b[ptr_b] in seen:
                    ptr_b += 1
                if ptr_b < len(list_b):
                    doc = list_b[ptr_b]
                    merged.append(doc)
                    team_b.add(doc)
                    seen.add(doc)
                    ptr_b += 1

        return merged, team_a, team_b

    def simulate_clicks(self, merged: List[int], relevance: Dict[int, float],
                        position_bias: bool = True) -> List[int]:
        """模拟用户点击（含位置偏差）"""
        clicks = []
        for pos, doc in enumerate(merged):
            rel = relevance.get(doc, 0.0)
            # 位置偏差：排名越靠后，被点击概率越低
            pos_factor = 1.0 / math.log2(pos + 2) if position_bias else 1.0
            click_prob = rel * pos_factor
            if np.random.random() < click_prob:
                clicks.append(doc)
        return clicks

    def judge(self, clicks: List[int], team_a: set, team_b: set):
        """根据点击判定胜负"""
        clicks_a = len([c for c in clicks if c in team_a])
        clicks_b = len([c for c in clicks if c in team_b])

        if clicks_a > clicks_b:
            self.wins_a += 1
        elif clicks_b > clicks_a:
            self.wins_b += 1
        else:
            self.ties += 1

    def get_result(self) -> Dict[str, float]:
        """获取实验结果"""
        total = self.wins_a + self.wins_b + self.ties
        if total == 0:
            return {'delta': 0.0, 'wins_a': 0, 'wins_b': 0, 'ties': 0}

        delta = (self.wins_a - self.wins_b) / total
        # 符号检验
        n_decided = self.wins_a + self.wins_b
        if n_decided > 0:
            p_value = scipy_stats.binom_test(
                self.wins_a, n_decided, 0.5
            ) if hasattr(scipy_stats, 'binom_test') else scipy_stats.binomtest(
                self.wins_a, n_decided, 0.5
            ).pvalue
        else:
            p_value = 1.0

        return {
            'delta': delta,
            'wins_a': self.wins_a,
            'wins_b': self.wins_b,
            'ties': self.ties,
            'total': total,
            'p_value': p_value,
            'a_better': delta > 0 and p_value < 0.05,
        }
