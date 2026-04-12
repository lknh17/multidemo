"""
V22 - 评估与 A/B 测试推理实验
================================
1. 检索指标对比（Recall/NDCG/MRR/mAP）
2. A/B 测试显著性分析
3. Bandit 收敛对比
4. 交错实验灵敏度分析
"""
import os, sys
import numpy as np
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger
from config import FullConfig
from metrics import RetrievalMetrics, ClassificationMetrics, FairnessMetrics
from model import ABTestFramework, BanditSelector, InterleavingExperiment


def demo_retrieval_metrics():
    """实验1：检索指标对比"""
    logger = get_logger("RetrievalMetrics")
    config = FullConfig()
    metrics = RetrievalMetrics(config.offline.k_values)

    logger.info("检索评估指标对比实验")
    logger.info("=" * 60)

    # 生成不同质量的排序结果
    n_docs = 200
    n_queries = 100

    for model_name, noise_level in [("强模型", 0.2), ("中等模型", 0.5), ("弱模型", 1.0)]:
        all_results = {}
        for q in range(n_queries):
            rng = np.random.RandomState(q)
            relevance = torch.zeros(n_docs)
            num_rel = rng.randint(5, 20)
            rel_positions = rng.choice(n_docs, num_rel, replace=False)
            relevance[rel_positions] = torch.tensor(
                rng.randint(1, 4, num_rel), dtype=torch.float32
            )

            # 模型分数 = 真实相关性 + 噪声
            scores = relevance * 0.8 + torch.randn(n_docs) * noise_level
            result = metrics.compute_all(scores, relevance)

            for k, v in result.items():
                all_results.setdefault(k, []).append(v)

        logger.info(f"\n  [{model_name}] (noise={noise_level})")
        for k in ['recall@5', 'recall@10', 'ndcg@5', 'ndcg@10', 'mrr', 'map']:
            if k in all_results:
                logger.info(f"    {k:<12}: {np.mean(all_results[k]):.4f} ± {np.std(all_results[k]):.4f}")


def demo_ab_test_significance():
    """实验2：A/B 测试显著性分析"""
    logger = get_logger("ABTestSignificance")
    config = FullConfig()
    ab = ABTestFramework(config.ab_test)

    logger.info("A/B 测试显著性分析")
    logger.info("=" * 60)

    # 不同效应大小
    sample_sizes = [100, 500, 1000, 5000]
    effect_sizes = [0.0, 0.01, 0.05, 0.10]

    for effect in effect_sizes:
        logger.info(f"\n  真实效应: {effect:.2%}")
        for n in sample_sizes:
            rng = np.random.RandomState(42)
            control = rng.normal(0.05, 0.02, n)
            treatment = rng.normal(0.05 + effect, 0.02, n)

            t_result = ab.welch_t_test(control, treatment)
            b_result = ab.bootstrap_ci(control, treatment, n_iterations=5000)

            sig_t = "✓" if t_result['significant'] else "✗"
            sig_b = "✓" if b_result['significant'] else "✗"
            logger.info(
                f"    n={n:>5}: t-test p={t_result['p_value']:.4f} {sig_t} | "
                f"Bootstrap CI=[{b_result['ci_lower']:.4f}, {b_result['ci_upper']:.4f}] {sig_b}"
            )

    # 样本量估计
    logger.info("\n  最小样本量估计 (基准CTR=5%):")
    for mde in [0.005, 0.01, 0.02]:
        n = ab.sample_size_estimation(0.05, mde)
        logger.info(f"    MDE={mde:.3f}: n={n:,} per group")


def demo_bandit_convergence():
    """实验3：Bandit 策略收敛对比"""
    logger = get_logger("BanditConvergence")
    config = FullConfig()

    true_rates = np.array([0.1, 0.15, 0.35, 0.25, 0.2])
    logger.info("多臂老虎机收敛对比")
    logger.info("=" * 60)
    logger.info(f"  真实奖励: {true_rates.tolist()}")
    logger.info(f"  最优臂: {np.argmax(true_rates)}")

    n_rounds = config.bandit.num_rounds
    n_runs = 10  # 多次运行取平均

    for strategy in ['epsilon_greedy', 'ucb1', 'thompson']:
        all_regrets = []
        all_optimal_pcts = []

        for run in range(n_runs):
            np.random.seed(run * 100)
            bandit = BanditSelector(config.bandit)
            optimal_count = 0

            for t in range(n_rounds):
                if strategy == 'epsilon_greedy':
                    arm = bandit.select_epsilon_greedy(t)
                elif strategy == 'ucb1':
                    arm = bandit.select_ucb1(t)
                else:
                    arm = bandit.select_thompson()

                reward = 1.0 if np.random.random() < true_rates[arm] else 0.0
                bandit.update(arm, reward)
                if arm == np.argmax(true_rates):
                    optimal_count += 1

            regret = bandit.cumulative_regret(true_rates)
            all_regrets.append(regret[-1])
            all_optimal_pcts.append(optimal_count / n_rounds)

        mean_regret = np.mean(all_regrets)
        std_regret = np.std(all_regrets)
        mean_optimal = np.mean(all_optimal_pcts)

        logger.info(f"\n  [{strategy}]")
        logger.info(f"    累积遗憾: {mean_regret:.1f} ± {std_regret:.1f}")
        logger.info(f"    最优臂选择率: {mean_optimal:.2%}")


def demo_interleaving_sensitivity():
    """实验4：交错实验灵敏度分析"""
    logger = get_logger("Interleaving")
    config = FullConfig()

    logger.info("交错实验灵敏度分析")
    logger.info("=" * 60)

    n_docs = 50
    n_queries_list = [50, 200, 500, 1000]

    # 不同的质量差距
    quality_gaps = [0.0, 0.05, 0.10, 0.20]

    for gap in quality_gaps:
        logger.info(f"\n  质量差距: {gap:.2f}")
        for n_queries in n_queries_list:
            interleaving = InterleavingExperiment(config.interleaving)

            for q in range(n_queries):
                rng = np.random.RandomState(q * 7 + 13)
                # 真实相关性
                relevance = {i: rng.random() * 0.5 for i in range(n_docs)}

                # 列表 A：较好排序
                items = list(range(n_docs))
                scores_a = [relevance[i] + rng.normal(0, 0.1) for i in items]
                list_a = [x for _, x in sorted(zip(scores_a, items), reverse=True)]

                # 列表 B：较差排序（加更多噪声）
                scores_b = [relevance[i] + rng.normal(0, 0.1 + gap) for i in items]
                list_b = [x for _, x in sorted(zip(scores_b, items), reverse=True)]

                merged, team_a, team_b = interleaving.team_draft(list_a, list_b)
                clicks = interleaving.simulate_clicks(merged, relevance)
                interleaving.judge(clicks, team_a, team_b)

            result = interleaving.get_result()
            sig = "✓" if result.get('p_value', 1.0) < 0.05 else "✗"
            logger.info(
                f"    n={n_queries:>5}: Δ={result['delta']:+.3f}, "
                f"A wins={result['wins_a']}, B wins={result['wins_b']}, "
                f"ties={result['ties']}, p={result.get('p_value', 1.0):.3f} {sig}"
            )


def main():
    set_seed(42)
    print("=" * 60)
    print("V22 - 评估体系与 A/B 测试 推理实验")
    print("=" * 60)

    demo_retrieval_metrics(); print()
    demo_ab_test_significance(); print()
    demo_bandit_convergence(); print()
    demo_interleaving_sensitivity()


if __name__ == "__main__":
    main()
