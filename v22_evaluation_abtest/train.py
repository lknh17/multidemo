"""
V22 - 评估体系与 A/B 测试训练脚本
===================================
python train.py --mode offline_eval
python train.py --mode ab_test
python train.py --mode bandit
"""
import os, sys, argparse
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, AverageMeter
from config import FullConfig
from metrics import RetrievalMetrics, ClassificationMetrics, FairnessMetrics
from model import ABTestFramework, BanditSelector
from dataset import (create_retrieval_dataloaders, create_abtest_dataloaders,
                     create_bandit_dataset, create_fairness_dataset)


def train_offline_eval(config, logger):
    """离线评估：计算检索指标 + 公平性分析"""
    logger.info("=" * 60)
    logger.info("Offline Evaluation: Retrieval Metrics")
    logger.info("=" * 60)

    train_loader, val_loader = create_retrieval_dataloaders(config)
    retrieval_metrics = RetrievalMetrics(config.offline.k_values)

    # 对每个 batch 计算检索指标
    all_metrics_a = {f'{m}@{k}': [] for m in ['recall', 'ndcg'] for k in config.offline.k_values}
    all_metrics_a.update({'mrr': [], 'map': []})
    all_metrics_b = {f'{m}@{k}': [] for m in ['recall', 'ndcg'] for k in config.offline.k_values}
    all_metrics_b.update({'mrr': [], 'map': []})

    for batch in tqdm(val_loader, desc="Evaluating"):
        relevance = batch['relevance']
        scores_a = batch['scores_a']
        scores_b = batch['scores_b']

        for i in range(relevance.shape[0]):
            rel = relevance[i]
            # 模型 A
            result_a = retrieval_metrics.compute_all(scores_a[i], rel)
            for key, val in result_a.items():
                if key in all_metrics_a:
                    all_metrics_a[key].append(val)

            # 模型 B
            result_b = retrieval_metrics.compute_all(scores_b[i], rel)
            for key, val in result_b.items():
                if key in all_metrics_b:
                    all_metrics_b[key].append(val)

    logger.info("\n模型对比（Model A vs Model B）:")
    logger.info(f"{'Metric':<15} {'Model A':>10} {'Model B':>10} {'Delta':>10}")
    logger.info("-" * 50)
    for key in all_metrics_a:
        mean_a = np.mean(all_metrics_a[key])
        mean_b = np.mean(all_metrics_b[key])
        delta = mean_a - mean_b
        logger.info(f"{key:<15} {mean_a:>10.4f} {mean_b:>10.4f} {delta:>+10.4f}")

    # 公平性评估
    logger.info("\n" + "=" * 60)
    logger.info("Fairness Evaluation")
    logger.info("=" * 60)
    fairness_ds = create_fairness_dataset(config)
    fairness_metrics = FairnessMetrics()

    preds = torch.tensor(fairness_ds.predictions, dtype=torch.long)
    labels = torch.tensor(fairness_ds.labels, dtype=torch.long)
    sensitive = torch.tensor(fairness_ds.sensitive, dtype=torch.long)

    dp = fairness_metrics.demographic_parity(preds, sensitive)
    eo = fairness_metrics.equalized_odds(preds, labels, sensitive)
    eop = fairness_metrics.equal_opportunity(preds, labels, sensitive)
    logger.info(f"  Demographic Parity Gap:  {dp:.4f}")
    logger.info(f"  Equalized Odds Gap:      {eo:.4f}")
    logger.info(f"  Equal Opportunity Gap:    {eop:.4f}")


def train_ab_test(config, logger):
    """A/B 测试模拟"""
    logger.info("=" * 60)
    logger.info("A/B Test Simulation")
    logger.info("=" * 60)

    ab_framework = ABTestFramework(config.ab_test)

    # 样本量估计
    for mde in [0.005, 0.01, 0.02, 0.05]:
        n = ab_framework.sample_size_estimation(baseline_rate=0.05, mde=mde)
        logger.info(f"  MDE={mde:.3f} → min samples: {n:,}")

    # 模拟实验日志
    train_loader, _ = create_abtest_dataloaders(config)
    logger.info(f"\n模拟 {config.num_train_samples} 用户实验...")

    for batch in tqdm(train_loader, desc="Collecting"):
        for i in range(len(batch['user_id'])):
            user_id = batch['user_id'][i].item()
            group = ab_framework.assign_group(user_id)
            source = batch['treatment'] if group == 'treatment' else batch['control']
            for metric_name in source:
                ab_framework.record_metric(group, metric_name, source[metric_name][i].item())

    # 运行分析
    logger.info("\n实验结果：")
    for metric_name in ['ctr', 'conversion', 'revenue', 'session_duration', 'bounce_rate']:
        result = ab_framework.run_analysis(metric_name)
        if 'error' in result:
            logger.info(f"  {metric_name}: {result['error']}")
            continue

        t = result['t_test']
        b = result['bootstrap']
        sig = "✓ 显著" if t['significant'] else "✗ 不显著"
        logger.info(f"\n  [{metric_name}] {sig}")
        logger.info(f"    Control:   mean={t['mean_control']:.4f}, n={result['n_control']}")
        logger.info(f"    Treatment: mean={t['mean_treatment']:.4f}, n={result['n_treatment']}")
        logger.info(f"    Lift: {t['lift']:.2%}")
        logger.info(f"    t-test: t={t['t_stat']:.3f}, p={t['p_value']:.4f}")
        logger.info(f"    Bootstrap CI: [{b['ci_lower']:.4f}, {b['ci_upper']:.4f}]")


def train_bandit(config, logger):
    """多臂老虎机模拟"""
    logger.info("=" * 60)
    logger.info("Multi-Armed Bandit Simulation")
    logger.info("=" * 60)

    bandit_ds = create_bandit_dataset(config)
    true_rates = bandit_ds.get_true_rates()
    logger.info(f"  真实奖励概率: {true_rates}")
    logger.info(f"  最优臂: {np.argmax(true_rates)} (rate={true_rates.max():.3f})")

    strategies = ['epsilon_greedy', 'ucb1', 'thompson']

    for strategy in strategies:
        bandit = BanditSelector(config.bandit)
        total_reward = 0.0

        for t in range(config.bandit.num_rounds):
            if strategy == 'epsilon_greedy':
                arm = bandit.select_epsilon_greedy(t)
            elif strategy == 'ucb1':
                arm = bandit.select_ucb1(t)
            else:
                arm = bandit.select_thompson()

            # 生成奖励
            reward = 1.0 if np.random.random() < true_rates[arm] else 0.0
            bandit.update(arm, reward)
            total_reward += reward

        stats = bandit.get_stats()
        regret = bandit.cumulative_regret(true_rates)

        logger.info(f"\n  [{strategy}]")
        logger.info(f"    Total reward: {total_reward:.0f}/{config.bandit.num_rounds}")
        logger.info(f"    Final regret: {regret[-1]:.1f}")
        logger.info(f"    Arm selections: {stats['counts'].astype(int).tolist()}")
        logger.info(f"    Mean rewards:   {[f'{r:.3f}' for r in stats['mean_rewards']]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="offline_eval",
                        choices=["offline_eval", "ab_test", "bandit"])
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    config = FullConfig()
    if args.epochs:
        config.num_epochs = args.epochs
    set_seed(config.seed)
    logger = get_logger("V22-Eval")

    {"offline_eval": train_offline_eval,
     "ab_test": train_ab_test,
     "bandit": train_bandit}[args.mode](config, logger)


if __name__ == "__main__":
    main()
