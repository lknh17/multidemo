"""
V25 - 端到端广告管线推理与实验
================================
1. 全管线延迟分解
2. 检索 Recall@K 各阶段分析
3. 多目标 Pareto 分析
4. 在线学习收敛
"""
import os, sys, time, torch
import torch.nn.functional as F
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from shared.utils import set_seed, get_logger, get_device
from config import FullConfig
from pipeline_modules import AdEncoder, AdMatcher, SafetyFilter, QualityGate
from model import E2EAdPipeline, CTRPredictor, MultiObjectiveRanker, OnlineLearner


def demo_latency_breakdown():
    """实验 1：全管线各阶段延迟分解"""
    logger = get_logger("Latency")
    device = get_device()
    config = FullConfig()
    D = config.creative.d_model

    encoder = AdEncoder(config.creative).to(device).eval()
    matcher = AdMatcher(config.matching, D).to(device).eval()
    safety = SafetyFilter(D).to(device).eval()
    quality = QualityGate(D).to(device).eval()

    # 模拟数据
    N_ads = 500
    query_img = torch.randn(1, 3, 224, 224).to(device)
    query_txt = torch.randint(1, 5000, (1, 64)).to(device)
    query_aud = torch.randn(1, 128).to(device)
    ad_imgs = torch.randn(N_ads, 3, 224, 224).to(device)
    ad_txts = torch.randint(1, 5000, (N_ads, 64)).to(device)
    ad_auds = torch.randn(N_ads, 128).to(device)

    timings = {}
    with torch.no_grad():
        # Encode query
        t0 = time.time()
        q_emb = encoder(query_img, query_txt, query_aud)
        timings['encode_query'] = time.time() - t0

        # Encode ads (batch)
        t0 = time.time()
        ad_embs = []
        bs = 32
        for i in range(0, N_ads, bs):
            end = min(i + bs, N_ads)
            emb = encoder(ad_imgs[i:end], ad_txts[i:end], ad_auds[i:end])
            ad_embs.append(emb)
        ad_emb_all = torch.cat(ad_embs, dim=0)
        timings['encode_ads'] = time.time() - t0

        # Retrieve
        t0 = time.time()
        top_idx, recall_scores = matcher.retrieve(q_emb, ad_emb_all, top_k=100)
        timings['retrieve'] = time.time() - t0

        # Rerank
        t0 = time.time()
        candidates = ad_emb_all[top_idx[0]].unsqueeze(0)
        rerank_scores = matcher.rerank(q_emb, candidates)
        timings['rerank'] = time.time() - t0

        # Safety
        t0 = time.time()
        safe_mask, _ = safety(candidates.squeeze(0))
        timings['safety_filter'] = time.time() - t0

        # Quality
        t0 = time.time()
        qual_mask, _ = quality(candidates.squeeze(0))
        timings['quality_gate'] = time.time() - t0

    total = sum(timings.values())
    logger.info("=" * 50)
    logger.info("全管线延迟分解")
    logger.info("=" * 50)
    for stage, t in timings.items():
        pct = t / total * 100
        bar = '█' * int(pct / 2)
        logger.info(f"  {stage:20s}: {t*1000:8.2f}ms ({pct:5.1f}%) {bar}")
    logger.info(f"  {'总计':20s}: {total*1000:8.2f}ms")
    logger.info(f"  广告库大小={N_ads}, 召回={100}, 安全通过={safe_mask.sum().item()}, 质量通过={qual_mask.sum().item()}")


def demo_recall_at_k():
    """实验 2：检索 Recall@K 各阶段"""
    logger = get_logger("Recall@K")
    device = get_device()
    config = FullConfig()
    D = config.creative.d_model

    encoder = AdEncoder(config.creative).to(device).eval()
    matcher = AdMatcher(config.matching, D).to(device).eval()

    # 构造已知 ground-truth 的检索场景
    N = 200
    torch.manual_seed(42)
    ad_embs = F.normalize(torch.randn(N, D).to(device), dim=-1)

    # 模拟 10 个 query，每个有 5 个相关文档
    n_queries = 10
    n_relevant = 5
    query_embs = []
    relevant_sets = []
    for q in range(n_queries):
        rel_idx = list(range(q * n_relevant, (q + 1) * n_relevant))
        relevant_sets.append(set(rel_idx))
        # query = 相关文档的均值 + 噪声
        query = ad_embs[rel_idx].mean(0) + torch.randn(D).to(device) * 0.1
        query_embs.append(F.normalize(query.unsqueeze(0), dim=-1))

    logger.info("=" * 50)
    logger.info("检索 Recall@K 分析")
    logger.info("=" * 50)

    for K in [5, 10, 20, 50, 100]:
        recalls = []
        with torch.no_grad():
            for q in range(n_queries):
                top_idx, _ = matcher.retrieve(query_embs[q], ad_embs, top_k=K)
                retrieved = set(top_idx[0].cpu().tolist())
                recall = len(retrieved & relevant_sets[q]) / n_relevant
                recalls.append(recall)
        avg_recall = sum(recalls) / len(recalls)
        bar = '█' * int(avg_recall * 20)
        logger.info(f"  Recall@{K:3d}: {avg_recall:.3f} {bar}")


def demo_pareto_analysis():
    """实验 3：多目标 Pareto 分析"""
    logger = get_logger("Pareto")
    device = get_device()
    config = FullConfig()
    D = config.creative.d_model

    model = MultiObjectiveRanker(D, config).to(device).eval()

    B = 50
    user_emb = torch.randn(B, D).to(device)
    ad_emb = torch.randn(B, D).to(device)
    ctx_emb = torch.randn(B, D).to(device)

    logger.info("=" * 50)
    logger.info("多目标 Pareto 权重分析")
    logger.info("=" * 50)

    weight_configs = [
        [0.7, 0.1, 0.1, 0.1],  # CTR 导向
        [0.1, 0.7, 0.1, 0.1],  # 相关性导向
        [0.1, 0.1, 0.7, 0.1],  # 多样性导向
        [0.1, 0.1, 0.1, 0.7],  # 新鲜度导向
        [0.25, 0.25, 0.25, 0.25],  # 均衡
    ]
    names = ['CTR导向', '相关性导向', '多样性导向', '新鲜度导向', '均衡策略']

    with torch.no_grad():
        for wc, name in zip(weight_configs, names):
            model.weights.data = torch.tensor(wc, device=device)
            outputs = model(user_emb, ad_emb, ctx_emb)
            logger.info(f"\n  策略: {name}")
            logger.info(f"    权重: ctr={wc[0]:.1f} rel={wc[1]:.1f} div={wc[2]:.1f} fresh={wc[3]:.1f}")
            logger.info(f"    平均 CTR 分:     {outputs['ctr_score'].mean():.4f}")
            logger.info(f"    平均 相关性分:   {outputs['relevance_score'].mean():.4f}")
            logger.info(f"    平均 多样性分:   {outputs['diversity_score'].mean():.4f}")
            logger.info(f"    平均 新鲜度分:   {outputs['freshness_score'].mean():.4f}")
            logger.info(f"    最终得分 std:    {outputs['final_score'].std():.4f}")


def demo_online_learning():
    """实验 4：在线学习收敛"""
    logger = get_logger("OnlineLearning")
    device = get_device()
    config = FullConfig()
    D = config.creative.d_model

    ctr_model = CTRPredictor(D).to(device)
    learner = OnlineLearner(ctr_model, lr=1e-3, ema_beta=0.99)

    logger.info("=" * 50)
    logger.info("在线学习收敛实验")
    logger.info("=" * 50)

    n_steps = 200
    window_size = 20
    recent_losses = []
    milestones = [1, 10, 25, 50, 100, 150, 200]

    for step in range(1, n_steps + 1):
        # 模拟在线数据流
        user = torch.randn(1, D).to(device)
        ad = torch.randn(1, D).to(device)
        ctx = torch.randn(1, D).to(device)
        click = torch.tensor([1.0 if torch.rand(1).item() > 0.8 else 0.0]).to(device)

        loss = learner.update(user, ad, ctx, click)
        recent_losses.append(loss)
        if len(recent_losses) > window_size:
            recent_losses.pop(0)

        if step in milestones:
            avg_recent = sum(recent_losses) / len(recent_losses)
            cum_avg = learner.get_avg_loss()
            bar = '█' * int((1 - min(avg_recent, 1)) * 20)
            logger.info(f"  Step {step:4d}: recent_loss={avg_recent:.4f}  cumulative={cum_avg:.4f}  {bar}")

    logger.info(f"\n  最终累积平均损失: {learner.get_avg_loss():.4f}")
    logger.info(f"  总更新步数: {learner.step_count}")


def main():
    set_seed(42)
    print("=" * 60)
    print("V25 - 端到端广告多模态管线 推理实验 (Capstone)")
    print("=" * 60)
    demo_latency_breakdown(); print()
    demo_recall_at_k(); print()
    demo_pareto_analysis(); print()
    demo_online_learning()


if __name__ == "__main__":
    main()
