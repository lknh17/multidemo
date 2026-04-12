"""
V22 - 检索 / 分类 / 公平性评估指标
====================================
"""
import torch
import numpy as np
from typing import List, Dict, Optional


class RetrievalMetrics:
    """检索评估指标集合"""

    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [1, 3, 5, 10, 20]

    def recall_at_k(self, scores: torch.Tensor, relevance: torch.Tensor, k: int) -> float:
        """Recall@K: Top-K 中召回相关文档的比例"""
        topk_idx = scores.argsort(descending=True)[:k]
        relevant_in_topk = relevance[topk_idx].sum().item()
        total_relevant = relevance.sum().item()
        return relevant_in_topk / max(total_relevant, 1e-8)

    def precision_at_k(self, scores: torch.Tensor, relevance: torch.Tensor, k: int) -> float:
        """Precision@K: Top-K 中相关文档的比例"""
        topk_idx = scores.argsort(descending=True)[:k]
        relevant_in_topk = relevance[topk_idx].sum().item()
        return relevant_in_topk / k

    def ndcg_at_k(self, scores: torch.Tensor, relevance: torch.Tensor, k: int) -> float:
        """NDCG@K: 归一化折损累积增益"""
        topk_idx = scores.argsort(descending=True)[:k]
        gains = (2.0 ** relevance[topk_idx].float() - 1.0)
        discounts = 1.0 / torch.log2(torch.arange(1, k + 1, dtype=torch.float32) + 1.0)
        dcg = (gains * discounts[:len(gains)]).sum().item()

        # 理想排序
        ideal_rel = relevance.float().sort(descending=True).values[:k]
        ideal_gains = (2.0 ** ideal_rel - 1.0)
        idcg = (ideal_gains * discounts[:len(ideal_gains)]).sum().item()
        return dcg / max(idcg, 1e-8)

    def mrr(self, scores: torch.Tensor, relevance: torch.Tensor) -> float:
        """MRR: 平均倒数排名"""
        ranked_idx = scores.argsort(descending=True)
        ranked_rel = relevance[ranked_idx]
        first_relevant = (ranked_rel > 0).nonzero(as_tuple=True)[0]
        if len(first_relevant) == 0:
            return 0.0
        rank = first_relevant[0].item() + 1
        return 1.0 / rank

    def average_precision(self, scores: torch.Tensor, relevance: torch.Tensor) -> float:
        """AP: 平均精度（用于计算 mAP）"""
        ranked_idx = scores.argsort(descending=True)
        ranked_rel = relevance[ranked_idx].float()
        cum_relevant = ranked_rel.cumsum(0)
        precision_at_k = cum_relevant / torch.arange(1, len(ranked_rel) + 1, dtype=torch.float32)
        ap = (precision_at_k * ranked_rel).sum().item() / max(relevance.sum().item(), 1e-8)
        return ap

    def compute_all(self, scores: torch.Tensor, relevance: torch.Tensor) -> Dict[str, float]:
        """计算所有指标"""
        results = {}
        for k in self.k_values:
            results[f'recall@{k}'] = self.recall_at_k(scores, relevance, k)
            results[f'precision@{k}'] = self.precision_at_k(scores, relevance, k)
            results[f'ndcg@{k}'] = self.ndcg_at_k(scores, relevance, k)
        results['mrr'] = self.mrr(scores, relevance)
        results['map'] = self.average_precision(scores, relevance)
        return results


class ClassificationMetrics:
    """分类评估指标"""

    @staticmethod
    def accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
        """准确率"""
        return (predictions == labels).float().mean().item()

    @staticmethod
    def f1_score(predictions: torch.Tensor, labels: torch.Tensor,
                 num_classes: int, average: str = 'macro') -> float:
        """F1 分数（macro/micro）"""
        if average == 'micro':
            tp = (predictions == labels).sum().item()
            total = len(labels)
            # micro: precision = recall = accuracy
            return tp / max(total, 1)

        # macro: 每个类别单独计算再平均
        f1_scores = []
        for c in range(num_classes):
            pred_c = (predictions == c)
            true_c = (labels == c)
            tp = (pred_c & true_c).sum().item()
            fp = (pred_c & ~true_c).sum().item()
            fn = (~pred_c & true_c).sum().item()
            precision = tp / max(tp + fp, 1e-8)
            recall = tp / max(tp + fn, 1e-8)
            f1 = 2 * precision * recall / max(precision + recall, 1e-8)
            f1_scores.append(f1)
        return np.mean(f1_scores)

    @staticmethod
    def auc_roc(scores: torch.Tensor, labels: torch.Tensor) -> float:
        """AUC-ROC（二分类，梯形法近似）"""
        scores_np = scores.detach().cpu().numpy()
        labels_np = labels.detach().cpu().numpy()

        # 按分数降序排列
        sorted_idx = np.argsort(-scores_np)
        sorted_labels = labels_np[sorted_idx]

        tp, fp = 0, 0
        total_pos = labels_np.sum()
        total_neg = len(labels_np) - total_pos

        tpr_prev, fpr_prev = 0.0, 0.0
        auc = 0.0

        for label in sorted_labels:
            if label == 1:
                tp += 1
            else:
                fp += 1
            tpr = tp / max(total_pos, 1)
            fpr = fp / max(total_neg, 1)
            # 梯形积分
            auc += (fpr - fpr_prev) * (tpr + tpr_prev) / 2
            tpr_prev, fpr_prev = tpr, fpr

        return auc


class FairnessMetrics:
    """公平性评估指标"""

    @staticmethod
    def demographic_parity(predictions: torch.Tensor,
                           sensitive_attr: torch.Tensor) -> float:
        """
        Demographic Parity 差距
        |P(Y_hat=1|A=0) - P(Y_hat=1|A=1)|
        """
        group_0_mask = (sensitive_attr == 0)
        group_1_mask = (sensitive_attr == 1)

        if group_0_mask.sum() == 0 or group_1_mask.sum() == 0:
            return 0.0

        rate_0 = predictions[group_0_mask].float().mean().item()
        rate_1 = predictions[group_1_mask].float().mean().item()
        return abs(rate_0 - rate_1)

    @staticmethod
    def equalized_odds(predictions: torch.Tensor, labels: torch.Tensor,
                       sensitive_attr: torch.Tensor) -> float:
        """
        Equalized Odds 差距（取 Y=0 和 Y=1 两种情况的最大差距）
        """
        max_gap = 0.0
        for y_val in [0, 1]:
            mask = (labels == y_val)
            g0 = mask & (sensitive_attr == 0)
            g1 = mask & (sensitive_attr == 1)

            if g0.sum() == 0 or g1.sum() == 0:
                continue

            rate_0 = predictions[g0].float().mean().item()
            rate_1 = predictions[g1].float().mean().item()
            max_gap = max(max_gap, abs(rate_0 - rate_1))

        return max_gap

    @staticmethod
    def equal_opportunity(predictions: torch.Tensor, labels: torch.Tensor,
                          sensitive_attr: torch.Tensor) -> float:
        """
        Equal Opportunity 差距（仅看正类 Y=1）
        |P(Y_hat=1|Y=1,A=0) - P(Y_hat=1|Y=1,A=1)|
        """
        pos_mask = (labels == 1)
        g0 = pos_mask & (sensitive_attr == 0)
        g1 = pos_mask & (sensitive_attr == 1)

        if g0.sum() == 0 or g1.sum() == 0:
            return 0.0

        rate_0 = predictions[g0].float().mean().item()
        rate_1 = predictions[g1].float().mean().item()
        return abs(rate_0 - rate_1)
