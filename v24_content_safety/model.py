"""
V24 - 安全守卫模型 & 合规检查
==============================
1. SafetyGuardModel：级联安全检查
2. ComplianceChecker：规则 + 模型混合合规
3. RobustClassifier：对抗训练包装器
4. CalibratedClassifier：Platt Scaling 校准
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from config import SafetyFullConfig, SafetyClassifierConfig, AdversarialConfig
from safety_modules import (ContentClassifier, ToxicityScorer,
                             AdversarialAttacker, InputSanitizer, SimpleViTBackbone)


class SafetyGuardModel(nn.Module):
    """级联安全检查模型：规则过滤 → 快速分类 → 深度检查"""

    def __init__(self, config: SafetyFullConfig):
        super().__init__()
        self.config = config
        # 快速分类器
        self.fast_classifier = ContentClassifier(config.safety_cls)
        # 输入清洗
        self.sanitizer = InputSanitizer()
        # 级联阈值
        self.reject_threshold = 0.9    # 高于此直接拒绝
        self.uncertain_threshold = 0.3  # 低于此直接通过

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 1. 输入清洗
        sanitized = self.sanitizer(images)

        # 2. 快速安全分类
        fast_out = self.fast_classifier(sanitized)
        probs = fast_out['probs']  # [B, K]
        max_prob = probs.max(dim=-1).values  # [B]

        # 3. 级联决策
        reject_mask = max_prob > self.reject_threshold
        accept_mask = max_prob < self.uncertain_threshold
        uncertain_mask = ~reject_mask & ~accept_mask

        # 状态码：0=通过, 1=拒绝, 2=需复审
        decision = torch.zeros(images.shape[0], dtype=torch.long, device=images.device)
        decision[reject_mask] = 1
        decision[uncertain_mask] = 2

        return {
            'probs': probs,
            'max_prob': max_prob,
            'decision': decision,
            'logits': fast_out['logits'],
            'cls_feat': fast_out['cls_feat'],
            'reject_count': reject_mask.sum(),
            'accept_count': accept_mask.sum(),
            'uncertain_count': uncertain_mask.sum(),
        }


class ComplianceChecker(nn.Module):
    """规则 + 模型混合合规检查"""

    # 关键词黑名单（示例）
    KEYWORD_RULES = {
        'nsfw': ['porn', 'xxx', 'nude'],
        'violence': ['kill', 'murder', 'attack'],
        'spam': ['buy now', 'click here', 'free money'],
        'hate': ['racial slur placeholder'],
    }

    def __init__(self, config: SafetyClassifierConfig):
        super().__init__()
        self.model_checker = ContentClassifier(config)
        # 规则权重
        self.rule_weight = 0.3
        self.model_weight = 0.7

    def rule_check(self, texts: List[str]) -> torch.Tensor:
        """基于关键词规则的快速检查"""
        B = len(texts)
        K = len(self.KEYWORD_RULES)
        scores = torch.zeros(B, K)
        for b, text in enumerate(texts):
            text_lower = text.lower()
            for k, (category, keywords) in enumerate(self.KEYWORD_RULES.items()):
                for kw in keywords:
                    if kw in text_lower:
                        scores[b, k] = 1.0
                        break
        return scores

    def forward(self, images: torch.Tensor,
                texts: Optional[List[str]] = None) -> Dict[str, torch.Tensor]:
        # 模型检查
        model_out = self.model_checker(images)
        model_scores = model_out['probs']  # [B, K]

        # 规则检查（仅文本部分）
        if texts is not None:
            rule_scores = self.rule_check(texts).to(images.device)
            # 融合分数：规则触发直接标记，否则用模型分数
            combined = torch.where(
                rule_scores > 0.5,
                torch.ones_like(model_scores),
                model_scores
            )
        else:
            combined = model_scores

        return {
            'model_scores': model_scores,
            'combined_scores': combined,
            'predictions': (combined > 0.5).float(),
            'logits': model_out['logits'],
        }


class RobustClassifier(nn.Module):
    """对抗训练包装器"""

    def __init__(self, config: SafetyFullConfig):
        super().__init__()
        self.classifier = ContentClassifier(config.safety_cls)
        self.attacker = AdversarialAttacker(config.adversarial)
        self.adv_ratio = config.adversarial.adversarial_ratio

    def forward(self, images: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        # 正常前向
        clean_out = self.classifier(images)

        if self.training and labels is not None:
            # 生成对抗样本
            B = images.shape[0]
            n_adv = int(B * self.adv_ratio)
            if n_adv > 0:
                adv_images = self.attacker.pgd(
                    self.classifier, images[:n_adv], labels[:n_adv],
                    loss_fn=nn.BCEWithLogitsLoss()
                )
                adv_out = self.classifier(adv_images)
                return {
                    'clean_logits': clean_out['logits'],
                    'clean_probs': clean_out['probs'],
                    'adv_logits': adv_out['logits'],
                    'adv_probs': adv_out['probs'],
                }

        return clean_out

    def adversarial_loss(self, clean_logits: torch.Tensor,
                         adv_logits: torch.Tensor,
                         labels: torch.Tensor) -> torch.Tensor:
        """对抗训练混合损失"""
        clean_loss = F.binary_cross_entropy_with_logits(clean_logits, labels)
        adv_loss = F.binary_cross_entropy_with_logits(adv_logits, labels)
        return 0.5 * clean_loss + 0.5 * adv_loss


class CalibratedClassifier(nn.Module):
    """Platt Scaling 校准分类器"""

    def __init__(self, base_classifier: nn.Module, num_classes: int = 8):
        super().__init__()
        self.base = base_classifier
        # Platt Scaling 参数：a * z + b
        self.platt_a = nn.Parameter(torch.ones(num_classes))
        self.platt_b = nn.Parameter(torch.zeros(num_classes))
        # 温度缩放
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            base_out = self.base(images)
        raw_logits = base_out['logits']

        # Platt Scaling
        calibrated_logits = self.platt_a * raw_logits + self.platt_b
        calibrated_probs = torch.sigmoid(calibrated_logits)

        # 温度缩放
        temp_logits = raw_logits / self.temperature
        temp_probs = torch.sigmoid(temp_logits)

        return {
            'raw_logits': raw_logits,
            'raw_probs': base_out['probs'],
            'platt_logits': calibrated_logits,
            'platt_probs': calibrated_probs,
            'temp_logits': temp_logits,
            'temp_probs': temp_probs,
        }

    def compute_ece(self, probs: torch.Tensor, labels: torch.Tensor,
                    n_bins: int = 10) -> torch.Tensor:
        """计算 Expected Calibration Error"""
        bin_boundaries = torch.linspace(0, 1, n_bins + 1, device=probs.device)
        ece = torch.zeros(1, device=probs.device)

        for i in range(n_bins):
            mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
            if mask.sum() > 0:
                avg_conf = probs[mask].mean()
                avg_acc = labels[mask].float().mean()
                ece += (mask.sum().float() / probs.numel()) * (avg_conf - avg_acc).abs()

        return ece
