"""
p05 强化学习 GRPO - 奖励函数

三种奖励函数：
1. correctness_reward — 数学答案正确性
2. format_reward — 推理步骤格式规范性
3. composite_reward — 加权组合

使用方式:
    from reward import composite_reward, correctness_reward, format_reward
"""

import os
import sys
import re
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import reward_config, RewardConfig
from dataset import extract_model_answer


# ============================================================
# 1. 正确性奖励
# ============================================================
def correctness_reward(
    response: str,
    ground_truth: float,
    cfg: RewardConfig = None,
    tolerance: float = 1e-5,
) -> float:
    """
    正确性奖励：检查数学答案是否正确。
    
    评分规则：
    - 答案完全正确 → +1.0
    - 答案错误但格式正确 → -0.5（部分分）
    - 无法提取答案 → -1.0
    
    Args:
        response: 模型生成的响应
        ground_truth: 标准答案（数值）
        cfg: 奖励配置
        tolerance: 数值比较容差
    """
    if cfg is None:
        cfg = reward_config
    
    # 提取模型答案
    model_answer = extract_model_answer(response)
    
    if model_answer is None:
        return cfg.no_answer_reward
    
    # 数值比较
    if abs(model_answer - ground_truth) < tolerance:
        return cfg.correct_reward
    
    # 相对误差 < 1%（处理浮点精度问题）
    if ground_truth != 0 and abs(model_answer - ground_truth) / abs(ground_truth) < 0.01:
        return cfg.correct_reward
    
    # 答案错误但能提取到数字（说明格式基本对）
    return cfg.incorrect_reward


# ============================================================
# 2. 格式奖励
# ============================================================
def format_reward(
    response: str,
    cfg: RewardConfig = None,
) -> float:
    """
    格式奖励：检查推理步骤的格式规范性。
    
    评分标准：
    - 包含清晰的推理步骤 → +0.3
    - 步骤数量在合理范围 → 额外 +0.1
    - 包含最终答案标记 → +0.2
    - 格式混乱 → 0.0
    
    Args:
        response: 模型生成的响应
        cfg: 奖励配置
    """
    if cfg is None:
        cfg = reward_config
    
    reward = 0.0
    
    # 检查是否包含推理步骤
    step_patterns = [
        r'步骤\s*\d+[：:]',          # 步骤 1：
        r'[Ss]tep\s*\d+[：:]',       # Step 1:
        r'第\s*\d+\s*步[：:]',       # 第 1 步：
        r'\d+\.\s+',                  # 1. 
        r'\d+\)\s+',                  # 1) 
    ]
    
    step_count = 0
    for pattern in step_patterns:
        matches = re.findall(pattern, response)
        step_count = max(step_count, len(matches))
    
    # 有推理步骤
    if step_count >= cfg.step_count_min:
        reward += cfg.has_steps_reward
    
    # 步骤数量在合理范围
    if cfg.step_count_min <= step_count <= cfg.step_count_max:
        reward += 0.1
    elif step_count > cfg.step_count_max:
        reward += 0.05  # 步骤太多，减半奖励
    
    # 包含最终答案标记
    has_answer_marker = any([
        "####" in response,
        "答案" in response,
        "answer" in response.lower(),
        "\\boxed" in response,
    ])
    if has_answer_marker:
        reward += cfg.has_final_answer_reward
    
    # 响应不能太短
    if len(response.strip()) < 20:
        reward = 0.0
    
    return reward


# ============================================================
# 3. 长度惩罚
# ============================================================
def length_penalty(
    response: str,
    cfg: RewardConfig = None,
) -> float:
    """
    长度惩罚：防止模型生成过长的响应。
    
    短于 max_response_length → 0（无惩罚）
    超过 → 按超出长度线性惩罚
    """
    if cfg is None:
        cfg = reward_config
    
    length = len(response)
    
    if length <= cfg.max_response_length:
        return 0.0
    
    excess = length - cfg.max_response_length
    return -excess * cfg.length_penalty_factor


# ============================================================
# 4. 组合奖励
# ============================================================
def composite_reward(
    response: str,
    ground_truth: float,
    correctness_weight: float = None,
    format_weight: float = None,
    length_weight: float = None,
    cfg: RewardConfig = None,
) -> dict:
    """
    组合奖励：加权组合三种奖励。
    
    返回包含各分项和总分的字典，便于分析和调试。
    
    Args:
        response: 模型生成的响应
        ground_truth: 标准答案
        correctness_weight: 正确性权重（默认从 config 读取）
        format_weight: 格式权重
        length_weight: 长度权重
        cfg: 奖励配置
    """
    from config import config as grpo_config
    
    if correctness_weight is None:
        correctness_weight = grpo_config.reward_correctness_weight
    if format_weight is None:
        format_weight = grpo_config.reward_format_weight
    if length_weight is None:
        length_weight = grpo_config.reward_length_weight
    
    # 计算各分项
    r_correct = correctness_reward(response, ground_truth, cfg)
    r_format = format_reward(response, cfg)
    r_length = length_penalty(response, cfg)
    
    # 加权求和
    total = (
        correctness_weight * r_correct
        + format_weight * r_format
        + length_weight * r_length
    )
    
    return {
        "total": total,
        "correctness": r_correct,
        "format": r_format,
        "length": r_length,
        "weights": {
            "correctness": correctness_weight,
            "format": format_weight,
            "length": length_weight,
        },
    }


# ============================================================
# 5. 批量评分
# ============================================================
def batch_reward(
    responses: List[str],
    ground_truths: List[float],
    reward_type: str = "composite",
    **kwargs,
) -> List[float]:
    """
    批量计算奖励分数。
    
    Args:
        responses: 模型响应列表
        ground_truths: 标准答案列表
        reward_type: 奖励类型 (correctness / format / composite)
    
    Returns:
        奖励分数列表
    """
    rewards = []
    
    for resp, gt in zip(responses, ground_truths):
        if reward_type == "correctness":
            r = correctness_reward(resp, gt, **kwargs)
        elif reward_type == "format":
            r = format_reward(resp, **kwargs)
        elif reward_type == "composite":
            result = composite_reward(resp, gt, **kwargs)
            r = result["total"]
        else:
            raise ValueError(f"未知奖励类型: {reward_type}")
        rewards.append(r)
    
    return rewards


# ============================================================
# 6. 奖励统计
# ============================================================
def reward_statistics(rewards: List[float]) -> dict:
    """计算奖励统计信息"""
    import statistics
    
    if not rewards:
        return {"mean": 0, "std": 0, "min": 0, "max": 0, "median": 0}
    
    return {
        "mean": statistics.mean(rewards),
        "std": statistics.stdev(rewards) if len(rewards) > 1 else 0,
        "min": min(rewards),
        "max": max(rewards),
        "median": statistics.median(rewards),
    }
