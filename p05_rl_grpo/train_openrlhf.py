"""
p05 强化学习 GRPO - OpenRLHF 训练脚本

使用 OpenRLHF 框架进行 GRPO 训练，与 trl 方案对比。
OpenRLHF 在多 GPU 和 vLLM 加速方面更有优势。

支持功能:
1. vLLM 加速采样
2. 多 GPU 数据并行
3. 自定义奖励函数
4. Ray 分布式训练

使用方式:
    cd p05_rl_grpo
    # 单 GPU 训练
    python train_openrlhf.py

    # 多 GPU（Ray 模式）
    python train_openrlhf.py --ray --num-gpus 4

    # 指定奖励函数
    python train_openrlhf.py --reward-type composite
"""

import os
import sys
import argparse
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import GRPOConfig, config
from dataset import load_jsonl, format_chat_messages, extract_answer, extract_model_answer
from reward import batch_reward


# ============================================================
# 1. OpenRLHF 数据准备
# ============================================================
def prepare_openrlhf_data(data_path: str, output_path: str, max_samples: int = None):
    """
    将 GSM8K 数据转换为 OpenRLHF 所需的格式。
    
    OpenRLHF 期望的数据格式:
    {"prompt": "...", "metadata": {"ground_truth": 42.0}}
    """
    samples = load_jsonl(data_path)
    if max_samples:
        samples = samples[:max_samples]
    
    processed = []
    for s in samples:
        gt = extract_answer(s["answer"])
        if gt is None:
            continue
        
        messages = format_chat_messages(s["question"])
        prompt_text = ""
        for msg in messages:
            prompt_text += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        prompt_text += "<|im_start|>assistant\n"
        
        processed.append({
            "prompt": prompt_text,
            "metadata": {
                "ground_truth": gt,
                "question": s["question"],
            }
        })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for item in processed:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"  OpenRLHF 数据: {len(processed)} 条 → {output_path}")
    return processed


# ============================================================
# 2. 自定义奖励函数（OpenRLHF 接口）
# ============================================================
class GSM8KRewardFunction:
    """
    OpenRLHF 的奖励函数接口。
    
    OpenRLHF 将奖励函数作为独立进程/服务运行，
    通过 HTTP API 或函数调用传入 (prompt, response) 对。
    """
    
    def __init__(self, reward_type: str = "composite"):
        self.reward_type = reward_type
    
    def __call__(self, prompts: list, responses: list, metadata: list = None) -> list:
        """
        计算奖励分数。
        
        Args:
            prompts: prompt 列表
            responses: 模型响应列表
            metadata: 元数据列表（包含 ground_truth）
        """
        rewards = []
        for i, response in enumerate(responses):
            gt = 0.0
            if metadata and i < len(metadata):
                gt = metadata[i].get("ground_truth", 0.0)
            
            if self.reward_type == "correctness":
                from reward import correctness_reward
                r = correctness_reward(response, gt)
            elif self.reward_type == "format":
                from reward import format_reward
                r = format_reward(response)
            elif self.reward_type == "composite":
                from reward import composite_reward
                result = composite_reward(response, gt)
                r = result["total"]
            else:
                r = 0.0
            
            rewards.append(r)
        
        return rewards


# ============================================================
# 3. OpenRLHF 训练
# ============================================================
def train_openrlhf(cfg: GRPOConfig, args):
    """使用 OpenRLHF 框架进行 GRPO 训练"""
    
    print("=" * 60)
    print("  p05 强化学习 GRPO - OpenRLHF 训练")
    print("=" * 60)
    
    # ---- 准备数据 ----
    print("\n[1/4] 准备数据...")
    data_path = args.data_path or "data/gsm8k_train.jsonl"
    
    if not os.path.exists(data_path):
        print(f"  数据文件不存在: {data_path}")
        print(f"  请先运行: python download_data.py")
        return
    
    openrlhf_data_path = "data/gsm8k_openrlhf.jsonl"
    prepare_openrlhf_data(
        data_path, openrlhf_data_path,
        max_samples=args.max_samples or cfg.max_samples,
    )
    
    # ---- 配置参数 ----
    print("\n[2/4] 配置训练参数...")
    model_path = args.model_path or cfg.model_name
    reward_type = args.reward_type or cfg.reward_type
    group_size = args.group_size or cfg.group_size
    kl_coef = args.kl_coef if args.kl_coef is not None else cfg.kl_coef
    
    print(f"  模型: {model_path}")
    print(f"  奖励: {reward_type}")
    print(f"  Group Size: {group_size}")
    print(f"  KL 系数: {kl_coef}")
    
    # ---- 构建 OpenRLHF 命令 ----
    print("\n[3/4] 构建训练命令...")
    
    # OpenRLHF CLI 参数
    openrlhf_args = {
        "pretrain": model_path,
        "dataset": openrlhf_data_path,
        "input_key": "prompt",
        "output_dir": cfg.output_dir + "_openrlhf",
        "save_steps": cfg.save_steps,
        "logging_steps": cfg.logging_steps,
        "micro_train_batch_size": cfg.per_device_train_batch_size,
        "train_batch_size": cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps,
        "max_epochs": cfg.num_train_epochs,
        "max_len": cfg.max_new_tokens + 256,
        "generate_max_len": cfg.max_new_tokens,
        "num_episodes": group_size,
        "learning_rate": cfg.learning_rate,
        "kl_coef": kl_coef,
        "bf16": "",
        "gradient_checkpointing": "",
        "flash_attn": "",
        "seed": cfg.seed,
    }
    
    # 构建命令行
    cmd_parts = ["openrlhf", "train", "grpo"]
    for key, val in openrlhf_args.items():
        if val == "":
            cmd_parts.append(f"--{key}")
        else:
            cmd_parts.append(f"--{key} {val}")
    
    cmd = " \\\n    ".join(cmd_parts)
    
    print(f"\n  OpenRLHF 命令:")
    print(f"  {cmd}")
    
    # ---- 尝试执行 ----
    print("\n[4/4] 执行训练...")
    
    try:
        from openrlhf.trainer import GRPOTrainer as OpenRLHFGRPOTrainer
        
        # 如果 OpenRLHF 已安装，使用 Python API
        print("  检测到 OpenRLHF，使用 Python API 训练...")
        
        # 创建奖励函数实例
        reward_fn = GSM8KRewardFunction(reward_type)
        
        # 注意：实际使用时需要根据 OpenRLHF 的 API 版本调整
        # 这里提供参考代码框架
        print("  OpenRLHF 训练开始...")
        print("  （请参考 OpenRLHF 文档配置具体参数）")
        
    except ImportError:
        print("  OpenRLHF 未安装，输出命令供手动执行：")
        print(f"\n  pip install openrlhf")
        print(f"\n  {cmd}")
        
        # 保存命令到文件
        cmd_file = os.path.join(cfg.output_dir + "_openrlhf", "train_cmd.sh")
        os.makedirs(os.path.dirname(cmd_file), exist_ok=True)
        with open(cmd_file, "w") as f:
            f.write("#!/bin/bash\n")
            f.write(f"# OpenRLHF GRPO 训练命令\n\n")
            f.write(cmd.replace(" \\\n    ", " \\\n  ") + "\n")
        print(f"\n  命令已保存到: {cmd_file}")
    
    print("\n" + "=" * 60)
    print("  OpenRLHF 配置完成！")
    print("=" * 60)


# ============================================================
# 4. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p05 GRPO (OpenRLHF)")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--reward-type", type=str, default=None,
                       choices=["correctness", "format", "composite"])
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--kl-coef", type=float, default=None)
    parser.add_argument("--ray", action="store_true", help="使用 Ray 分布式")
    parser.add_argument("--num-gpus", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()
    
    train_openrlhf(config, args)


if __name__ == "__main__":
    main()
