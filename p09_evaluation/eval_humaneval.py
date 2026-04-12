"""
p09 评测体系 - HumanEval 代码评测

实现 HumanEval 评测:
- 给定函数签名和 docstring，生成函数体
- 运行测试用例验证正确性
- 计算 pass@k 指标

使用方式:
    python eval_humaneval.py
    python eval_humaneval.py --model outputs/p03_sft/final --k 1 5
"""

import os
import sys
import json
import argparse
import random
import signal
import tempfile
import subprocess
import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config


# ============================================================
# 1. HumanEval 数据加载
# ============================================================
def load_humaneval_data(max_samples: int = 164) -> List[Dict]:
    """
    加载 HumanEval 数据集
    
    格式: {'task_id': str, 'prompt': str, 'canonical_solution': str, 'test': str, 'entry_point': str}
    """
    try:
        from datasets import load_dataset
        dataset = load_dataset("openai_humaneval", split="test")
        samples = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            samples.append({
                "task_id": item["task_id"],
                "prompt": item["prompt"],
                "canonical_solution": item["canonical_solution"],
                "test": item["test"],
                "entry_point": item["entry_point"],
            })
        return samples
    except Exception as e:
        print(f"  ⚠️ 加载 HumanEval 失败: {e}")
        print(f"  → 使用模拟数据进行演示")
        return _generate_mock_data(min(max_samples, 10))


def _generate_mock_data(n: int = 10) -> List[Dict]:
    """生成模拟 HumanEval 数据"""
    tasks = [
        {
            "task_id": "HumanEval/0",
            "prompt": 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """Check if in given list of numbers, are any two numbers closer to each other than given threshold.\n    """\n',
            "canonical_solution": '    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n    return False\n',
            "test": '\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n',
            "entry_point": "has_close_elements",
        },
        {
            "task_id": "HumanEval/1",
            "prompt": 'def separate_paren_groups(paren_string: str) -> List[str]:\n    """Input to this function is a string containing multiple groups of nested parentheses.\n    Separate those groups into separate strings and return the list of those.\n    """\n',
            "canonical_solution": '    result = []\n    current = ""\n    depth = 0\n    for c in paren_string:\n        if c == "(":\n            depth += 1\n            current += c\n        elif c == ")":\n            depth -= 1\n            current += c\n            if depth == 0:\n                result.append(current)\n                current = ""\n    return result\n',
            "test": '\ndef check(candidate):\n    assert candidate("(()()) ((())) () ((())()())") == ["(()())", "((()))", "()", "((())()())"]\n',
            "entry_point": "separate_paren_groups",
        },
        {
            "task_id": "HumanEval/2",
            "prompt": 'def truncate_number(number: float) -> float:\n    """Given a positive floating point number, return its fractional part.\n    """\n',
            "canonical_solution": "    return number % 1.0\n",
            "test": '\ndef check(candidate):\n    assert candidate(3.5) == 0.5\n    assert abs(candidate(1.33) - 0.33) < 1e-6\n',
            "entry_point": "truncate_number",
        },
    ]
    return tasks[:n]


# ============================================================
# 2. 模型加载与推理
# ============================================================
def load_model(model_path: str, device: str = "auto"):
    """加载模型和 tokenizer"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"  加载模型: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if config.bf16 else torch.float32,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer, device


def generate_completion(model, tokenizer, prompt: str,
                        device: str, max_new_tokens: int = 512,
                        temperature: float = 0.2) -> str:
    """生成代码补全"""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if temperature > 0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95
    
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
    
    new_ids = output_ids[0, inputs["input_ids"].shape[1]:]
    completion = tokenizer.decode(new_ids, skip_special_tokens=True)
    
    # 截断到第一个完整函数结束（遇到新函数定义或类定义时停止）
    lines = completion.split("\n")
    result_lines = []
    for line in lines:
        # 遇到新的顶层定义则停止
        if result_lines and (line.startswith("def ") or line.startswith("class ")):
            break
        result_lines.append(line)
    
    return "\n".join(result_lines)


# ============================================================
# 3. 代码执行与测试
# ============================================================
def run_test(prompt: str, completion: str, test_code: str,
             entry_point: str, timeout: int = 10) -> bool:
    """
    执行测试用例，判断生成的代码是否通过
    
    方法: 将 prompt + completion + test 拼接，在子进程中执行
    """
    # 拼接完整代码
    full_code = prompt + completion + "\n" + test_code
    full_code += f"\ncheck({entry_point})\n"
    
    # 添加必要的 import
    header = "from typing import List, Optional, Tuple, Dict, Any\nimport math\n"
    full_code = header + full_code
    
    try:
        # 写入临时文件
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                          delete=False, encoding="utf-8") as f:
            f.write(full_code)
            tmp_path = f.name
        
        # 在子进程中执行（有超时限制）
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )
        
        # 清理
        os.unlink(tmp_path)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        try:
            os.unlink(tmp_path)
        except:
            pass
        return False
    except Exception:
        return False


# ============================================================
# 4. pass@k 计算
# ============================================================
def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """
    计算 pass@k
    
    n: 每个任务生成的总样本数
    c: 通过测试的样本数
    k: pass@k 中的 k
    
    公式: 1 - C(n-c, k) / C(n, k)
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


# ============================================================
# 5. 主评测流程
# ============================================================
def evaluate_humaneval(model_path: str, cfg=None) -> Dict:
    """
    运行 HumanEval 评测
    
    返回: {
        'pass_at_k': {k: float},
        'total_tasks': int,
        'details': [...],
    }
    """
    if cfg is None:
        cfg = config.humaneval
    
    model, tokenizer, device = load_model(model_path, config.device)
    
    data = load_humaneval_data(cfg.max_samples)
    
    task_results = defaultdict(list)  # task_id -> [bool, bool, ...]
    details = []
    
    print(f"\n  共 {len(data)} 个编程任务，每题生成 {cfg.num_samples_per_task} 个样本")
    
    for task in tqdm(data, desc="  HumanEval 评测"):
        task_passed = []
        
        for sample_idx in range(cfg.num_samples_per_task):
            completion = generate_completion(
                model, tokenizer, task["prompt"], device,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature if cfg.num_samples_per_task > 1 else 0.0,
            )
            
            passed = run_test(
                task["prompt"], completion, task["test"],
                task["entry_point"], timeout=cfg.timeout_seconds,
            )
            task_passed.append(passed)
        
        task_results[task["task_id"]] = task_passed
        
        details.append({
            "task_id": task["task_id"],
            "entry_point": task["entry_point"],
            "num_passed": sum(task_passed),
            "num_total": len(task_passed),
        })
    
    # 计算 pass@k
    pass_at_k = {}
    for k in cfg.k_values:
        scores = []
        for task_id, results in task_results.items():
            n = len(results)
            c = sum(results)
            if n >= k:
                scores.append(compute_pass_at_k(n, c, k))
        pass_at_k[k] = np.mean(scores) if scores else 0.0
    
    return {
        "pass_at_k": pass_at_k,
        "total_tasks": len(data),
        "details": details,
    }


# ============================================================
# 6. 结果输出
# ============================================================
def print_humaneval_results(results: Dict, model_name: str):
    """格式化输出 HumanEval 结果"""
    print(f"\n{'='*60}")
    print(f"  HumanEval 代码评测结果 — {model_name}")
    print(f"{'='*60}")
    
    print(f"\n  总任务数: {results['total_tasks']}")
    print(f"\n  Pass@k 指标:")
    for k, score in results["pass_at_k"].items():
        print(f"    pass@{k}: {score:.1%}")
    
    # 展示部分结果
    print(f"\n  --- 部分任务结果 ---")
    for detail in results["details"][:8]:
        status = "✅" if detail["num_passed"] > 0 else "❌"
        print(f"  [{status}] {detail['task_id']}: "
              f"{detail['entry_point']} — "
              f"通过 {detail['num_passed']}/{detail['num_total']}")
    print()


# ============================================================
# 7. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="HumanEval 代码生成评测")
    parser.add_argument("--model", type=str, default=config.models.base_model,
                        help="模型路径")
    parser.add_argument("--max-samples", type=int, default=config.humaneval.max_samples,
                        help="最大评测任务数")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5],
                        help="pass@k 的 k 值")
    parser.add_argument("--output", type=str, default=None,
                        help="结果保存路径 (JSON)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  p09 评测体系 — HumanEval 代码评测")
    print("=" * 60)
    
    cfg = config.humaneval
    cfg.max_samples = args.max_samples
    cfg.k_values = args.k
    
    results = evaluate_humaneval(args.model, cfg)
    print_humaneval_results(results, args.model)
    
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        # pass_at_k 的 key 从 int 转 str（JSON 兼容）
        save_results = results.copy()
        save_results["pass_at_k"] = {str(k): v for k, v in results["pass_at_k"].items()}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(save_results, f, ensure_ascii=False, indent=2)
        print(f"  结果已保存到: {args.output}")


if __name__ == "__main__":
    main()
