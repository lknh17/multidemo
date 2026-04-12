"""
p10 最佳实践总结 - 一键流水线

支持 p02→p03→p04→p05 全流程顺序执行:
- 阶段跳过和恢复
- 每阶段自动保存 checkpoint
- 失败后可从断点恢复

使用方式:
    python pipeline_runner.py
    python pipeline_runner.py --skip p02 --resume-from p04
"""

import os
import sys
import json
import time
import argparse
import subprocess
from typing import List, Optional, Dict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 流水线阶段定义
# ============================================================
PIPELINE_STAGES = [
    {
        "id": "p02",
        "name": "继续预训练",
        "script": "p02_continual_pretrain/train.py",
        "output_dir": "outputs/p02_pretrain",
        "description": "在中文 Wikipedia 上继续预训练",
    },
    {
        "id": "p03",
        "name": "SFT 指令微调",
        "script": "p03_sft_finetuning/train.py",
        "output_dir": "outputs/p03_sft",
        "description": "用指令数据进行监督微调",
    },
    {
        "id": "p04",
        "name": "DPO 偏好对齐",
        "script": "p04_dpo_alignment/train.py",
        "output_dir": "outputs/p04_dpo",
        "description": "用偏好数据进行 DPO 对齐",
    },
    {
        "id": "p05",
        "name": "RL 强化学习",
        "script": "p05_rl_grpo/train.py",
        "output_dir": "outputs/p05_rl",
        "description": "用 GRPO 进行强化学习优化",
    },
]

# 进度文件
PROGRESS_FILE = "outputs/pipeline_progress.json"


# ============================================================
# 2. 进度管理
# ============================================================
def load_progress() -> Dict:
    """加载流水线进度"""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"completed": [], "current": None, "started_at": None}


def save_progress(progress: Dict):
    """保存流水线进度"""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)


# ============================================================
# 3. 阶段执行
# ============================================================
def run_stage(stage: Dict, dry_run: bool = False) -> bool:
    """
    执行一个训练阶段
    
    返回: True 成功, False 失败
    """
    script = stage["script"]
    root_dir = os.path.join(os.path.dirname(__file__), "..")
    script_path = os.path.join(root_dir, script)
    
    if not os.path.exists(script_path):
        print(f"  ⚠️ 脚本不存在: {script_path}")
        print(f"  → 跳过此阶段（脚本待创建）")
        return True  # 不阻塞后续阶段
    
    print(f"\n  🚀 执行: python {script}")
    
    if dry_run:
        print(f"  [DRY RUN] 跳过实际执行")
        return True
    
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            cwd=os.path.dirname(script_path),
            timeout=7200,  # 2 小时超时
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ⚠️ 阶段超时（超过 2 小时）")
        return False
    except Exception as e:
        print(f"  ❌ 执行失败: {e}")
        return False


# ============================================================
# 4. 流水线主逻辑
# ============================================================
def run_pipeline(skip_stages: List[str] = None,
                 resume_from: str = None,
                 dry_run: bool = False):
    """
    运行完整流水线
    
    参数:
        skip_stages: 要跳过的阶段 ID 列表
        resume_from: 从指定阶段恢复
        dry_run: 只打印不执行
    """
    skip_stages = skip_stages or []
    
    print("=" * 60)
    print("  p10 最佳实践 — 一键全流程流水线")
    print("=" * 60)
    
    # 加载进度
    progress = load_progress()
    
    # 确定起始阶段
    start_idx = 0
    if resume_from:
        for i, stage in enumerate(PIPELINE_STAGES):
            if stage["id"] == resume_from:
                start_idx = i
                break
        print(f"\n  📌 从 {resume_from} 恢复执行")
    elif progress["completed"]:
        # 自动从上次断点恢复
        for i, stage in enumerate(PIPELINE_STAGES):
            if stage["id"] not in progress["completed"]:
                start_idx = i
                break
        if start_idx > 0:
            print(f"\n  📌 自动恢复: 已完成 {progress['completed']}")
    
    # 打印计划
    print(f"\n  --- 执行计划 ---")
    for i, stage in enumerate(PIPELINE_STAGES):
        if i < start_idx:
            status = "⏭️ 已跳过"
        elif stage["id"] in skip_stages:
            status = "⏭️ 用户跳过"
        elif stage["id"] in progress["completed"]:
            status = "✅ 已完成"
        else:
            status = "⏳ 待执行"
        print(f"  {stage['id']}: {stage['name']} — {status}")
    
    # 执行
    total_start = time.time()
    
    for i, stage in enumerate(PIPELINE_STAGES):
        if i < start_idx:
            continue
        if stage["id"] in skip_stages:
            print(f"\n  ⏭️ 跳过: {stage['id']} {stage['name']}")
            continue
        if stage["id"] in progress["completed"] and not resume_from:
            print(f"\n  ✅ 已完成: {stage['id']} {stage['name']}")
            continue
        
        print(f"\n{'='*60}")
        print(f"  阶段 {stage['id']}: {stage['name']}")
        print(f"  {stage['description']}")
        print(f"{'='*60}")
        
        progress["current"] = stage["id"]
        progress["started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        save_progress(progress)
        
        stage_start = time.time()
        success = run_stage(stage, dry_run)
        elapsed = time.time() - stage_start
        
        if success:
            print(f"\n  ✅ {stage['id']} 完成 (耗时 {elapsed/60:.1f} 分钟)")
            progress["completed"].append(stage["id"])
            progress["current"] = None
            save_progress(progress)
        else:
            print(f"\n  ❌ {stage['id']} 失败")
            print(f"  → 修复问题后运行: python pipeline_runner.py --resume-from {stage['id']}")
            return
    
    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  🎉 全流程完成! 总耗时: {total_elapsed/60:.1f} 分钟")
    print(f"{'='*60}")


# ============================================================
# 5. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="一键全流程流水线")
    parser.add_argument("--skip", nargs="+", default=[], help="跳过的阶段 (如 p02 p04)")
    parser.add_argument("--resume-from", type=str, default=None, help="从指定阶段恢复")
    parser.add_argument("--dry-run", action="store_true", help="只打印计划不执行")
    parser.add_argument("--reset", action="store_true", help="重置进度")
    args = parser.parse_args()
    
    if args.reset:
        if os.path.exists(PROGRESS_FILE):
            os.remove(PROGRESS_FILE)
            print("  进度已重置")
        return
    
    run_pipeline(
        skip_stages=args.skip,
        resume_from=args.resume_from,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
