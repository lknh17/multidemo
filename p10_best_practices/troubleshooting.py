"""
p10 最佳实践总结 - 故障排查工具

自动诊断训练中的常见问题:
- GPU OOM 检测与建议
- Loss 异常检测（NaN/不下降/震荡）
- 训练卡住检测
- 环境问题检测
- 综合诊断报告

使用方式:
    python troubleshooting.py
    python troubleshooting.py --check gpu
    python troubleshooting.py --log-file outputs/trainer_state.json
"""

import os
import sys
import json
import argparse
import subprocess
import shutil
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. GPU/显存诊断
# ============================================================
def check_gpu() -> Dict:
    """检测 GPU 状态和显存使用"""
    result = {"status": "unknown", "issues": [], "suggestions": []}
    
    try:
        import torch
        if not torch.cuda.is_available():
            result["status"] = "no_gpu"
            result["issues"].append("未检测到 GPU")
            result["suggestions"].append("确认 NVIDIA 驱动已安装: nvidia-smi")
            result["suggestions"].append("确认 CUDA 版本与 PyTorch 匹配")
            return result
        
        gpu_count = torch.cuda.device_count()
        result["gpu_count"] = gpu_count
        result["devices"] = []
        
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_mem = props.total_mem / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            
            device_info = {
                "name": props.name,
                "total_gb": round(total_mem, 1),
                "allocated_gb": round(allocated, 1),
                "reserved_gb": round(reserved, 1),
                "free_gb": round(total_mem - reserved, 1),
            }
            result["devices"].append(device_info)
            
            # OOM 风险检测
            usage_ratio = reserved / total_mem if total_mem > 0 else 0
            if usage_ratio > 0.95:
                result["issues"].append(f"GPU {i}: 显存使用率 {usage_ratio:.0%}，即将 OOM")
                result["suggestions"].append(f"GPU {i}: 开启 gradient_checkpointing=True")
                result["suggestions"].append(f"GPU {i}: 减小 batch_size 或 max_seq_length")
            elif usage_ratio > 0.85:
                result["issues"].append(f"GPU {i}: 显存使用率 {usage_ratio:.0%}，较高")
            
            # 显存大小建议
            if total_mem < 16:
                result["suggestions"].append(f"GPU {i} ({total_mem:.0f}GB): 建议使用 LoRA 训练")
            elif total_mem < 24:
                result["suggestions"].append(f"GPU {i} ({total_mem:.0f}GB): 使用 24G preset 配置")
        
        result["status"] = "ok" if not result["issues"] else "warning"
        
    except ImportError:
        result["status"] = "error"
        result["issues"].append("PyTorch 未安装")
        result["suggestions"].append("pip install torch")
    
    return result


# ============================================================
# 2. Loss 异常诊断
# ============================================================
def check_loss(log_file: str = None, losses: List[float] = None) -> Dict:
    """检测 Loss 曲线中的异常"""
    result = {"status": "ok", "issues": [], "suggestions": []}
    
    # 从 trainer_state.json 加载
    if log_file and os.path.exists(log_file):
        with open(log_file, "r") as f:
            state = json.load(f)
        log_history = state.get("log_history", [])
        losses = [e["loss"] for e in log_history if "loss" in e]
    
    if not losses:
        result["status"] = "no_data"
        result["issues"].append("未找到 loss 数据")
        return result
    
    import math
    
    # 检测 NaN
    nan_indices = [i for i, l in enumerate(losses) if math.isnan(l) or math.isinf(l)]
    if nan_indices:
        result["status"] = "critical"
        result["issues"].append(f"Loss 出现 NaN/Inf (步数: {nan_indices[:5]})")
        result["suggestions"].append("降低学习率 (lr ÷ 2 或 ÷ 5)")
        result["suggestions"].append("检查数据中是否有异常值")
        result["suggestions"].append("使用 bf16 代替 fp16 (不需要 loss scaling)")
        result["suggestions"].append("增大 max_grad_norm (梯度裁剪)")
        return result
    
    # 检测 Loss 不下降
    if len(losses) > 20:
        first_quarter = sum(losses[:len(losses)//4]) / (len(losses)//4)
        last_quarter = sum(losses[-len(losses)//4:]) / (len(losses)//4)
        
        if last_quarter >= first_quarter * 0.98:
            result["status"] = "warning"
            result["issues"].append(f"Loss 几乎不下降 (前1/4平均: {first_quarter:.4f}, 后1/4平均: {last_quarter:.4f})")
            result["suggestions"].append("学习率可能太小，尝试增大 2-5 倍")
            result["suggestions"].append("检查数据质量（是否有大量空/重复样本）")
            result["suggestions"].append("检查 labels 是否正确设置（-100 mask）")
    
    # 检测 Loss 震荡
    if len(losses) > 10:
        diffs = [abs(losses[i] - losses[i-1]) for i in range(1, len(losses))]
        avg_diff = sum(diffs) / len(diffs)
        avg_loss = sum(losses) / len(losses)
        volatility = avg_diff / avg_loss if avg_loss > 0 else 0
        
        if volatility > 0.3:
            result["status"] = "warning"
            result["issues"].append(f"Loss 震荡剧烈 (波动率: {volatility:.1%})")
            result["suggestions"].append("增大 batch_size 或 gradient_accumulation_steps")
            result["suggestions"].append("降低学习率")
    
    # 检测 Loss 突然飙升
    for i in range(1, len(losses)):
        if losses[i] > losses[i-1] * 2 and losses[i-1] > 0.1:
            result["status"] = "warning"
            result["issues"].append(f"步数 {i}: Loss 突然飙升 ({losses[i-1]:.3f} → {losses[i]:.3f})")
            result["suggestions"].append("可能遇到了质量极差的数据 batch")
            result["suggestions"].append("降低 max_grad_norm 加强梯度裁剪")
            break
    
    return result


# ============================================================
# 3. 训练卡住检测
# ============================================================
def check_training_stuck(log_file: str = None) -> Dict:
    """检测训练是否卡住"""
    result = {"status": "ok", "issues": [], "suggestions": []}
    
    if log_file and os.path.exists(log_file):
        import time
        mod_time = os.path.getmtime(log_file)
        elapsed = time.time() - mod_time
        
        if elapsed > 3600:  # 超过 1 小时没更新
            result["status"] = "warning"
            result["issues"].append(f"日志文件超过 {elapsed/3600:.1f} 小时未更新")
            result["suggestions"].append("检查 GPU 利用率: nvidia-smi")
            result["suggestions"].append("检查是否死锁: 可能是 DeepSpeed 通信问题")
            result["suggestions"].append("尝试 Ctrl+C 后从 checkpoint 恢复")
    
    # 检查 GPU 利用率
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if output.returncode == 0:
            utils = [int(x.strip()) for x in output.stdout.strip().split("\n") if x.strip()]
            for i, util in enumerate(utils):
                if util < 5:
                    result["issues"].append(f"GPU {i} 利用率仅 {util}%")
                    result["suggestions"].append("训练可能卡在数据加载，增加 num_workers")
    except:
        pass
    
    return result


# ============================================================
# 4. 环境检测
# ============================================================
def check_environment() -> Dict:
    """检测训练环境"""
    result = {"status": "ok", "issues": [], "suggestions": [], "info": {}}
    
    # Python 版本
    result["info"]["python"] = sys.version.split()[0]
    
    # PyTorch
    try:
        import torch
        result["info"]["torch"] = torch.__version__
        result["info"]["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            result["info"]["cuda_version"] = torch.version.cuda
    except ImportError:
        result["issues"].append("PyTorch 未安装")
    
    # Transformers
    try:
        import transformers
        result["info"]["transformers"] = transformers.__version__
    except ImportError:
        result["issues"].append("transformers 未安装")
        result["suggestions"].append("pip install transformers>=4.36.0")
    
    # Flash Attention
    try:
        import flash_attn
        result["info"]["flash_attn"] = flash_attn.__version__
    except ImportError:
        result["suggestions"].append("建议安装 flash-attn 提升训练速度")
    
    # 磁盘空间
    disk = shutil.disk_usage("/")
    free_gb = disk.free / 1024**3
    if free_gb < 10:
        result["issues"].append(f"磁盘剩余空间不足: {free_gb:.1f} GB")
        result["suggestions"].append("清理旧的 checkpoint 或增加磁盘空间")
    result["info"]["disk_free_gb"] = round(free_gb, 1)
    
    if result["issues"]:
        result["status"] = "warning"
    
    return result


# ============================================================
# 5. 综合诊断
# ============================================================
def run_full_diagnosis(log_file: str = None) -> Dict:
    """运行全面诊断"""
    print("=" * 60)
    print("  p10 最佳实践 — 故障排查诊断")
    print("=" * 60)
    
    checks = {
        "环境检测": check_environment(),
        "GPU 状态": check_gpu(),
        "Loss 异常": check_loss(log_file),
        "训练卡住": check_training_stuck(log_file),
    }
    
    status_icons = {"ok": "✅", "warning": "⚠️", "critical": "❌", "error": "❌",
                    "no_gpu": "⚠️", "no_data": "➖", "unknown": "❓"}
    
    for name, result in checks.items():
        icon = status_icons.get(result["status"], "❓")
        print(f"\n  {icon} {name}: {result['status']}")
        
        if result.get("info"):
            for k, v in result["info"].items():
                print(f"     {k}: {v}")
        
        if result.get("devices"):
            for dev in result["devices"]:
                print(f"     {dev['name']}: {dev['total_gb']}GB (空闲 {dev['free_gb']}GB)")
        
        for issue in result.get("issues", []):
            print(f"     ⚠ {issue}")
        
        for sug in result.get("suggestions", []):
            print(f"     → {sug}")
    
    print(f"\n{'='*60}")
    return checks


# ============================================================
# 6. 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="训练故障排查工具")
    parser.add_argument("--check", choices=["all", "gpu", "loss", "env", "stuck"],
                        default="all", help="检查项目")
    parser.add_argument("--log-file", type=str, default=None,
                        help="trainer_state.json 路径")
    args = parser.parse_args()
    
    if args.check == "all":
        run_full_diagnosis(args.log_file)
    elif args.check == "gpu":
        result = check_gpu()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.check == "loss":
        result = check_loss(args.log_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.check == "env":
        result = check_environment()
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.check == "stuck":
        result = check_training_stuck(args.log_file)
        print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
