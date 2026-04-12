"""
p01 环境搭建 - 环境验证脚本

全面检查训练环境的各项配置是否正确：
1. Python 版本
2. PyTorch + CUDA 可用性
3. GPU 硬件信息
4. 关键依赖库版本
5. Flash Attention 可用性
6. DeepSpeed 可用性
7. 版本兼容性矩阵检查
8. 模型是否已下载

运行方式:
    cd p01_env_setup
    python verify_env.py
"""

import os
import sys
import platform
import importlib
from typing import Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 检查项定义
# ============================================================
class CheckResult:
    """单项检查的结果"""
    
    def __init__(self, name: str, status: str, detail: str, suggestion: str = ""):
        """
        Args:
            name: 检查项名称
            status: "PASS" | "WARN" | "FAIL"
            detail: 检查结果详情
            suggestion: 失败时的修复建议
        """
        self.name = name
        self.status = status
        self.detail = detail
        self.suggestion = suggestion
    
    def __str__(self):
        icons = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}
        icon = icons.get(self.status, "❓")
        line = f"  {icon} {self.name}: {self.detail}"
        if self.suggestion:
            line += f"\n     💡 {self.suggestion}"
        return line


# ============================================================
# 2. 各项检查函数
# ============================================================
def check_python() -> CheckResult:
    """检查 Python 版本（需要 3.10+）"""
    version = platform.python_version()
    major, minor = sys.version_info.major, sys.version_info.minor
    
    if major >= 3 and minor >= 10:
        return CheckResult("Python 版本", "PASS", f"{version}")
    else:
        return CheckResult(
            "Python 版本", "FAIL", f"{version}（需要 3.10+）",
            "请安装 Python 3.10+: conda create -n llm python=3.10"
        )


def check_pytorch() -> CheckResult:
    """检查 PyTorch 安装和 CUDA 可用性"""
    try:
        import torch
        version = torch.__version__
        cuda_available = torch.cuda.is_available()
        cuda_version = torch.version.cuda if cuda_available else "N/A"
        
        if cuda_available:
            return CheckResult("PyTorch + CUDA", "PASS", 
                             f"PyTorch {version} | CUDA {cuda_version}")
        else:
            return CheckResult(
                "PyTorch + CUDA", "WARN", 
                f"PyTorch {version}（CUDA 不可用，仅 CPU）",
                "GPU 训练需要 CUDA。请安装 CUDA 版 PyTorch: pip install torch --index-url https://download.pytorch.org/whl/cu118"
            )
    except ImportError:
        return CheckResult(
            "PyTorch", "FAIL", "未安装",
            "pip install torch --index-url https://download.pytorch.org/whl/cu118"
        )


def check_gpu() -> CheckResult:
    """检查 GPU 硬件信息"""
    try:
        import torch
        if not torch.cuda.is_available():
            return CheckResult("GPU 硬件", "WARN", "无可用 GPU")
        
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        
        # 检查 GPU 架构
        capability = torch.cuda.get_device_capability(0)
        arch_name = {
            (7, 0): "Volta (V100)",
            (7, 5): "Turing (RTX 20x0/T4)",
            (8, 0): "Ampere (A100)",
            (8, 6): "Ampere (RTX 30x0/A6000)",
            (8, 9): "Ada Lovelace (RTX 40x0)",
            (9, 0): "Hopper (H100)",
        }.get(capability, f"SM {capability[0]}.{capability[1]}")
        
        # bf16 需要 Ampere+ (SM 8.0+)
        bf16_support = capability[0] >= 8
        bf16_str = "支持" if bf16_support else "不支持（将使用 fp16）"
        
        detail = (
            f"{gpu_name} x{gpu_count} | "
            f"显存: {vram:.1f}GB | "
            f"架构: {arch_name} | "
            f"bf16: {bf16_str}"
        )
        
        if vram >= 20:
            return CheckResult("GPU 硬件", "PASS", detail)
        else:
            return CheckResult(
                "GPU 硬件", "WARN", detail,
                f"显存 {vram:.0f}GB 较小，部分实验可能需要调低 batch_size"
            )
    except Exception as e:
        return CheckResult("GPU 硬件", "FAIL", str(e))


def check_library(name: str, min_version: Optional[str] = None, 
                  import_name: Optional[str] = None) -> CheckResult:
    """通用库检查函数"""
    try:
        lib = importlib.import_module(import_name or name)
        version = getattr(lib, "__version__", "unknown")
        
        if min_version and version != "unknown":
            from packaging.version import Version
            try:
                if Version(version) < Version(min_version):
                    return CheckResult(
                        name, "WARN", f"{version}（推荐 {min_version}+）",
                        f"pip install {name}>={min_version}"
                    )
            except Exception:
                pass
        
        return CheckResult(name, "PASS", f"{version}")
    except ImportError:
        return CheckResult(
            name, "FAIL" if min_version else "WARN", "未安装",
            f"pip install {name}"
        )


def check_flash_attention() -> CheckResult:
    """检查 Flash Attention 安装"""
    try:
        import flash_attn
        version = flash_attn.__version__
        return CheckResult("Flash Attention", "PASS", f"{version}")
    except ImportError:
        return CheckResult(
            "Flash Attention", "WARN", "未安装（可选，但推荐）",
            "pip install flash-attn --no-build-isolation"
        )


def check_deepspeed() -> CheckResult:
    """检查 DeepSpeed 安装和 CUDA 扩展编译状态"""
    try:
        import deepspeed
        version = deepspeed.__version__
        
        # 检查 CUDA 扩展是否可用
        try:
            from deepspeed.ops.op_builder import CPUAdamBuilder
            cpu_adam = CPUAdamBuilder().is_compatible()
            ext_status = "CUDA 扩展可用" if cpu_adam else "CUDA 扩展未编译（JIT 编译）"
        except Exception:
            ext_status = "未检查 CUDA 扩展"
        
        return CheckResult("DeepSpeed", "PASS", f"{version} | {ext_status}")
    except ImportError:
        return CheckResult(
            "DeepSpeed", "FAIL", "未安装",
            "pip install deepspeed"
        )


def check_model_downloaded(model_name: str) -> CheckResult:
    """检查模型是否已下载到本地缓存"""
    try:
        from huggingface_hub import try_to_load_from_cache
        from transformers import AutoConfig
        
        # 尝试从缓存加载配置文件
        config_path = try_to_load_from_cache(model_name, "config.json")
        
        if config_path is not None and os.path.exists(str(config_path)):
            return CheckResult(f"模型 {model_name}", "PASS", "已下载")
        else:
            return CheckResult(
                f"模型 {model_name}", "WARN", "未下载",
                f"python download_model.py --model base"
            )
    except Exception:
        return CheckResult(f"模型 {model_name}", "WARN", "无法检查")


# ============================================================
# 3. 主流程
# ============================================================
def main():
    from config import config
    
    print("=" * 60)
    print("  实践大模型 - 环境验证报告")
    print("=" * 60)
    
    results = []
    
    # ---- 系统环境 ----
    print("\n📋 系统环境")
    print("-" * 40)
    results.append(check_python())
    results.append(check_pytorch())
    results.append(check_gpu())
    for r in results[-3:]:
        print(r)
    
    # ---- 核心依赖 ----
    print("\n📦 核心依赖")
    print("-" * 40)
    core_libs = [
        ("transformers", "4.45.0"),
        ("datasets", "2.20.0"),
        ("accelerate", "0.34.0"),
        ("peft", "0.12.0"),
        ("trl", "0.11.0"),
    ]
    for name, min_ver in core_libs:
        r = check_library(name, min_ver)
        results.append(r)
        print(r)
    
    # ---- 训练框架 ----
    print("\n🔧 训练框架")
    print("-" * 40)
    r_ds = check_deepspeed()
    r_fa = check_flash_attention()
    results.extend([r_ds, r_fa])
    print(r_ds)
    print(r_fa)
    
    # ---- 量化工具 ----
    print("\n🗜️  量化工具")
    print("-" * 40)
    quant_libs = [
        ("bitsandbytes", None),
        ("auto_gptq", None, "auto_gptq"),
        ("autoawq", None, "awq"),
    ]
    for item in quant_libs:
        name = item[0]
        min_ver = item[1] if len(item) > 1 else None
        import_name = item[2] if len(item) > 2 else None
        r = check_library(name, min_ver, import_name)
        results.append(r)
        print(r)
    
    # ---- 可选工具 ----
    print("\n📊 可选工具")
    print("-" * 40)
    optional_libs = [
        ("matplotlib", "3.7.0"),
        ("wandb", None),
        ("scipy", None),
        ("tabulate", None),
    ]
    for name, min_ver in optional_libs:
        r = check_library(name, min_ver)
        results.append(r)
        print(r)
    
    # ---- 模型下载状态 ----
    print("\n🤖 模型下载状态")
    print("-" * 40)
    for model_name in [config.base_model_name, config.chat_model_name]:
        r = check_model_downloaded(model_name)
        results.append(r)
        print(r)
    
    # ---- 汇总 ----
    pass_count = sum(1 for r in results if r.status == "PASS")
    warn_count = sum(1 for r in results if r.status == "WARN")
    fail_count = sum(1 for r in results if r.status == "FAIL")
    
    print("\n" + "=" * 60)
    print(f"  验证结果: ✅ {pass_count} 通过 | ⚠️  {warn_count} 警告 | ❌ {fail_count} 失败")
    
    if fail_count > 0:
        print("\n  ❌ 存在必须修复的问题，请按上方建议处理后重新验证")
    elif warn_count > 0:
        print("\n  ⚠️  存在可选项未安装，核心功能可正常使用")
    else:
        print("\n  🎉 所有检查通过！环境配置完美！")
    
    print(f"\n  GPU Preset: {config.gpu_preset}")
    print("=" * 60)


if __name__ == "__main__":
    main()
