"""
p07 模型量化 - GGUF 格式转换

将 HuggingFace 模型转换为 GGUF 格式（适用于 llama.cpp / Ollama / LM Studio）：
1. 导出模型为 GGUF 基础格式（FP16）
2. 对 GGUF 文件执行不同量化级别（Q2_K → Q8_0）
3. 生成多个量化版本

使用方式:
    python quantize_gguf.py
    python quantize_gguf.py --quant-type Q4_K_M
    python quantize_gguf.py --all-types           # 生成所有量化级别
"""

import os
import sys
import time
import argparse
import subprocess
import shutil
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config, gguf_config


def check_llama_cpp(llama_cpp_path: str) -> bool:
    """检查 llama.cpp 是否已编译"""
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    quantize_binary = os.path.join(llama_cpp_path, "build", "bin", "llama-quantize")
    
    if not os.path.exists(convert_script):
        # 尝试备选路径
        convert_script = os.path.join(llama_cpp_path, "convert.py")
    
    if not os.path.exists(quantize_binary):
        quantize_binary = os.path.join(llama_cpp_path, "llama-quantize")
    
    return os.path.exists(convert_script), os.path.exists(quantize_binary)


def setup_llama_cpp(llama_cpp_path: str):
    """下载并编译 llama.cpp"""
    if os.path.exists(llama_cpp_path):
        print(f"  llama.cpp 目录已存在: {llama_cpp_path}")
        return
    
    print("[Setup] 克隆 llama.cpp...")
    subprocess.run(
        ["git", "clone", "--depth", "1", "https://github.com/ggerganov/llama.cpp", llama_cpp_path],
        check=True,
    )
    
    print("[Setup] 编译 llama.cpp...")
    build_dir = os.path.join(llama_cpp_path, "build")
    os.makedirs(build_dir, exist_ok=True)
    subprocess.run(["cmake", "..", "-DLLAMA_CUDA=ON"], cwd=build_dir, check=True)
    subprocess.run(["cmake", "--build", ".", "--config", "Release", "-j"], cwd=build_dir, check=True)
    
    print("[Setup] llama.cpp 编译完成")


def convert_to_gguf_fp16(
    model_name: str,
    output_dir: str,
    llama_cpp_path: str,
) -> str:
    """将 HuggingFace 模型转换为 GGUF FP16 格式"""
    
    os.makedirs(output_dir, exist_ok=True)
    output_fp16 = os.path.join(output_dir, "model-fp16.gguf")
    
    if os.path.exists(output_fp16):
        print(f"  FP16 GGUF 已存在: {output_fp16}")
        return output_fp16
    
    # 查找转换脚本
    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    if not os.path.exists(convert_script):
        convert_script = os.path.join(llama_cpp_path, "convert.py")
    
    print(f"  转换脚本: {convert_script}")
    print(f"  输入模型: {model_name}")
    print(f"  输出文件: {output_fp16}")
    
    cmd = [
        sys.executable, convert_script,
        model_name,
        "--outfile", output_fp16,
        "--outtype", "f16",
    ]
    
    subprocess.run(cmd, check=True)
    return output_fp16


def quantize_gguf_file(
    input_gguf: str,
    quant_type: str,
    output_dir: str,
    llama_cpp_path: str,
) -> str:
    """对 GGUF 文件执行指定量化"""
    
    output_file = os.path.join(output_dir, f"model-{quant_type.lower()}.gguf")
    
    if os.path.exists(output_file):
        print(f"  {quant_type} 已存在: {output_file}")
        return output_file
    
    # 查找 quantize 二进制
    quantize_bin = os.path.join(llama_cpp_path, "build", "bin", "llama-quantize")
    if not os.path.exists(quantize_bin):
        quantize_bin = os.path.join(llama_cpp_path, "llama-quantize")
    
    cmd = [quantize_bin, input_gguf, output_file, quant_type]
    
    start_time = time.time()
    subprocess.run(cmd, check=True)
    quant_time = time.time() - start_time
    
    file_size = os.path.getsize(output_file) / (1024 ** 3)
    print(f"  {quant_type}: {file_size:.2f} GB, 耗时 {quant_time:.1f}s")
    
    return output_file


def quantize_gguf(
    model_name: str = None,
    quant_types: list = None,
    output_dir: str = None,
    llama_cpp_path: str = None,
    all_types: bool = False,
):
    """执行 GGUF 量化流程"""
    
    model_name = model_name or config.model_name
    output_dir = output_dir or gguf_config.output_dir
    llama_cpp_path = llama_cpp_path or gguf_config.llama_cpp_path
    
    if all_types:
        quant_types = gguf_config.quant_types
    elif quant_types is None:
        quant_types = [gguf_config.default_quant_type]
    
    print("=" * 60)
    print("GGUF 量化")
    print("=" * 60)
    print(f"  模型:       {model_name}")
    print(f"  量化级别:   {', '.join(quant_types)}")
    print(f"  输出目录:   {output_dir}")
    print(f"  llama.cpp:  {llama_cpp_path}")
    print("=" * 60)
    
    # 1. 检查 / 安装 llama.cpp
    print("\n[1/3] 检查 llama.cpp...")
    has_convert, has_quantize = check_llama_cpp(llama_cpp_path)
    if not (has_convert and has_quantize):
        print("  llama.cpp 未找到，开始安装...")
        setup_llama_cpp(llama_cpp_path)
    else:
        print("  llama.cpp 已就绪")
    
    # 2. 转换为 FP16 GGUF
    print("\n[2/3] 转换为 GGUF FP16...")
    fp16_gguf = convert_to_gguf_fp16(model_name, output_dir, llama_cpp_path)
    fp16_size = os.path.getsize(fp16_gguf) / (1024 ** 3)
    print(f"  FP16 GGUF: {fp16_size:.2f} GB")
    
    # 3. 量化
    print("\n[3/3] 执行量化...")
    results = {}
    for qt in quant_types:
        print(f"\n  量化 {qt}...")
        try:
            output_file = quantize_gguf_file(fp16_gguf, qt, output_dir, llama_cpp_path)
            file_size = os.path.getsize(output_file) / (1024 ** 3)
            results[qt] = {
                "path": output_file,
                "size_gb": file_size,
                "compression": fp16_size / file_size,
            }
        except Exception as e:
            print(f"  ❌ {qt} 量化失败: {e}")
            results[qt] = {"error": str(e)}
    
    # 汇总
    print("\n" + "=" * 60)
    print("GGUF 量化结果汇总")
    print("=" * 60)
    print(f"{'量化级别':<12} {'大小':>8} {'压缩比':>8}")
    print("-" * 30)
    for qt, info in results.items():
        if "error" not in info:
            print(f"{qt:<12} {info['size_gb']:>7.2f}G {info['compression']:>7.1f}x")
        else:
            print(f"{qt:<12} {'失败':>8}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="GGUF 量化")
    parser.add_argument("--model", type=str, default=None, help="模型名称或路径")
    parser.add_argument("--quant-type", type=str, default=None, help="量化级别 (如 Q4_K_M)")
    parser.add_argument("--all-types", action="store_true", help="生成所有量化级别")
    parser.add_argument("--output-dir", type=str, default=None, help="输出目录")
    parser.add_argument("--llama-cpp", type=str, default=None, help="llama.cpp 路径")
    args = parser.parse_args()
    
    quant_types = [args.quant_type] if args.quant_type else None
    
    quantize_gguf(
        model_name=args.model,
        quant_types=quant_types,
        output_dir=args.output_dir,
        llama_cpp_path=args.llama_cpp,
        all_types=args.all_types,
    )


if __name__ == "__main__":
    main()
