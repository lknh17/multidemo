"""
p08 推理部署 - SGLang 服务启动

使用 SGLang 启动推理服务，支持：
- RadixAttention（基数树缓存，自动复用前缀 KV Cache）
- 混合分块调度（prefill + decode 同批处理）
- 数据并行 + 张量并行
- OpenAI 兼容 API

使用方式:
    python serve_sglang.py
    python serve_sglang.py --tp 2
    python serve_sglang.py --dp 2
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config, sglang_config


def build_sglang_command(args) -> list:
    """构建 SGLang 启动命令"""
    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", args.model or config.model_path,
        "--port", str(args.port or config.sglang_port),
        "--host", "0.0.0.0",
    ]

    # 张量并行
    tp = args.tp or sglang_config.tp_size
    cmd.extend(["--tp", str(tp)])

    # 数据并行
    dp = args.dp or sglang_config.dp_size
    if dp > 1:
        cmd.extend(["--dp", str(dp)])

    # 精度
    cmd.extend(["--dtype", sglang_config.dtype])

    # 显存占比
    cmd.extend(["--mem-fraction-static", str(sglang_config.mem_fraction_static)])

    # 上下文长度
    cmd.extend(["--context-length", str(sglang_config.context_length)])

    # 分块 prefill
    cmd.extend(["--chunked-prefill-size", str(sglang_config.chunked_prefill_size)])

    # 调度策略
    cmd.extend(["--schedule-policy", sglang_config.schedule_policy])
    cmd.extend(["--schedule-conservativeness", str(sglang_config.schedule_conservativeness)])

    # RadixAttention 缓存
    if sglang_config.disable_radix_cache:
        cmd.append("--disable-radix-cache")

    # 模型名称
    if sglang_config.served_model_name:
        cmd.extend(["--served-model-name", sglang_config.served_model_name])

    # API Key
    if sglang_config.api_key and sglang_config.api_key != "EMPTY":
        cmd.extend(["--api-key", sglang_config.api_key])

    return cmd


def print_config(args):
    """打印服务配置摘要"""
    model = args.model or config.model_path
    port = args.port or config.sglang_port
    tp = args.tp or sglang_config.tp_size
    dp = args.dp or sglang_config.dp_size

    print("=" * 60)
    print("  SGLang 推理服务配置")
    print("=" * 60)
    print(f"  模型:           {model}")
    print(f"  端口:           {port}")
    print(f"  张量并行:       {tp} GPU(s)")
    print(f"  数据并行:       {dp}")
    print(f"  显存占比:       {sglang_config.mem_fraction_static * 100:.0f}%")
    print(f"  上下文长度:     {sglang_config.context_length}")
    print(f"  调度策略:       {sglang_config.schedule_policy}")
    print(f"  RadixAttention: {'✓' if not sglang_config.disable_radix_cache else '✗'}")
    print(f"  混合分块:       {'✓' if sglang_config.enable_mixed_chunk else '✗'}")
    print(f"  分块 Prefill:   {sglang_config.chunked_prefill_size}")
    print("=" * 60)
    print(f"\n  API 地址: http://0.0.0.0:{port}/v1")
    print(f"  兼容 OpenAI SDK\n")


def check_sglang_installed():
    """检查 SGLang 是否安装"""
    try:
        import sglang
        print(f"  SGLang 版本: {sglang.__version__}")
        return True
    except ImportError:
        print("  ✗ SGLang 未安装！请执行: pip install sglang[all]")
        return False


def save_launch_info(cmd: list, args):
    """保存启动信息"""
    os.makedirs(config.log_dir, exist_ok=True)
    info = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(cmd),
        "model": args.model or config.model_path,
        "port": args.port or config.sglang_port,
        "tp": args.tp or sglang_config.tp_size,
        "dp": args.dp or sglang_config.dp_size,
    }
    path = os.path.join(config.log_dir, "sglang_launch.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  启动信息已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="启动 SGLang 推理服务")
    parser.add_argument("--model", type=str, default=None, help="模型路径")
    parser.add_argument("--port", type=int, default=None, help="服务端口")
    parser.add_argument("--tp", type=int, default=None, help="张量并行 GPU 数")
    parser.add_argument("--dp", type=int, default=None, help="数据并行数")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令")
    args = parser.parse_args()

    print("\n🚀 启动 SGLang 推理服务\n")

    # 检查环境
    if not check_sglang_installed():
        sys.exit(1)

    # 打印配置
    print_config(args)

    # 构建命令
    cmd = build_sglang_command(args)
    print(f"  完整命令:\n  {' '.join(cmd)}\n")

    # 保存启动信息
    save_launch_info(cmd, args)

    if args.dry_run:
        print("  [Dry Run] 仅打印命令，不实际启动。")
        return

    # 启动服务
    print("  正在启动 SGLang 服务...\n")
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\n\n  ⏹ 服务已停止")
        process.terminate()


if __name__ == "__main__":
    main()
