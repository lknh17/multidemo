"""
p08 推理部署 - vLLM 服务启动

使用 vLLM 启动 OpenAI 兼容的 API 服务，支持：
- PagedAttention 高效 KV Cache 管理
- Continuous Batching 动态批处理
- Tensor Parallelism 多卡并行
- 前缀缓存（多轮对话加速）

使用方式:
    python serve_vllm.py
    python serve_vllm.py --tensor-parallel 2
    python serve_vllm.py --quantization awq
"""

import os
import sys
import subprocess
import argparse
import json
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config, vllm_config


def build_vllm_command(args) -> list:
    """构建 vLLM 启动命令"""
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model or config.model_path,
        "--port", str(args.port or config.vllm_port),
        "--host", "0.0.0.0",
        "--max-model-len", str(args.max_model_len or config.max_model_len),
    ]

    # 张量并行
    tp = args.tensor_parallel or vllm_config.tensor_parallel_size
    cmd.extend(["--tensor-parallel-size", str(tp)])

    # GPU 显存利用率
    cmd.extend(["--gpu-memory-utilization", str(vllm_config.gpu_memory_utilization)])

    # 精度
    cmd.extend(["--dtype", vllm_config.dtype])

    # 量化
    if args.quantization or vllm_config.quantization:
        q = args.quantization or vllm_config.quantization
        cmd.extend(["--quantization", q])

    # PagedAttention 参数
    cmd.extend(["--block-size", str(vllm_config.block_size)])
    cmd.extend(["--swap-space", str(vllm_config.swap_space)])
    cmd.extend(["--max-num-seqs", str(vllm_config.max_num_seqs)])

    # Continuous Batching
    cmd.extend(["--max-num-batched-tokens", str(vllm_config.max_num_batched_tokens)])
    if vllm_config.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")

    # 前缀缓存
    if vllm_config.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")

    # Speculative Decoding
    if vllm_config.speculative_model:
        cmd.extend(["--speculative-model", vllm_config.speculative_model])
        cmd.extend(["--num-speculative-tokens", str(vllm_config.num_speculative_tokens)])

    # 模型名称
    if vllm_config.served_model_name:
        cmd.extend(["--served-model-name", vllm_config.served_model_name])

    # API Key
    if vllm_config.api_key and vllm_config.api_key != "EMPTY":
        cmd.extend(["--api-key", vllm_config.api_key])

    # LoRA
    if vllm_config.enable_lora:
        cmd.append("--enable-lora")
        cmd.extend(["--max-loras", str(vllm_config.max_loras)])

    # 调试模式
    if args.eager:
        cmd.append("--enforce-eager")

    if vllm_config.disable_log_requests:
        cmd.append("--disable-log-requests")

    return cmd


def print_config(args):
    """打印服务配置摘要"""
    model = args.model or config.model_path
    port = args.port or config.vllm_port
    tp = args.tensor_parallel or vllm_config.tensor_parallel_size

    print("=" * 60)
    print("  vLLM 推理服务配置")
    print("=" * 60)
    print(f"  模型:           {model}")
    print(f"  端口:           {port}")
    print(f"  张量并行:       {tp} GPU(s)")
    print(f"  显存利用率:     {vllm_config.gpu_memory_utilization * 100:.0f}%")
    print(f"  最大上下文:     {args.max_model_len or config.max_model_len}")
    print(f"  最大并发序列:   {vllm_config.max_num_seqs}")
    print(f"  块大小:         {vllm_config.block_size}")
    print(f"  分块 Prefill:   {'✓' if vllm_config.enable_chunked_prefill else '✗'}")
    print(f"  前缀缓存:       {'✓' if vllm_config.enable_prefix_caching else '✗'}")
    print(f"  投机解码:       {'✓ ' + str(vllm_config.speculative_model) if vllm_config.speculative_model else '✗'}")
    print(f"  量化:           {args.quantization or vllm_config.quantization or '无'}")
    print("=" * 60)
    print(f"\n  API 地址: http://0.0.0.0:{port}/v1")
    print(f"  兼容 OpenAI SDK，可直接使用 openai.ChatCompletion\n")


def check_vllm_installed():
    """检查 vLLM 是否安装"""
    try:
        import vllm
        print(f"  vLLM 版本: {vllm.__version__}")
        return True
    except ImportError:
        print("  ✗ vLLM 未安装！请执行: pip install vllm")
        return False


def check_model_exists(model_path: str) -> bool:
    """检查模型是否存在"""
    if os.path.isdir(model_path):
        return True
    # HuggingFace 模型名（远程加载）
    if "/" in model_path and not os.path.exists(model_path):
        print(f"  模型 {model_path} 不在本地，将从 HuggingFace 下载...")
        return True
    return False


def save_launch_info(cmd: list, args):
    """保存启动信息供调试使用"""
    os.makedirs(config.log_dir, exist_ok=True)
    info = {
        "timestamp": datetime.now().isoformat(),
        "command": " ".join(cmd),
        "model": args.model or config.model_path,
        "port": args.port or config.vllm_port,
        "tensor_parallel": args.tensor_parallel or vllm_config.tensor_parallel_size,
    }
    path = os.path.join(config.log_dir, "vllm_launch.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    print(f"  启动信息已保存: {path}")


def main():
    parser = argparse.ArgumentParser(description="启动 vLLM 推理服务")
    parser.add_argument("--model", type=str, default=None, help="模型路径或 HuggingFace 名称")
    parser.add_argument("--port", type=int, default=None, help="服务端口")
    parser.add_argument("--tensor-parallel", type=int, default=None, help="张量并行 GPU 数")
    parser.add_argument("--max-model-len", type=int, default=None, help="最大上下文长度")
    parser.add_argument("--quantization", type=str, default=None, choices=["awq", "gptq", "squeezellm"], help="量化方法")
    parser.add_argument("--eager", action="store_true", help="禁用 CUDA Graph（调试用）")
    parser.add_argument("--dry-run", action="store_true", help="只打印命令，不实际启动")
    args = parser.parse_args()

    print("\n🚀 启动 vLLM 推理服务\n")

    # 检查环境
    if not check_vllm_installed():
        sys.exit(1)

    # 打印配置
    print_config(args)

    # 构建命令
    cmd = build_vllm_command(args)
    print(f"  完整命令:\n  {' '.join(cmd)}\n")

    # 保存启动信息
    save_launch_info(cmd, args)

    if args.dry_run:
        print("  [Dry Run] 仅打印命令，不实际启动。")
        return

    # 启动服务
    print("  正在启动 vLLM 服务...\n")
    try:
        process = subprocess.Popen(cmd)
        process.wait()
    except KeyboardInterrupt:
        print("\n\n  ⏹ 服务已停止")
        process.terminate()


if __name__ == "__main__":
    main()
