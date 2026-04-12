"""
p01 环境搭建 - 模型下载脚本

下载实践系列所需的基座模型：
1. Qwen2.5-0.5B — 贯穿 p02-p05 全流程的基座模型
2. Qwen2.5-0.5B-Instruct — 用于对比的 Chat 版本
3. Qwen2.5-VL-2B-Instruct — p06 MLLM 阶段使用

支持国内 hf-mirror.com 加速，断点续传。

使用方式:
    cd p01_env_setup
    python download_model.py                    # 下载全部
    python download_model.py --model base       # 只下载基座模型
    python download_model.py --no-mirror        # 不使用镜像
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EnvConfig, config


# ============================================================
# 1. 模型下载函数
# ============================================================
def download_model(
    model_name: str,
    cache_dir: str,
    use_mirror: bool = True,
    mirror_url: str = "https://hf-mirror.com",
) -> str:
    """
    从 HuggingFace Hub 下载模型。
    
    为什么用 snapshot_download 而不是 from_pretrained？
    - snapshot_download 只下载文件，不加载到内存
    - from_pretrained 会下载并加载模型，可能 OOM
    - snapshot_download 支持断点续传，更适合大模型
    - 下载后文件缓存在本地，后续 from_pretrained 直接用缓存
    
    Args:
        model_name: HuggingFace 模型仓库名（如 "Qwen/Qwen2.5-0.5B"）
        cache_dir: 本地缓存目录
        use_mirror: 是否使用国内镜像
        mirror_url: 镜像地址
    
    Returns:
        本地模型路径
    """
    from huggingface_hub import snapshot_download
    
    # 设置镜像
    if use_mirror:
        os.environ["HF_ENDPOINT"] = mirror_url
        print(f"  使用镜像: {mirror_url}")
    
    print(f"  模型: {model_name}")
    print(f"  缓存: {cache_dir}")
    print(f"  下载中...")
    
    start_time = time.time()
    
    local_path = snapshot_download(
        repo_id=model_name,
        cache_dir=cache_dir,
        # resume_download=True 支持断点续传
        # 如果下载中断，重新运行会从断点继续
        resume_download=True,
    )
    
    elapsed = time.time() - start_time
    print(f"  ✅ 完成！耗时: {elapsed:.1f}s")
    print(f"  路径: {local_path}")
    
    return local_path


# ============================================================
# 2. 模型信息展示
# ============================================================
def show_model_info(model_name: str):
    """
    展示模型的基本信息（参数量、架构、词表大小等）。
    
    加载模型配置（不加载权重）来获取这些信息，
    这样不需要 GPU 显存，在 CPU 上就能查看。
    """
    from transformers import AutoConfig
    
    print(f"\n{'='*50}")
    print(f" 模型信息: {model_name}")
    print(f"{'='*50}")
    
    try:
        cfg = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        
        # 估算参数量（基于模型配置）
        hidden = getattr(cfg, 'hidden_size', '?')
        layers = getattr(cfg, 'num_hidden_layers', '?')
        heads = getattr(cfg, 'num_attention_heads', '?')
        vocab = getattr(cfg, 'vocab_size', '?')
        intermediate = getattr(cfg, 'intermediate_size', '?')
        max_pos = getattr(cfg, 'max_position_embeddings', '?')
        
        print(f"  模型类型:      {cfg.model_type}")
        print(f"  隐藏维度:      {hidden}")
        print(f"  层数:          {layers}")
        print(f"  注意力头数:    {heads}")
        print(f"  FFN 中间维度:  {intermediate}")
        print(f"  词表大小:      {vocab}")
        print(f"  最大位置:      {max_pos}")
        
        # 粗略估算参数量
        if isinstance(hidden, int) and isinstance(layers, int) and isinstance(vocab, int):
            # 近似公式: 12 * L * d^2（attention + FFN）+ vocab * d（embedding）
            if isinstance(intermediate, int):
                params_per_layer = (
                    4 * hidden * hidden          # QKV + output projection
                    + 2 * hidden * intermediate  # FFN up + down
                    + 3 * hidden * intermediate  # FFN gate (SwiGLU)
                ) / 1e9  # 简化估算
                # 更准确的估算
                attn_params = 4 * hidden * hidden  # Q, K, V, O
                ffn_params = 3 * hidden * intermediate  # gate, up, down (SwiGLU)
                layer_params = attn_params + ffn_params
                total_params = (layer_params * layers + vocab * hidden) / 1e9
            else:
                total_params = (12 * layers * hidden ** 2 + vocab * hidden) / 1e9
            
            print(f"  估算参数量:    ~{total_params:.2f}B")
            print(f"  bf16 显存:     ~{total_params * 2:.2f} GB（仅加载）")
            print(f"  fp32 显存:     ~{total_params * 4:.2f} GB（仅加载）")
    
    except Exception as e:
        print(f"  ⚠️ 无法获取模型信息: {e}")
    
    print(f"{'='*50}\n")


# ============================================================
# 3. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="下载实践系列所需模型")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["all", "base", "chat", "vl"],
        help="下载哪些模型: all=全部, base=基座, chat=对话版, vl=多模态"
    )
    parser.add_argument(
        "--no-mirror", action="store_true",
        help="不使用国内镜像（海外服务器使用）"
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="自定义缓存目录（默认: ~/.cache/huggingface/hub）"
    )
    args = parser.parse_args()
    
    cache_dir = args.cache_dir or config.model_cache_dir
    use_mirror = not args.no_mirror
    
    print("=" * 60)
    print("  实践大模型 - 模型下载")
    print("=" * 60)
    print(f"  镜像加速: {'开启' if use_mirror else '关闭'}")
    print(f"  缓存目录: {cache_dir}")
    print()
    
    # 需要下载的模型列表
    models_to_download = []
    
    if args.model in ("all", "base"):
        models_to_download.append(("基座模型", config.base_model_name))
    if args.model in ("all", "chat"):
        models_to_download.append(("Chat 模型", config.chat_model_name))
    if args.model in ("all", "vl"):
        models_to_download.append(("多模态模型", config.vl_model_name))
    
    # 逐个下载
    for i, (desc, model_name) in enumerate(models_to_download, 1):
        print(f"\n[{i}/{len(models_to_download)}] 下载 {desc}")
        print("-" * 40)
        
        path = download_model(
            model_name=model_name,
            cache_dir=cache_dir,
            use_mirror=use_mirror,
        )
        
        # 展示模型信息
        show_model_info(model_name)
    
    print("\n" + "=" * 60)
    print("  ✅ 所有模型下载完成！")
    print()
    print("  下一步:")
    print("    python verify_env.py     # 验证环境")
    print("    python gpu_benchmark.py  # GPU 性能测试")
    print("    python inference.py      # 体验模型推理")
    print("=" * 60)


if __name__ == "__main__":
    main()
