"""
p08 推理部署 - 配置文件

管理推理部署的所有参数，支持 vLLM / SGLang / Ollama 三大框架，
涵盖模型路径、服务端口、性能调优、压力测试等配置。

使用方式:
    from config import config, vllm_config, sglang_config, ollama_config
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ============================================================
# 1. 通用服务配置
# ============================================================
@dataclass
class ServingConfig:
    """推理服务通用配置"""

    # ---- 模型 ----
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"   # HuggingFace 模型名
    model_path: str = "models/Qwen2.5-0.5B-Instruct"  # 本地模型路径
    tokenizer_path: Optional[str] = None               # 自定义 tokenizer 路径（None 则跟 model_path 一致）

    # ---- 服务端口 ----
    vllm_port: int = 8000                              # vLLM 服务端口
    sglang_port: int = 8001                            # SGLang 服务端口
    ollama_port: int = 11434                           # Ollama 服务端口（默认 11434）

    # ---- 通用推理参数 ----
    max_model_len: int = 2048                          # 最大上下文长度
    max_new_tokens: int = 512                          # 最大生成 token 数
    temperature: float = 0.7                           # 采样温度
    top_p: float = 0.9                                 # Top-p 采样
    top_k: int = 50                                    # Top-k 采样
    repetition_penalty: float = 1.05                   # 重复惩罚

    # ---- 路径 ----
    output_dir: str = "outputs/serving"
    log_dir: str = "logs/serving"
    seed: int = 42


# ============================================================
# 2. vLLM 配置
# ============================================================
@dataclass
class VLLMConfig:
    """vLLM 推理引擎配置"""

    # ---- 核心参数 ----
    tensor_parallel_size: int = 1                      # 张量并行 GPU 数量
    gpu_memory_utilization: float = 0.90               # GPU 显存利用率（0.0~1.0）
    dtype: str = "auto"                                # 推理精度: auto / float16 / bfloat16
    quantization: Optional[str] = None                 # 量化方法: awq / gptq / squeezellm / None

    # ---- PagedAttention ----
    block_size: int = 16                               # KV Cache 分页块大小（16 或 32）
    swap_space: int = 4                                # CPU swap 空间（GB）
    max_num_seqs: int = 256                            # 最大并发序列数

    # ---- Continuous Batching ----
    max_num_batched_tokens: int = 4096                 # 每批最大 token 数
    enable_chunked_prefill: bool = True                # 启用分块 prefill（降低 TTFT）

    # ---- Speculative Decoding ----
    speculative_model: Optional[str] = None            # 投机解码草稿模型
    num_speculative_tokens: int = 5                    # 每步投机 token 数

    # ---- API 配置 ----
    api_key: str = "EMPTY"                             # API Key（本地部署可留空）
    served_model_name: Optional[str] = None            # 对外暴露的模型名称
    enable_prefix_caching: bool = True                 # 前缀缓存（多轮对话加速）
    disable_log_requests: bool = False                 # 关闭请求日志

    # ---- 高级 ----
    enforce_eager: bool = False                        # 禁用 CUDA Graph（调试用）
    max_loras: int = 1                                 # 最大同时加载 LoRA 数
    enable_lora: bool = False                          # 是否启用 LoRA 服务


# ============================================================
# 3. SGLang 配置
# ============================================================
@dataclass
class SGLangConfig:
    """SGLang 推理引擎配置"""

    # ---- 核心参数 ----
    tp_size: int = 1                                   # 张量并行大小
    dp_size: int = 1                                   # 数据并行大小
    dtype: str = "auto"                                # 推理精度
    mem_fraction_static: float = 0.88                  # 静态显存占比

    # ---- RadixAttention ----
    context_length: int = 2048                         # 上下文长度
    chunked_prefill_size: int = 8192                   # 分块 prefill 大小
    enable_mixed_chunk: bool = True                    # 混合分块（prefill + decode 同批）

    # ---- 调度 ----
    schedule_policy: str = "lpm"                       # 调度策略: lpm / random / fcfs
    schedule_conservativeness: float = 1.0             # 调度保守度

    # ---- API ----
    api_key: str = "EMPTY"
    served_model_name: Optional[str] = None
    disable_radix_cache: bool = False                  # 关闭 RadixAttention 缓存


# ============================================================
# 4. Ollama 配置
# ============================================================
@dataclass
class OllamaConfig:
    """Ollama 本地部署配置"""

    # ---- 模型 ----
    gguf_path: str = "models/Qwen2.5-0.5B-Instruct-Q4_K_M.gguf"  # GGUF 量化模型
    model_tag: str = "qwen2.5-0.5b-custom"             # Ollama 模型标签

    # ---- Modelfile 参数 ----
    num_ctx: int = 2048                                # 上下文窗口大小
    num_gpu: int = 99                                  # 使用 GPU 层数（99 = 全部）
    num_thread: int = 8                                # CPU 线程数
    repeat_penalty: float = 1.1                        # 重复惩罚

    # ---- 系统提示 ----
    system_prompt: str = "你是一个有帮助的AI助手。请用中文回答问题。"


# ============================================================
# 5. 压力测试配置
# ============================================================
@dataclass
class BenchmarkConfig:
    """压力测试配置"""

    # ---- 测试目标 ----
    target_url: str = "http://localhost:8000/v1"       # 被测服务地址
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"    # 请求中的模型名

    # ---- 并发配置 ----
    concurrency_levels: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32])
    num_requests_per_level: int = 50                   # 每个并发级别的请求数

    # ---- 测试数据 ----
    prompt_length: int = 128                           # 输入 prompt 的 token 长度
    max_tokens: int = 256                              # 最大生成 token 数
    temperature: float = 0.7

    # ---- 测试 prompt 模板 ----
    test_prompts: List[str] = field(default_factory=lambda: [
        "请详细解释什么是Transformer架构，以及它在深度学习中的重要性。",
        "用Python写一个快速排序算法，并解释其时间复杂度。",
        "请分析人工智能在医疗领域的应用前景和挑战。",
        "解释什么是注意力机制，它是如何工作的？",
        "请写一篇关于量子计算的科普文章，要通俗易懂。",
    ])

    # ---- 输出 ----
    output_dir: str = "outputs/benchmark"
    save_raw_results: bool = True                      # 保存原始请求结果


# ============================================================
# 6. GPU Preset
# ============================================================
def create_config_24g() -> ServingConfig:
    """24G GPU (4090) 的优化配置"""
    cfg = ServingConfig()
    cfg.max_model_len = 2048
    return cfg


def create_config_48g() -> ServingConfig:
    """48G GPU (A6000) 的优化配置"""
    cfg = ServingConfig()
    cfg.max_model_len = 4096
    return cfg


def create_vllm_config_24g() -> VLLMConfig:
    """24G vLLM 优化配置"""
    vcfg = VLLMConfig()
    vcfg.gpu_memory_utilization = 0.90
    vcfg.max_num_seqs = 128
    vcfg.max_num_batched_tokens = 2048
    return vcfg


def create_vllm_config_48g() -> VLLMConfig:
    """48G vLLM 优化配置"""
    vcfg = VLLMConfig()
    vcfg.gpu_memory_utilization = 0.92
    vcfg.max_num_seqs = 256
    vcfg.max_num_batched_tokens = 8192
    return vcfg


# 默认配置
config = ServingConfig()
config_48g = create_config_48g()
vllm_config = VLLMConfig()
sglang_config = SGLangConfig()
ollama_config = OllamaConfig()
benchmark_config = BenchmarkConfig()
