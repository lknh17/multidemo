"""
p08 推理部署 - API 客户端

通过 OpenAI 兼容 API 调用推理服务，支持：
- 流式响应（逐 token 输出）
- 多轮对话（维护上下文）
- 批量推理（并发请求）

适用于 vLLM / SGLang / Ollama 等任何 OpenAI 兼容服务。

使用方式:
    python inference.py
    python inference.py --url http://localhost:8001/v1 --stream
    python inference.py --mode batch
    python inference.py --mode chat
"""

import os
import sys
import json
import time
import argparse
import asyncio
from typing import List, Dict, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import config

try:
    from openai import OpenAI
except ImportError:
    print("请安装 openai SDK: pip install openai")
    sys.exit(1)

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    HAS_AIOHTTP = False


# ============================================================
# 1. 基础推理：单次请求
# ============================================================
def single_inference(
    client: OpenAI,
    model: str,
    prompt: str,
    system_prompt: str = "你是一个有帮助的AI助手。",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """单次推理（非流式）"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    content = response.choices[0].message.content
    usage = response.usage
    print(f"  [Token 用量] prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}")
    return content


# ============================================================
# 2. 流式推理：逐 Token 输出
# ============================================================
def stream_inference(
    client: OpenAI,
    model: str,
    prompt: str,
    system_prompt: str = "你是一个有帮助的AI助手。",
    max_tokens: int = 512,
    temperature: float = 0.7,
) -> str:
    """流式推理（逐 token 输出）"""
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=True,
    )

    full_content = ""
    start_time = time.perf_counter()
    first_token_time = None
    token_count = 0

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            print(delta.content, end="", flush=True)
            full_content += delta.content
            token_count += 1

    end_time = time.perf_counter()
    print()

    # 性能指标
    ttft = (first_token_time - start_time) if first_token_time else 0
    total_time = end_time - start_time
    tps = token_count / total_time if total_time > 0 else 0

    print(f"  [性能] TTFT={ttft:.3f}s  总耗时={total_time:.3f}s  TPS={tps:.1f}  tokens={token_count}")
    return full_content


# ============================================================
# 3. 多轮对话
# ============================================================
def chat_mode(client: OpenAI, model: str, max_tokens: int = 512):
    """交互式多轮对话"""
    print("=" * 50)
    print("  多轮对话模式 (输入 'quit' 退出)")
    print("=" * 50)

    messages = [
        {"role": "system", "content": "你是一个有帮助的AI助手。请用中文回答。"}
    ]

    while True:
        user_input = input("\n🧑 你: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            print("  对话结束。")
            break
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})

        print("🤖 助手: ", end="")
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=True,
        )

        assistant_content = ""
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                print(delta.content, end="", flush=True)
                assistant_content += delta.content
        print()

        messages.append({"role": "assistant", "content": assistant_content})
        print(f"  [上下文长度: {len(messages)} 条消息]")


# ============================================================
# 4. 批量推理
# ============================================================
async def batch_inference_async(
    url: str,
    model: str,
    prompts: List[str],
    max_tokens: int = 256,
    concurrency: int = 4,
) -> List[Dict]:
    """批量异步推理"""
    if not HAS_AIOHTTP:
        print("  ⚠ 批量推理需要 aiohttp，请安装: pip install aiohttp")
        return []

    results = []
    semaphore = asyncio.Semaphore(concurrency)

    async def process_one(idx: int, prompt: str):
        async with semaphore:
            payload = {
                "model": model,
                "messages": [
                    {"role": "system", "content": "你是一个有帮助的AI助手。"},
                    {"role": "user", "content": prompt},
                ],
                "max_tokens": max_tokens,
                "temperature": 0.7,
            }
            start = time.perf_counter()
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        f"{url}/chat/completions",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60),
                    ) as resp:
                        data = await resp.json()
                        elapsed = time.perf_counter() - start
                        content = data["choices"][0]["message"]["content"]
                        return {
                            "index": idx,
                            "prompt": prompt,
                            "response": content,
                            "time": elapsed,
                            "tokens": data.get("usage", {}).get("total_tokens", 0),
                        }
            except Exception as e:
                return {
                    "index": idx,
                    "prompt": prompt,
                    "response": None,
                    "error": str(e),
                    "time": time.perf_counter() - start,
                }

    tasks = [process_one(i, p) for i, p in enumerate(prompts)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda x: x["index"])


def run_batch(url: str, model: str, max_tokens: int = 256, concurrency: int = 4):
    """运行批量推理"""
    prompts = [
        "什么是Transformer架构？用一段话概括。",
        "请用Python实现二分查找。",
        "解释一下什么是KV Cache，为什么它对推理很重要？",
        "请简述 PagedAttention 的核心思想。",
        "什么是continuous batching？它相比static batching有什么优势？",
        "请解释speculative decoding的原理。",
        "vLLM和SGLang的主要区别是什么？",
        "如何评估推理服务的性能？需要关注哪些指标？",
    ]

    print("=" * 60)
    print(f"  批量推理 ({len(prompts)} 个请求, 并发={concurrency})")
    print("=" * 60)

    start = time.perf_counter()
    results = asyncio.run(batch_inference_async(url, model, prompts, max_tokens, concurrency))
    total_time = time.perf_counter() - start

    # 打印结果
    success_count = 0
    for r in results:
        status = "✓" if r.get("response") else "✗"
        if r.get("response"):
            success_count += 1
        preview = (r.get("response") or r.get("error", ""))[:80]
        print(f"\n  [{status}] #{r['index']} ({r['time']:.2f}s)")
        print(f"      Q: {r['prompt'][:60]}...")
        print(f"      A: {preview}...")

    print(f"\n  汇总: {success_count}/{len(prompts)} 成功, 总耗时 {total_time:.2f}s")
    print(f"  吞吐: {success_count / total_time:.2f} req/s")

    # 保存结果
    os.makedirs(config.output_dir, exist_ok=True)
    path = os.path.join(config.output_dir, "batch_results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  结果已保存: {path}")


# ============================================================
# 5. 演示入口
# ============================================================
def demo_single_and_stream(client: OpenAI, model: str):
    """演示单次推理和流式推理"""
    test_prompts = [
        "请用简单的语言解释什么是注意力机制。",
        "用Python写一个冒泡排序，并解释时间复杂度。",
    ]

    # 非流式
    print("\n" + "=" * 60)
    print("  1. 非流式推理")
    print("=" * 60)
    for prompt in test_prompts[:1]:
        print(f"\n  Q: {prompt}")
        start = time.perf_counter()
        answer = single_inference(client, model, prompt)
        elapsed = time.perf_counter() - start
        print(f"  A: {answer[:200]}...")
        print(f"  耗时: {elapsed:.2f}s")

    # 流式
    print("\n" + "=" * 60)
    print("  2. 流式推理")
    print("=" * 60)
    for prompt in test_prompts:
        print(f"\n  Q: {prompt}")
        print("  A: ", end="")
        stream_inference(client, model, prompt)


def main():
    parser = argparse.ArgumentParser(description="推理服务 API 客户端")
    parser.add_argument("--url", type=str, default=f"http://localhost:{config.vllm_port}/v1", help="服务地址")
    parser.add_argument("--model", type=str, default=config.model_name, help="模型名称")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "stream", "chat", "batch"], help="运行模式")
    parser.add_argument("--prompt", type=str, default=None, help="自定义 prompt")
    parser.add_argument("--max-tokens", type=int, default=512, help="最大生成 token 数")
    parser.add_argument("--concurrency", type=int, default=4, help="批量推理并发数")
    args = parser.parse_args()

    print(f"\n🔗 连接服务: {args.url}")
    print(f"📦 模型: {args.model}\n")

    if args.mode == "batch":
        run_batch(args.url, args.model, args.max_tokens, args.concurrency)
        return

    client = OpenAI(base_url=args.url, api_key="EMPTY")

    if args.mode == "chat":
        chat_mode(client, args.model, args.max_tokens)
    elif args.mode == "stream":
        prompt = args.prompt or "请解释什么是KV Cache，以及它在LLM推理中的作用。"
        print(f"  Q: {prompt}")
        print("  A: ", end="")
        stream_inference(client, args.model, prompt, max_tokens=args.max_tokens)
    else:
        demo_single_and_stream(client, args.model)


if __name__ == "__main__":
    main()
