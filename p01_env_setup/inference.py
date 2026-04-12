"""
p01 环境搭建 - 交互式推理脚本

加载 Qwen2.5-0.5B 模型进行交互式推理，
让用户亲身体验基座模型的能力和局限。

支持两种模式:
1. Base 模型（续写模式）—— 只会续写文本，不会对话
2. Instruct 模型（对话模式）—— 能理解指令并回答

通过对比两种模型，直观感受 SFT 的作用。

使用方式:
    cd p01_env_setup
    python inference.py                    # 默认 base 模型
    python inference.py --chat             # 使用 Instruct 模型
    python inference.py --interactive      # 交互式对话
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import EnvConfig, config


# ============================================================
# 1. 模型加载
# ============================================================
def load_model(model_name: str, dtype: str = "bf16"):
    """
    加载模型和 tokenizer。
    
    Args:
        model_name: 模型名称
        dtype: 加载精度 ("bf16" / "fp16" / "fp32")
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    dtype_map = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)
    
    print(f"加载模型: {model_name} ({dtype})")
    start = time.time()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    
    elapsed = time.time() - start
    
    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  加载耗时: {elapsed:.1f}s")
    print(f"  参数量:   {num_params:.2f}B")
    print(f"  设备:     {model.device}")
    print(f"  精度:     {torch_dtype}")
    
    return model, tokenizer


# ============================================================
# 2. 文本生成
# ============================================================
def generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    do_sample: bool = True,
    stream: bool = False,
) -> str:
    """
    生成文本。
    
    采样参数详解:
    - temperature: 温度参数，控制输出的随机程度
      - 0.0: 完全确定性（贪心），每次选概率最高的 token
      - 0.7: 适中，保持多样性的同时不会太离谱
      - 1.0: 标准采样
      - >1.0: 更随机、更有创造力
    
    - top_p (nucleus sampling): 核采样
      - 只从累积概率达到 top_p 的最小 token 集合中采样
      - 0.9: 保留概率前 90% 的 token
      - 比固定 top_k 更灵活
    
    - top_k: 只从概率前 k 个 token 中采样
      - 50: 保留前 50 个候选
    """
    import torch
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    start = time.time()
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            do_sample=do_sample,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    elapsed = time.time() - start
    
    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    num_tokens = len(new_tokens)
    tokens_per_sec = num_tokens / elapsed if elapsed > 0 else 0
    
    return response.strip(), {
        "tokens": num_tokens,
        "time_s": round(elapsed, 2),
        "tokens_per_sec": round(tokens_per_sec, 1),
    }


# ============================================================
# 3. 预设测试用例
# ============================================================
def run_preset_tests(model, tokenizer, is_chat: bool = False):
    """运行预设的测试用例，展示模型能力"""
    
    test_cases = [
        ("中文续写", "中国是一个"),
        ("知识问答", "太阳系中最大的行星是"),
        ("故事创作", "从前有一个小女孩，她住在森林边的小木屋里"),
        ("代码能力", "def fibonacci(n):\n    "),
        ("数学推理", "2+3=5, 4+5=9, 7+8="),
        ("英文能力", "The capital of France is"),
    ]
    
    if is_chat:
        # Chat 模型使用对话格式
        test_cases = [
            ("中文对话", "你好，请自我介绍一下"),
            ("知识问答", "请解释什么是量子纠缠"),
            ("故事创作", "请写一个50字的小故事"),
            ("代码能力", "请用Python实现斐波那契数列"),
            ("数学推理", "小明有5个苹果，给了小红2个，又买了3个，现在有几个？"),
            ("英文能力", "Please explain what is machine learning in one sentence"),
        ]
    
    for name, prompt in test_cases:
        print(f"\n{'─'*60}")
        print(f"  📝 {name}")
        print(f"  输入: {prompt}")
        
        response, stats = generate(
            model, tokenizer, prompt,
            max_new_tokens=200,
            temperature=0.7,
        )
        
        # 截断显示
        display = response[:300] + "..." if len(response) > 300 else response
        print(f"  输出: {display}")
        print(f"  统计: {stats['tokens']} tokens | {stats['time_s']}s | {stats['tokens_per_sec']} tok/s")


# ============================================================
# 4. 交互式对话
# ============================================================
def interactive_chat(model, tokenizer, is_chat: bool = False):
    """交互式对话模式"""
    
    print("\n" + "=" * 60)
    print("  🤖 交互式推理（输入 'quit' 退出）")
    print("=" * 60)
    
    mode = "对话" if is_chat else "续写"
    print(f"  模式: {mode}")
    print(f"  提示: Base 模型只会续写文本，不会回答问题")
    print()
    
    while True:
        try:
            prompt = input("📝 你: ").strip()
            if prompt.lower() in ("quit", "exit", "q"):
                print("  👋 再见！")
                break
            if not prompt:
                continue
            
            response, stats = generate(
                model, tokenizer, prompt,
                max_new_tokens=300,
                temperature=0.7,
            )
            
            print(f"🤖 模型: {response}")
            print(f"   ({stats['tokens']} tokens, {stats['time_s']}s, {stats['tokens_per_sec']} tok/s)")
            print()
        
        except KeyboardInterrupt:
            print("\n  👋 再见！")
            break


# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="交互式推理")
    parser.add_argument("--chat", action="store_true",
                       help="使用 Instruct (Chat) 模型")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="交互式对话模式")
    parser.add_argument("--model", type=str, default=None,
                       help="自定义模型名称")
    parser.add_argument("--dtype", type=str, default="bf16",
                       choices=["bf16", "fp16", "fp32"])
    args = parser.parse_args()
    
    # 选择模型
    if args.model:
        model_name = args.model
    elif args.chat:
        model_name = config.chat_model_name
    else:
        model_name = config.base_model_name
    
    print("=" * 60)
    print("  实践大模型 - 模型推理体验")
    print("=" * 60)
    
    model, tokenizer = load_model(model_name, args.dtype)
    
    if args.interactive:
        interactive_chat(model, tokenizer, is_chat=args.chat)
    else:
        print(f"\n{'='*60}")
        print(f"  预设测试用例")
        print(f"{'='*60}")
        run_preset_tests(model, tokenizer, is_chat=args.chat)
        
        print(f"\n\n{'='*60}")
        print("  💡 提示: 运行 python inference.py --interactive 进入交互模式")
        print("  💡 提示: 运行 python inference.py --chat 对比 Instruct 模型效果")
        print("=" * 60)


if __name__ == "__main__":
    main()
