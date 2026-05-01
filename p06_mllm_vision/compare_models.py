#!/usr/bin/env python3
"""
对比不同微调模型的推理效果
"""
import os, sys, time
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config
from inference import load_model, generate_with_image

# 对比的模型
MODELS = {
    "原始模型": config.model_name,
    "LoRA微调": "outputs/mllm_vision/final",
    "全参微调": "full_output/final",
}

# 测试问题
TEST_CASES = [
    {"image": "data/images/000000294160.jpg", "question": "请详细描述这张图片中的内容。"},
    {"image": "data/images/000000101702.jpg", "question": "图片中的动物在做什么？"},
]


def main():
    print("=" * 70)
    print("  多模态微调效果对比")
    print("=" * 70)
    
    results = {}
    
    for name, path in MODELS.items():
        if not os.path.exists(path) and "/" not in path:
            print(f"\n⚠️  跳过 [{name}]: 路径不存在 {path}")
            continue
        
        print(f"\n{'─'*70}")
        print(f"  加载 [{name}]: {path}")
        print(f"{'─'*70}")
        
        try:
            model, tokenizer, processor = load_model(path)
        except Exception as e:
            print(f"  ❌ 加载失败: {e}")
            continue
        
        results[name] = []
        
        for i, case in enumerate(TEST_CASES):
            print(f"\n  📷 测试 {i+1}: {case['question'][:40]}...")
            start = time.time()
            try:
                response = generate_with_image(
                    model, tokenizer, processor,
                    case["image"], case["question"],
                    max_new_tokens=256,
                )
            except Exception as e:
                response = f"[推理失败: {e}]"
            elapsed = time.time() - start
            
            results[name].append({
                "question": case["question"],
                "response": response,
                "time": elapsed,
            })
            print(f"  ⏱️ {elapsed:.1f}s")
            print(f"  🤖 {response[:200]}...")
        
        # 释放显存
        import torch
        del model
        torch.cuda.empty_cache()
    
    # 对比汇总
    print(f"\n\n{'='*70}")
    print("  对比汇总")
    print(f"{'='*70}")
    
    for i, case in enumerate(TEST_CASES):
        print(f"\n{'─'*70}")
        print(f"  问题 {i+1}: {case['question']}")
        print(f"  图片: {case['image']}")
        print(f"{'─'*70}")
        
        for name in results:
            r = results[name][i]
            print(f"\n  [{name}] ({r['time']:.1f}s):")
            print(f"    {r['response'][:300]}")
    
    print(f"\n{'='*70}")
    print("  对比完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
