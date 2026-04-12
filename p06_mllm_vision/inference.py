"""
p06 MLLM 多模态视觉微调 - 推理脚本

加载微调后的 Qwen2.5-VL 模型，进行多模态推理：
1. 图像描述（Image Captioning）
2. 视觉问答（VQA）
3. OCR 文字识别
4. 图像推理

使用方式:
    cd p06_mllm_vision
    python inference.py
    python inference.py --model-path outputs/mllm_vision/final
    python inference.py --image path/to/image.jpg --question "描述这张图片"
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config


# ============================================================
# 测试用例
# ============================================================
TEST_CASES = [
    {
        "name": "图像描述",
        "question": "请详细描述这张图片中的内容。",
        "description": "测试模型的基础视觉理解和描述能力",
    },
    {
        "name": "视觉问答 (VQA)",
        "question": "图片中有几个人？他们在做什么？",
        "description": "测试模型的计数和行为理解能力",
    },
    {
        "name": "OCR 文字识别",
        "question": "请读出图片中的所有文字。",
        "description": "测试模型的 OCR 能力",
    },
    {
        "name": "空间关系推理",
        "question": "描述图片中各个物体的空间位置关系。",
        "description": "测试模型的空间推理能力",
    },
    {
        "name": "细粒度识别",
        "question": "图片中的主要物体是什么？请给出尽可能详细的信息。",
        "description": "测试模型的细粒度识别能力",
    },
]


# ============================================================
# 1. 模型加载
# ============================================================
def load_model(model_path: str):
    """加载 Qwen2.5-VL 模型"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
    
    print(f"  加载模型: {model_path}")
    
    try:
        processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )
        tokenizer = processor.tokenizer
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        processor = None
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    
    return model, tokenizer, processor


# ============================================================
# 2. 多模态推理
# ============================================================
def generate_with_image(
    model,
    tokenizer,
    processor,
    image_path: str,
    question: str,
    max_new_tokens: int = 512,
) -> str:
    """
    使用图像和问题进行多模态推理。
    
    流程：
    1. 加载并预处理图像
    2. 构造 Qwen2.5-VL 的消息格式
    3. 编码为模型输入
    4. 生成回答
    """
    import torch
    from PIL import Image
    
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 构造消息
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": question},
            ]
        }
    ]
    
    # 使用 processor 编码（如果可用）
    if processor is not None:
        try:
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[image],
                padding=True, return_tensors="pt"
            ).to(model.device)
        except Exception:
            # Fallback: 仅文本模式
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
    else:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    # 生成
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # 解码（只取新生成部分）
    input_len = inputs["input_ids"].shape[1]
    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    
    return response


# ============================================================
# 3. 纯文本推理（无图像时的 fallback）
# ============================================================
def generate_text_only(model, tokenizer, question: str, max_new_tokens: int = 512) -> str:
    """纯文本推理（用于对比）"""
    import torch
    
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ============================================================
# 4. 创建测试图像
# ============================================================
def create_test_image(save_path: str = "data/test_image.jpg"):
    """创建一张简单的测试图像（带文字和图形）"""
    from PIL import Image, ImageDraw, ImageFont
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    img = Image.new("RGB", (640, 480), color=(240, 248, 255))
    draw = ImageDraw.Draw(img)
    
    # 绘制简单图形
    draw.rectangle([50, 50, 200, 200], fill=(65, 105, 225), outline=(0, 0, 139), width=3)
    draw.ellipse([250, 80, 400, 230], fill=(220, 20, 60), outline=(139, 0, 0), width=3)
    draw.polygon([(500, 50), (590, 200), (410, 200)], fill=(34, 139, 34), outline=(0, 100, 0), width=3)
    
    # 添加文字
    try:
        font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", 28)
    except Exception:
        font = ImageFont.load_default()
    
    draw.text((50, 280), "MLLM 视觉理解测试", fill=(0, 0, 0), font=font)
    draw.text((50, 330), "蓝色方块 · 红色圆形 · 绿色三角", fill=(100, 100, 100), font=font)
    draw.text((50, 380), "Qwen2.5-VL-2B", fill=(65, 105, 225), font=font)
    draw.text((50, 420), "2024 多模态微调实验", fill=(100, 100, 100), font=font)
    
    img.save(save_path, quality=95)
    print(f"  ✅ 测试图像已保存: {save_path}")
    return save_path


# ============================================================
# 5. 主流程
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MLLM 多模态推理")
    parser.add_argument("--model-path", type=str, default=None,
                       help="模型路径（默认: 原始 Qwen2.5-VL-2B）")
    parser.add_argument("--image", type=str, default=None,
                       help="输入图像路径")
    parser.add_argument("--question", type=str, default=None,
                       help="输入问题")
    parser.add_argument("--base-model", type=str, default=None)
    args = parser.parse_args()
    
    model_path = args.model_path or config.model_name
    
    print("=" * 60)
    print("  p06 MLLM 多模态视觉微调 - 推理测试")
    print("=" * 60)
    
    # 加载模型
    print("\n加载模型...")
    model, tokenizer, processor = load_model(model_path)
    
    # 单张图像推理模式
    if args.image and args.question:
        print(f"\n{'─'*60}")
        print(f"  🖼️ 图像: {args.image}")
        print(f"  ❓ 问题: {args.question}")
        print(f"{'─'*60}")
        
        start = time.time()
        response = generate_with_image(
            model, tokenizer, processor, args.image, args.question
        )
        elapsed = time.time() - start
        
        print(f"  🤖 回答: {response}")
        print(f"  ⏱️ 耗时: {elapsed:.2f}s")
        return
    
    # 批量测试模式
    test_image = args.image or "data/test_image.jpg"
    
    if not os.path.exists(test_image):
        print("\n创建测试图像...")
        test_image = create_test_image(test_image)
    
    print(f"\n测试图像: {test_image}")
    
    for case in TEST_CASES:
        print(f"\n{'─'*60}")
        print(f"  📋 测试: {case['name']}")
        print(f"  📝 说明: {case['description']}")
        print(f"  ❓ 问题: {case['question']}")
        print(f"{'─'*60}")
        
        start = time.time()
        try:
            response = generate_with_image(
                model, tokenizer, processor, test_image, case["question"]
            )
        except Exception as e:
            response = f"[推理失败: {e}]"
        elapsed = time.time() - start
        
        display = response[:500] + ("..." if len(response) > 500 else "")
        print(f"  🤖 回答: {display}")
        print(f"  ⏱️ 耗时: {elapsed:.2f}s")
    
    print(f"\n{'='*60}")
    print("  推理测试完成！")
    print("  提示: 使用 --image 和 --question 参数可测试自定义图像")
    print("=" * 60)


if __name__ == "__main__":
    main()
