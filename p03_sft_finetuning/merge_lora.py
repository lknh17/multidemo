"""
p03 SFT 指令微调 - LoRA 权重合并

将 LoRA adapter 权重合并回基座模型，生成完整模型。
合并后的模型可以直接用于推理，不需要 PEFT 库。

合并原理:
    W_merged = W_base + B × A × (alpha / r)
    其中 B: [d, r], A: [r, k] 是 LoRA 低秩矩阵

使用方式:
    cd p03_sft_finetuning

    # 合并 LoRA
    python merge_lora.py --adapter-path outputs/sft_lora/final

    # 合并 QLoRA（需要先反量化）
    python merge_lora.py --adapter-path outputs/sft_qlora/final

    # 指定输出目录
    python merge_lora.py --adapter-path outputs/sft_lora/final --output-dir outputs/merged
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import config


# ============================================================
# 1. 合并 LoRA 权重
# ============================================================
def merge_lora_weights(
    adapter_path: str,
    base_model_name: str = None,
    output_dir: str = None,
    save_format: str = "safetensors",
):
    """
    将 LoRA adapter 合并回基座模型。
    
    合并步骤:
    1. 加载基座模型
    2. 加载 LoRA adapter
    3. 调用 merge_and_unload() 将 LoRA 权重融合进基座
    4. 保存合并后的完整模型
    
    Args:
        adapter_path: LoRA adapter 路径
        base_model_name: 基座模型名称（None 则从 adapter 配置读取）
        output_dir: 输出目录
        save_format: 保存格式（safetensors / pytorch）
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel, PeftConfig
    
    # 读取 adapter 的配置（获取基座模型名称）
    peft_config = PeftConfig.from_pretrained(adapter_path)
    base_name = base_model_name or peft_config.base_model_name_or_path
    
    print("=" * 60)
    print("  p03 LoRA 权重合并")
    print("=" * 60)
    print(f"\n  基座模型:   {base_name}")
    print(f"  Adapter:    {adapter_path}")
    print(f"  LoRA rank:  {peft_config.r}")
    print(f"  LoRA alpha: {peft_config.lora_alpha}")
    
    # ---- 加载基座模型 ----
    print("\n[1/4] 加载基座模型...")
    tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=True)
    
    model = AutoModelForCausalLM.from_pretrained(
        base_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="cpu",  # 在 CPU 上合并，避免 OOM
    )
    
    # ---- 加载 LoRA Adapter ----
    print("[2/4] 加载 LoRA adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)
    
    # ---- 合并权重 ----
    print("[3/4] 合并权重 (merge_and_unload)...")
    # merge_and_unload() 将 LoRA 矩阵 B×A 乘以 scaling factor，
    # 加到原始权重上，然后移除 LoRA 层
    model = model.merge_and_unload()
    
    # 验证：合并后不应该有 LoRA 相关的参数
    lora_params = [n for n, p in model.named_parameters() if "lora" in n.lower()]
    if lora_params:
        print(f"  ⚠️ 仍有 LoRA 参数: {lora_params[:5]}")
    else:
        print("  ✅ LoRA 参数已完全合并")
    
    # ---- 保存合并后的模型 ----
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(adapter_path), "..", "merged")
    
    print(f"[4/4] 保存合并模型到: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    if save_format == "safetensors":
        model.save_pretrained(output_dir, safe_serialization=True)
    else:
        model.save_pretrained(output_dir, safe_serialization=False)
    
    # 保存 tokenizer
    tokenizer.save_pretrained(output_dir)
    
    # 打印文件列表
    files = os.listdir(output_dir)
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in files)
    
    print(f"\n  输出文件:")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(output_dir, f))
        print(f"    {f}: {size/1024/1024:.1f} MB")
    print(f"  总大小: {total_size/1024/1024:.1f} MB")
    
    print("\n" + "=" * 60)
    print("  ✅ 合并完成！")
    print(f"  合并模型路径: {output_dir}")
    print("  对比推理: python inference.py --merged-path " + output_dir)
    print("=" * 60)


# ============================================================
# 2. 主入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="p03 LoRA 权重合并")
    parser.add_argument("--adapter-path", type=str, required=True,
                       help="LoRA adapter 路径")
    parser.add_argument("--base-model", type=str, default=None,
                       help="基座模型名称（默认从 adapter 配置读取）")
    parser.add_argument("--output-dir", type=str, default=None,
                       help="合并模型输出目录")
    parser.add_argument("--save-format", type=str, default="safetensors",
                       choices=["safetensors", "pytorch"],
                       help="保存格式")
    args = parser.parse_args()
    
    merge_lora_weights(
        adapter_path=args.adapter_path,
        base_model_name=args.base_model,
        output_dir=args.output_dir,
        save_format=args.save_format,
    )


if __name__ == "__main__":
    main()
