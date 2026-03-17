#!/usr/bin/env python3
"""
阶段 5: 量化部署 - 模型优化与部署

学习目标：
1. 理解模型量化的原理
2. 掌握 4bit/8bit 量化技术
3. 实现模型合并和导出
4. 学习推理优化技巧
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
import os

def load_base_model_with_quantization(model_path: str, quantization: str = "none"):
    """加载基础模型（可选量化）"""
    print(f"📥 Loading model with {quantization} quantization...")

    if quantization == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
    elif quantization == "8bit":
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            trust_remote_code=True,
            device_map="auto"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    return model

def merge_lora_weights(base_model_path: str, lora_path: str, output_path: str):
    """合并 LoRA 权重到基础模型"""
    print("🔄 Merging LoRA weights...")

    # 加载基础模型
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    # 加载 LoRA 权重
    model = PeftModel.from_pretrained(base_model, lora_path)

    # 合并权重
    print("🔧 Merging...")
    merged_model = model.merge_and_unload()

    # 保存合并后的模型
    print(f"💾 Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("✅ Merge complete!")
    return merged_model, tokenizer

def compare_model_sizes(base_path: str, lora_path: str, merged_path: str = None):
    """对比模型大小"""
    print("\n📊 Model Size Comparison:")

    def get_dir_size(path):
        if not os.path.exists(path):
            return 0
        total = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total += os.path.getsize(fp)
        return total / (1024 ** 3)  # GB

    base_size = get_dir_size(base_path)
    lora_size = get_dir_size(lora_path)

    print(f"   Base model: {base_size:.2f} GB")
    print(f"   LoRA weights: {lora_size:.2f} GB ({lora_size/base_size*100:.1f}% of base)")

    if merged_path and os.path.exists(merged_path):
        merged_size = get_dir_size(merged_path)
        print(f"   Merged model: {merged_size:.2f} GB")

def benchmark_inference(model, tokenizer, prompt: str, num_runs: int = 5):
    """推理性能测试"""
    import time

    print(f"\n⏱️  Benchmarking inference ({num_runs} runs)...")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # 预热
    with torch.no_grad():
        model.generate(**inputs, max_new_tokens=50)

    # 测试
    times = []
    for i in range(num_runs):
        start = time.time()
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)
        end = time.time()
        times.append(end - start)

    avg_time = sum(times) / len(times)
    tokens_per_sec = 50 / avg_time

    print(f"   Average time: {avg_time:.2f}s")
    print(f"   Tokens/sec: {tokens_per_sec:.1f}")

    return avg_time

def main():
    print("=" * 60)
    print("阶段 5: 量化部署 - 模型优化与部署")
    print("=" * 60)

    base_model_path = "/app/models/Qwen2.5-0.5B"
    lora_path = "/app/outputs/final"
    merged_path = "/app/outputs/merged"

    # 1. 对比模型大小
    print("\n📊 步骤 1: 对比模型大小")
    if os.path.exists(lora_path):
        compare_model_sizes(base_model_path, lora_path)
    else:
        print("⚠️  LoRA 模型不存在，跳过对比")

    # 2. 合并 LoRA 权重
    if os.path.exists(lora_path):
        print("\n🔄 步骤 2: 合并 LoRA 权重")
        merged_model, tokenizer = merge_lora_weights(
            base_model_path,
            lora_path,
            merged_path
        )
        compare_model_sizes(base_model_path, lora_path, merged_path)
    else:
        print("\n⚠️  步骤 2: LoRA 模型不存在，使用基础模型")
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        merged_model = load_base_model_with_quantization(base_model_path, "none")

    # 3. 测试不同量化方式
    print("\n🔬 步骤 3: 测试不同量化方式")

    test_prompt = "请介绍一下 LoRA 微调技术。"

    print("\n--- FP16 (无量化) ---")
    fp16_model = load_base_model_with_quantization(base_model_path, "none")
    benchmark_inference(fp16_model, tokenizer, test_prompt)

    print("\n--- 8bit 量化 ---")
    int8_model = load_base_model_with_quantization(base_model_path, "8bit")
    benchmark_inference(int8_model, tokenizer, test_prompt)

    print("\n--- 4bit 量化 ---")
    int4_model = load_base_model_with_quantization(base_model_path, "4bit")
    benchmark_inference(int4_model, tokenizer, test_prompt)

    # 4. 推理示例
    print("\n💬 步骤 4: 推理示例")
    inputs = tokenizer(test_prompt, return_tensors="pt").to(merged_model.device)
    with torch.no_grad():
        outputs = merged_model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Prompt: {test_prompt}")
    print(f"   Response: {response}")

    print("\n✅ 阶段 5 完成！")
    print("\n📚 关键概念：")
    print("   - 量化: 降低模型精度以减少显存和加速推理")
    print("   - 4bit: 最激进的量化，显存占用最小")
    print("   - 8bit: 平衡量化，精度损失小")
    print("   - 合并权重: 将 LoRA 权重合并到基础模型")
    print("   - 推理优化: 量化、批处理、KV cache")

if __name__ == "__main__":
    main()
