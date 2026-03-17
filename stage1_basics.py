#!/usr/bin/env python3
"""
阶段 1: 微调基础 - 了解 LLM 微调的基本概念

学习目标：
1. 理解什么是 LLM 微调
2. 了解预训练 vs 微调的区别
3. 认识 Qwen2.5-0.5B 模型
4. 掌握模型加载和基本推理
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model():
    """加载 Qwen2.5-0.5B 模型"""
    model_path = "/app/models/Qwen2.5-0.5B"

    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    print("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("✅ Model loaded successfully!")
    print(f"   Model parameters: {model.num_parameters() / 1e6:.2f}M")
    print(f"   Device: {model.device}")

    return tokenizer, model

def simple_inference(tokenizer, model, prompt: str):
    """简单推理示例"""
    print(f"\n💬 Prompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"🤖 Response: {response}")

    return response

def main():
    print("=" * 60)
    print("阶段 1: 微调基础 - 模型加载与推理")
    print("=" * 60)

    # 加载模型
    tokenizer, model = load_model()

    # 测试推理
    prompts = [
        "你好，请介绍一下自己。",
        "什么是大语言模型？",
        "解释一下什么是微调。"
    ]

    for prompt in prompts:
        simple_inference(tokenizer, model, prompt)
        print()

    print("✅ 阶段 1 完成！")
    print("\n📚 关键概念：")
    print("   - 预训练模型：在大规模数据上训练的基础模型")
    print("   - 微调：在特定任务数据上继续训练，适应特定领域")
    print("   - Qwen2.5-0.5B：5亿参数的中文友好小模型")
    print("   - 推理：使用模型生成文本的过程")

if __name__ == "__main__":
    main()
