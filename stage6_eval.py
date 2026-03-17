#!/usr/bin/env python3
"""
阶段 6: 模型评估 - 评估微调效果

学习目标：
1. 理解模型评估指标
2. 掌握自动评估方法
3. 实现人工评估流程
4. 对比微调前后效果
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from typing import List, Dict
import numpy as np

def load_model(model_path: str, lora_path: str = None):
    """加载模型"""
    print(f"📥 Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if lora_path:
        print(f"📥 Loading LoRA weights from {lora_path}...")
        model = PeftModel.from_pretrained(model, lora_path)

    return model, tokenizer

def calculate_perplexity(model, tokenizer, texts: List[str]) -> float:
    """计算困惑度"""
    print("📊 Calculating perplexity...")

    total_loss = 0
    total_tokens = 0

    model.eval()
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss.item()
            num_tokens = inputs['input_ids'].size(1)

            total_loss += loss * num_tokens
            total_tokens += num_tokens

    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)

    return perplexity

def generate_responses(model, tokenizer, prompts: List[str]) -> List[str]:
    """生成回复"""
    responses = []

    model.eval()
    with torch.no_grad():
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)

    return responses

def evaluate_quality(responses: List[str]) -> Dict[str, float]:
    """评估回复质量"""
    print("📊 Evaluating response quality...")

    metrics = {
        'avg_length': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'repetition_rate': 0
    }

    lengths = []
    repetitions = 0

    for response in responses:
        length = len(response)
        lengths.append(length)

        # 检测重复（简单方法：检查连续重复的短语）
        words = response.split()
        if len(words) > 10:
            for i in range(len(words) - 5):
                phrase = ' '.join(words[i:i+3])
                if response.count(phrase) > 1:
                    repetitions += 1
                    break

    if lengths:
        metrics['avg_length'] = sum(lengths) / len(lengths)
        metrics['min_length'] = min(lengths)
        metrics['max_length'] = max(lengths)
        metrics['repetition_rate'] = repetitions / len(responses)

    return metrics

def compare_models(base_model_path: str, finetuned_model_path: str, test_prompts: List[str]):
    """对比基础模型和微调模型"""
    print("=" * 60)
    print("对比基础模型 vs 微调模型")
    print("=" * 60)

    # 加载基础模型
    print("\n--- 基础模型 ---")
    base_model, tokenizer = load_model(base_model_path)
    base_responses = generate_responses(base_model, tokenizer, test_prompts)

    # 加载微调模型
    print("\n--- 微调模型 ---")
    if finetuned_model_path.endswith('.safetensors') or '/checkpoint-' in finetuned_model_path:
        # LoRA 模式
        finetuned_model, _ = load_model(base_model_path, finetuned_model_path)
    else:
        # 合并模式
        finetuned_model, _ = load_model(finetuned_model_path)

    finetuned_responses = generate_responses(finetuned_model, tokenizer, test_prompts)

    # 对比结果
    print("\n" + "=" * 60)
    print("对比结果")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts):
        print(f"\n【测试 {i+1}】")
        print(f"Prompt: {prompt}")
        print(f"\n基础模型回复:")
        print(base_responses[i])
        print(f"\n微调模型回复:")
        print(finetuned_responses[i])
        print("-" * 60)

    # 质量评估
    print("\n📊 质量评估:")
    base_metrics = evaluate_quality(base_responses)
    finetuned_metrics = evaluate_quality(finetuned_responses)

    print("\n基础模型:")
    for key, value in base_metrics.items():
        print(f"   {key}: {value:.2f}")

    print("\n微调模型:")
    for key, value in finetuned_metrics.items():
        print(f"   {key}: {value:.2f}")

def load_test_set(test_file: str = "/app/data/processed/val.jsonl") -> List[Dict]:
    """加载测试集"""
    if not os.path.exists(test_file):
        print(f"⚠️  测试文件不存在: {test_file}")
        return []

    test_data = []
    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))

    return test_data[:10]  # 只取前 10 条

def main():
    print("=" * 60)
    print("阶段 6: 模型评估 - 评估微调效果")
    print("=" * 60)

    base_model_path = "/app/models/Qwen2.5-0.5B"
    finetuned_model_path = "/app/outputs/final"  # 或 /app/outputs/merged

    # 测试 prompts
    test_prompts = [
        "请介绍一下 LoRA 微调技术。",
        "如何准备高质量的训练数据？",
        "什么是模型量化？",
        "解释一下学习率的作用。",
        "如何评估模型的效果？"
    ]

    # 1. 对比基础模型和微调模型
    print("\n📊 步骤 1: 对比模型效果")
    import os
    if os.path.exists(finetuned_model_path):
        compare_models(base_model_path, finetuned_model_path, test_prompts)
    else:
        print("⚠️  微调模型不存在，跳过对比")
        print("   请先完成阶段 4 的训练")

        # 只测试基础模型
        print("\n--- 基础模型测试 ---")
        base_model, tokenizer = load_model(base_model_path)
        base_responses = generate_responses(base_model, tokenizer, test_prompts)

        for i, (prompt, response) in enumerate(zip(test_prompts, base_responses)):
            print(f"\n【测试 {i+1}】")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            print("-" * 60)

    # 2. 计算困惑度
    print("\n📊 步骤 2: 计算困惑度")
    test_data = load_test_set()

    if test_data:
        test_texts = [item['input'] for item in test_data if item.get('input')]

        if test_texts:
            base_model, tokenizer = load_model(base_model_path)
            base_ppl = calculate_perplexity(base_model, tokenizer, test_texts)
            print(f"   基础模型困惑度: {base_ppl:.2f}")

            if os.path.exists(finetuned_model_path):
                finetuned_model, _ = load_model(base_model_path, finetuned_model_path)
                finetuned_ppl = calculate_perplexity(finetuned_model, tokenizer, test_texts)
                print(f"   微调模型困惑度: {finetuned_ppl:.2f}")
                print(f"   改进: {(base_ppl - finetuned_ppl) / base_ppl * 100:.1f}%")
    else:
        print("   ⚠️  测试数据不存在，跳过困惑度计算")

    print("\n✅ 阶段 6 完成！")
    print("\n📚 关键概念：")
    print("   - 困惑度 (Perplexity): 越低越好，表示模型预测能力")
    print("   - 自动评估: 使用指标量化模型效果")
    print("   - 人工评估: 主观判断回复质量")
    print("   - A/B 测试: 对比不同模型的效果")

if __name__ == "__main__":
    import os
    main()
