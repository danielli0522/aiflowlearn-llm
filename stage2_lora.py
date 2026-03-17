#!/usr/bin/env python3
"""
阶段 2: LoRA 原理 - 参数高效微调技术

学习目标：
1. 理解 LoRA 的核心思想
2. 掌握 LoRA 的关键参数
3. 实现 LoRA 模型配置
4. 对比全量微调 vs LoRA 微调
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType

def load_base_model():
    """加载基础模型"""
    model_path = "/app/models/Qwen2.5-0.5B"

    print("📥 Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return tokenizer, model

def apply_lora(model, r=8, lora_alpha=16, lora_dropout=0.05):
    """应用 LoRA 配置"""
    print(f"\n🔧 Applying LoRA configuration...")
    print(f"   r (rank): {r}")
    print(f"   lora_alpha: {lora_alpha}")
    print(f"   lora_dropout: {lora_dropout}")

    lora_config = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    lora_model = get_peft_model(model, lora_config)

    return lora_model, lora_config

def print_trainable_parameters(model):
    """打印可训练参数统计"""
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"\n📊 Parameter Statistics:")
    print(f"   Total parameters: {all_param / 1e6:.2f}M")
    print(f"   Trainable parameters: {trainable_params / 1e6:.2f}M")
    print(f"   Trainable ratio: {100 * trainable_params / all_param:.2f}%")

def compare_lora_configs():
    """对比不同 LoRA 配置"""
    print("\n" + "=" * 60)
    print("对比不同 LoRA 配置")
    print("=" * 60)

    _, base_model = load_base_model()

    configs = [
        {"r": 4, "lora_alpha": 8, "name": "小配置 (r=4)"},
        {"r": 8, "lora_alpha": 16, "name": "标准配置 (r=8)"},
        {"r": 16, "lora_alpha": 32, "name": "大配置 (r=16)"},
    ]

    for config in configs:
        print(f"\n--- {config['name']} ---")
        lora_model, _ = apply_lora(
            base_model,
            r=config['r'],
            lora_alpha=config['lora_alpha']
        )
        print_trainable_parameters(lora_model)

def main():
    print("=" * 60)
    print("阶段 2: LoRA 原理 - 参数高效微调")
    print("=" * 60)

    # 加载基础模型
    tokenizer, base_model = load_base_model()

    print("\n--- 基础模型参数 ---")
    print_trainable_parameters(base_model)

    # 应用 LoRA
    lora_model, lora_config = apply_lora(base_model)

    print("\n--- LoRA 模型参数 ---")
    print_trainable_parameters(lora_model)

    # 打印 LoRA 模型结构
    print("\n📋 LoRA Model Structure:")
    print(lora_model)

    # 对比不同配置
    compare_lora_configs()

    print("\n✅ 阶段 2 完成！")
    print("\n📚 关键概念：")
    print("   - LoRA: Low-Rank Adaptation，低秩适应")
    print("   - r (rank): 低秩矩阵的秩，越大参数越多")
    print("   - lora_alpha: 缩放因子，控制 LoRA 权重的影响")
    print("   - target_modules: 应用 LoRA 的目标层（通常是注意力层）")
    print("   - 参数效率: LoRA 只训练 <1% 的参数，大幅降低成本")

if __name__ == "__main__":
    main()
