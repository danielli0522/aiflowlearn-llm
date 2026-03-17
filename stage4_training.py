#!/usr/bin/env python3
"""
阶段 4: 训练优化 - 执行 LoRA 微调训练

学习目标：
1. 理解训练超参数的作用
2. 掌握训练监控和日志
3. 实现完整的训练流程
4. 学习训练优化技巧
"""

import os
import yaml
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

def load_config(config_path: str = "/app/config/base_config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def load_model_and_tokenizer(config):
    """加载模型和分词器"""
    model_path = config['model']['local_path']

    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        padding_side='right'
    )

    # 设置 pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("📥 Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    return model, tokenizer

def apply_lora(model, config):
    """应用 LoRA 配置"""
    lora_cfg = config['lora']

    print("🔧 Applying LoRA...")
    lora_config = LoraConfig(
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['lora_alpha'],
        lora_dropout=lora_cfg['lora_dropout'],
        target_modules=lora_cfg['target_modules'],
        bias=lora_cfg['bias'],
        task_type=TaskType.CAUSAL_LM
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model

def load_and_prepare_data(config, tokenizer):
    """加载和准备数据"""
    data_cfg = config['data']

    print("📂 Loading dataset...")
    dataset = load_dataset(
        'json',
        data_files={
            'train': data_cfg['train_file'],
            'validation': data_cfg['val_file']
        }
    )

    print(f"   Train samples: {len(dataset['train'])}")
    print(f"   Validation samples: {len(dataset['validation'])}")

    def preprocess_function(examples):
        """预处理函数"""
        # 构造完整的输入文本
        texts = []
        for instruction, input_text, output in zip(
            examples['instruction'],
            examples['input'],
            examples['output']
        ):
            if input_text:
                text = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n{output}"
            else:
                text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
            texts.append(text)

        # Tokenize
        model_inputs = tokenizer(
            texts,
            max_length=config['model']['max_length'],
            truncation=True,
            padding=False
        )

        # 设置 labels（用于计算 loss）
        model_inputs['labels'] = model_inputs['input_ids'].copy()

        return model_inputs

    print("🔄 Preprocessing dataset...")
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=dataset['train'].column_names
    )

    return tokenized_dataset

def create_trainer(model, tokenizer, dataset, config):
    """创建 Trainer"""
    train_cfg = config['training']

    print("⚙️  Creating training arguments...")
    training_args = TrainingArguments(
        output_dir=train_cfg['output_dir'],
        num_train_epochs=train_cfg['num_train_epochs'],
        per_device_train_batch_size=train_cfg['per_device_train_batch_size'],
        per_device_eval_batch_size=train_cfg['per_device_eval_batch_size'],
        gradient_accumulation_steps=train_cfg['gradient_accumulation_steps'],
        learning_rate=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
        warmup_steps=train_cfg['warmup_steps'],
        logging_steps=train_cfg['logging_steps'],
        save_steps=train_cfg['save_steps'],
        eval_steps=train_cfg['eval_steps'],
        save_total_limit=train_cfg['save_total_limit'],
        fp16=train_cfg['fp16'],
        gradient_checkpointing=train_cfg['gradient_checkpointing'],
        optim=train_cfg['optim'],
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        report_to="none"  # 不使用 wandb 等
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    print("🏋️  Creating Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'],
        data_collator=data_collator
    )

    return trainer

def main():
    print("=" * 60)
    print("阶段 4: 训练优化 - LoRA 微调训练")
    print("=" * 60)

    # 1. 加载配置
    print("\n📋 步骤 1: 加载配置")
    config = load_config()
    print(f"   Model: {config['model']['name']}")
    print(f"   LoRA r: {config['lora']['r']}")
    print(f"   Epochs: {config['training']['num_train_epochs']}")
    print(f"   Learning rate: {config['training']['learning_rate']}")

    # 2. 加载模型
    print("\n📥 步骤 2: 加载模型")
    model, tokenizer = load_model_and_tokenizer(config)

    # 3. 应用 LoRA
    print("\n🔧 步骤 3: 应用 LoRA")
    model = apply_lora(model, config)

    # 4. 准备数据
    print("\n📂 步骤 4: 准备数据")
    dataset = load_and_prepare_data(config, tokenizer)

    # 5. 创建 Trainer
    print("\n⚙️  步骤 5: 创建 Trainer")
    trainer = create_trainer(model, tokenizer, dataset, config)

    # 6. 开始训练
    print("\n🚀 步骤 6: 开始训练")
    print("=" * 60)
    trainer.train()

    # 7. 保存模型
    print("\n💾 步骤 7: 保存模型")
    output_dir = config['training']['output_dir']
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print(f"   模型已保存到: {output_dir}/final")

    print("\n✅ 阶段 4 完成！")
    print("\n📚 关键概念：")
    print("   - Learning rate: 学习率，控制参数更新步长")
    print("   - Batch size: 批次大小，影响训练速度和显存")
    print("   - Gradient accumulation: 梯度累积，模拟大 batch")
    print("   - Warmup: 学习率预热，避免训练初期震荡")
    print("   - FP16: 混合精度训练，节省显存")

if __name__ == "__main__":
    main()
