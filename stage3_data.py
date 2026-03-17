#!/usr/bin/env python3
"""
阶段 3: 数据工程 - 准备高质量训练数据

学习目标：
1. 理解训练数据的格式要求
2. 掌握数据清洗和预处理技巧
3. 实现数据质量检查
4. 构建训练数据集
"""

import json
import os
from pathlib import Path
from typing import List, Dict
import re

SOURCE_DIR = "/Users/lshl124/openclaw-aigc-project/digital-twin/data/cleaned/growth"
OUTPUT_DIR = "/app/data/processed"

def load_markdown_files(data_dir: str) -> List[Dict[str, str]]:
    """加载所有 Markdown 文件"""
    data_dir = Path(data_dir)

    if not data_dir.exists():
        print(f"⚠️  数据目录不存在: {data_dir}")
        return []

    markdown_files = list(data_dir.glob("*.md"))
    print(f"📂 找到 {len(markdown_files)} 个 Markdown 文件")

    documents = []
    for md_file in markdown_files:
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    'filename': md_file.name,
                    'content': content,
                    'size': len(content)
                })
        except Exception as e:
            print(f"⚠️  读取文件失败 {md_file.name}: {e}")

    return documents

def clean_text(text: str) -> str:
    """清洗文本"""
    # 移除多余空行
    text = re.sub(r'\n{3,}', '\n\n', text)

    # 移除行首行尾空格
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # 移除特殊字符（保留中文、英文、数字、标点）
    # text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9\s\.,!?;:，。！？；：、]', '', text)

    return text.strip()

def extract_training_samples(content: str, filename: str) -> List[Dict[str, str]]:
    """从内容中提取训练样本"""
    samples = []

    # 按段落分割
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    for i, para in enumerate(paragraphs):
        # 跳过太短的段落
        if len(para) < 30:
            continue

        # 跳过纯标题
        if para.startswith('#') and '\n' not in para:
            continue

        # 清洗文本
        cleaned = clean_text(para)

        if len(cleaned) < 30:
            continue

        # 构造 Alpaca 格式样本
        sample = {
            'instruction': '请根据以下成长日记内容，进行总结和分析。',
            'input': cleaned,
            'output': f'这是一段关于个人成长的记录。{cleaned[:100]}...',  # 简化版输出
            'metadata': {
                'source': filename,
                'index': i,
                'length': len(cleaned)
            }
        }

        samples.append(sample)

    return samples

def check_data_quality(samples: List[Dict]) -> Dict:
    """检查数据质量"""
    stats = {
        'total_samples': len(samples),
        'avg_length': 0,
        'min_length': float('inf'),
        'max_length': 0,
        'empty_samples': 0,
        'duplicate_samples': 0
    }

    if not samples:
        return stats

    lengths = []
    seen_inputs = set()

    for sample in samples:
        input_text = sample.get('input', '')
        length = len(input_text)

        lengths.append(length)

        if length == 0:
            stats['empty_samples'] += 1

        if input_text in seen_inputs:
            stats['duplicate_samples'] += 1
        else:
            seen_inputs.add(input_text)

    if lengths:
        stats['avg_length'] = sum(lengths) / len(lengths)
        stats['min_length'] = min(lengths)
        stats['max_length'] = max(lengths)

    return stats

def save_jsonl(data: List[Dict], output_path: str):
    """保存为 JSONL 格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            # 移除 metadata（训练时不需要）
            train_item = {
                'instruction': item['instruction'],
                'input': item['input'],
                'output': item['output']
            }
            f.write(json.dumps(train_item, ensure_ascii=False) + '\n')

    print(f"✅ 保存 {len(data)} 个样本到 {output_path}")

def main():
    print("=" * 60)
    print("阶段 3: 数据工程 - 准备训练数据")
    print("=" * 60)

    # 1. 加载原始数据
    print("\n📥 步骤 1: 加载原始数据")
    documents = load_markdown_files(SOURCE_DIR)

    if not documents:
        print("⚠️  没有找到数据文件，使用示例数据")
        documents = [{
            'filename': 'example.md',
            'content': '# 示例数据\n\n这是一个示例段落，用于演示数据处理流程。\n\n这是第二个段落。',
            'size': 100
        }]

    print(f"   总文件数: {len(documents)}")
    print(f"   总大小: {sum(d['size'] for d in documents) / 1024:.2f} KB")

    # 2. 提取训练样本
    print("\n🔄 步骤 2: 提取训练样本")
    all_samples = []
    for doc in documents:
        samples = extract_training_samples(doc['content'], doc['filename'])
        all_samples.extend(samples)
        print(f"   {doc['filename']}: {len(samples)} 个样本")

    print(f"   总样本数: {len(all_samples)}")

    # 3. 数据质量检查
    print("\n📊 步骤 3: 数据质量检查")
    stats = check_data_quality(all_samples)
    print(f"   总样本数: {stats['total_samples']}")
    print(f"   平均长度: {stats['avg_length']:.0f} 字符")
    print(f"   最短长度: {stats['min_length']} 字符")
    print(f"   最长长度: {stats['max_length']} 字符")
    print(f"   空样本数: {stats['empty_samples']}")
    print(f"   重复样本数: {stats['duplicate_samples']}")

    # 4. 分割训练集和验证集
    print("\n✂️  步骤 4: 分割数据集")
    split_idx = int(len(all_samples) * 0.9)
    train_samples = all_samples[:split_idx]
    val_samples = all_samples[split_idx:]

    print(f"   训练集: {len(train_samples)} 个样本")
    print(f"   验证集: {len(val_samples)} 个样本")

    # 5. 保存数据
    print("\n💾 步骤 5: 保存数据")
    save_jsonl(train_samples, f"{OUTPUT_DIR}/train.jsonl")
    save_jsonl(val_samples, f"{OUTPUT_DIR}/val.jsonl")

    # 6. 展示样本
    print("\n📝 样本预览:")
    if train_samples:
        sample = train_samples[0]
        print(f"   Instruction: {sample['instruction']}")
        print(f"   Input: {sample['input'][:100]}...")
        print(f"   Output: {sample['output'][:100]}...")

    print("\n✅ 阶段 3 完成！")
    print("\n📚 关键概念：")
    print("   - Alpaca 格式: instruction + input + output")
    print("   - 数据清洗: 去重、过滤、格式化")
    print("   - 数据分割: 训练集 90% + 验证集 10%")
    print("   - 质量检查: 长度、重复、空值")

if __name__ == "__main__":
    main()
