#!/usr/bin/env python3
"""
数据准备脚本 - 处理 digital-twin 的成长日记数据
将 Markdown 文件转换为 LLM 微调训练格式
"""

import os
import json
from pathlib import Path
from typing import List, Dict
import re

# 数据源路径
SOURCE_DATA_DIR = "/Users/lshl124/openclaw-aigc-project/digital-twin/data/cleaned/growth"
OUTPUT_DIR = "/app/data/processed"

def read_markdown_files(data_dir: str) -> List[Dict[str, str]]:
    """读取所有 Markdown 文件"""
    data_dir = Path(data_dir)
    markdown_files = list(data_dir.glob("*.md"))

    print(f"📂 Found {len(markdown_files)} markdown files")

    documents = []
    for md_file in markdown_files:
        with open(md_file, 'r', encoding='utf-8') as f:
            content = f.read()
            documents.append({
                'filename': md_file.name,
                'content': content
            })

    return documents

def extract_qa_pairs(content: str) -> List[Dict[str, str]]:
    """从 Markdown 内容中提取问答对"""
    qa_pairs = []

    # 简单策略：按段落分割，每个段落作为一个训练样本
    paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]

    for para in paragraphs:
        # 跳过太短的段落
        if len(para) < 20:
            continue

        # 跳过纯标题
        if para.startswith('#'):
            continue

        # 构造问答对（简单示例）
        qa_pairs.append({
            'instruction': '请根据以下内容进行总结和分析：',
            'input': para,
            'output': para  # 实际应用中可以用 LLM 生成更好的输出
        })

    return qa_pairs

def convert_to_alpaca_format(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """转换为 Alpaca 格式"""
    alpaca_data = []

    for doc in documents:
        qa_pairs = extract_qa_pairs(doc['content'])
        alpaca_data.extend(qa_pairs)

    return alpaca_data

def save_jsonl(data: List[Dict], output_path: str):
    """保存为 JSONL 格式"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Saved {len(data)} samples to {output_path}")

def main():
    print("🔄 Starting data preparation...")

    # 读取 Markdown 文件
    documents = read_markdown_files(SOURCE_DATA_DIR)

    if not documents:
        print("⚠️  No markdown files found!")
        return

    # 转换为训练格式
    print("🔄 Converting to Alpaca format...")
    alpaca_data = convert_to_alpaca_format(documents)

    # 分割训练集和验证集
    split_idx = int(len(alpaca_data) * 0.9)
    train_data = alpaca_data[:split_idx]
    val_data = alpaca_data[split_idx:]

    # 保存
    save_jsonl(train_data, f"{OUTPUT_DIR}/train.jsonl")
    save_jsonl(val_data, f"{OUTPUT_DIR}/val.jsonl")

    print(f"\n📊 Data Statistics:")
    print(f"   Total samples: {len(alpaca_data)}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    print(f"\n✅ Data preparation complete!")

if __name__ == "__main__":
    main()
