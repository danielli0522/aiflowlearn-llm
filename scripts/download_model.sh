#!/bin/bash
set -e

MODEL_NAME="Qwen/Qwen2.5-0.5B"
MODEL_DIR="/app/models/Qwen2.5-0.5B"

echo "🔍 Checking for Qwen2.5-0.5B model..."

if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    echo "✅ Model already exists at $MODEL_DIR"
    exit 0
fi

echo "📥 Downloading Qwen2.5-0.5B from HuggingFace..."
echo "   This may take a few minutes (model size: ~1GB)"

mkdir -p "$MODEL_DIR"

python3 << EOF
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = "$MODEL_NAME"
save_dir = "$MODEL_DIR"

print(f"Downloading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.save_pretrained(save_dir)

print(f"Downloading model...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype="auto"
)
model.save_pretrained(save_dir)

print(f"✅ Model downloaded successfully to {save_dir}")
EOF

echo "✅ Download complete!"
