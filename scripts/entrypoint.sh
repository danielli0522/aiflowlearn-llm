#!/bin/bash
set -e

echo "🚀 Starting AIFlowLearn LLM Training Environment..."

# Check and download model if needed
bash /app/scripts/download_model.sh

echo "✅ Environment ready!"
echo ""
echo "📚 Available commands:"
echo "  python scripts/prepare_data.py    - Prepare training data"
echo "  python train.py                   - Start training"
echo "  python evaluate.py                - Evaluate model"
echo ""

# Keep container running
exec "$@"
