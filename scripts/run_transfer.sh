#!/bin/bash
set -e

echo "🔁 Running Transfer Learning Experiments..."

mkdir -p logs

uv run python ../src/train_transfer_resnet.py \
  | tee logs/transfer_full.log

uv run python ../src/train_transfer_multimodal_resnet.py \
  | tee logs/transfer_freeze_unfreeze.log

echo "✅ Transfer Learning Experiments Complete"