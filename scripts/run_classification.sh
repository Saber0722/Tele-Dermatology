#!/bin/bash
set -e

echo "📷 Running Classification Experiments..."

mkdir -p logs

echo "1️⃣ EfficientNet Baseline"
uv run python ../src/train_baseline.py \
  | tee logs/efficientnet_baseline.log

echo "2️⃣ EfficientNet Mask-Crop"
uv run python ../src/train_effecient_mask_crop.py \
  | tee logs/efficientnet_mask_crop.log

echo "3️⃣ ResNet Baseline"
uv run python ../src/train_resnet_baseline.py \
  | tee logs/resnet_baseline.log

echo "4️⃣ ResNet Mask-Crop"
uv run python ../src/train_resnet_mask_crop.py \
  | tee logs/resnet_mask_crop.log

echo "✅ Classification Experiments Complete"