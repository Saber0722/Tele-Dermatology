#!/bin/bash
set -e

echo "🧬 Running Segmentation Training..."

mkdir -p logs

uv run python ../src/train_segmentation_isic.py \
  | tee logs/segmentation.log

echo "🧬 Generating Bounding Boxes..."

uv run python ../src/generate_mask_bboxes.py \
  | tee logs/bbox_generation.log

echo "✅ Segmentation Pipeline Complete"