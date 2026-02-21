#!/bin/bash
set -e

echo "======================================"
echo "🚀 FULL TELE-DERMATOLOGY PIPELINE"
echo "======================================"

./scripts/run_segmentation.sh
./scripts/run_classification.sh
./scripts/run_transfer.sh

echo ""
echo "======================================"
echo "✅ ALL EXPERIMENTS COMPLETED"
echo "======================================"