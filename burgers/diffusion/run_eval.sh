#!/bin/bash
# Evaluation script runner for autoencoder reconstruction
#
# This script evaluates the trained autoencoder model on:
# - Full train set (3600 samples) and test set (400 samples) for metrics
# - 4 samples for visualization
#
# Usage:
#   bash run_eval.sh
#   # Or modify parameters below as needed

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate fundiff_env

# Navigate to the diffusion directory
cd /root/autodl-fs/fundiff/burgers/diffusion

# Run evaluation script
# - eval_num_samples=-1 means use ALL samples for metrics (3600 train + 400 test)
# - num_samples=4 means visualize 4 test samples
python eval_autoencoder.py \
    --config=configs/autoencoder.py:fae \
    --config.training.use_pde=True \
    --num_samples=4 \
    --compute_metrics=True \
    --eval_num_samples=-1 \
    --output_dir=evaluation_results

echo ""
echo "Evaluation completed!"
echo "Results saved to: evaluation_results/"
echo "  - metrics.json: Full train/test metrics"
echo "  - reconstruction_*.png: Visualization images"
echo "  - summary.png: Summary visualization"
