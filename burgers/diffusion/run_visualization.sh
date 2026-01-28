#!/bin/bash
# Visualization-only script runner for autoencoder reconstruction
#
# This script only generates visualizations (4 samples) without computing metrics
#
# Usage:
#   bash run_visualization.sh

# Activate conda environment
source /root/miniconda3/etc/profile.d/conda.sh
conda activate fundiff_env

# Navigate to the diffusion directory
cd /root/autodl-fs/fundiff/burgers/diffusion

# Run visualization-only script
python eval_visualization_only.py \
    --config=configs/autoencoder.py:fae \
    --config.training.use_pde=True \
    --num_samples=4 \
    --output_dir=evaluation_results

echo ""
echo "Visualization completed!"
echo "Results saved to: evaluation_results/"
echo "  - reconstruction_*.png: Individual sample visualizations"
echo "  - summary.png: Summary visualization"
