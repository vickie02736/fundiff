#!/usr/bin/env python
"""
Evaluation script for autoencoder reconstruction visualization and metrics.

This script evaluates a trained autoencoder model and provides:
1. Full metrics evaluation on train and test sets
2. Visualization of reconstruction results
3. Error analysis (MAE, MSE, R2, Relative L2, PSNR, SSIM)

Usage:
    python eval_autoencoder.py \\
        --config=configs/autoencoder.py:fae \\
        --config.training.use_pde=True \\
        --num_samples=4 \\
        --eval_num_samples=-1 \\
        --compute_metrics=True

Parameters:
    --num_samples: Number of test samples to visualize (default: 4)
    --eval_num_samples: Number of samples for metrics evaluation. Use -1 for all samples (default: -1)
    --compute_metrics: Whether to compute metrics (default: True)
    --output_dir: Directory to save results (default: evaluation_results)

Dataset splits:
    - Train: 3600 samples
    - Test: 400 samples
    - Total: 4000 samples
"""

import os
import sys

# Add parent directory to Python path to enable imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import jax
jax.config.update("jax_default_matmul_precision", "highest")

from absl import app
from absl import flags
from ml_collections import config_flags
from tqdm import tqdm

import jax.numpy as jnp
from jax import vmap
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.models import Encoder, Decoder
from function_diffusion.utils.model_utils import (
    create_autoencoder_state,
    create_optimizer,
)
from function_diffusion.utils.data_utils import create_dataloader
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    restore_checkpoint,
)

from model_utils import create_encoder_step, create_decoder_step, create_eval_step
from function_diffusion.utils.data_utils import BatchParser
from burgers.data_utils import create_dataset

import matplotlib.pyplot as plt
import numpy as np

# Try to import SSIM from skimage, fallback to manual implementation if not available
try:
    from skimage.metrics import structural_similarity as ssim_skimage
    HAS_SSIM = True
except ImportError:
    HAS_SSIM = False
    print("Warning: skimage not available, SSIM will use manual implementation")

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "configs/autoencoder.py:fae",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

flags.DEFINE_integer(
    "num_samples",
    4,
    "Number of test samples to visualize.",
    short_name="n"
)

flags.DEFINE_string(
    "output_dir",
    "evaluation_results",
    "Directory to save visualization results.",
)

flags.DEFINE_integer(
    "eval_num_samples",
    -1,
    "Number of samples to use for metrics evaluation. Use -1 or <=0 to use all samples.",
    short_name="e"
)

flags.DEFINE_boolean(
    "compute_metrics",
    True,
    "Whether to compute metrics on train and test sets.",
)

flags.DEFINE_string(
    "ckpt_dir",
    "",
    "Direct path to the checkpoint directory. If provided, overrides the default path construction.",
)


def restore_autoencoder_state(config, encoder, decoder):
    """Restore autoencoder state from checkpoint."""
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_autoencoder_state(config, encoder, decoder, tx)

    # Determine checkpoint path
    if FLAGS.ckpt_dir:
        ckpt_path = FLAGS.ckpt_dir
    else:
        # Create checkpoint manager using default logic
        job_name = f"{config.model.model_name}_use_pde_{config.training.use_pde}"
        # Use absolute path of the script's directory as base
        base_dir = os.path.dirname(os.path.abspath(__file__))
        ckpt_path = os.path.join(base_dir, job_name, "ckpt")
        
        # Fallback to current working directory if script directory doesn't have it
        if not os.path.exists(ckpt_path):
            ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    
    if not os.path.exists(ckpt_path):
        raise ValueError(f"Checkpoint path does not exist: {ckpt_path}\n"
                         f"Please ensure the path is correct or specify it using --ckpt_dir.")
    
    print(f"Using checkpoint path: {ckpt_path}")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)

    # Restore the model from the checkpoint
    state = restore_checkpoint(ckpt_mngr, state)
    print(f"Restored model from step {state.step}")

    return state


def compute_reconstruction_error(pred, target):
    """Compute relative L2 error."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    return jnp.linalg.norm(pred_flat - target_flat) / (jnp.linalg.norm(target_flat) + 1e-10)

def compute_all_metrics(pred, target):
    """Compute all metrics: MAE, MSE, R2, Relative L2, PSNR."""
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # MAE: Mean Absolute Error
    mae = jnp.mean(jnp.abs(pred_flat - target_flat))
    
    # MSE: Mean Squared Error
    mse = jnp.mean((pred_flat - target_flat) ** 2)
    
    # R2: Coefficient of Determination
    ss_res = jnp.sum((target_flat - pred_flat) ** 2)
    ss_tot = jnp.sum((target_flat - jnp.mean(target_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-10))
    
    # Relative L2: Normalized L2 error
    relative_l2 = jnp.linalg.norm(pred_flat - target_flat) / (jnp.linalg.norm(target_flat) + 1e-10)
    
    # PSNR: Peak Signal-to-Noise Ratio
    # PSNR = 20 * log10(MAX) - 10 * log10(MSE)
    # For normalized data [0,1], MAX = 1
    max_val = 1.0
    psnr = 20 * jnp.log10(max_val) - 10 * jnp.log10(mse + 1e-10)
    
    return mae, mse, r2, relative_l2, psnr

def compute_ssim(pred, target):
    """Compute SSIM (Structural Similarity Index)."""
    pred_np = np.array(pred)
    target_np = np.array(target)
    
    # Normalize to [0, 1] range
    pred_min, pred_max = pred_np.min(), pred_np.max()
    target_min, target_max = target_np.min(), target_np.max()
    
    pred_norm = (pred_np - pred_min) / (pred_max - pred_min + 1e-10)
    target_norm = (target_np - target_min) / (target_max - target_min + 1e-10)
    
    if HAS_SSIM:
        ssim_value = ssim_skimage(target_norm, pred_norm, data_range=1.0)
    else:
        # Manual SSIM implementation (simplified)
        mu1, mu2 = target_norm.mean(), pred_norm.mean()
        sigma1_sq = ((target_norm - mu1) ** 2).mean()
        sigma2_sq = ((pred_norm - mu2) ** 2).mean()
        sigma12 = ((target_norm - mu1) * (pred_norm - mu2)).mean()
        
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        ssim_value = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
                     ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2) + 1e-10)
    
    return float(ssim_value)


def visualize_reconstruction(original, reconstructed, error, residual, save_path, idx):
    """Visualize reconstruction results."""
    fig = plt.figure(figsize=(16, 4))
    
    # Original
    plt.subplot(1, 4, 1)
    plt.title('Original', fontsize=12)
    plt.imshow(original, cmap='coolwarm', aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    
    # Reconstructed
    plt.subplot(1, 4, 2)
    plt.title('Reconstructed', fontsize=12)
    plt.imshow(reconstructed, cmap='coolwarm', aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    
    # Absolute Error
    plt.subplot(1, 4, 3)
    plt.title(f'Absolute Error\n(Rel. L2: {error:.4e})', fontsize=12)
    abs_error = np.abs(original - reconstructed)
    plt.imshow(abs_error, cmap='coolwarm', aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    
    # PDE Residual
    plt.subplot(1, 4, 4)
    plt.title('PDE Residual', fontsize=12)
    plt.imshow(np.abs(residual), cmap='coolwarm', aspect='auto', origin='lower')
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('t')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'reconstruction_{idx:03d}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main(argv):
    # Initialize model
    print("Initializing models...")
    encoder = Encoder(**FLAGS.config.model.encoder)
    decoder = Decoder(**FLAGS.config.model.decoder)
    
    # Restore model state
    print("Restoring checkpoint...")
    state = restore_autoencoder_state(FLAGS.config, encoder, decoder)
    
    # Device setup
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    
    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((num_devices,)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())
    
    # Create encoder and decoder steps
    print("Creating encoder and decoder steps...")
    encoder_step = create_encoder_step(encoder, mesh)
    decoder_step = create_decoder_step(decoder, mesh)
    eval_step = create_eval_step(encoder, decoder, mesh)
    
    # Load datasets
    print("Loading datasets...")
    train_dataset, test_dataset = create_dataset(FLAGS.config)
    print(f"Dataset loaded, train size: {len(train_dataset)}, test size: {len(test_dataset)}")
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=FLAGS.config.dataset.test_batch_size,
        num_workers=FLAGS.config.dataset.num_workers,
        shuffle=False
    )
    test_loader = create_dataloader(
        test_dataset,
        batch_size=FLAGS.config.dataset.test_batch_size,
        num_workers=FLAGS.config.dataset.num_workers,
        shuffle=False
    )
    
    # Create batch parser
    sample_batch = next(iter(train_loader))
    sample_batch_jax = jnp.array(sample_batch)
    b, h, w, c = sample_batch_jax.shape
    batch_parser = BatchParser(FLAGS.config, h, w)
    
    # Create uniform grid for evaluation (same as training)
    h, w = 200, 200
    x_coords = jnp.linspace(0, 1, h)
    y_coords = jnp.linspace(0, 1, w)
    x_coords, y_coords = jnp.meshgrid(x_coords, y_coords, indexing='ij')
    coords = jnp.hstack([x_coords.reshape(-1, 1), y_coords.reshape(-1, 1)])
    coords = multihost_utils.host_local_array_to_global_array(coords, mesh, P())
    
    # Create output directory
    output_dir = FLAGS.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Function to evaluate metrics on a dataset
    def evaluate_metrics(loader, encoder_step, decoder_step, eval_step, batch_parser, params, num_samples, dataset_name):
        """Evaluate metrics on a dataset.
        
        Args:
            num_samples: Number of samples to evaluate. If None or <= 0, use all samples.
        """
        total_mae = 0.0
        total_mse = 0.0
        total_r2 = 0.0
        total_relative_l2 = 0.0
        total_psnr = 0.0
        total_ssim = 0.0
        count = 0
        rng_key = jax.random.PRNGKey(42)
        
        use_all_samples = (num_samples is None or num_samples <= 0)
        if use_all_samples:
            print(f"\nEvaluating {dataset_name} set (all samples)...")
        else:
            print(f"\nEvaluating {dataset_name} set ({num_samples} samples)...")
        
        for batch_idx, batch in enumerate(tqdm(loader, desc=f"Processing {dataset_name}")):
            if not use_all_samples and count >= num_samples:
                break
                
            rng_key, subkey = jax.random.split(rng_key)
            batch = jnp.array(batch)
            
            # Prepare batch for evaluation (use full resolution)
            batch_eval = batch_parser.random_query(batch, downsample=1, rng_key=subkey)
            batch_eval = multihost_utils.host_local_array_to_global_array(
                batch_eval, mesh, P("batch")
            )
            
            # Get predictions using eval_step
            coords, x, y_true = batch_eval
            u_pred = eval_step(params, batch_eval)
            y_true = jnp.squeeze(y_true)
            
            # Compute metrics per sample
            batch_size = u_pred.shape[0]
            for i in range(batch_size):
                if not use_all_samples and count >= num_samples:
                    break
                
                # Compute all metrics
                mae, mse, r2, rel_l2, psnr = compute_all_metrics(u_pred[i], y_true[i])
                ssim_val = compute_ssim(u_pred[i], y_true[i])
                
                total_mae += mae
                total_mse += mse
                total_r2 += r2
                total_relative_l2 += rel_l2
                total_psnr += psnr
                total_ssim += ssim_val
                count += 1
        
        if count == 0:
            count = 1  # Avoid division by zero
        
        metrics = {
            'mae': float(total_mae / count),
            'mse': float(total_mse / count),
            'r2': float(total_r2 / count),
            'relative_l2': float(total_relative_l2 / count),
            'psnr': float(total_psnr / count),
            'ssim': float(total_ssim / count),
            'num_samples': count
        }
        
        return metrics
    
    # Compute metrics on train and test sets
    if FLAGS.compute_metrics:
        # Use all samples if eval_num_samples <= 0, otherwise use the specified number
        # For train: 3600 samples, for test: 400 samples
        eval_num_samples = None if FLAGS.eval_num_samples <= 0 else FLAGS.eval_num_samples
        train_metrics = evaluate_metrics(
            train_loader, encoder_step, decoder_step, eval_step, batch_parser, 
            state.params, eval_num_samples, "train"
        )
        
        test_metrics = evaluate_metrics(
            test_loader, encoder_step, decoder_step, eval_step, batch_parser,
            state.params, eval_num_samples, "test"
        )
        
        # Print metrics summary
        print("\n" + "="*80)
        print("EVALUATION METRICS SUMMARY")
        print("="*80)
        print(f"\n{'Metric':<20} {'Train':<30} {'Test':<30}")
        print("-"*80)
        print(f"{'MAE':<20} {train_metrics['mae']:<30.6e} {test_metrics['mae']:<30.6e}")
        print(f"{'MSE':<20} {train_metrics['mse']:<30.6e} {test_metrics['mse']:<30.6e}")
        print(f"{'R2':<20} {train_metrics['r2']:<30.6f} {test_metrics['r2']:<30.6f}")
        print(f"{'Relative L2':<20} {train_metrics['relative_l2']:<30.6e} {test_metrics['relative_l2']:<30.6e}")
        print(f"{'PSNR (dB)':<20} {train_metrics['psnr']:<30.4f} {test_metrics['psnr']:<30.4f}")
        print(f"{'SSIM':<20} {train_metrics['ssim']:<30.6f} {test_metrics['ssim']:<30.6f}")
        print(f"{'Num Samples':<20} {train_metrics['num_samples']:<30} {test_metrics['num_samples']:<30}")
        print("="*80)
        
        # Save metrics to file
        import json
        metrics_dict = {
            'train': train_metrics,
            'test': test_metrics
        }
        metrics_path = os.path.join(output_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"\nMetrics saved to: {metrics_path}")
    
    # Visualization loop (using test set)
    print(f"\nStarting evaluation on {FLAGS.num_samples} samples...")
    all_errors = []
    all_originals = []
    all_reconstructed = []
    all_residuals = []
    
    rng_key = jax.random.PRNGKey(42)
    sample_count = 0
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Processing batches")):
        if sample_count >= FLAGS.num_samples:
            break
            
        # Convert to JAX array
        batch = jnp.array(batch)
        batch_size = batch.shape[0]
        
        # Prepare batch format: (coords, input, target)
        # For reconstruction, we use the same data as input and target
        u_true = batch  # Shape: (B, H, W, C)
        
        # Create batch in format expected by encoder: (coords_mask, input, coords_mask)
        # For full resolution, we use all ones
        u_batch = (jnp.ones_like(u_true), u_true, jnp.ones_like(u_true))
        
        # Shard the batch across devices
        u_batch = multihost_utils.host_local_array_to_global_array(
            u_batch, mesh, P("batch")
        )
        
        # Encode
        z = encoder_step(state.params[0], u_batch)
        
        # Decode
        u_pred, r_pred = decoder_step(state.params[1], z, coords)
        
        # Reshape predictions
        u_pred = u_pred.reshape(-1, h, w)
        u_true_reshaped = u_true.reshape(-1, h, w)
        r_pred = r_pred.reshape(-1, h, w)
        
        # Compute errors
        errors = vmap(compute_reconstruction_error)(u_pred, u_true_reshaped)
        
        # Store results
        for i in range(batch_size):
            if sample_count >= FLAGS.num_samples:
                break
                
            all_originals.append(u_true_reshaped[i])
            all_reconstructed.append(u_pred[i])
            all_residuals.append(r_pred[i])
            all_errors.append(errors[i])
            sample_count += 1
    
    # Convert to numpy for visualization
    all_originals = [np.array(x) for x in all_originals]
    all_reconstructed = [np.array(x) for x in all_reconstructed]
    all_residuals = [np.array(x) for x in all_residuals]
    all_errors = [float(x) for x in all_errors]
    
    # Print visualization statistics
    print("\n" + "="*50)
    print("Visualization Statistics (Test Set Samples):")
    print("="*50)
    print(f"Number of samples: {len(all_errors)}")
    print(f"Mean relative L2 error: {np.mean(all_errors):.6e}")
    print(f"Std relative L2 error:  {np.std(all_errors):.6e}")
    print(f"Min relative L2 error:  {np.min(all_errors):.6e}")
    print(f"Max relative L2 error:  {np.max(all_errors):.6e}")
    print("="*50)
    
    # Visualize each sample
    print(f"\nGenerating visualizations...")
    for idx, (orig, recon, error, res) in enumerate(
        zip(all_originals, all_reconstructed, all_errors, all_residuals)
    ):
        visualize_reconstruction(orig, recon, error, res, output_dir, idx)
        print(f"  Saved visualization {idx+1}/{len(all_originals)}")
    
    # Create summary visualization with all samples
    print("\nCreating summary visualization...")
    num_samples_to_show = min(FLAGS.num_samples, 4)
    fig, axes = plt.subplots(num_samples_to_show, 3, figsize=(15, 5*num_samples_to_show))
    
    if num_samples_to_show == 1:
        axes = axes.reshape(1, -1)
    
    for idx in range(num_samples_to_show):
        # Original
        axes[idx, 0].set_title(f'Sample {idx+1}: Original', fontsize=12)
        im0 = axes[idx, 0].imshow(all_originals[idx], cmap='coolwarm', aspect='auto', origin='lower')
        plt.colorbar(im0, ax=axes[idx, 0])
        axes[idx, 0].set_xlabel('x')
        axes[idx, 0].set_ylabel('t')
        
        # Reconstructed
        axes[idx, 1].set_title(f'Reconstructed (Error: {all_errors[idx]:.4e})', fontsize=12)
        im1 = axes[idx, 1].imshow(all_reconstructed[idx], cmap='coolwarm', aspect='auto', origin='lower')
        plt.colorbar(im1, ax=axes[idx, 1])
        axes[idx, 1].set_xlabel('x')
        axes[idx, 1].set_ylabel('t')
        
        # Error
        abs_error = np.abs(all_originals[idx] - all_reconstructed[idx])
        axes[idx, 2].set_title('Absolute Error', fontsize=12)
        im2 = axes[idx, 2].imshow(abs_error, cmap='coolwarm', aspect='auto', origin='lower')
        plt.colorbar(im2, ax=axes[idx, 2])
        axes[idx, 2].set_xlabel('x')
        axes[idx, 2].set_ylabel('t')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved summary to: {summary_path}")
    
    print("\n" + "="*70)
    print(f"Evaluation complete! Results saved to: {os.path.abspath(output_dir)}")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - metrics.json: Full train/test metrics evaluation")
    print(f"  - reconstruction_*.png: Individual sample visualizations")
    print(f"  - summary.png: Summary visualization")
    print("="*70)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
