#!/usr/bin/env python
"""
Evaluation script for autoencoder reconstruction visualization only (no metrics).

This script only generates visualizations without computing metrics.

Usage:
    python eval_visualization_only.py \\
        --config=configs/autoencoder.py:fae \\
        --config.training.use_pde=True \\
        --num_samples=4
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

from model_utils import create_encoder_step, create_decoder_step
from burgers.data_utils import create_dataset

import matplotlib.pyplot as plt
import numpy as np

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
    
    # Load test dataset
    print("Loading test dataset...")
    _, test_dataset = create_dataset(FLAGS.config)
    test_loader = create_dataloader(
        test_dataset,
        batch_size=FLAGS.config.dataset.test_batch_size,
        num_workers=FLAGS.config.dataset.num_workers,
        shuffle=False
    )
    
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
    
    # Visualization loop (using test set)
    print(f"\nGenerating visualizations for {FLAGS.num_samples} samples...")
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
    print("Visualization Statistics:")
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
    print(f"Visualization complete! Results saved to: {os.path.abspath(output_dir)}")
    print("="*70)
    print("\nGenerated files:")
    print(f"  - reconstruction_*.png: Individual sample visualizations")
    print(f"  - summary.png: Summary visualization")
    print("="*70)


if __name__ == "__main__":
    flags.mark_flags_as_required(["config"])
    app.run(main)
