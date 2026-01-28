import os
import json
import time

import ml_collections
import wandb

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils, multihost_utils
from jax.sharding import Mesh, PartitionSpec as P

from function_diffusion.models import Encoder, Decoder

from function_diffusion.utils.model_utils import (
    create_optimizer,
    create_autoencoder_state,
    compute_total_params,
)
from function_diffusion.utils.checkpoint_utils import (
    create_checkpoint_manager,
    save_checkpoint,
    restore_checkpoint,
)
from function_diffusion.utils.data_utils import create_dataloader, BatchParser

from burgers.data_utils import create_dataset
from model_utils import create_train_step, create_encoder_step, create_decoder_step, create_eval_step
from jax import vmap


def train_and_evaluate(config: ml_collections.ConfigDict):
    # Initialize model
    encoder = Encoder(**config.model.encoder)
    decoder = Decoder(**config.model.decoder)
    # Create learning rate schedule and optimizer
    lr, tx = create_optimizer(config)

    # Create train state
    state = create_autoencoder_state(config, encoder, decoder, tx)
    num_params = compute_total_params(state)
    print(f"Model storage cost: {num_params * 4 / 1024 / 1024:.2f} MB of parameters")

    # Device count
    num_local_devices = jax.local_device_count()
    num_devices = jax.device_count()
    print(f"Number of devices: {num_devices}")
    print(f"Number of local devices: {num_local_devices}")

    # Create sharding for data parallelism
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), "batch")
    state = multihost_utils.host_local_array_to_global_array(state, mesh, P())

    # Create loss and train step functions
    train_step = create_train_step(encoder, decoder, mesh)
    eval_step = create_eval_step(encoder, decoder, mesh)

    # Two-stage training state
    current_use_pde = getattr(config.training, 'use_pde', False)
    stage2_start_step = getattr(config.training, 'stage2_start_step', None)
    plateau_patience = getattr(config.training, 'plateau_patience', 10)
    plateau_threshold = getattr(config.training, 'plateau_threshold', 1e-4)
    best_loss = float('inf')
    patience_counter = 0
    stage = 1 if not current_use_pde else 2
    
    print(f"Starting in Stage {stage}: use_pde={current_use_pde}")

    # Create dataloaders
    print("Creating dataloaders...")
    train_dataset, test_dataset = create_dataset(config)
    print(f"Dataset created, train size: {len(train_dataset)}, test size: {len(test_dataset)}")
    
    print(f"Creating DataLoader with {config.dataset.num_workers} workers...")
    train_loader = create_dataloader(train_dataset,
                                     batch_size=config.dataset.train_batch_size,
                                     num_workers=config.dataset.num_workers)
    test_loader = create_dataloader(test_dataset,
                                    batch_size=config.dataset.test_batch_size,
                                    num_workers=config.dataset.num_workers,
                                    shuffle=False)
    print("DataLoader created successfully")

    # Create batch parser
    print("Loading first batch for batch parser initialization...")
    sample_batch = next(iter(train_loader))
    print(f"First batch loaded, shape: {sample_batch.shape}")
    
    # Convert sample batch to JAX array to determine shape
    sample_batch_jax = jnp.array(sample_batch)
    b, h, w, c = sample_batch_jax.shape
    print(f"Batch shape: {b}, {h}, {w}, {c}")
    
    print("Creating BatchParser...")
    batch_parser = BatchParser(config, h, w)
    print("BatchParser created successfully")

    # Create checkpoint manager
    job_name = f"{config.model.model_name}_use_pde_{config.training.use_pde}"
    ckpt_path = os.path.join(os.getcwd(), job_name, "ckpt")
    if jax.process_index() == 0:
        if not os.path.isdir(ckpt_path):
            os.makedirs(ckpt_path)

        # Save config
        config_dict = config.to_dict()
        config_path = os.path.join(os.getcwd(), job_name, "config.json")
        with open(config_path, "w") as json_file:
            json.dump(config_dict, json_file, indent=4)

        # Initialize W&B
        wandb_config = config.wandb
        wandb.init(project=wandb_config.project, name=job_name, config=config)

    # Create checkpoint manager
    print("Creating checkpoint manager...")
    ckpt_mngr = create_checkpoint_manager(config.saving, ckpt_path)
    print("Checkpoint manager created successfully")

    # Pre-compute constants outside the training loop for better performance
    print("Pre-computing constants...")
    downsample_factors = jnp.array([1, 2, 5]) if config.training.random_resolution else None
    print("Constants pre-computed")

    # Helper function to compute metrics
    def compute_metrics(pred, target):
        """Compute MSE and relative L2 error."""
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        mse = jnp.mean((pred_flat - target_flat) ** 2)
        relative_l2 = jnp.linalg.norm(pred_flat - target_flat) / (jnp.linalg.norm(target_flat) + 1e-10)
        return mse, relative_l2

    # Evaluation function
    def evaluate_model(loader, batch_parser, eval_step, params, num_samples=100):
        """Evaluate model on a dataset."""
        from model_utils import loss_fn
        
        total_mse = 0.0
        total_relative_l2 = 0.0
        total_loss_data = 0.0
        total_loss_res = 0.0
        count = 0
        num_batches_evaluated = 0
        rng_key = jax.random.PRNGKey(42)
        
        for batch_idx, batch in enumerate(loader):
            if count >= num_samples:
                break
                
            rng_key, subkey = jax.random.split(rng_key)
            batch = jnp.array(batch)
            
            # Prepare batch for evaluation (use full resolution)
            batch_eval = batch_parser.random_query(batch, downsample=1, rng_key=subkey)
            batch_eval = multihost_utils.host_local_array_to_global_array(
                batch_eval, mesh, P("batch")
            )
            
            # Get predictions
            u_pred = eval_step(params, batch_eval)
            coords, x, y_true = batch_eval
            y_true = jnp.squeeze(y_true)
            
            # Compute metrics per sample
            batch_size = u_pred.shape[0]
            for i in range(batch_size):
                if count >= num_samples:
                    break
                mse, rel_l2 = compute_metrics(u_pred[i], y_true[i])
                total_mse += mse
                total_relative_l2 += rel_l2
                count += 1
            
            # Also compute loss components for reference
            _, (loss_data, loss_res) = loss_fn(encoder, decoder, params, batch_eval, use_pde=config.training.use_pde)
            total_loss_data += loss_data
            total_loss_res += loss_res
            num_batches_evaluated += 1
        
        if count == 0:
            count = 1  # Avoid division by zero
        if num_batches_evaluated == 0:
            num_batches_evaluated = 1  # Avoid division by zero
        
        return {
            'mse': total_mse / count,
            'relative_l2': total_relative_l2 / count,
            'loss_data': total_loss_data / num_batches_evaluated,
            'loss_res': total_loss_res / num_batches_evaluated,
        }

    # Training loop
    print("Starting training loop...")
    print(f"Training configuration: use_pde={config.training.use_pde}, random_resolution={config.training.random_resolution}")
    rng_key = jax.random.PRNGKey(0)
    for epoch in range(10000):
        start_time = time.time()
        if epoch % 10 == 0:
            print(f"Epoch {epoch}, getting batches from DataLoader...")
        for batch_idx, batch in enumerate(train_loader):
            if epoch == 0 and batch_idx == 0:
                print(f"First batch received, converting to JAX array...")
            rng_key, subkey = jax.random.split(rng_key)
            # Convert PyTorch tensor to JAX array (optimized: single operation)
            batch = jnp.array(batch)

            if config.training.random_resolution:
                key1, key2 = jax.random.split(subkey)
                random_downsample = jax.random.choice(key1, downsample_factors)
                batch = batch_parser.random_query(batch, random_downsample, rng_key=key2)
            else:
                batch = batch_parser.random_query(batch, downsample=1, rng_key=subkey)

            if epoch == 0 and batch_idx == 0:
                print(f"First batch processed, converting to global array...")
            
            batch = multihost_utils.host_local_array_to_global_array(
                batch, mesh, P("batch")
            )
            
            if epoch == 0 and batch_idx == 0:
                print(f"Calling train_step (JIT compilation will happen on first call)...")
                import time as time_module
                step_start = time_module.time()
            
            state, loss, loss_data, loss_res = train_step(state, batch, current_use_pde)
            
            if epoch == 0 and batch_idx == 0:
                print(f"First train_step completed in {time_module.time() - step_start:.2f}s (includes JIT compilation)")

        # Logging
        step = int(state.step)
        if epoch % config.logging.log_interval == 0:
            # Log metrics
            loss = loss.item()
            loss_data = loss_data.item()
            loss_res = loss_res.item()

            end_time = time.time()
            log_dict = {
                "train/loss": loss,
                "train/loss_data": loss_data,
                "train/loss_res": loss_res,
                "train/stage": stage,
                "lr": lr(step)
            }

            if jax.process_index() == 0:
                wandb.log(log_dict, step)  # Log metrics to W&B
                print("step: {}, stage: {}, loss data: {:.3e}, loss res: {:.3e}, time: {:.3e}".format(step, stage, loss_data, loss_res, end_time - start_time))

        # Evaluation on train and test sets
        eval_interval = getattr(config.logging, 'eval_interval', 10)  # Default to every 10 epochs
        if epoch % eval_interval == 0:
            print(f"Evaluating on train and test sets at epoch {epoch}...")
            
            # Evaluate on train set
            train_metrics = evaluate_model(train_loader, batch_parser, eval_step, state.params, num_samples=50)
            
            # Evaluate on test set
            test_metrics = evaluate_model(test_loader, batch_parser, eval_step, state.params, num_samples=50)
            
            current_eval_loss = test_metrics['loss_data']
            
            # Stage switching logic
            if stage == 1:
                # 1. Switch by step count
                if stage2_start_step is not None and step >= stage2_start_step:
                    print(f"Switching to Stage 2 at step {step} (step limit reached)")
                    current_use_pde = True
                    stage = 2
                
                # 2. Switch by plateau detection
                elif current_eval_loss < best_loss * (1 - plateau_threshold):
                    best_loss = current_eval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= plateau_patience:
                        print(f"Switching to Stage 2 at epoch {epoch}, step {step} (plateau detected)")
                        current_use_pde = True
                        stage = 2

            # Log train metrics
            train_log_dict = {
                f"train_metrics/mse": train_metrics['mse'].item(),
                f"train_metrics/relative_l2": train_metrics['relative_l2'].item(),
                f"train_metrics/loss_data": train_metrics['loss_data'].item(),
                f"train_metrics/loss_res": train_metrics['loss_res'].item(),
            }
            
            # Log test metrics
            test_log_dict = {
                f"test_metrics/mse": test_metrics['mse'].item(),
                f"test_metrics/relative_l2": test_metrics['relative_l2'].item(),
                f"test_metrics/loss_data": test_metrics['loss_data'].item(),
                f"test_metrics/loss_res": test_metrics['loss_res'].item(),
            }
            
            all_log_dict = {**train_log_dict, **test_log_dict}
            
            if jax.process_index() == 0:
                wandb.log(all_log_dict, step)
                print(f"  Train - MSE: {train_metrics['mse']:.6e}, Rel L2: {train_metrics['relative_l2']:.6e}, Loss Data: {train_metrics['loss_data']:.6e}, Loss Res: {train_metrics['loss_res']:.6e}")
                print(f"  Test  - MSE: {test_metrics['mse']:.6e}, Rel L2: {test_metrics['relative_l2']:.6e}, Loss Data: {test_metrics['loss_data']:.6e}, Loss Res: {test_metrics['loss_res']:.6e}")

        # Save checkpoint
        if epoch % config.saving.save_interval == 0:
            save_checkpoint(ckpt_mngr, state)

        if step >= config.training.max_steps:
            break

    # Save final checkpoint
    print("Training finished, saving final checkpoint...")
    save_checkpoint(ckpt_mngr, state)
    ckpt_mngr.wait_until_finished()




