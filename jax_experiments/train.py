"""
Gemma from scratch (JAX/Flax variant)

This script provides a comprehensive training pipeline for a Gemma-like model using JAX/Flax.
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" # Suppress verbose TF/XLA logs
import argparse
from datetime import datetime
from itertools import cycle
import time
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
import optax
from flax import serialization # Added import

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from jax_experiments.config import GEMMA3_CONFIG_CUSTOM
from .model import Gemma3Model
from .config import GEMMA3_CONFIG_CUSTOM

# Reusing the MemmapDataset from the PyTorch implementation logic
class MemmapDataset(Dataset):
    """A PyTorch Dataset that loads sequences from a memory-mapped .bin file."""

    def __init__(self, data_path, sequence_length, dtype=np.uint16):
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.sequence_length].astype(np.int64)
        y = self.data[idx + 1 : idx + 1 + self.sequence_length].astype(np.int64)
        return np.array(x), np.array(y) # Return numpy arrays for JAX

def create_train_state(rng, config, learning_rate_fn, sequence_length):
    """Creates initial `TrainState`."""
    model = Gemma3Model(config)
    # Initialize parameters
    dummy_input = jnp.ones((1, sequence_length), dtype=jnp.int32)
    params = model.init(rng, dummy_input)["params"]
    
    # Optimizer
    tx = optax.adamw(
        learning_rate=learning_rate_fn,
        b1=0.9,
        b2=0.95,
        eps=1e-9,
        weight_decay=0.1
    )
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )

@jax.jit
def train_step(state, batch, accumulation_steps):
    """Training step with gradient accumulation via reshaping."""
    # batch: (accum_steps, micro_batch, seq_len)
    inputs, targets = batch
    
    def loss_fn(params):
        logits = state.apply_fn({'params': params}, inputs)
        # logits: (accum_steps, micro_batch, seq_len, vocab)
        # targets: (accum_steps, micro_batch, seq_len)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
        return jnp.mean(loss)

    # To handle accumulation, we can simply reshape inputs to (batch_size * accum_steps, ...)
    # But real accumulation usually sums gradients. 
    # Here, JAX can handle large batches automatically if they fit in memory (or via pmap/sharding).
    # Since we are single device presumably, we will just treat it as one large batch for simplicity
    # unless we strictly want to save memory. 
    # For exact parity with "gradient accumulation" to save memory on single GPU, 
    # we would need `jax.lax.scan` over the accumulation steps.
    
    # Let's implement the scan version for memory efficiency.
    
    def micro_batch_loss_fn(params, micro_batch):
        mb_inputs, mb_targets = micro_batch
        logits = state.apply_fn({'params': params}, mb_inputs)
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, mb_targets)
        return jnp.mean(loss)

    grad_fn = jax.value_and_grad(micro_batch_loss_fn)

    def accumulate_grads(accum_grads, micro_batch):
        loss, grads = grad_fn(state.params, micro_batch)
        new_accum_grads = jax.tree.map(lambda x, y: x + y, accum_grads, grads)
        return new_accum_grads, loss

    # Initialize accumulated gradients with zeros
    init_accum_grads = jax.tree.map(jnp.zeros_like, state.params)

    # inputs shape: (accum_steps, micro_batch, seq)
    # Scan over the first dimension (accum_steps)
    final_accum_grads, losses = jax.lax.scan(accumulate_grads, init_accum_grads, (inputs, targets))
    
    # Average gradients and loss
    loss = jnp.mean(losses)
    # Normalize gradients by accumulation steps
    grads = jax.tree.map(lambda x: x / accumulation_steps, final_accum_grads)
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss

@jax.jit
def eval_step(state, batch):
    inputs, targets = batch
    logits = state.apply_fn({'params': state.params}, inputs)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
    
    # Calculate accuracy
    preds = jnp.argmax(logits, axis=-1)
    accuracy = jnp.mean(preds == targets)
    
    return jnp.mean(loss), accuracy

def evaluate_model(state, eval_iters, iterators):
    metrics = {}
    
    for split, loader_iter in iterators.items():
        total_loss = 0.0
        total_acc = 0.0
        
        for _ in range(eval_iters):
            inputs, targets = next(loader_iter)
            inputs = jnp.array(inputs.numpy())
            targets = jnp.array(targets.numpy())
            
            loss, acc = eval_step(state, (inputs, targets))
            total_loss += loss
            total_acc += acc
            
        avg_loss = total_loss / eval_iters
        metrics[f"{split}_loss"] = avg_loss
        metrics[f"{split}_perplexity"] = math.exp(avg_loss)
        metrics[f"{split}_accuracy"] = total_acc / eval_iters
        
    return metrics

def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def main(args):
    print(f"JAX Devices: {jax.devices()}")
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    best_model_params_path = os.path.join(models_dir, f"best_model_jax_{timestamp}.params")
    
    writer = SummaryWriter(log_dir=f"runs/gemma_jax_{timestamp}")
    
    rng = jax.random.PRNGKey(args.seed)
    
    # Schedule
    num_optimizer_steps = args.max_iters // args.gradient_accumulation_steps
    warmup_optimizer_steps = args.warmup_steps // args.gradient_accumulation_steps
    
    # Ensure valid decay_steps (total steps must be > warmup steps)
    if num_optimizer_steps <= warmup_optimizer_steps:
        print(f"Warning: max_iters ({args.max_iters}) is less than or equal to warmup_steps ({args.warmup_steps}). Adjusting schedule to warmup-only.")
        warmup_optimizer_steps = num_optimizer_steps - 1 # Ensure strictly less than total if possible, or just accept short warmup
        # Actually, if we want "warmup only" or "warmup then flat", we just need valid args.
        # If we set decay_steps = num_optimizer_steps, and warmup = num, then decay duration is 0.
        # optax might require decay_duration > 0.
        # Let's set num_optimizer_steps = warmup + 1
        num_optimizer_steps = warmup_optimizer_steps + 1

    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.learning_rate,
        warmup_steps=warmup_optimizer_steps,
        decay_steps=num_optimizer_steps,
        end_value=args.min_lr
    )

    # Initialize model
    rng, init_rng = jax.random.split(rng)
    state = create_train_state(init_rng, GEMMA3_CONFIG_CUSTOM, lr_schedule, args.sequence_length)
    
    print(f"Model initialized. Params count: {sum(x.size for x in jax.tree_util.tree_leaves(state.params))}")

    # Data Loaders
    # We use PyTorch DataLoader for convenience, but with a custom collate_fn or converting to numpy
    train_dataset = MemmapDataset(
        os.path.join(args.data_dir, "train.bin"), args.sequence_length
    )
    
    # Helper to get numpy batches
    def get_numpy_loader(dataset, batch_size, shuffle=True):
        return DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle, 
            num_workers=os.cpu_count() // 2,
            collate_fn=numpy_collate,
            persistent_workers=True,
            drop_last=True
        )

    # For gradient accumulation, we want the loader to yield (accum_steps * micro_batch_size) items, 
    # OR we handle fetching `accum_steps` batches in the loop.
    # Handling in loop is easier to implement without changing loader batch size logic much.
    # BUT `train_step` expects (accum_steps, micro_batch, ...).
    # So we should probably set loader batch size to `micro_batch_size` and fetch multiple times.
    
    micro_batch_size = args.batch_size
    train_loader = get_numpy_loader(train_dataset, micro_batch_size)
    val_dataset = MemmapDataset(os.path.join(args.data_dir, "val.bin"), args.sequence_length)
    val_loader = get_numpy_loader(val_dataset, micro_batch_size, shuffle=False)
    train_eval_loader = get_numpy_loader(train_dataset, micro_batch_size, shuffle=False)

    eval_iterators = {
        "train": cycle(train_eval_loader),
        "val": cycle(val_loader),
    }

    train_iter = iter(train_loader)
    
    print(f"Starting training run {timestamp}.")
    start_time = time.time()
    
    train_loss_list = []
    validation_loss_list = []
    best_val_loss = float("inf")
    best_iter_num = 0

    # Main loop
    # In this implementation, one 'iter_num' corresponds to one Gradient Accumulation step (one update)
    # similar to the PyTorch implementation logic (although PyTorch loop counts every micro-step).
    # Wait, PyTorch loop: `for iter_num in pbar`. If `accum > 1`, it updates optimizer only every `accum` steps.
    # So `iter_num` in PyTorch is a MICRO-step.
    
    # To match PyTorch structure:
    # We will run the loop `max_iters` times.
    # But JAX `scan` approach naturally processes `accum_steps` at once. 
    # So one call to `train_step` does `accum_steps` micro-steps and 1 update.
    # This means our loop should run `max_iters // accum_steps` times.
    # Let's adjust `pbar` range.
    
    total_updates = args.max_iters // args.gradient_accumulation_steps
    pbar = tqdm(range(total_updates))
    
    for update_step in pbar:
        # Fetch `gradient_accumulation_steps` batches
        batch_inputs = []
        batch_targets = []
        
        for _ in range(args.gradient_accumulation_steps):
            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                inputs, targets = next(train_iter)
            batch_inputs.append(inputs)
            batch_targets.append(targets)
            
        # Stack to (accum_steps, micro_batch, seq)
        inputs = np.stack(batch_inputs)
        targets = np.stack(batch_targets)
        
        # JAX Inputs
        inputs = jnp.array(inputs)
        targets = jnp.array(targets)
        
        state, loss = train_step(state, (inputs, targets), args.gradient_accumulation_steps)
        
        # Logging
        current_iter = (update_step + 1) * args.gradient_accumulation_steps
        
        if (update_step + 1) % args.log_interval == 0:
            writer.add_scalar("Train/loss_step", loss.item() * args.gradient_accumulation_steps, current_iter)
            pbar.set_description(f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f}")

        if (update_step + 1) % args.eval_interval == 0:
            metrics = evaluate_model(state, args.eval_iters, eval_iterators)
            
            print(f"\nStep {current_iter}: val_loss {metrics['val_loss']:.4f}")
            
            for key, value in metrics.items():
                writer.add_scalar(f"Metrics/{key}", value, current_iter)
            
            train_loss_list.append(metrics["train_loss"])
            validation_loss_list.append(metrics["val_loss"])

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_iter_num = current_iter
                # Saving JAX checkpoint (simplified)
                with open(best_model_params_path, 'wb') as f:
                     f.write(serialization.to_bytes(state.params))

    writer.close()
    print(f"Training completed in {time.time() - start_time:.2f}s")
    print(f"Best model saved at {best_model_params_path}")

    # Save final model parameters
    final_model_params_path = os.path.join(models_dir, f"final_model_jax_{timestamp}.params")
    with open(final_model_params_path, 'wb') as f:
        f.write(serialization.to_bytes(state.params))
    print(f"Final model saved at {final_model_params_path}")

    # Plot
    plt.plot(train_loss_list, "g", label="train_loss")
    plt.plot(validation_loss_list, "r", label="validation_loss")
    plt.savefig(f"{timestamp}_loss_plot_jax.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./tinystories_data")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--max-iters", type=int, default=150_000)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--log-interval", type=int, default=10)
    
    args = parser.parse_args()
    
    # Helper for max_iters calc similar to original
    if args.max_iters == -1:
        # Simplified logic, just assuming user knows what they are doing or copying the logic:
        train_data_path = os.path.join(args.data_dir, "train.bin")
        train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
        num_tokens = len(train_data)
        num_possible_sequences = num_tokens - args.sequence_length
        iters_for_one_epoch = num_possible_sequences // args.batch_size
        args.max_iters = iters_for_one_epoch
        
    import multiprocessing
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass # Context already set

    main(args)
