"""
Gemma from scratch

This script provides a comprehensive training pipeline for a Gemma-like model.
It includes features such as mixed-precision training, gradient accumulation,
learning rate scheduling with warmup and cosine decay, and logging with TensorBoard.
"""

import os
import argparse
from contextlib import nullcontext
from datetime import datetime
from itertools import cycle
import time
import math
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM


class MemmapDataset(Dataset):
    """A PyTorch Dataset that loads sequences from a memory-mapped .bin file."""

    def __init__(self, data_path, sequence_length, dtype=np.uint16):
        # Memory-map the file
        self.data = np.memmap(data_path, dtype=dtype, mode="r")
        self.sequence_length = sequence_length

    def __len__(self):
        # Return the total number of possible sequences
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        # Get a single sequence and its target
        x = torch.from_numpy(
            self.data[idx : idx + self.sequence_length].astype(np.int64)
        )
        y = torch.from_numpy(
            self.data[idx + 1 : idx + 1 + self.sequence_length].astype(np.int64)
        )
        return x, y


def evaluate_model(model, eval_iters, ctx, iterators, device):
    """Calculates loss, perplexity, and accuracy for train and validation splits."""
    metrics = {}
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():
        for split, loader_iter in iterators.items():
            total_loss = 0.0
            total_correct_predictions = 0
            total_predictions = 0

            # Simply get the next batch from the persistent iterator
            for k in range(eval_iters):
                inputs, targets = next(loader_iter)

                # Move data to the correct device
                if device == "cuda":
                    inputs, targets = (
                        inputs.to(device, non_blocking=True),
                        targets.to(device, non_blocking=True),
                    )
                else:
                    inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                with ctx:
                    logits, loss = model(inputs, targets)

                # Accumulate metrics on the GPU
                total_loss += loss
                preds = torch.argmax(logits, dim=-1)
                total_correct_predictions += (preds == targets).sum()
                total_predictions += targets.numel()

            # Transfer final metrics to CPU once
            avg_loss = (total_loss / eval_iters).item()
            accuracy = total_correct_predictions.item() / total_predictions

            metrics[f"{split}_loss"] = avg_loss
            metrics[f"{split}_perplexity"] = math.exp(avg_loss)
            metrics[f"{split}_accuracy"] = accuracy

    model.train()  # Switch back to training mode
    return metrics


def main(args):
    """Main function to orchestrate the training process."""
    # --- Setup ---
    # Create a unique timestamp for this training run for checkpointing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    # Set up a dedicated directory for saving models
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    best_model_params_path = os.path.join(models_dir, f"best_model_{timestamp}.pt")

    # TensorBoard setup for real-time monitoring of training
    writer = SummaryWriter(log_dir=f"runs/gemma_{timestamp}")

    # Set a fixed seed for reproducibility
    torch.manual_seed(args.seed)

    # Determine the optimal device and data type for training
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = "bfloat16"  # MPS always supports bfloat16
    elif torch.cuda.is_available():
        device = "cuda"
        # Use bfloat16 if supported for better performance, else fall back to float16
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        device = "cpu"
        dtype = "bfloat16"  # bfloat16 is also good on modern CPUs

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    print(f"Using device: {device}")
    print(f"dtype: {dtype}")

    # Use autocast for mixed-precision training to save memory and speed up computation
    ctx = (
        torch.amp.autocast(device_type=device, dtype=ptdtype)
        if device != "cpu"
        else nullcontext()
    )

    # GradScaler is only needed for float16 to prevent underflow of small gradients
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    # --- Model and Optimizer ---
    model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)

    # Load checkpoint if specified
    if args.resume_from:
        if os.path.exists(args.resume_from):
            print(f"Loading weights from checkpoint: {args.resume_from}")
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=True)
            
            # Fix the keys if the model has been compiled (remove "_orig_mod." prefix)
            state_dict = {}
            for key, value in checkpoint.items():
                if key.startswith("_orig_mod."):
                    new_key = key.replace("_orig_mod.", "")
                    state_dict[new_key] = value
                else:
                    state_dict[key] = value
            
            model.load_state_dict(state_dict)
        else:
            print(f"Checkpoint file not found: {args.resume_from}")
            return

    model.to(device)

    # Compile the model
    if torch.__version__ >= "2.0":
        print("Compiling the model (this can take a while)")
        model = torch.compile(model)

    # AdamW is a robust optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-9,
        fused=(
            True if device == "cuda" else False
        ),  # Enable fused implementation on CUDA
    )

    # The scheduler implements a linear warmup followed by a cosine decay.
    # This helps stabilize training in the beginning and converge better at the end.
    num_optimizer_steps = args.max_iters // args.gradient_accumulation_steps
    warmup_optimizer_steps = args.warmup_steps // args.gradient_accumulation_steps

    scheduler_warmup = LinearLR(optimizer, total_iters=args.warmup_steps)
    scheduler_decay = CosineAnnealingLR(
        optimizer,
        T_max=(num_optimizer_steps - warmup_optimizer_steps),
        eta_min=args.min_lr,
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[args.warmup_steps],
    )

    best_val_loss = float("inf")

    # --- Create a parallel DataLoader for training data ---
    train_dataset = MemmapDataset(
        os.path.join(args.data_dir, "train.bin"), args.sequence_length
    )
    # Set num_workers > 0 to enable parallel loading. A good starting point is half your CPU cores.
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,  # A safe default
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,  # Reuse workers
    )

    val_dataset = MemmapDataset(
        os.path.join(args.data_dir, "val.bin"), args.sequence_length
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=os.cpu_count() // 2,  # A safe default
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,  # Reuse workers
    )

    # Create a SECOND, DEDICATED data loader for evaluating on the training set.
    # It uses the same dataset but is a completely separate object.
    # No shuffling is needed for evaluation.
    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        num_workers=os.cpu_count() // 2,  # A safe default
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
    )

    # Create persistent iterators for evaluation ONCE before the training loop.
    # This avoids the expensive re-initialization lag.
    eval_iterators = {
        "train": cycle(train_eval_loader),
        "val": cycle(val_loader),
    }

    # --- Training Loop ---
    train_loss_list = []
    validation_loss_list = []
    train_iter = iter(train_loader)
    print(
        f"Starting training run {timestamp}. Saving best model to {best_model_params_path}"
    )
    start_time = time.time()  # Start the timer
    pbar = tqdm(range(args.max_iters))
    best_iter_num = 0
    for iter_num in pbar:
        is_accumulating = (iter_num + 1) % args.gradient_accumulation_steps != 0
        is_last_iter = iter_num + 1 == args.max_iters

        # Get the next batch from the parallel loader
        try:
            inputs, targets = next(train_iter)
        except StopIteration:
            # Re-initialize if the epoch finishes
            train_iter = iter(train_loader)
            inputs, targets = next(train_iter)

        # Move batch to the correct device
        if device == "cuda":
            inputs, targets = (
                inputs.to(device, non_blocking=True),
                targets.to(device, non_blocking=True),
            )
        else:
            inputs, targets = inputs.to(device), targets.to(device)

        # Accumulate gradients over multiple steps to simulate a larger batch size
        with ctx:
            _, loss = model(inputs, targets)
            loss = loss / args.gradient_accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Perform an optimizer step only after accumulating gradients for the specified number of steps.
        if not (is_accumulating) or is_last_iter:
            # Clip gradients to prevent them from exploding, which can destabilize training
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # Using max_norm=1.0 is also a common choice

            # Log gradient norm after it has been computed
            writer.add_scalar("Train/gradient_norm", grad_norm, iter_num)

            # Step the optimizer, update the scaler, and reset gradients
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Log the unscaled loss for interpretability
        if (iter_num + 1) % args.log_interval == 0:
            writer.add_scalar(
                "Train/learning_rate", optimizer.param_groups[0]["lr"], iter_num
            )
            # Log the unscaled loss for interpretability
            writer.add_scalar(
                "Train/loss_step",
                loss.item() * args.gradient_accumulation_steps,
                iter_num,
            )
            pbar.set_description(
                f"Loss: {loss.item() * args.gradient_accumulation_steps:.4f}"
            )

        # Periodically evaluate the model on the validation set to track progress
        if (iter_num + 1) % args.eval_interval == 0 and iter_num != 0:
            metrics = evaluate_model(
                model,
                args.eval_iters,
                ctx,
                eval_iterators,
                device,
            )

            print(
                f"\nIteration {iter_num + 1}: "
                f"val_loss {metrics['val_loss']:.4f}, "
                f"val_perplexity {metrics['val_perplexity']:.4f}, "
                f"val_accuracy {metrics['val_accuracy']:.4f}"
            )

            # Log all evaluation metrics to TensorBoard
            for key, value in metrics.items():
                writer.add_scalar(f"Metrics/{key}", value, iter_num + 1)

            train_loss_list.append(metrics["train_loss"])
            validation_loss_list.append(metrics["val_loss"])

            # Save a checkpoint if the validation loss has improved
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                best_iter_num = iter_num + 1
                torch.save(model.state_dict(), best_model_params_path)

        # Save a checkpoint every 1 million iterations
        if (iter_num + 1) % 1_000_000 == 0:
            millions = (iter_num + 1) // 1_000_000
            checkpoint_path = os.path.join(models_dir, f"model_{timestamp}@{millions}M.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"\nSaved periodic checkpoint to {checkpoint_path}")

    writer.close()
    end_time = time.time()  # Stop the timer
    training_duration = end_time - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds.")
    final_model_path = os.path.join(
        models_dir, f"best_model_{timestamp}@{best_iter_num}.pt"
    )
    os.rename(best_model_params_path, final_model_path)
    print(f"Best model saved at: {final_model_path}")

    # Create and save a plot of training and validation loss
    plt.plot(train_loss_list, "g", label="train_loss")
    plt.plot(validation_loss_list, "r", label="validation_loss")
    plt.xlabel(f"Log Intervals (iter x interval = {args.eval_interval})")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{timestamp}_loss_plot.png")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Gemma-like model from scratch."
    )

    # Data arguments
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./tinystories_data",
        help="Directory with train.bin and val.bin.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Max learning rate."
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=5e-5,
        help="Minimum learning rate for cosine decay.",
    )
    parser.add_argument(
        "--max_iters",
        type=int,
        default=150_000,
        help="Total training iterations. Set to -1 to run for exactly one epoch.",
    )  # max iters of TinyStories: 14_746_016
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )

    # Model and data parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=128,
        help="Context window length (block size).",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=32,
        help="Steps to accumulate gradients.",
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval_interval", type=int, default=500, help="How often to run evaluation."
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        default=200,
        help="Number of iterations for evaluation.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="How often to log training metrics.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint file to load model weights from.",
    )

    args = parser.parse_args()

    # If max_iters is set to -1, calculate the number of iterations for one full epoch.
    if args.max_iters == -1:
        print("max_iters set to -1. Calculating iterations for one full epoch.")
        train_data_path = os.path.join(args.data_dir, "train.bin")
        # Note: This memory-maps the file, not loading it all into RAM, so it's efficient.
        train_data = np.memmap(train_data_path, dtype=np.uint16, mode="r")
        num_tokens = len(train_data)
        num_possible_sequences = num_tokens - args.sequence_length
        iters_for_one_epoch = num_possible_sequences // args.batch_size
        print(f"  Training data has {num_tokens:,} tokens.")
        print(
            f"  Number of possible sequences of length {args.sequence_length}: {num_possible_sequences:,}"
        )
        print(
            f"  With a batch size of {args.batch_size}, one epoch is approx. {iters_for_one_epoch:,} iterations."
        )
        args.max_iters = iters_for_one_epoch

    main(args)
