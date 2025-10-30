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
import time
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM


def evaluate_model(
    model, eval_iters, ctx, data_dir, sequence_length, batch_size, device_type, device
):
    """Calculates loss, perplexity, and accuracy for train and validation splits."""
    metrics = {}
    model.eval()
    with torch.inference_mode():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            correct_predictions = 0
            total_predictions = 0

            for k in range(eval_iters):
                inputs, targets = get_batch(
                    split, data_dir, sequence_length, batch_size, device_type, device
                )
                with ctx:
                    logits, loss = model(inputs, targets)

                losses[k] = loss.item()

                # Calculate accuracy by comparing the predicted token (argmax) with the target token
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == targets).sum().item()
                total_predictions += targets.numel()

            split_loss = losses.mean()
            metrics[f"{split}_loss"] = split_loss
            # Perplexity is the exponential of the cross-entropy loss
            metrics[f"{split}_perplexity"] = torch.exp(split_loss)
            metrics[f"{split}_accuracy"] = correct_predictions / total_predictions

    model.train()  # Switch back to training mode
    return metrics


# Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications


def get_batch(split, data_dir, sequence_length, batch_size, device_type, device):
    """
    Loads a batch of data efficiently using memory-mapping.

    This function memory-maps the binary data file to avoid loading the entire dataset
    into RAM. It then randomly samples starting positions for sequences to form a batch.

    It recreates np.memmap every batch to avoid a memory leak, as per
    https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    """
    # Use np.memmap to treat the file on disk as a NumPy array without loading it all into memory.
    file_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(file_path, dtype=np.uint16, mode="r")

    # Randomly select starting indices for the batches
    ix = torch.randint(len(data) - sequence_length, (batch_size,))
    x = torch.from_numpy(np.stack([data[i : i + sequence_length] for i in ix])).long()
    y = torch.from_numpy(
        np.stack([data[i + 1 : i + 1 + sequence_length] for i in ix])
    ).long()

    # Move tensors to the specified device.
    # If on CUDA, pin memory for faster (asynchronous -> non_blocking=True) transfer.
    if device_type == "cuda":
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


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
    model.to(device)

    # AdamW is a robust optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-9,
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

    # --- Training Loop ---
    train_loss_list = []
    validation_loss_list = []
    print(
        f"Starting training run {timestamp}. Saving best model to {best_model_params_path}"
    )
    start_time = time.time()  # Start the timer
    pbar = tqdm(range(args.max_iters))
    for iter_num in pbar:
        # Accumulate gradients over multiple steps to simulate a larger batch size
        with ctx:
            X, y = get_batch(
                "train",
                args.data_dir,
                args.sequence_length,
                args.batch_size,
                device,
                device,
            )
            _, loss = model(X, y)
            loss = loss / args.gradient_accumulation_steps

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Perform an optimizer step only after accumulating gradients for the specified number of steps.
        if ((iter_num + 1) % args.gradient_accumulation_steps == 0) or (
            iter_num + 1 == args.max_iters
        ):
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
        if iter_num % args.log_interval == 0:
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
        if iter_num % args.eval_interval == 0 and iter_num != 0:
            metrics = evaluate_model(
                model,
                args.eval_iters,
                ctx,
                args.data_dir,
                args.sequence_length,
                args.batch_size,
                device,
                device,
            )

            print(
                f"\nIteration {iter_num}: "
                f"val_loss {metrics['val_loss']:.4f}, "
                f"val_perplexity {metrics['val_perplexity']:.4f}, "
                f"val_accuracy {metrics['val_accuracy']:.4f}"
            )

            # Log all evaluation metrics to TensorBoard
            for key, value in metrics.items():
                writer.add_scalar(f"Metrics/{key}", value, iter_num)

            train_loss_list.append(metrics["train_loss"])
            validation_loss_list.append(metrics["val_loss"])

            # Save a checkpoint if the validation loss has improved
            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                torch.save(model.state_dict(), best_model_params_path)

    writer.close()
    end_time = time.time()  # Stop the timer
    training_duration = end_time - start_time
    print(f"\nTraining completed in {training_duration:.2f} seconds.")
    print(f"Best model saved at: {best_model_params_path}")

    # Create and save a plot of training and validation loss
    plt.plot(train_loss_list, "g", label="train_loss")
    plt.plot(validation_loss_list, "r", label="validation_loss")
    plt.xlabel(f"Iterations (x{iter_num})")
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
