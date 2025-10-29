"""Gemma from scratch"""

import os
import argparse
from contextlib import nullcontext
from datetime import datetime
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
                X, Y = get_batch(
                    split, data_dir, sequence_length, batch_size, device_type, device
                )
                with ctx:
                    logits, loss = model(X, Y)

                losses[k] = loss.item()

                # Calculate accuracy
                preds = torch.argmax(logits, dim=-1)
                correct_predictions += (preds == Y).sum().item()
                total_predictions += Y.numel()

            split_loss = losses.mean()
            metrics[f"{split}_loss"] = split_loss
            # Calculate perplexity from the mean loss
            metrics[f"{split}_perplexity"] = torch.exp(split_loss)
            metrics[f"{split}_accuracy"] = correct_predictions / total_predictions

    model.train()
    return metrics


# Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications


def get_batch(split, data_dir, sequence_length, batch_size, device_type, device):
    """Loads a batch of data from the appropriate binary file."""
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    file_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(file_path, dtype=np.uint16, mode="r")

    ix = torch.randint(len(data) - sequence_length, (batch_size,))
    x = torch.from_numpy(np.array([data[i : i + sequence_length] for i in ix])).long()
    y = torch.from_numpy(
        np.array([data[i + 1 : i + 1 + sequence_length] for i in ix])
    ).long()

    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
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

    # TensorBoard setup
    writer = SummaryWriter(log_dir=f"runs/gemma_{timestamp}")

    torch.manual_seed(123)

    # Set the device (mps for Apple Silicon, cuda for NVIDIA, cpu as fallback)
    # bfloat16 is good on modern CPUs and GPUs.
    # On CUDA, check for bfloat16 support. MPS always supports it.
    dtype = "bfloat16"
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
        if not torch.cuda.is_bf16_supported():
            dtype = "float16"
    else:
        device = "cpu"

    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    print(f"Using device: {device}")
    print(f"dtype: {dtype}")

    ctx = (
        torch.amp.autocast(device_type=device, dtype=ptdtype)
        if device != "cpu"
        else nullcontext()
    )

    # Enabled for float16, bfloat16 doesn't need it.
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    # --- Model and Optimizer ---
    model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-9,
    )

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
    for iter_num in tqdm(range(args.max_iters)):
        # Forward and backward pass
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

        scaler.scale(loss).backward()

        # Optimizer step
        if ((iter_num + 1) % args.gradient_accumulation_steps == 0) or (
            iter_num + 1 == args.max_iters
        ):
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )  # Using max_norm=1.0 is also a common choice
            writer.add_scalar("Train/gradient_norm", grad_norm, iter_num)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # Logging
        writer.add_scalar(
            "Train/learning_rate", optimizer.param_groups[0]["lr"], iter_num
        )
        writer.add_scalar(
            "Train/loss_step", loss.item() * args.gradient_accumulation_steps, iter_num
        )

        # Evaluation
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

            for key, value in metrics.items():
                writer.add_scalar(f"Metrics/{key}", value, iter_num)

            train_loss_list.append(metrics["train_loss"])
            validation_loss_list.append(metrics["val_loss"])

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                torch.save(model.state_dict(), best_model_params_path)

    writer.close()
    print(f"\nTraining finished. Best model saved at: {best_model_params_path}")

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

    # Preprocessed training data
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
    )
    # max iters of TinyStories: 14_746_016
    parser.add_argument(
        "--warmup_steps", type=int, default=1000, help="Number of warmup steps."
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

    args = parser.parse_args()
    # Override the max_iters argument
    if args.max_iters == -1:
        print("max_iters set to -1. Calculating iterations for one full epoch.")
        train_data_path = os.path.join(args.data_dir, "train.bin")
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
