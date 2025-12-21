"""
Gemma from scratch

This script provides a comprehensive training pipeline for a Gemma-like model.
It includes features such as mixed-precision training, gradient accumulation,
learning rate scheduling with warmup and cosine decay, and logging with TensorBoard.
"""

import os
import argparse
from datetime import datetime
from itertools import cycle
import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from gemma_scratch.model import Gemma3Model
from gemma_scratch.config import GEMMA3_CONFIG_CUSTOM
from gemma_scratch.dataset import MemmapDataset
from gemma_scratch.trainer import GemmaTrainer
from gemma_scratch.training_utils import (
    set_seed,
    get_device_settings,
    plot_loss_curves,
    load_checkpoint,
)


def main(args):
    """Main function to orchestrate the training process."""
    # --- Setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    writer = SummaryWriter(log_dir=f"runs/gemma_{timestamp}")
    set_seed(args.seed)
    device, dtype, ctx, scaler = get_device_settings()

    # --- Model and Optimizer ---
    model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)

    # Load checkpoint if specified
    if args.resume_from:
        if not load_checkpoint(model, args.resume_from, device):
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
        fused=(True if device == "cuda" else False),
    )

    # Scheduler
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

    # --- Data Loading ---
    train_dataset = MemmapDataset(
        os.path.join(args.data_dir, "train.bin"), args.sequence_length
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=os.cpu_count() // 2,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
    )

    val_dataset = MemmapDataset(
        os.path.join(args.data_dir, "val.bin"), args.sequence_length
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=os.cpu_count() // 2,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
    )

    train_eval_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=os.cpu_count() // 2,
        pin_memory=True if device == "cuda" else False,
        persistent_workers=True,
    )

    eval_iterators = {
        "train": cycle(train_eval_loader),
        "val": cycle(val_loader),
    }

    # --- Training ---
    trainer = GemmaTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ctx=ctx,
        train_loader=train_loader,
        eval_iterators=eval_iterators,
        writer=writer,
        args=args,
        device=device,
        timestamp=timestamp,
    )

    train_loss, val_loss = trainer.train()

    plot_loss_curves(train_loss, val_loss, timestamp, args.eval_interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a Gemma-like model from scratch."
    )

    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./tinystories_data",
        help="Directory with train.bin and val.bin.",
    )

    # Training hyperparameters
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Max learning rate."
    )
    parser.add_argument(
        "--min-lr",
        type=float,
        default=5e-5,
        help="Minimum learning rate for cosine decay.",
    )
    parser.add_argument(
        "--max-iters",
        type=int,
        default=150_000,
        help="Total training iterations. Set to -1 to run for exactly one epoch.",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=1000, help="Number of warmup steps."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for reproducibility."
    )

    # Model and data parameters
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--block-size",
        type=int,
        default=128,
        help="Block sequence lenght.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=32,
        help="Steps to accumulate gradients.",
    )

    # Evaluation parameters
    parser.add_argument(
        "--eval-interval", type=int, default=500, help="How often to run evaluation."
    )
    parser.add_argument(
        "--eval-iters",
        type=int,
        default=200,
        help="Number of iterations for evaluation.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="How often to log training metrics.",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to a checkpoint file to load model weights from.",
    )

    args = parser.parse_args()

    # Fix for sequence_length not being explicitly in arguments but used
    args.sequence_length = args.block_size

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
