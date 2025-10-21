"""Gemma from scratch"""

import os
from contextlib import nullcontext
import numpy as np
from tqdm.auto import tqdm
from datasets import load_dataset
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tokenizer import gpt2_tokenizer as enc
from model import Gemma3Model
from config import GEMMA3_CONFIG_CUSTOM


def process(example):
    """encoding text"""
    ids = enc.encode_ordinary(
        example["text"]
    )  # encode_ordinary ignores any special tokens
    out = {"ids": ids, "len": len(ids)}
    return out


def prepare_data(data):
    """preparing a large text dataset for machine learning model training"""
    if not os.path.exists("train.bin"):
        tokenized = data.map(
            process,
            remove_columns=["text"],
            desc="tokenizing the splits",
            num_proc=os.cpu_count(),
        )
        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset["len"], dtype=np.uint64)
            filename = f"{split}.bin"
            dtype = np.uint16  # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))
            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
                # Batch together samples for faster write
                batch = dset.shard(
                    num_shards=total_batches, index=batch_idx, contiguous=True
                )  # .with_format('numpy')
                arr_batch = np.concatenate(batch["ids"])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()


def estimate_loss(model):
    # def estimate_loss(model, eval_iters, ctx)
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ["train", "val"]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


# Some functions from https://github.com/karpathy/nanoGPT/blob/master/train.py with slight modifications
# block size = context window


def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == "train":
        data = np.memmap("train.bin", dtype=np.uint16, mode="r")
    else:
        data = np.memmap("validation.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + 1 + block_size]).astype(np.int64))
            for i in ix
        ]
    )
    if device_type == "cuda":
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = (
            x.pin_memory().to(device, non_blocking=True),
            y.pin_memory().to(device, non_blocking=True),
        )
    else:
        x, y = x.to(device), y.to(device)
    return x, y


if __name__ == "__main__":
    ds = load_dataset("roneneldan/TinyStories")
    prepare_data(ds)

    torch.manual_seed(123)
    model = Gemma3Model(GEMMA3_CONFIG_CUSTOM)

    # Training configuration
    learning_rate = 1e-4  # more stable training, earlier 1e-4
    max_iters = 150_000  # increase from 25000
    warmup_steps = 1000  # smoother initial train, earlier 100
    min_lr = 5e-5  # lower rate, earlier 5e-4
    eval_iters = 500  # increased from 100
    batch_size = 32  # changed from 16, better gradient estimate
    block_size = 128  # changed from 64, capture longer range dependencies

    gradient_accumulation_steps = 32  # reduced from 50

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = (
        "cuda" if "cuda" in device else "cpu"
    )  # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler

    # How to use autocast https://wandb.ai/wandb_fc/tips/reports/How-To-Use-Autocast-in-PyTorch--VmlldzoyMTk4NTky
    # dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    dtype = (
        "bfloat16"
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else "float16"
    )  # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[dtype]

    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )

    # torch.set_default_device(device) TODO: REMOVE
    torch.manual_seed(42)

    ##PUT IN WEIGHT DECAY, CHANGED BETA2 to 0.95
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        eps=1e-9,
    )  # weight decay for regularization

    num_optimizer_steps = max_iters // gradient_accumulation_steps
    warmup_optimizer_steps = warmup_steps // gradient_accumulation_steps

    scheduler_warmup = LinearLR(
        optimizer, total_iters=warmup_steps
    )  # Implement linear warmup
    scheduler_decay = CosineAnnealingLR(
        optimizer, T_max=(num_optimizer_steps - warmup_optimizer_steps), eta_min=min_lr
    )  # Implement lr decay
    scheduler = SequentialLR(
        optimizer,
        schedulers=[scheduler_warmup, scheduler_decay],
        milestones=[warmup_steps],
    )  # Switching from warmup to decay

    # https://stackoverflow.com/questions/72534859/is-gradscaler-necessary-with-mixed-precision-training-with-pytorch
    scaler = torch.amp.GradScaler(device="cuda", enabled=(dtype == "float16"))

    best_val_loss = float("inf")
    best_model_params_path = "best_model_params.pt"
    train_loss_list, validation_loss_list = [], []

    # Ensure model is on the correct device
    model = model.to(device)

    # In your training loop
    for iter_num in tqdm(range(max_iters)):
        with ctx:
            X, y = get_batch("train")
            logits, loss = model(X, y)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        # Step the optimizer and scheduler only after accumulating gradients
        if ((iter_num + 1) % gradient_accumulation_steps == 0) or (
            iter_num + 1 == max_iters
        ):
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if iter_num % eval_iters == 0 and iter_num != 0:
            losses = estimate_loss(model)
            print(
                f"Iteration {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
            # Now, this will report the learning rate as it's being updated by the scheduler
            print(f"The current learning rate: {optimizer.param_groups[0]['lr']:.5f}")
            train_loss_list.append(losses["train"].detach().cpu())
            validation_loss_list.append(losses["val"].detach().cpu())

            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(model.state_dict(), best_model_params_path)

    plt.plot(train_loss_list, "g", label="train_loss")
    plt.plot(validation_loss_list, "r", label="validation_loss")
    plt.xlabel(f"Iterations (x{eval_iters})")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()
