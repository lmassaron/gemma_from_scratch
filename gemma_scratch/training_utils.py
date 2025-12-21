import os
import torch
import matplotlib.pyplot as plt
from contextlib import nullcontext


def set_seed(seed):
    torch.manual_seed(seed)


def load_checkpoint(model, resume_from_path, device):
    if os.path.exists(resume_from_path):
        print(f"Loading weights from checkpoint: {resume_from_path}")
        checkpoint = torch.load(
            resume_from_path, map_location=device, weights_only=True
        )

        state_dict = {}
        for key, value in checkpoint.items():
            if key.startswith("_orig_mod."):
                new_key = key.replace("_orig_mod.", "")
                state_dict[new_key] = value
            else:
                state_dict[key] = value

        model.load_state_dict(state_dict)
        return True
    else:
        print(f"Checkpoint file not found: {resume_from_path}")
        return False


def get_device_settings():
    """Determines the optimal device and data type for training."""
    if torch.backends.mps.is_available():
        device = "mps"
        dtype = "bfloat16"
    elif torch.cuda.is_available():
        device = "cuda"
        dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    else:
        device = "cpu"
        dtype = "bfloat16"

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

    # GradScaler is only needed for float16
    scaler = torch.amp.GradScaler(enabled=(dtype == "float16"))

    return device, dtype, ctx, scaler


def plot_loss_curves(train_loss_list, validation_loss_list, timestamp, eval_interval):
    plt.plot(train_loss_list, "g", label="train_loss")
    plt.plot(validation_loss_list, "r", label="validation_loss")
    plt.xlabel(f"Log Intervals (iter x interval = {eval_interval})")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"{timestamp}_loss_plot.png")
    plt.close()
