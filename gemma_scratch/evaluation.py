import math
import torch


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
