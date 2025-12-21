import os
import time
import torch
from tqdm.auto import tqdm
from gemma_scratch.evaluation import evaluate_model


class GemmaTrainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        scaler,
        ctx,
        train_loader,
        eval_iterators,
        writer,
        args,
        device,
        timestamp,
        models_dir="models",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.ctx = ctx
        self.train_loader = train_loader
        self.eval_iterators = eval_iterators
        self.writer = writer
        self.args = args
        self.device = device
        self.timestamp = timestamp
        self.models_dir = models_dir

        os.makedirs(models_dir, exist_ok=True)
        self.best_model_params_path = os.path.join(
            models_dir, f"best_model_{timestamp}.pt"
        )

        self.train_loss_list = []
        self.validation_loss_list = []

    def train(self):
        train_iter = iter(self.train_loader)
        print(
            f"Starting training run {self.timestamp}. Saving best model to {self.best_model_params_path}"
        )

        start_time = time.time()
        pbar = tqdm(range(self.args.max_iters))
        best_val_loss = float("inf")
        best_iter_num = 0

        for iter_num in pbar:
            is_accumulating = (
                iter_num + 1
            ) % self.args.gradient_accumulation_steps != 0
            is_last_iter = iter_num + 1 == self.args.max_iters

            try:
                inputs, targets = next(train_iter)
            except StopIteration:
                train_iter = iter(self.train_loader)
                inputs, targets = next(train_iter)

            if self.device == "cuda":
                inputs, targets = (
                    inputs.to(self.device, non_blocking=True),
                    targets.to(self.device, non_blocking=True),
                )
            else:
                inputs, targets = inputs.to(self.device), targets.to(self.device)

            with self.ctx:
                _, loss = self.model(inputs, targets)
                loss = loss / self.args.gradient_accumulation_steps

            self.scaler.scale(loss).backward()

            if not is_accumulating or is_last_iter:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=1.0
                )
                self.writer.add_scalar("Train/gradient_norm", grad_norm, iter_num)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()

            if (iter_num + 1) % self.args.log_interval == 0:
                self.writer.add_scalar(
                    "Train/learning_rate",
                    self.optimizer.param_groups[0]["lr"],
                    iter_num,
                )
                self.writer.add_scalar(
                    "Train/loss_step",
                    loss.item() * self.args.gradient_accumulation_steps,
                    iter_num,
                )
                pbar.set_description(
                    f"Loss: {loss.item() * self.args.gradient_accumulation_steps:.4f}"
                )

            if (iter_num + 1) % self.args.eval_interval == 0 and iter_num != 0:
                metrics = evaluate_model(
                    self.model,
                    self.args.eval_iters,
                    self.ctx,
                    self.eval_iterators,
                    self.device,
                )

                print(
                    f"\nIteration {iter_num + 1}: "
                    f"val_loss {metrics['val_loss']:.4f}, "
                    f"val_perplexity {metrics['val_perplexity']:.4f}, "
                    f"val_accuracy {metrics['val_accuracy']:.4f}"
                )

                for key, value in metrics.items():
                    self.writer.add_scalar(f"Metrics/{key}", value, iter_num + 1)

                self.train_loss_list.append(metrics["train_loss"])
                self.validation_loss_list.append(metrics["val_loss"])

                if metrics["val_loss"] < best_val_loss:
                    best_val_loss = metrics["val_loss"]
                    best_iter_num = iter_num + 1
                    torch.save(self.model.state_dict(), self.best_model_params_path)

            if (iter_num + 1) % 1_000_000 == 0:
                millions = (iter_num + 1) // 1_000_000
                checkpoint_path = os.path.join(
                    self.models_dir, f"model_{self.timestamp}@{millions}M.pt"
                )
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"\nSaved periodic checkpoint to {checkpoint_path}")

        self.writer.close()
        end_time = time.time()
        print(f"\nTraining completed in {end_time - start_time:.2f} seconds.")

        final_model_path = os.path.join(
            self.models_dir, f"best_model_{self.timestamp}@{best_iter_num}.pt"
        )
        if os.path.exists(self.best_model_params_path):
            os.rename(self.best_model_params_path, final_model_path)
        print(f"Best model saved at: {final_model_path}")

        return self.train_loss_list, self.validation_loss_list
