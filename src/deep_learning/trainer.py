import copy
from typing import Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deep_learning.data_loader import seed_everything

Mode = Literal["min", "max"]


class EarlyStopping:
    def __init__(self, patience: int, mode: Mode = "min") -> None:
        self.patience = patience
        self.mode = mode
        self.best_score: float | None = None
        self.counter = 0
        self.should_stop = False
        self.best_state_dict: dict[str, torch.Tensor] | None = None

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score
        return score > self.best_score

    def __call__(self, val_loss: float, model: nn.Module) -> None:
        if self._is_improvement(val_loss):
            self.best_score = val_loss
            self.counter = 0
            self.best_state_dict = copy.deepcopy(model.state_dict())
            return

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True


def _binary_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = (torch.sigmoid(logits) >= 0.5).float()
    return (preds == targets).float().mean().item()


def _use_non_blocking(device: torch.device | str) -> bool:
    return str(device).startswith("cuda")


def _to_device(
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    device: torch.device | str,
    non_blocking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_x = batch_x.to(device, non_blocking=non_blocking)
    batch_y = batch_y.to(device, non_blocking=non_blocking).unsqueeze(1)
    return batch_x, batch_y


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device | str,
    use_amp: bool = False,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    non_blocking = _use_non_blocking(device)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    for batch_x, batch_y in loader:
        batch_x, batch_y = _to_device(batch_x, batch_y, device, non_blocking)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and scaler is not None:
            with torch.amp.autocast("cuda"):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_acc += _binary_accuracy(logits.detach(), batch_y)
        num_batches += 1

    if num_batches == 0:
        return {"loss": 0.0, "acc": 0.0}

    return {
        "loss": total_loss / num_batches,
        "acc": total_acc / num_batches,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device | str,
    use_amp: bool = False,
) -> dict[str, object]:
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    num_batches = 0
    y_true: list[float] = []
    y_prob: list[float] = []
    non_blocking = _use_non_blocking(device)

    for batch_x, batch_y in loader:
        batch_x, batch_y = _to_device(batch_x, batch_y, device, non_blocking)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(batch_x)
                loss = criterion(logits, batch_y)
        else:
            logits = model(batch_x)
            loss = criterion(logits, batch_y)

        probs = torch.sigmoid(logits)

        total_loss += loss.item()
        total_acc += _binary_accuracy(logits, batch_y)
        num_batches += 1

        y_true.extend(batch_y.squeeze(1).cpu().tolist())
        y_prob.extend(probs.squeeze(1).cpu().tolist())

    if num_batches == 0:
        return {"loss": 0.0, "acc": 0.0, "y_true": [], "y_prob": []}

    return {
        "loss": total_loss / num_batches,
        "acc": total_acc / num_batches,
        "y_true": y_true,
        "y_prob": y_prob,
    }


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
    patience: int,
    lr: float,
    device: torch.device | str,
    seed: int,
    use_amp: bool = False,
) -> dict[str, object]:
    seed_everything(seed)

    if use_amp and not str(device).startswith("cuda"):
        use_amp = False

    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    early_stopping = EarlyStopping(patience=patience, mode="min")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
    }

    for epoch in range(max_epochs):
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, device, use_amp=use_amp
        )
        val_metrics = evaluate(
            model, val_loader, criterion, device, use_amp=use_amp
        )

        history["train_loss"].append(train_metrics["loss"])
        history["val_loss"].append(val_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["val_acc"].append(val_metrics["acc"])

        print(
            f"Epoch {epoch + 1}/{max_epochs} | "
            f"train_loss={train_metrics['loss']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['acc']:.4f}, val_acc={val_metrics['acc']:.4f}"
        )

        early_stopping(val_metrics["loss"], model)
        if early_stopping.should_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}.")
            break

    best_state_dict = early_stopping.best_state_dict
    if best_state_dict is None:
        best_state_dict = copy.deepcopy(model.state_dict())

    return {
        "history": history,
        "best_state_dict": best_state_dict,
    }
