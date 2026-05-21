import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from config.config import OUTPUTS_FIGURES_DIR


@torch.no_grad()
def predict(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device | str,
) -> tuple[list[float], list[float], list[int]]:
    model.eval()
    y_true: list[float] = []
    y_prob: list[float] = []
    y_pred: list[int] = []

    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits).squeeze(1).cpu().tolist()
        labels = batch_y.cpu().tolist()
        preds = [1 if prob >= 0.5 else 0 for prob in probs]

        y_true.extend(labels)
        y_prob.extend(probs)
        y_pred.extend(preds)

    return y_true, y_prob, y_pred


def compute_metrics(
    y_true: list[float] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    y_prob: list[float] | np.ndarray,
) -> dict[str, float]:
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    y_prob_arr = np.asarray(y_prob)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "precision": float(precision_score(y_true_arr, y_pred_arr, zero_division=0)),
        "recall": float(recall_score(y_true_arr, y_pred_arr, zero_division=0)),
        "f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true_arr, y_prob_arr))
    except ValueError:
        metrics["roc_auc"] = float("nan")

    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        metrics["pr_auc"] = float(auc(recall_vals, precision_vals))
    except ValueError:
        metrics["pr_auc"] = float("nan")

    return metrics


def plot_loss_curve(history: dict[str, list[float]], save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_curve(history: dict[str, list[float]], save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_acc"]) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_matrix_dl(
    cm: np.ndarray,
    save_path: Path | str,
    title: str,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_roc_curve(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    save_path: Path | str,
    title: str,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    plt.figure(figsize=(6, 5))
    try:
        fpr, tpr, _ = roc_curve(y_true_arr, y_prob_arr)
        roc_auc = roc_auc_score(y_true_arr, y_prob_arr)
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    except ValueError:
        plt.plot([0, 1], [0, 1], label="AUC = N/A")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pr_curve(
    y_true: list[float] | np.ndarray,
    y_prob: list[float] | np.ndarray,
    save_path: Path | str,
    title: str,
) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)

    plt.figure(figsize=(6, 5))
    try:
        precision_vals, recall_vals, _ = precision_recall_curve(y_true_arr, y_prob_arr)
        pr_auc = auc(recall_vals, precision_vals)
        plt.plot(recall_vals, precision_vals, label=f"AUC = {pr_auc:.3f}")
    except ValueError:
        plt.plot([0, 1], [1, 0], label="AUC = N/A")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def save_run_artifacts(
    run_dir: Path | str,
    history: dict[str, list[float]],
    metrics: dict[str, float],
    model_state_dict: dict[str, torch.Tensor],
    y_true: list[float] | np.ndarray | None = None,
    y_prob: list[float] | np.ndarray | None = None,
) -> None:
    run_dir = Path(run_dir)
    run_id = run_dir.name
    figures_dir = OUTPUTS_FIGURES_DIR / run_id

    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "history.json", "w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    torch.save(model_state_dict, run_dir / "model.pt")

    plot_loss_curve(history, figures_dir / "loss.png")
    plot_accuracy_curve(history, figures_dir / "acc.png")

    if y_true is not None and y_prob is not None:
        y_true_arr = np.asarray(y_true)
        y_prob_arr = np.asarray(y_prob)
        y_pred_arr = (y_prob_arr >= 0.5).astype(int)
        cm = confusion_matrix(y_true_arr, y_pred_arr)

        plot_confusion_matrix_dl(cm, figures_dir / "cm.png", title=f"Confusion Matrix — {run_id}")
        plot_roc_curve(y_true_arr, y_prob_arr, figures_dir / "roc.png", title=f"ROC Curve — {run_id}")
        plot_pr_curve(y_true_arr, y_prob_arr, figures_dir / "pr.png", title=f"PR Curve — {run_id}")
