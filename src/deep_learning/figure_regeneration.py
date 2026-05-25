import json
from pathlib import Path

import config.config as cfg
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score

from deep_learning.data_loader import (
    BatadalDataCache,
    SkabDataCache,
    build_batadal_cache,
    build_dataloaders,
    build_skab_cache,
)
from deep_learning.evaluator import (
    plot_accuracy_curve,
    plot_confusion_matrix_dl,
    plot_loss_curve,
    plot_pr_curve,
    plot_roc_curve,
    predict,
)
from deep_learning.models import build_model
from deep_learning.results_aggregator import parse_run_id
from deep_learning.scenarios import apply_scenario


def _fold_to_int(fold_raw: str) -> int | None:
    if fold_raw == "NA":
        return None
    return int(fold_raw)


def _build_test_loader(
    meta: dict[str, str | int],
    skab_cache: SkabDataCache | None,
    batadal_cache: BatadalDataCache | None,
) -> tuple[int, object]:
    dataset = str(meta["dataset"])
    scenario = str(meta["scenario"])
    seed = int(meta["seed"])
    fold = _fold_to_int(str(meta["fold"]))

    if dataset == "skab":
        if fold is None:
            raise ValueError(f"SKAB run requires numeric fold, got {meta['fold']}")
        if skab_cache is None:
            skab_cache = build_skab_cache(cfg.DL_SKAB_N_FOLDS)
        X, y = skab_cache.X, skab_cache.y
        train_idx, val_idx, test_idx = skab_cache.splits[fold]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_te, y_te = X[test_idx], y[test_idx]
    elif dataset == "batadal":
        if batadal_cache is None:
            batadal_cache = build_batadal_cache()
        X_tr = batadal_cache.X_tr
        y_tr = batadal_cache.y_tr
        X_val = batadal_cache.X_val
        y_val = batadal_cache.y_val
        X_te = batadal_cache.X_te
        y_te = batadal_cache.y_te
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    n_features = X_tr.shape[1]
    X_te = apply_scenario(scenario, X_te, cfg, seed)
    _, _, test_loader = build_dataloaders(
        X_tr,
        y_tr,
        X_val,
        y_val,
        X_te,
        y_te,
        cfg.DL_SEQUENCE_LENGTH,
        cfg.DL_STRIDE,
        cfg.BATCH_SIZE,
        seed,
    )
    return n_features, test_loader


def regenerate_history_plots(run_dir: Path, figures_dir: Path) -> None:
    history_path = run_dir / "history.json"
    if not history_path.exists():
        raise FileNotFoundError(f"Missing history.json in {run_dir}")

    with open(history_path, encoding="utf-8") as file:
        history = json.load(file)

    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(history, figures_dir / "loss.png")
    plot_accuracy_curve(history, figures_dir / "acc.png")


def regenerate_test_plots(
    run_dir: Path,
    figures_dir: Path,
    skab_cache: SkabDataCache | None = None,
    batadal_cache: BatadalDataCache | None = None,
) -> dict[str, float]:
    model_path = run_dir / "model.pt"
    metrics_path = run_dir / "metrics.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Missing model.pt in {run_dir}")

    run_id = run_dir.name
    meta = parse_run_id(run_id)
    model_name = str(meta["model"])
    n_features, test_loader = _build_test_loader(meta, skab_cache, batadal_cache)

    model = build_model(model_name, input_size=n_features)
    state_dict = torch.load(model_path, map_location=cfg.DL_DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model = model.to(cfg.DL_DEVICE)

    y_true, y_prob, y_pred = predict(model, test_loader, cfg.DL_DEVICE)
    y_true_arr = np.asarray(y_true)
    y_prob_arr = np.asarray(y_prob)
    y_pred_arr = np.asarray(y_pred)

    figures_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true_arr, y_pred_arr)
    plot_confusion_matrix_dl(cm, figures_dir / "cm.png", title=f"Confusion Matrix — {run_id}")
    plot_roc_curve(y_true_arr, y_prob_arr, figures_dir / "roc.png", title=f"ROC Curve — {run_id}")
    plot_pr_curve(y_true_arr, y_prob_arr, figures_dir / "pr.png", title=f"PR Curve — {run_id}")

    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as file:
            saved = json.load(file)
        return {
            "saved_f1": saved.get("f1"),
            "recomputed_f1": float(f1_score(y_true_arr, y_pred_arr, zero_division=0)),
        }

    return {}


def regenerate_figures_for_run(
    run_id: str,
    outputs_dir: Path | None = None,
    skab_cache: SkabDataCache | None = None,
    batadal_cache: BatadalDataCache | None = None,
    history_only: bool = False,
    test_only: bool = False,
) -> None:
    outputs_dir = outputs_dir or cfg.OUTPUTS_DIR
    run_dir = outputs_dir / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    figures_dir = cfg.OUTPUTS_FIGURES_DIR / run_id

    if not test_only:
        regenerate_history_plots(run_dir, figures_dir)
        print(f"  [A1] loss.png, acc.png <- history.json")

    if not history_only:
        check = regenerate_test_plots(run_dir, figures_dir, skab_cache, batadal_cache)
        print(f"  [A2] cm.png, roc.png, pr.png <- model.pt")
        if check:
            print(
                f"       metrics check: saved_f1={check.get('saved_f1')}, "
                f"recomputed_f1={check.get('recomputed_f1')}"
            )
