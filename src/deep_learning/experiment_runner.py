from typing import Any

import config.config as cfg
from deep_learning.data_loader import (
    BatadalDataCache,
    SkabDataCache,
    build_dataloaders,
    compute_train_pos_weight,
    load_batadal_multivariate,
    load_skab_multivariate,
    make_skab_groupkfold_splits,
    seed_everything,
)
from deep_learning.evaluator import compute_metrics, predict, save_run_artifacts
from deep_learning.models import build_model
from deep_learning.scenarios import apply_scenario
from deep_learning.trainer import fit


def run_single_experiment(
    model_name: str,
    dataset_name: str,
    scenario: str,
    seed: int,
    fold: int | None = None,
    skab_cache: SkabDataCache | None = None,
    batadal_cache: BatadalDataCache | None = None,
    save_plots: bool = True,
    use_weighted_bce: bool = False,
) -> dict[str, Any]:
    seed_everything(seed)
    cfg.create_required_dirs()

    dataset_name = dataset_name.lower()
    model_name = model_name.lower()
    scenario = scenario.lower()

    if dataset_name == "skab":
        if fold is None:
            raise ValueError("fold must be specified when dataset_name is 'skab'.")

        if skab_cache is not None:
            X, y = skab_cache.X, skab_cache.y
            splits = skab_cache.splits
        else:
            X, y, groups = load_skab_multivariate()
            splits = make_skab_groupkfold_splits(X, y, groups, cfg.DL_SKAB_N_FOLDS)

        if fold < 0 or fold >= len(splits):
            raise ValueError(
                f"fold must be between 0 and {len(splits) - 1}, received {fold}."
            )

        train_idx, val_idx, test_idx = splits[fold]
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_te, y_te = X[test_idx], y[test_idx]
    elif dataset_name == "batadal":
        if batadal_cache is not None:
            X_tr = batadal_cache.X_tr
            y_tr = batadal_cache.y_tr
            X_val = batadal_cache.X_val
            y_val = batadal_cache.y_val
            X_te = batadal_cache.X_te
            y_te = batadal_cache.y_te
        else:
            X_tr, y_tr, X_val, y_val, X_te, y_te = load_batadal_multivariate()
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported datasets: skab, batadal")

    n_features = X_tr.shape[1]
    X_te = apply_scenario(scenario, X_te, cfg, seed)

    train_loader, val_loader, test_loader = build_dataloaders(
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

    suffix = cfg.DL_WEIGHTED_BCE_SUFFIX if use_weighted_bce else ""
    run_id = (
        f"{dataset_name}_{model_name}_{scenario}_seed{seed}_"
        f"fold{fold if fold is not None else 'NA'}{suffix}"
    )
    run_dir = cfg.OUTPUTS_DIR / run_id

    pos_weight_value: float | None = None
    if use_weighted_bce:
        pos_weight_value = compute_train_pos_weight(
            y_tr,
            cfg.DL_SEQUENCE_LENGTH,
            cfg.DL_STRIDE,
            max_weight=cfg.DL_POS_WEIGHT_MAX,
        )

    print(f"Starting experiment: {run_id}")
    print(
        f"Train windows: {len(train_loader.dataset)}, "
        f"Val windows: {len(val_loader.dataset)}, "
        f"Test windows: {len(test_loader.dataset)}"
    )
    if use_weighted_bce:
        print(f"Using weighted BCE | pos_weight={pos_weight_value:.4f}")

    model = build_model(model_name, input_size=n_features)
    fit_result = fit(
        model,
        train_loader,
        val_loader,
        cfg.MAX_EPOCHS,
        cfg.EARLY_STOPPING_PATIENCE,
        cfg.DL_LEARNING_RATE,
        cfg.DL_DEVICE,
        seed,
        use_amp=cfg.DL_USE_AMP,
        pos_weight=pos_weight_value,
    )

    history = fit_result["history"]
    best_state_dict = fit_result["best_state_dict"]

    model.load_state_dict(best_state_dict)
    y_true, y_prob, y_pred = predict(model, test_loader, cfg.DL_DEVICE)
    metrics = compute_metrics(y_true, y_pred, y_prob)
    metrics["loss_type"] = "bce_weighted" if use_weighted_bce else "bce"
    metrics["pos_weight"] = float(pos_weight_value if pos_weight_value is not None else 1.0)

    save_run_artifacts(
        run_dir,
        history,
        metrics,
        best_state_dict,
        y_true,
        y_prob,
        save_plots=save_plots,
    )

    print(
        f"Finished experiment: {run_id} | "
        f"f1={metrics['f1']:.4f}, accuracy={metrics['accuracy']:.4f}"
    )

    return {
        **metrics,
        "run_id": run_id,
        "run_dir": str(run_dir),
    }
