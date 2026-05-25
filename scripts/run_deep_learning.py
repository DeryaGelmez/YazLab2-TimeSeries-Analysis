import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

for path in (PROJECT_ROOT, SRC_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import config.config as cfg
from deep_learning.data_loader import build_batadal_cache, build_skab_cache
from deep_learning.experiment_runner import run_single_experiment
from deep_learning.results_aggregator import (
    aggregate_mean_std,
    collect_run_metrics,
    export_raw_results,
    export_table_1,
    export_table_2,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deep learning experiments for SKAB and BATADAL datasets."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["lstm", "gru", "cnn1d"],
        choices=["lstm", "gru", "cnn1d"],
        help="Models to train.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["skab", "batadal"],
        choices=["skab", "batadal"],
        help="Datasets to evaluate.",
    )
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["original", "noise", "unseen"],
        choices=["original", "noise", "unseen"],
        help="Evaluation scenarios.",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Random seeds. Defaults to config.RANDOM_SEEDS.",
    )
    parser.add_argument(
        "--skab-folds",
        nargs="+",
        type=int,
        default=None,
        help="SKAB GroupKFold indices. Defaults to range(DL_SKAB_N_FOLDS).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Debug mode: MAX_EPOCHS=2, single seed, single SKAB fold.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Skip per-run plot generation (metrics.json, history.json, model.pt still saved).",
    )
    return parser.parse_args()


def build_experiment_plan(args: argparse.Namespace) -> tuple[list[tuple], int]:
    seeds = args.seeds if args.seeds is not None else list(cfg.RANDOM_SEEDS)
    skab_folds = (
        args.skab_folds
        if args.skab_folds is not None
        else list(range(cfg.DL_SKAB_N_FOLDS))
    )

    if args.smoke:
        cfg.MAX_EPOCHS = 2
        seeds = [seeds[0]]
        skab_folds = [skab_folds[0]]

    experiments: list[tuple[str, str, str, int, int | None]] = []

    for dataset in args.datasets:
        for model in args.models:
            for scenario in args.scenarios:
                for seed in seeds:
                    if dataset == "skab":
                        for fold in skab_folds:
                            experiments.append((model, dataset, scenario, seed, fold))
                    else:
                        experiments.append((model, dataset, scenario, seed, None))

    return experiments, cfg.MAX_EPOCHS


def main() -> None:
    args = parse_args()
    cfg.create_required_dirs()

    experiments, max_epochs = build_experiment_plan(args)
    total = len(experiments)
    run_dirs: list[Path] = []
    save_plots = not args.fast

    skab_cache = None
    batadal_cache = None
    if "skab" in args.datasets:
        print("Preloading SKAB data and GroupKFold splits...")
        skab_cache = build_skab_cache(cfg.DL_SKAB_N_FOLDS)
        print(
            f"SKAB cached: {skab_cache.X.shape[0]} rows, "
            f"{len(skab_cache.splits)} folds"
        )
    if "batadal" in args.datasets:
        print("Preloading BATADAL data...")
        batadal_cache = build_batadal_cache()
        print(
            f"BATADAL cached: train={len(batadal_cache.X_tr)}, "
            f"val={len(batadal_cache.X_val)}, test={len(batadal_cache.X_te)}"
        )

    print(
        f"Starting deep learning pipeline | runs={total}, max_epochs={max_epochs}, "
        f"device={cfg.DL_DEVICE}, batch_size={cfg.BATCH_SIZE}, "
        f"num_workers={cfg.DL_NUM_WORKERS}, pin_memory={cfg.DL_PIN_MEMORY}, "
        f"amp={cfg.DL_USE_AMP}, save_plots={save_plots}"
    )

    for index, (model, dataset, scenario, seed, fold) in enumerate(experiments, start=1):
        run_id = (
            f"{dataset}_{model}_{scenario}_seed{seed}_"
            f"fold{fold if fold is not None else 'NA'}"
        )
        print(f"[{index}/{total}] {run_id}")

        result = run_single_experiment(
            model_name=model,
            dataset_name=dataset,
            scenario=scenario,
            seed=seed,
            fold=fold,
            skab_cache=skab_cache,
            batadal_cache=batadal_cache,
            save_plots=save_plots,
        )
        run_dirs.append(Path(result["run_dir"]))
        print(f"[{index}/{total}] completed {run_id} | f1={result['f1']:.4f}")

    raw_df = collect_run_metrics(run_dirs)
    agg_df = aggregate_mean_std(raw_df)

    export_raw_results(raw_df, cfg.OUTPUTS_METRICS_DIR / "raw_results.csv")
    export_table_1(agg_df, cfg.OUTPUTS_METRICS_DIR / "table1.md")
    export_table_2(agg_df, cfg.OUTPUTS_METRICS_DIR / "table2.md")

    print(f"Saved report tables to {cfg.OUTPUTS_METRICS_DIR}")


if __name__ == "__main__":
    main()
