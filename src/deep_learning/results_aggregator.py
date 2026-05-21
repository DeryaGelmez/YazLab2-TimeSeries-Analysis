import re
from pathlib import Path

import numpy as np
import pandas as pd

from config.config import OUTPUTS_METRICS_DIR

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "pr_auc",
]

MODEL_ORDER = ["lstm", "gru", "cnn1d"]
MODEL_LABELS = {
    "lstm": "LSTM",
    "gru": "GRU",
    "cnn1d": "1D-CNN",
}
DATASET_ORDER = ["skab", "batadal"]
DATASET_LABELS = {
    "skab": "SKAB",
    "batadal": "BATADAL",
}


def parse_run_id(run_id: str) -> dict[str, str | int]:
    fold_match = re.search(r"_fold(?P<fold>NA|\d+)$", run_id)
    if fold_match is None:
        raise ValueError(f"Could not parse fold from run_id: {run_id}")

    fold_raw = fold_match.group("fold")
    body = run_id[: fold_match.start()]

    seed_match = re.search(r"_seed(?P<seed>\d+)$", body)
    if seed_match is None:
        raise ValueError(f"Could not parse seed from run_id: {run_id}")

    seed = int(seed_match.group("seed"))
    body = body[: seed_match.start()]

    scenario = None
    for candidate in ("original", "noise", "unseen"):
        suffix = f"_{candidate}"
        if body.endswith(suffix):
            scenario = candidate
            body = body[: -len(suffix)]
            break

    if scenario is None:
        raise ValueError(f"Could not parse scenario from run_id: {run_id}")

    model = None
    for candidate in MODEL_ORDER:
        suffix = f"_{candidate}"
        if body.endswith(suffix):
            model = candidate
            body = body[: -len(suffix)]
            break

    if model is None:
        raise ValueError(f"Could not parse model from run_id: {run_id}")

    dataset = body
    if dataset not in DATASET_LABELS:
        raise ValueError(f"Unknown dataset '{dataset}' in run_id: {run_id}")

    return {
        "dataset": dataset,
        "model": model,
        "scenario": scenario,
        "seed": seed,
        "fold": fold_raw,
    }


def collect_run_metrics(run_dirs: list[Path | str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for run_dir in run_dirs:
        run_dir = Path(run_dir)
        metrics_path = run_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        run_meta = parse_run_id(run_dir.name)
        metrics = pd.read_json(metrics_path, typ="series").to_dict()

        row = {
            **run_meta,
            **{col: metrics.get(col, np.nan) for col in METRIC_COLUMNS},
        }
        rows.append(row)

    if not rows:
        return pd.DataFrame(
            columns=["dataset", "model", "scenario", "seed", "fold", *METRIC_COLUMNS]
        )

    df = pd.DataFrame(rows)
    df["seed"] = df["seed"].astype(int)
    return df


def aggregate_mean_std(
    df: pd.DataFrame,
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["dataset", "model", "scenario"]

    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "f1_mean", "f1_std"])

    agg = (
        df.groupby(group_cols, as_index=False)[METRIC_COLUMNS]
        .agg(["mean", "std"])
        .reset_index()
    )

    agg.columns = [
        col[0] if col[1] == "" else f"{col[0]}_{col[1]}"
        for col in agg.columns.to_flat_index()
    ]
    return agg


def _format_mean_std(mean_value: float, std_value: float) -> str:
    if pd.isna(mean_value):
        return "-"
    if pd.isna(std_value):
        return f"{mean_value:.3f}"
    return f"{mean_value:.3f} ± {std_value:.3f}"


def _lookup_metric(
    agg_df: pd.DataFrame,
    dataset: str,
    model: str,
    scenario: str,
    metric: str = "f1",
) -> str:
    mask = (
        (agg_df["dataset"] == dataset)
        & (agg_df["model"] == model)
        & (agg_df["scenario"] == scenario)
    )
    subset = agg_df.loc[mask]
    if subset.empty:
        return "-"

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    return _format_mean_std(subset.iloc[0][mean_col], subset.iloc[0][std_col])


def export_table_1(agg_df: pd.DataFrame, save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    original_df = agg_df[agg_df["scenario"] == "original"] if not agg_df.empty else agg_df
    rows: list[dict[str, str]] = []

    for model in MODEL_ORDER:
        row = {"Model": MODEL_LABELS[model]}
        for dataset in DATASET_ORDER:
            row[DATASET_LABELS[dataset]] = _lookup_metric(
                original_df, dataset, model, "original", metric="f1"
            )
        rows.append(row)

    table_df = pd.DataFrame(rows)

    markdown_lines = [
        "#### Tablo 1: Model Performansı ve Stabilitesi (Ortalama F1-score ± Standart Sapma)",
        "",
        "| Model | SKAB | BATADAL |",
        "| :--- | :---: | :---: |",
    ]
    for _, row in table_df.iterrows():
        markdown_lines.append(
            f"| **{row['Model']}** | {row['SKAB']} | {row['BATADAL']} |"
        )

    save_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    table_df.to_csv(save_path.with_suffix(".csv"), index=False)


def export_table_2(agg_df: pd.DataFrame, save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []

    for model in MODEL_ORDER:
        original_values: list[str] = []
        noise_values: list[str] = []
        unseen_values: list[str] = []

        for dataset in DATASET_ORDER:
            original = _lookup_metric(agg_df, dataset, model, "original", metric="f1")
            noise = _lookup_metric(agg_df, dataset, model, "noise", metric="f1")
            unseen = _lookup_metric(agg_df, dataset, model, "unseen", metric="f1")

            if original != "-":
                original_values.append(f"{DATASET_LABELS[dataset]}: {original}")
            if noise != "-":
                noise_values.append(f"{DATASET_LABELS[dataset]}: {noise}")
            if unseen != "-":
                unseen_values.append(f"{DATASET_LABELS[dataset]}: {unseen}")

        noise_effect = " / ".join(
            [
                ", ".join(original_values) if original_values else "-",
                ", ".join(noise_values) if noise_values else "-",
            ]
        )
        unseen_analysis = ", ".join(unseen_values) if unseen_values else "-"

        rows.append(
            {
                "Model": MODEL_LABELS[model],
                "Gürültü Etkisi (F1) (Orijinal / Gürültülü)": noise_effect,
                "Unseen Analizi (F1 ± std)": unseen_analysis,
            }
        )

    table_df = pd.DataFrame(rows)

    markdown_lines = [
        "#### Tablo 2: Gürültü Etkisi ve Unseen Senaryo Analizi",
        "",
        "| Model | Gürültü Etkisi (F1) <br> (Orijinal / Gürültülü) | Unseen Analizi <br> (F1 ± std) |",
        "| :--- | :--- | :--- |",
    ]
    for _, row in table_df.iterrows():
        markdown_lines.append(
            f"| **{row['Model']}** | {row['Gürültü Etkisi (F1) (Orijinal / Gürültülü)']} | "
            f"{row['Unseen Analizi (F1 ± std)']} |"
        )

    save_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")
    table_df.to_csv(save_path.with_suffix(".csv"), index=False)


def export_raw_results(df: pd.DataFrame, save_path: Path | str) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)


def generate_report_tables(
    run_dirs: list[Path | str],
    output_dir: Path | str | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir) if output_dir is not None else OUTPUTS_METRICS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = collect_run_metrics(run_dirs)
    agg_df = aggregate_mean_std(raw_df)

    export_raw_results(raw_df, output_dir / "raw_results.csv")
    export_table_1(agg_df, output_dir / "table1.md")
    export_table_2(agg_df, output_dir / "table2.md")

    return raw_df
