"""
İstatistiksel test scripti.

Kullanım:
    python -m scripts.run_statistical_tests

Çıktı:
    results/metrics/wilcoxon_dl_skab.csv
    results/metrics/wilcoxon_dl_batadal.csv
    results/metrics/statistical_test_summary.json
"""
import json
import pandas as pd
from pathlib import Path

from config.config import OUTPUTS_METRICS_DIR, METRICS_DIR
from src.utils.statistical_tests import run_wilcoxon_pairwise


def load_dl_results(suffix: str = "") -> pd.DataFrame | None:
    path = OUTPUTS_METRICS_DIR / f"raw_results{suffix}.csv"
    if not path.exists():
        print(f"  [UYARI] Dosya bulunamadı: {path}")
        return None
    return pd.read_csv(path)


def run_wilcoxon_for_dataset(df: pd.DataFrame, dataset: str, scenario: str = "original"):
    subset = df[
        (df["dataset"].str.upper() == dataset.upper()) &
        (df["scenario"].str.lower() == scenario.lower())
    ].copy()

    if subset.empty:
        print(f"  [UYARI] {dataset} / {scenario} için veri bulunamadı.")
        return None

    score_col = next(
        (c for c in ["f1", "f1_score", "test_f1"] if c in subset.columns),
        None
    )
    model_col = next(
        (c for c in ["model", "model_name"] if c in subset.columns),
        None
    )

    if score_col is None or model_col is None:
        print(f"  [UYARI] Gerekli sütunlar bulunamadı. Mevcut: {list(subset.columns)}")
        return None

    print(f"  {dataset} / {scenario}: {subset[model_col].unique()}, n={len(subset)}")

    return run_wilcoxon_pairwise(subset, model_col=model_col, score_col=score_col)


def main():
    print("İstatistiksel Testler Başladı\n")

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    summary = {}

    for suffix, grid_name in [("", "Grid1_BCE"), ("_wbce", "Grid2_WBCE")]:
        df = load_dl_results(suffix)
        if df is None:
            continue

        print(f"\n=== {grid_name} ===")

        for dataset in ["SKAB", "BATADAL"]:
            for scenario in ["original", "gaussian_noise", "unseen"]:
                label = f"{grid_name}_{dataset}_{scenario}"
                print(f"\n--- {label} ---")

                result_df = run_wilcoxon_for_dataset(df, dataset, scenario)
                if result_df is None or result_df.empty:
                    continue

                out_path = METRICS_DIR / f"wilcoxon_{grid_name}_{dataset}_{scenario}.csv"
                result_df.to_csv(out_path, index=False)
                print(result_df[["model_1", "model_2", "p_value", "significant"]].to_string(index=False))

                summary[label] = result_df.to_dict(orient="records")

    summary_path = METRICS_DIR / "statistical_test_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n\nÖzet JSON kaydedildi: {summary_path}")
    print("İstatistiksel Testler Tamamlandı")


if __name__ == "__main__":
    main()
