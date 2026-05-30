"""
Automata deney sonuçlarını birleştiren özet modülü.

Üç senaryo (original, gaussian_noise, controlled_unseen) ve
iki veri seti (SKAB, BATADAL) için tek bir karşılaştırma tablosu üretir.
"""
import pandas as pd

from config.config import (
    METRICS_DIR,
    LOGS_DIR,
)


def get_best_parameter_result(csv_path, dataset_name):
    """
    Parametre analizi CSV'sinden en iyi kombinasyonu döndürür.
    GroupKFold CSV'si varsa (mean±std sütunlu) onu tercih eder.
    """
    df = pd.read_csv(csv_path)

    # GroupKFold CSV formatı: f1_score_mean, f1_score_std sütunları
    if "f1_score_mean" in df.columns:
        best_row = df.sort_values(by="f1_score_mean", ascending=False).iloc[0]
        return {
            "dataset": dataset_name,
            "evaluation": "GroupKFold",
            "best_window_size": best_row["window_size"],
            "best_alphabet_size": best_row["alphabet_size"],
            "best_state_count": best_row.get("state_count_mean", None),
            "best_transition_density": best_row.get("transition_density_mean", None),
            "best_accuracy": best_row["accuracy_mean"],
            "best_accuracy_std": best_row.get("accuracy_std", 0),
            "best_precision": best_row["precision_mean"],
            "best_precision_std": best_row.get("precision_std", 0),
            "best_recall": best_row["recall_mean"],
            "best_recall_std": best_row.get("recall_std", 0),
            "best_f1_score": best_row["f1_score_mean"],
            "best_f1_score_std": best_row.get("f1_score_std", 0),
        }

    # Klasik (single-run) format
    best_row = df.sort_values(by="f1_score", ascending=False).iloc[0]
    return {
        "dataset": dataset_name,
        "evaluation": "single_fold",
        "best_window_size": best_row["window_size"],
        "best_alphabet_size": best_row["alphabet_size"],
        "best_state_count": best_row.get("state_count", None),
        "best_transition_density": best_row.get("transition_density", None),
        "best_accuracy": best_row["accuracy"],
        "best_accuracy_std": 0,
        "best_precision": best_row["precision"],
        "best_precision_std": 0,
        "best_recall": best_row["recall"],
        "best_recall_std": 0,
        "best_f1_score": best_row["f1_score"],
        "best_f1_score_std": 0,
    }


def _load_original_scenario():
    """
    run_skab_automata / run_batadal_automata tarafından kaydedilen
    base run sonuçlarını yükler. Her iki log formatını destekler.
    """
    records = []
    for dataset, fname in [("SKAB", "automata_skab_results.csv"),
                            ("BATADAL", "automata_batadal_results.csv")]:
        path = LOGS_DIR / fname
        if not path.exists():
            continue
        df = pd.read_csv(path, on_bad_lines="skip")
        if df.empty:
            continue
        last = df.iloc[-1]
        # Her iki log formatı için fallback: test_ prefix'li veya düz sütunlar
        accuracy = last.get("accuracy", last.get("test_accuracy", None))
        precision = last.get("precision", last.get("test_precision", None))
        recall = last.get("recall", last.get("test_recall", None))
        f1 = last.get("f1_score", last.get("test_f1_score", None))
        records.append({
            "dataset": dataset,
            "scenario": "original",
            "window_size": last.get("window_size", None),
            "alphabet_size": last.get("alphabet_size", None),
            "accuracy_mean": accuracy,
            "accuracy_std": 0.0,
            "precision_mean": precision,
            "precision_std": 0.0,
            "recall_mean": recall,
            "recall_std": 0.0,
            "f1_score_mean": f1,
            "f1_score_std": 0.0,
        })
    return records


def _load_summary_file(path, scenario_label):
    """Summary CSV'den tek bir senaryo satırı döndürür."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    row = df.iloc[-1].to_dict()
    row["scenario"] = scenario_label
    # Normalize column names to mean/std style
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        if metric in row and f"{metric}_mean" not in row:
            row[f"{metric}_mean"] = row[metric]
            row[f"{metric}_std"] = 0.0
    return row


def build_scenario_comparison_table():
    """
    SKAB ve BATADAL için 3 senaryo × 4 metrik karşılaştırma tablosu oluşturur.
    """
    records = _load_original_scenario()

    noise_files = [
        (LOGS_DIR / "skab_noise_experiment_summary.csv", "SKAB", "gaussian_noise"),
        (LOGS_DIR / "batadal_noise_experiment_summary.csv", "BATADAL", "gaussian_noise"),
    ]
    for path, dataset, scenario in noise_files:
        row = _load_summary_file(path, scenario)
        if row is not None:
            row["dataset"] = dataset
            records.append(row)
        else:
            # Fallback: load per-seed file and compute mean/std
            seed_path = LOGS_DIR / f"{dataset.lower()}_noise_experiment_results.csv"
            if seed_path.exists():
                df = pd.read_csv(seed_path)
                records.append({
                    "dataset": dataset,
                    "scenario": scenario,
                    "window_size": df["window_size"].iloc[0] if "window_size" in df else None,
                    "alphabet_size": df["alphabet_size"].iloc[0] if "alphabet_size" in df else None,
                    "accuracy_mean": df["accuracy"].mean(),
                    "accuracy_std": df["accuracy"].std(),
                    "precision_mean": df["precision"].mean(),
                    "precision_std": df["precision"].std(),
                    "recall_mean": df["recall"].mean(),
                    "recall_std": df["recall"].std(),
                    "f1_score_mean": df["f1_score"].mean(),
                    "f1_score_std": df["f1_score"].std(),
                })

    unseen_files = [
        (LOGS_DIR / "skab_controlled_unseen_summary.csv", "SKAB", "controlled_unseen"),
        (LOGS_DIR / "batadal_controlled_unseen_summary.csv", "BATADAL", "controlled_unseen"),
    ]
    for path, dataset, scenario in unseen_files:
        row = _load_summary_file(path, scenario)
        if row is not None:
            row["dataset"] = dataset
            records.append(row)
        else:
            seed_path = LOGS_DIR / f"{dataset.lower()}_controlled_unseen_results.csv"
            if seed_path.exists():
                df = pd.read_csv(seed_path)
                records.append({
                    "dataset": dataset,
                    "scenario": scenario,
                    "window_size": df["window_size"].iloc[0] if "window_size" in df else None,
                    "alphabet_size": df["alphabet_size"].iloc[0] if "alphabet_size" in df else None,
                    "accuracy_mean": df["accuracy"].mean(),
                    "accuracy_std": df["accuracy"].std(),
                    "precision_mean": df["precision"].mean(),
                    "precision_std": df["precision"].std(),
                    "recall_mean": df["recall"].mean(),
                    "recall_std": df["recall"].std(),
                    "f1_score_mean": df["f1_score"].mean(),
                    "f1_score_std": df["f1_score"].std(),
                })

    keep_cols = [
        "dataset", "scenario", "window_size", "alphabet_size",
        "accuracy_mean", "accuracy_std",
        "precision_mean", "precision_std",
        "recall_mean", "recall_std",
        "f1_score_mean", "f1_score_std",
    ]

    df = pd.DataFrame(records)
    existing_cols = [c for c in keep_cols if c in df.columns]
    return df[existing_cols].reset_index(drop=True)


def main():
    print("Automata özet tabloları oluşturuluyor...")

    # Best parameter summary — SKAB için GKF CSV'si öncelikli
    best_records = []
    for dataset, fname in [
        ("SKAB", "skab_automata_parameter_analysis_gkf.csv"),
        ("BATADAL", "batadal_automata_parameter_analysis.csv"),
    ]:
        path = METRICS_DIR / fname
        if path.exists():
            best_records.append(get_best_parameter_result(path, dataset))

    if best_records:
        best_df = pd.DataFrame(best_records)
        best_path = METRICS_DIR / "automata_best_parameter_summary.csv"
        best_df.to_csv(best_path, index=False)
        print("\nEn iyi parametre özet tablosu:")
        print(best_df)
        print(f"Kaydedildi: {best_path}")

    # Scenario comparison table
    scenario_df = build_scenario_comparison_table()
    scenario_path = METRICS_DIR / "automata_scenario_summary.csv"
    scenario_df.to_csv(scenario_path, index=False)

    print("\nSenaryo özet tablosu (original + noise + unseen):")
    print(scenario_df.to_string(index=False))
    print(f"Kaydedildi: {scenario_path}")

    print("\nAutomata özet tabloları tamamlandı.")


if __name__ == "__main__":
    main()
