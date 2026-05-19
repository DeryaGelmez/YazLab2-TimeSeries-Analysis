import pandas as pd

from config.config import (
    METRICS_DIR,
    LOGS_DIR
)


def get_best_parameter_result(csv_path, dataset_name):
    df = pd.read_csv(csv_path)

    best_row = df.sort_values(
        by="f1_score",
        ascending=False
    ).iloc[0]

    return {
        "dataset": dataset_name,
        "best_window_size": best_row["window_size"],
        "best_alphabet_size": best_row["alphabet_size"],
        "best_state_count": best_row["state_count"],
        "best_transition_density": best_row["transition_density"],
        "best_accuracy": best_row["accuracy"],
        "best_precision": best_row["precision"],
        "best_recall": best_row["recall"],
        "best_f1_score": best_row["f1_score"]
    }


def main():
    print("Automata özet tabloları oluşturuluyor...")

    skab_best = get_best_parameter_result(
        METRICS_DIR / "skab_automata_parameter_analysis.csv",
        "SKAB"
    )

    batadal_best = get_best_parameter_result(
        METRICS_DIR / "batadal_automata_parameter_analysis.csv",
        "BATADAL"
    )

    best_results_df = pd.DataFrame([
        skab_best,
        batadal_best
    ])

    best_results_path = METRICS_DIR / "automata_best_parameter_summary.csv"

    best_results_df.to_csv(
        best_results_path,
        index=False
    )

    print("\nEn iyi parametre özet tablosu:")
    print(best_results_df)
    print("\nKaydedildi:")
    print(best_results_path)

    scenario_files = [
        LOGS_DIR / "skab_noise_experiment_results.csv",
        LOGS_DIR / "skab_controlled_unseen_results.csv",
        LOGS_DIR / "batadal_noise_experiment_results.csv",
        LOGS_DIR / "batadal_controlled_unseen_results.csv"
    ]

    scenario_dfs = []

    for file_path in scenario_files:
        if file_path.exists():
            scenario_dfs.append(pd.read_csv(file_path))

    if scenario_dfs:
        scenario_summary_df = pd.concat(
            scenario_dfs,
            ignore_index=True
        )

        scenario_summary_path = METRICS_DIR / "automata_scenario_summary.csv"

        scenario_summary_df.to_csv(
            scenario_summary_path,
            index=False
        )

        print("\nSenaryo özet tablosu:")
        print(scenario_summary_df)
        print("\nKaydedildi:")
        print(scenario_summary_path)

    print("\nAutomata özet tabloları tamamlandı.")


if __name__ == "__main__":
    main()