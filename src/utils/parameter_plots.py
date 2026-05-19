import pandas as pd
import matplotlib.pyplot as plt

from config.config import (
    METRICS_DIR,
    FIGURES_DIR
)


def plot_metric_vs_alphabet(
    csv_path,
    metric_name,
    dataset_name,
    save_path
):
    """
    Plots metric values against alphabet size.
    """

    df = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))

    for window_size in sorted(df["window_size"].unique()):

        subset = df[df["window_size"] == window_size]

        plt.plot(
            subset["alphabet_size"],
            subset[metric_name],
            marker="o",
            label=f"window={window_size}"
        )

    plt.xlabel("Alphabet Size")
    plt.ylabel(metric_name)
    plt.title(f"{dataset_name} | {metric_name} vs Alphabet Size")

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def main():

    print("Parametre duyarlılık grafikleri oluşturuluyor...")

    skab_csv = (
        METRICS_DIR /
        "skab_automata_parameter_analysis.csv"
    )

    batadal_csv = (
        METRICS_DIR /
        "batadal_automata_parameter_analysis.csv"
    )

    metrics_to_plot = [
        "f1_score",
        "recall",
        "state_count",
        "transition_density"
    ]

    for metric in metrics_to_plot:

        plot_metric_vs_alphabet(
            csv_path=skab_csv,
            metric_name=metric,
            dataset_name="SKAB",
            save_path=(
                FIGURES_DIR /
                f"skab_{metric}_vs_alphabet.png"
            )
        )

        plot_metric_vs_alphabet(
            csv_path=batadal_csv,
            metric_name=metric,
            dataset_name="BATADAL",
            save_path=(
                FIGURES_DIR /
                f"batadal_{metric}_vs_alphabet.png"
            )
        )

    print("Tüm parametre grafikleri kaydedildi.")


if __name__ == "__main__":
    main()