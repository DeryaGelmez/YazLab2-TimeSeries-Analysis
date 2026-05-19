from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_confusion_matrix(
    confusion_matrix,
    class_names,
    save_path,
    title="Confusion Matrix"
):
    """
    Plots and saves confusion matrix figure.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_transition_heatmap(
    transition_probabilities,
    save_path,
    title="Transition Probability Heatmap"
):
    """
    Plots and saves transition probability heatmap.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap_df = pd.DataFrame(
        transition_probabilities
    ).fillna(0)

    plt.figure(figsize=(12, 10))

    sns.heatmap(
        heatmap_df,
        cmap="Blues",
        cbar=True
    )

    plt.xlabel("Current State")
    plt.ylabel("Next State")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()