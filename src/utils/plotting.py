from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


print("plotting module loaded")


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