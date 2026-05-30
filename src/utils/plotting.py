from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


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


def plot_state_diagram(
    transition_probabilities,
    save_path,
    threshold=0.1,
    title="Automata State Diagram"
):
    """
    Plots automata state transition diagram.
    """

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    G = nx.DiGraph()

    for current_state, next_states in transition_probabilities.items():

        for next_state, probability in next_states.items():

            if probability >= threshold:

                G.add_edge(
                    current_state,
                    next_state,
                    weight=round(probability, 2)
                )

    plt.figure(figsize=(14, 12))

    pos = nx.spring_layout(
        G,
        seed=42
    )

    nx.draw_networkx_nodes(
        G,
        pos,
        node_size=1200,
        node_color="lightblue"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        font_size=8
    )

    nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowstyle="->",
        arrowsize=15
    )

    edge_labels = nx.get_edge_attributes(
        G,
        "weight"
    )

    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels,
        font_size=7
    )

    plt.title(title)

    plt.axis("off")

    plt.tight_layout()

    plt.savefig(save_path)

    plt.close()


def plot_roc_curve(
    y_true,
    y_scores,
    save_path,
    title="ROC Curve"
):
    """
    Automata modeli için ROC eğrisi çizer.
    y_scores: anomali skoru (yüksek = daha anormal).
              path probability düşük → anomali yüksek olduğundan
              y_scores = 1 - path_probability olarak geçilmeli.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        print(f"  [UYARI] ROC eğrisi atlandı: y_true tek sınıf içeriyor ({title})")
        return float("nan")

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="steelblue", lw=2,
             label=f"ROC (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return roc_auc


def plot_pr_curve(
    y_true,
    y_scores,
    save_path,
    title="Precision-Recall Curve"
):
    """
    Automata modeli için Precision-Recall eğrisi çizer.
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores, dtype=float)

    if len(np.unique(y_true)) < 2:
        print(f"  [UYARI] PR eğrisi atlandı: y_true tek sınıf içeriyor ({title})")
        return float("nan")

    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)

    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, color="darkorange", lw=2,
             label=f"PR (AP = {ap:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return ap