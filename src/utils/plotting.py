from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


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