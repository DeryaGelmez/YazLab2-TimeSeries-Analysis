import pandas as pd

from config.config import (
    SKAB_TRAIN_PC1,
    SKAB_TEST_PC1,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_ALPHABET_SIZE,
    AUTOMATA_PROBABILITY_THRESHOLD,
    LOGS_DIR,
    FIGURES_DIR
)

from src.automata.paa import apply_paa
from src.automata.sax import apply_sax
from src.automata.pattern import extract_patterns, get_unique_states
from src.automata.transition import (
    count_transitions,
    calculate_transition_probabilities,
    calculate_path_probability,
    calculate_transition_density
)
from src.automata.levenshtein import find_nearest_pattern
from src.utils.metrics import calculate_classification_metrics
from src.utils.logger import log_experiment_result
from src.utils.plotting import (
    plot_confusion_matrix,
    plot_transition_heatmap,
    plot_state_diagram
)


def prepare_automata_sequence(pc1_values, window_size, alphabet_size):
    

    n_segments = len(pc1_values) // window_size

    paa_values = apply_paa(
        pc1_values,
        n_segments=n_segments
    )

    sax_sequence = apply_sax(
        paa_values,
        alphabet_size=alphabet_size
    )

    patterns = extract_patterns(
        sax_sequence,
        window_size=window_size
    )

    return patterns


def predict_with_automata(
    test_patterns,
    train_states,
    transition_probabilities,
    threshold
):

    predictions = []
    mapped_patterns = []

    for i in range(len(test_patterns) - 1):

        current_pattern = test_patterns[i]
        next_pattern = test_patterns[i + 1]

        status = "seen"
        mapped_current = current_pattern
        mapped_next = next_pattern

        if current_pattern not in train_states:
            mapped_current, _ = find_nearest_pattern(
                current_pattern,
                train_states
            )
            status = "unseen"

        if next_pattern not in train_states:
            mapped_next, _ = find_nearest_pattern(
                next_pattern,
                train_states
            )
            status = "unseen"

        probability = calculate_path_probability(
            [mapped_current, mapped_next],
            transition_probabilities
        )

        if probability < threshold:
            prediction = 1
        else:
            prediction = 0

        predictions.append(prediction)

        mapped_patterns.append({
            "time_step": i,
            "state": current_pattern,
            "mapped_state": mapped_current,
            "next_pattern": next_pattern,
            "mapped_next_pattern": mapped_next,
            "status": status,
            "probability": probability,
            "decision": "anomaly" if prediction == 1 else "normal"
        })

    return predictions, mapped_patterns


def main():
    print("SKAB Automata Pipeline Başladı")

    train_df = pd.read_csv(SKAB_TRAIN_PC1)
    test_df = pd.read_csv(SKAB_TEST_PC1)

    train_pc1 = train_df["PC1"].values
    test_pc1 = test_df["PC1"].values

    y_test_original = test_df["anomaly"].values

    print("Train PC1 uzunluğu:", len(train_pc1))
    print("Test PC1 uzunluğu:", len(test_pc1))

    train_patterns = prepare_automata_sequence(
        train_pc1,
        window_size=DEFAULT_WINDOW_SIZE,
        alphabet_size=DEFAULT_ALPHABET_SIZE
    )

    test_patterns = prepare_automata_sequence(
        test_pc1,
        window_size=DEFAULT_WINDOW_SIZE,
        alphabet_size=DEFAULT_ALPHABET_SIZE
    )

    train_states = get_unique_states(train_patterns)

    transition_counts = count_transitions(train_patterns)

    transition_probabilities = calculate_transition_probabilities(
        transition_counts
    )

    transition_density = calculate_transition_density(
        transition_counts,
        train_states
    )

    plot_transition_heatmap(
        transition_probabilities=transition_probabilities,
        save_path=FIGURES_DIR / "skab_automata_transition_heatmap.png",
        title="SKAB Automata Transition Probability Heatmap"
    )

    print("\nTransition probability heatmap kaydedildi.")

    plot_state_diagram(
        transition_probabilities=transition_probabilities,
        save_path=FIGURES_DIR / "skab_automata_state_diagram.png",
        threshold=0.3,
        title="SKAB Automata State Diagram"
    )

    print("\nState diagram kaydedildi.")

    print("Train pattern sayısı:", len(train_patterns))
    print("Test pattern sayısı:", len(test_patterns))
    print("State sayısı:", len(train_states))
    print("Transition density:", transition_density)

    y_pred, explanations = predict_with_automata(
        test_patterns,
        train_states,
        transition_probabilities,
        threshold=AUTOMATA_PROBABILITY_THRESHOLD
    )

    y_test = y_test_original[:len(y_pred)]

    metrics = calculate_classification_metrics(
        y_test,
        y_pred
    )

    print("\nPerformans Sonuçları:")
    print(metrics)

    plot_confusion_matrix(
    confusion_matrix=metrics["confusion_matrix"],
    class_names=["Normal", "Anomaly"],
    save_path=FIGURES_DIR / "skab_automata_confusion_matrix.png",
    title="SKAB Automata Confusion Matrix"
    )

    print("\nConfusion matrix görseli kaydedildi.")

    log_experiment_result(
        LOGS_DIR / "automata_skab_results.csv",
        {
            "dataset": "SKAB",
            "model": "Automata",
            "window_size": DEFAULT_WINDOW_SIZE,
            "alphabet_size": DEFAULT_ALPHABET_SIZE,
            "threshold": AUTOMATA_PROBABILITY_THRESHOLD,
            "state_count": len(train_states),
            "transition_density": transition_density,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        }
    )

    explanations_df = pd.DataFrame(explanations)

    explanation_output_path = LOGS_DIR / "skab_automata_explanations.csv"

    explanations_df.to_csv(
        explanation_output_path,
        index=False,
        encoding="utf-8"
    )

    print("\nAçıklanabilirlik çıktıları kaydedildi:")
    print(explanation_output_path)

    print("\nİlk 5 açıklama örneği:")
    for item in explanations[:5]:
        print(item)

    print("\nSKAB Automata Pipeline Tamamlandı")


if __name__ == "__main__":
    main()