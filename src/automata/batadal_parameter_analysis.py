import pandas as pd

from config.config import (
    BATADAL_TRAIN_PC1,
    BATADAL_TEST_PC1,
    WINDOW_SIZES,
    ALPHABET_SIZES,
    AUTOMATA_PROBABILITY_THRESHOLD,
    METRICS_DIR
)

from src.automata.run_batadal_automata import (
    prepare_automata_sequence,
    predict_with_automata
)

from src.automata.pattern import get_unique_states
from src.automata.transition import (
    count_transitions,
    calculate_transition_probabilities,
    calculate_transition_density
)
from src.utils.metrics import calculate_classification_metrics


def run_single_experiment(
    train_pc1,
    test_pc1,
    y_test_original,
    window_size,
    alphabet_size,
    threshold
):
    train_patterns = prepare_automata_sequence(
        train_pc1,
        window_size=window_size,
        alphabet_size=alphabet_size
    )

    test_patterns = prepare_automata_sequence(
        test_pc1,
        window_size=window_size,
        alphabet_size=alphabet_size
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

    y_pred, _ = predict_with_automata(
        test_patterns,
        train_states,
        transition_probabilities,
        threshold=threshold
    )

    y_test = y_test_original[:len(y_pred)]

    metrics = calculate_classification_metrics(
        y_test,
        y_pred
    )

    return {
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "state_count": len(train_states),
        "transition_density": transition_density,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"]
    }


def main():
    print("BATADAL Automata Parametre Analizi Başladı")

    train_df = pd.read_csv(BATADAL_TRAIN_PC1)
    test_df = pd.read_csv(BATADAL_TEST_PC1)

    train_pc1 = train_df["PC1"].values
    test_pc1 = test_df["PC1"].values
    y_test_original = test_df["anomaly"].values

    results = []

    for window_size in WINDOW_SIZES:
        for alphabet_size in ALPHABET_SIZES:

            print(
                f"Deney çalışıyor | window_size={window_size}, "
                f"alphabet_size={alphabet_size}"
            )

            experiment_result = run_single_experiment(
                train_pc1=train_pc1,
                test_pc1=test_pc1,
                y_test_original=y_test_original,
                window_size=window_size,
                alphabet_size=alphabet_size,
                threshold=AUTOMATA_PROBABILITY_THRESHOLD
            )

            results.append(experiment_result)

    results_df = pd.DataFrame(results)

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = METRICS_DIR / "batadal_automata_parameter_analysis.csv"

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\nBATADAL parametre analizi tamamlandı.")
    print(results_df)
    print("\nSonuçlar kaydedildi:")
    print(output_path)


if __name__ == "__main__":
    main()