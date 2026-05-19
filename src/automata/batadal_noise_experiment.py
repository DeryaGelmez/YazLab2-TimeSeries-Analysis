import numpy as np
import pandas as pd

from config.config import (
    BATADAL_TRAIN_PC1,
    BATADAL_TEST_PC1,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_ALPHABET_SIZE,
    AUTOMATA_PROBABILITY_THRESHOLD,
    GAUSSIAN_NOISE_STD,
    LOGS_DIR
)

from src.automata.run_batadal_automata import (
    prepare_automata_sequence,
    predict_with_automata
)

from src.automata.pattern import get_unique_states
from src.automata.transition import (
    count_transitions,
    calculate_transition_probabilities
)

from src.utils.metrics import calculate_classification_metrics


def add_gaussian_noise(values, noise_std, seed=42):
    np.random.seed(seed)

    noise = np.random.normal(
        loc=0,
        scale=noise_std,
        size=len(values)
    )

    return values + noise


def main():
    print("BATADAL Gaussian Noise Deneyi Başladı")

    train_df = pd.read_csv(BATADAL_TRAIN_PC1)
    test_df = pd.read_csv(BATADAL_TEST_PC1)

    train_pc1 = train_df["PC1"].values
    test_pc1 = test_df["PC1"].values
    y_test_original = test_df["anomaly"].values

    noisy_test_pc1 = add_gaussian_noise(
        values=test_pc1,
        noise_std=GAUSSIAN_NOISE_STD,
        seed=42
    )

    train_patterns = prepare_automata_sequence(
        train_pc1,
        window_size=DEFAULT_WINDOW_SIZE,
        alphabet_size=DEFAULT_ALPHABET_SIZE
    )

    noisy_test_patterns = prepare_automata_sequence(
        noisy_test_pc1,
        window_size=DEFAULT_WINDOW_SIZE,
        alphabet_size=DEFAULT_ALPHABET_SIZE
    )

    train_states = get_unique_states(train_patterns)

    transition_counts = count_transitions(train_patterns)

    transition_probabilities = calculate_transition_probabilities(
        transition_counts
    )

    y_pred, explanations = predict_with_automata(
        noisy_test_patterns,
        train_states,
        transition_probabilities,
        threshold=AUTOMATA_PROBABILITY_THRESHOLD
    )

    y_test = y_test_original[:len(y_pred)]

    metrics = calculate_classification_metrics(
        y_test,
        y_pred
    )

    print("\nGaussian Noise Performans Sonuçları:")
    print(metrics)

    result_df = pd.DataFrame([{
        "dataset": "BATADAL",
        "scenario": "gaussian_noise",
        "model": "Automata",
        "window_size": DEFAULT_WINDOW_SIZE,
        "alphabet_size": DEFAULT_ALPHABET_SIZE,
        "noise_std": GAUSSIAN_NOISE_STD,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"]
    }])

    output_path = LOGS_DIR / "batadal_noise_experiment_results.csv"

    result_df.to_csv(
        output_path,
        index=False
    )

    print("\nGaussian noise deney sonucu kaydedildi:")
    print(output_path)


if __name__ == "__main__":
    main()