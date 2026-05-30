import numpy as np
import pandas as pd

from config.config import (
    SKAB_TRAIN_PC1,
    SKAB_TEST_PC1,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_ALPHABET_SIZE,
    AUTOMATA_PROBABILITY_THRESHOLD,
    GAUSSIAN_NOISE_STD,
    RANDOM_SEEDS,
    LOGS_DIR
)

from src.automata.run_skab_automata import (
    prepare_automata_sequence,
    predict_with_automata
)

from src.automata.pattern import get_unique_states
from src.automata.transition import (
    count_transitions,
    calculate_transition_probabilities
)

from src.utils.metrics import calculate_classification_metrics


def add_gaussian_noise(values, noise_std, seed):
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0, scale=noise_std, size=len(values))
    return values + noise


def main():
    print("SKAB Gaussian Noise Deneyi Başladı (5 seed)")

    train_df = pd.read_csv(SKAB_TRAIN_PC1)
    test_df = pd.read_csv(SKAB_TEST_PC1)

    train_pc1 = train_df["PC1"].values
    test_pc1 = test_df["PC1"].values
    y_test_original = test_df["anomaly"].values

    train_patterns = prepare_automata_sequence(
        train_pc1,
        window_size=DEFAULT_WINDOW_SIZE,
        alphabet_size=DEFAULT_ALPHABET_SIZE
    )
    train_states = get_unique_states(train_patterns)
    transition_counts = count_transitions(train_patterns)
    transition_probabilities = calculate_transition_probabilities(transition_counts)

    seed_results = []

    for seed in RANDOM_SEEDS:
        noisy_test_pc1 = add_gaussian_noise(
            values=test_pc1,
            noise_std=GAUSSIAN_NOISE_STD,
            seed=seed
        )

        noisy_test_patterns = prepare_automata_sequence(
            noisy_test_pc1,
            window_size=DEFAULT_WINDOW_SIZE,
            alphabet_size=DEFAULT_ALPHABET_SIZE
        )

        y_pred, _ = predict_with_automata(
            noisy_test_patterns,
            train_states,
            transition_probabilities,
            threshold=AUTOMATA_PROBABILITY_THRESHOLD
        )

        y_test = y_test_original[:len(y_pred)]
        metrics = calculate_classification_metrics(y_test, y_pred)

        seed_results.append({
            "seed": seed,
            "dataset": "SKAB",
            "scenario": "gaussian_noise",
            "model": "Automata",
            "window_size": DEFAULT_WINDOW_SIZE,
            "alphabet_size": DEFAULT_ALPHABET_SIZE,
            "noise_std": GAUSSIAN_NOISE_STD,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        })

        print(
            f"  Seed {seed}: F1={metrics['f1_score']:.4f}, "
            f"Recall={metrics['recall']:.4f}"
        )

    results_df = pd.DataFrame(seed_results)

    summary = {
        "dataset": "SKAB",
        "scenario": "gaussian_noise",
        "model": "Automata",
        "window_size": DEFAULT_WINDOW_SIZE,
        "alphabet_size": DEFAULT_ALPHABET_SIZE,
        "noise_std": GAUSSIAN_NOISE_STD,
        "accuracy_mean": results_df["accuracy"].mean(),
        "accuracy_std": results_df["accuracy"].std(),
        "precision_mean": results_df["precision"].mean(),
        "precision_std": results_df["precision"].std(),
        "recall_mean": results_df["recall"].mean(),
        "recall_std": results_df["recall"].std(),
        "f1_score_mean": results_df["f1_score"].mean(),
        "f1_score_std": results_df["f1_score"].std(),
    }

    print("\nGaussian Noise Performans Özeti (mean±std, 5 seed):")
    print(f"  F1: {summary['f1_score_mean']:.4f} ± {summary['f1_score_std']:.4f}")
    print(f"  Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")
    print(f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}")

    output_path = LOGS_DIR / "skab_noise_experiment_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSeed bazlı sonuçlar kaydedildi: {output_path}")

    summary_path = LOGS_DIR / "skab_noise_experiment_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    print(f"Özet kaydedildi: {summary_path}")

    print("\nSKAB Gaussian Noise Deneyi Tamamlandı")


if __name__ == "__main__":
    main()
