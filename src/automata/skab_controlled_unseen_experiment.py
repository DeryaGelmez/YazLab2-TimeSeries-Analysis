import random
import pandas as pd

from config.config import (
    SKAB_TRAIN_PC1,
    SKAB_TEST_PC1,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_ALPHABET_SIZE,
    AUTOMATA_PROBABILITY_THRESHOLD,
    UNSEEN_RATIO,
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

from src.automata.levenshtein import find_nearest_pattern
from src.utils.metrics import calculate_classification_metrics


def create_unseen_pattern(pattern, alphabet_size):
    alphabet = [chr(i) for i in range(97, 97 + alphabet_size)]
    pattern_chars = list(pattern)
    current_last_char = pattern_chars[-1]
    possible_chars = [char for char in alphabet if char != current_last_char]
    pattern_chars[-1] = random.choice(possible_chars)
    return "".join(pattern_chars)


def create_controlled_unseen_test_patterns(
    test_patterns,
    train_states,
    alphabet_size,
    unseen_ratio=UNSEEN_RATIO,
    seed=42
):
    random.seed(seed)
    modified_patterns = test_patterns.copy()
    n_unseen = int(len(test_patterns) * unseen_ratio)
    candidate_indices = list(range(len(test_patterns)))
    selected_indices = random.sample(candidate_indices, n_unseen)

    for idx in selected_indices:
        original_pattern = modified_patterns[idx]
        new_pattern = create_unseen_pattern(original_pattern, alphabet_size)
        attempts = 0
        while new_pattern in train_states and attempts < 10:
            new_pattern = create_unseen_pattern(original_pattern, alphabet_size)
            attempts += 1
        modified_patterns[idx] = new_pattern

    return modified_patterns


def main():
    print("SKAB Kontrollü Unseen Deneyi Başladı (5 seed)")
    print(f"Kullanılan DEFAULT_WINDOW_SIZE: {DEFAULT_WINDOW_SIZE}")
    print(f"Kullanılan DEFAULT_ALPHABET_SIZE: {DEFAULT_ALPHABET_SIZE}")

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
    test_patterns = prepare_automata_sequence(
        test_pc1,
        window_size=DEFAULT_WINDOW_SIZE,
        alphabet_size=DEFAULT_ALPHABET_SIZE
    )
    train_states = get_unique_states(train_patterns)
    transition_counts = count_transitions(train_patterns)
    transition_probabilities = calculate_transition_probabilities(transition_counts)

    seed_results = []
    all_unseen_records = []

    for seed in RANDOM_SEEDS:
        controlled_test_patterns = create_controlled_unseen_test_patterns(
            test_patterns=test_patterns,
            train_states=train_states,
            alphabet_size=DEFAULT_ALPHABET_SIZE,
            unseen_ratio=UNSEEN_RATIO,
            seed=seed
        )

        unseen_records = []
        for pattern in controlled_test_patterns:
            if pattern not in train_states:
                nearest_pattern, distance = find_nearest_pattern(
                    pattern, train_states
                )
                unseen_records.append({
                    "seed": seed,
                    "unseen_pattern": pattern,
                    "mapped_pattern": nearest_pattern,
                    "distance": distance
                })

        all_unseen_records.extend(unseen_records)

        y_pred, _ = predict_with_automata(
            controlled_test_patterns,
            train_states,
            transition_probabilities,
            threshold=AUTOMATA_PROBABILITY_THRESHOLD
        )

        y_test = y_test_original[1:1 + len(y_pred)]
        metrics = calculate_classification_metrics(y_test, y_pred)

        seed_results.append({
            "seed": seed,
            "dataset": "SKAB",
            "scenario": "controlled_unseen",
            "window_size": DEFAULT_WINDOW_SIZE,
            "alphabet_size": DEFAULT_ALPHABET_SIZE,
            "unseen_ratio_target": 0.10,
            "unseen_count": len(unseen_records),
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"]
        })

        print(
            f"  Seed {seed}: F1={metrics['f1_score']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"Unseen count={len(unseen_records)}"
        )

    results_df = pd.DataFrame(seed_results)

    summary = {
        "dataset": "SKAB",
        "scenario": "controlled_unseen",
        "window_size": DEFAULT_WINDOW_SIZE,
        "alphabet_size": DEFAULT_ALPHABET_SIZE,
        "unseen_ratio_target": 0.10,
        "accuracy_mean": results_df["accuracy"].mean(),
        "accuracy_std": results_df["accuracy"].std(),
        "precision_mean": results_df["precision"].mean(),
        "precision_std": results_df["precision"].std(),
        "recall_mean": results_df["recall"].mean(),
        "recall_std": results_df["recall"].std(),
        "f1_score_mean": results_df["f1_score"].mean(),
        "f1_score_std": results_df["f1_score"].std(),
    }

    print("\nKontrollü Unseen Performans Özeti (mean±std, 5 seed):")
    print(f"  F1: {summary['f1_score_mean']:.4f} ± {summary['f1_score_std']:.4f}")
    print(f"  Recall: {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}")

    results_df.to_csv(LOGS_DIR / "skab_controlled_unseen_results.csv", index=False)
    pd.DataFrame(all_unseen_records).to_csv(
        LOGS_DIR / "skab_controlled_unseen_mapping.csv", index=False
    )
    pd.DataFrame([summary]).to_csv(
        LOGS_DIR / "skab_controlled_unseen_summary.csv", index=False
    )

    print("\nSKAB Kontrollü Unseen Deneyi Tamamlandı")


if __name__ == "__main__":
    main()
