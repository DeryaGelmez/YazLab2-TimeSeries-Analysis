import random
import pandas as pd

from config.config import (
    SKAB_TRAIN_PC1,
    SKAB_TEST_PC1,
    WINDOW_SIZE,
    ALPHABET_SIZE,
    AUTOMATA_PROBABILITY_THRESHOLD,
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

    possible_chars = [
        char for char in alphabet
        if char != current_last_char
    ]

    pattern_chars[-1] = random.choice(possible_chars)

    return "".join(pattern_chars)


def create_controlled_unseen_test_patterns(
    test_patterns,
    train_states,
    alphabet_size,
    unseen_ratio=0.10,
    seed=42
):

    random.seed(seed)

    modified_patterns = test_patterns.copy()

    n_unseen = int(len(test_patterns) * unseen_ratio)

    candidate_indices = list(range(len(test_patterns)))

    selected_indices = random.sample(
        candidate_indices,
        n_unseen
    )

    for idx in selected_indices:

        original_pattern = modified_patterns[idx]
        new_pattern = create_unseen_pattern(
            original_pattern,
            alphabet_size
        )

        attempts = 0

        while new_pattern in train_states and attempts < 10:
            new_pattern = create_unseen_pattern(
                original_pattern,
                alphabet_size
            )
            attempts += 1

        modified_patterns[idx] = new_pattern

    return modified_patterns


def main():
    print("SKAB Kontrollü Unseen Deneyi Başladı")

    print("Kullanılan WINDOW_SIZE:", WINDOW_SIZE)
    print("Kullanılan ALPHABET_SIZE:", ALPHABET_SIZE)

    train_df = pd.read_csv(SKAB_TRAIN_PC1)
    test_df = pd.read_csv(SKAB_TEST_PC1)

    train_pc1 = train_df["PC1"].values
    test_pc1 = test_df["PC1"].values
    y_test_original = test_df["anomaly"].values

    train_patterns = prepare_automata_sequence(
        train_pc1,
        window_size=WINDOW_SIZE,
        alphabet_size=ALPHABET_SIZE
    )

    test_patterns = prepare_automata_sequence(
        test_pc1,
        window_size=WINDOW_SIZE,
        alphabet_size=ALPHABET_SIZE
    )

    train_states = get_unique_states(train_patterns)

    transition_counts = count_transitions(train_patterns)
    transition_probabilities = calculate_transition_probabilities(
        transition_counts
    )

    controlled_test_patterns = create_controlled_unseen_test_patterns(
        test_patterns=test_patterns,
        train_states=train_states,
        alphabet_size=ALPHABET_SIZE,
        unseen_ratio=0.10,
        seed=42
    )

    unseen_records = []

    for pattern in controlled_test_patterns:
        if pattern not in train_states:

            nearest_pattern, distance = find_nearest_pattern(
                pattern,
                train_states
            )

            unseen_records.append({
                "unseen_pattern": pattern,
                "mapped_pattern": nearest_pattern,
                "distance": distance
            })

    print("Toplam test pattern sayısı:", len(controlled_test_patterns))
    print("Kontrollü unseen pattern sayısı:", len(unseen_records))
    print(
        "Kontrollü unseen oranı:",
        len(unseen_records) / len(controlled_test_patterns)
    )

    y_pred, explanations = predict_with_automata(
        controlled_test_patterns,
        train_states,
        transition_probabilities,
        threshold=AUTOMATA_PROBABILITY_THRESHOLD
    )

    y_test = y_test_original[:len(y_pred)]

    metrics = calculate_classification_metrics(
        y_test,
        y_pred
    )

    print("\nKontrollü Unseen Performans Sonuçları:")
    print(metrics)

    unseen_df = pd.DataFrame(unseen_records)
    unseen_output_path = LOGS_DIR / "skab_controlled_unseen_mapping.csv"

    unseen_df.to_csv(
        unseen_output_path,
        index=False
    )

    result_df = pd.DataFrame([{
        "dataset": "SKAB",
        "scenario": "controlled_unseen",
        "window_size": WINDOW_SIZE,
        "alphabet_size": ALPHABET_SIZE,
        "unseen_ratio_target": 0.10,
        "unseen_count": len(unseen_records),
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1_score": metrics["f1_score"]
    }])

    result_output_path = LOGS_DIR / "skab_controlled_unseen_results.csv"

    result_df.to_csv(
        result_output_path,
        index=False
    )

    print("\nUnseen mapping çıktısı kaydedildi:")
    print(unseen_output_path)

    print("\nUnseen deney sonucu kaydedildi:")
    print(result_output_path)

    print("\nİlk 10 unseen mapping örneği:")
    print(unseen_df.head(10))


if __name__ == "__main__":
    main()