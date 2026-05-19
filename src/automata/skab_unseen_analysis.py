import pandas as pd

from config.config import (
    SKAB_TRAIN_PC1,
    SKAB_TEST_PC1,
    WINDOW_SIZE,
    ALPHABET_SIZE,
    RESULTS_DIR
)

from src.automata.run_skab_automata import (
    prepare_automata_sequence
)

from src.automata.pattern import get_unique_states

from src.automata.levenshtein import find_nearest_pattern


def main():

    print("SKAB Unseen Pattern Analizi Başladı")

    train_df = pd.read_csv(SKAB_TRAIN_PC1)
    test_df = pd.read_csv(SKAB_TEST_PC1)

    train_pc1 = train_df["PC1"].values
    test_pc1 = test_df["PC1"].values

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

    unseen_results = []

    unseen_count = 0

    for pattern in test_patterns:

        if pattern not in train_states:

            unseen_count += 1

            closest_pattern, distance = find_nearest_pattern(
                pattern,
                train_states
)

            unseen_results.append({
                "unseen_pattern": pattern,
                "mapped_pattern": closest_pattern,
                "distance": distance
            })

    total_patterns = len(test_patterns)

    unseen_ratio = unseen_count / total_patterns

    print(f"\nToplam test pattern sayısı: {total_patterns}")
    print(f"Unseen pattern sayısı: {unseen_count}")
    print(f"Unseen oranı: {unseen_ratio:.4f}")

    unseen_df = pd.DataFrame(unseen_results)

    output_path = (
        RESULTS_DIR /
        "logs" /
        "skab_unseen_patterns.csv"
    )

    unseen_df.to_csv(
        output_path,
        index=False
    )

    print("\nUnseen pattern çıktıları kaydedildi:")
    print(output_path)

    print("\nİlk 10 unseen örneği:")
    print(unseen_df.head(10))


if __name__ == "__main__":
    main()