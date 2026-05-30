"""
SKAB veri seti üzerinde GroupKFold ile automata değerlendirme.

PDF gereksinimi:
  "SKAB veri seti için source_file sütunu grup değişkeni olarak kullanılmalıdır.
   GroupKFold veya mümkünse StratifiedGroupKFold uygulanmalıdır.
   Sonuçlar fold ortalaması ve standart sapması ile raporlanmalıdır."
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config.config import (
    DEFAULT_WINDOW_SIZE,
    DEFAULT_ALPHABET_SIZE,
    AUTOMATA_PROBABILITY_THRESHOLD,
    METRICS_DIR,
    LOGS_DIR,
)
from src.automata.paa import apply_paa
from src.automata.sax import apply_sax
from src.automata.pattern import extract_patterns, get_unique_states
from src.automata.transition import (
    count_transitions,
    calculate_transition_probabilities,
    calculate_path_probability,
    calculate_transition_density,
)
from src.automata.levenshtein import find_nearest_pattern
from src.utils.metrics import calculate_classification_metrics

RAW_SKAB_PATH = Path("data/raw/SKAB/skab_combined.csv")
N_SPLITS = 5

EXCLUDED_COLS = {"datetime", "changepoint", "source_group", "source_file", "anomaly"}


def prepare_pc1(values, window_size, alphabet_size):
    n_segments = len(values) // window_size
    paa_vals = apply_paa(values, n_segments=n_segments)
    sax_seq = apply_sax(paa_vals, alphabet_size=alphabet_size)
    patterns = extract_patterns(sax_seq, window_size=window_size)
    return patterns


def predict(test_patterns, train_states, transition_probs, threshold):
    """3-state path probability: P(prev→cur) × P(cur→next). Döndürür (predictions, scores)."""
    predictions = []
    scores = []
    for i in range(1, len(test_patterns) - 1):
        prev = test_patterns[i - 1]
        cur = test_patterns[i]
        nxt = test_patterns[i + 1]
        if prev not in train_states:
            prev, _ = find_nearest_pattern(prev, list(train_states))
        if cur not in train_states:
            cur, _ = find_nearest_pattern(cur, list(train_states))
        if nxt not in train_states:
            nxt, _ = find_nearest_pattern(nxt, list(train_states))
        p_prev_cur = transition_probs.get(prev, {}).get(cur, 0.0)
        p_cur_nxt = transition_probs.get(cur, {}).get(nxt, 0.0)
        prob = p_prev_cur * p_cur_nxt
        predictions.append(1 if prob < threshold else 0)
        scores.append(prob)
    return predictions, scores


def run_groupkfold(
    window_size=DEFAULT_WINDOW_SIZE,
    alphabet_size=DEFAULT_ALPHABET_SIZE,
    threshold=AUTOMATA_PROBABILITY_THRESHOLD,
):
    df = pd.read_csv(RAW_SKAB_PATH, sep=";", encoding="cp1254")

    feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
    X = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.mean())

    y = df["anomaly"]
    groups = df["source_file"]

    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train = X.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        pca = PCA(n_components=1)
        train_pc1 = pca.fit_transform(X_train_scaled).ravel()
        test_pc1 = pca.transform(X_test_scaled).ravel()

        train_patterns = prepare_pc1(train_pc1, window_size, alphabet_size)
        test_patterns = prepare_pc1(test_pc1, window_size, alphabet_size)

        train_states = get_unique_states(train_patterns)
        counts = count_transitions(train_patterns)
        probs = calculate_transition_probabilities(counts)
        density = calculate_transition_density(counts, train_states)

        y_pred, _ = predict(test_patterns, train_states, probs, threshold)
        y_true = y_test[1:1 + len(y_pred)]

        metrics = calculate_classification_metrics(y_true, y_pred)

        fold_results.append({
            "fold": fold + 1,
            "window_size": window_size,
            "alphabet_size": alphabet_size,
            "threshold": threshold,
            "state_count": len(train_states),
            "transition_density": density,
            "accuracy": metrics["accuracy"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1_score": metrics["f1_score"],
        })

        print(
            f"  Fold {fold + 1}: F1={metrics['f1_score']:.4f}  "
            f"Recall={metrics['recall']:.4f}  "
            f"States={len(train_states)}"
        )

    results_df = pd.DataFrame(fold_results)

    summary = {
        "dataset": "SKAB",
        "model": "Automata",
        "window_size": window_size,
        "alphabet_size": alphabet_size,
        "threshold": threshold,
        "n_folds": N_SPLITS,
        "accuracy_mean": results_df["accuracy"].mean(),
        "accuracy_std": results_df["accuracy"].std(),
        "precision_mean": results_df["precision"].mean(),
        "precision_std": results_df["precision"].std(),
        "recall_mean": results_df["recall"].mean(),
        "recall_std": results_df["recall"].std(),
        "f1_score_mean": results_df["f1_score"].mean(),
        "f1_score_std": results_df["f1_score"].std(),
        "state_count_mean": results_df["state_count"].mean(),
        "transition_density_mean": results_df["transition_density"].mean(),
    }

    return results_df, summary


def main():
    print(
        f"SKAB Automata GroupKFold Değerlendirmesi "
        f"(window={DEFAULT_WINDOW_SIZE}, alphabet={DEFAULT_ALPHABET_SIZE})"
    )

    fold_df, summary = run_groupkfold()

    print("\nFold Sonuçları:")
    print(fold_df[["fold", "f1_score", "recall", "precision", "state_count"]].to_string(index=False))

    print(
        f"\nGenel Sonuç (mean±std, {N_SPLITS} fold):\n"
        f"  F1:        {summary['f1_score_mean']:.4f} ± {summary['f1_score_std']:.4f}\n"
        f"  Recall:    {summary['recall_mean']:.4f} ± {summary['recall_std']:.4f}\n"
        f"  Precision: {summary['precision_mean']:.4f} ± {summary['precision_std']:.4f}\n"
        f"  Accuracy:  {summary['accuracy_mean']:.4f} ± {summary['accuracy_std']:.4f}"
    )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    fold_df.to_csv(METRICS_DIR / "skab_automata_gkf_fold_results.csv", index=False)
    pd.DataFrame([summary]).to_csv(
        METRICS_DIR / "skab_automata_gkf_summary.csv", index=False
    )

    print("\nSonuçlar kaydedildi.")


if __name__ == "__main__":
    main()
