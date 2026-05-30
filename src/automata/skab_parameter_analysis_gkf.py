"""
SKAB Automata Parametre Analizi — GroupKFold ile.

PDF gereksinimi:
  "SKAB veri seti için GroupKFold uygulanmalıdır.
   Sonuçlar fold ortalaması ve standart sapması ile raporlanmalıdır."

Tüm 16 parametre kombinasyonu (window: 3-6, alphabet: 3-6) için
5-fold GroupKFold çalıştırılır; her kombinasyon için mean±std raporlanır.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config.config import (
    WINDOW_SIZES,
    ALPHABET_SIZES,
    AUTOMATA_PROBABILITY_THRESHOLD,
    METRICS_DIR,
)
from src.automata.paa import apply_paa
from src.automata.sax import apply_sax
from src.automata.pattern import extract_patterns, get_unique_states
from src.automata.transition import (
    count_transitions,
    calculate_transition_probabilities,
    calculate_transition_density,
)
from src.automata.levenshtein import find_nearest_pattern
from src.utils.metrics import calculate_classification_metrics

RAW_SKAB_PATH = Path("data/raw/SKAB/skab_combined.csv")
N_SPLITS = 5
EXCLUDED_COLS = {"datetime", "changepoint", "source_group", "source_file", "anomaly"}


def _prepare_pc1(values, window_size, alphabet_size):
    n_segments = len(values) // window_size
    paa_vals = apply_paa(values, n_segments=n_segments)
    sax_seq = apply_sax(paa_vals, alphabet_size=alphabet_size)
    return extract_patterns(sax_seq, window_size=window_size)


def _predict(test_patterns, train_states, transition_probs, threshold):
    """3-state path probability prediction."""
    predictions = []
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
        p1 = transition_probs.get(prev, {}).get(cur, 0.0)
        p2 = transition_probs.get(cur, {}).get(nxt, 0.0)
        predictions.append(1 if (p1 * p2) < threshold else 0)
    return predictions


def run_single_combo(df, X, y, groups, window_size, alphabet_size, threshold):
    """Bir (window_size, alphabet_size) kombinasyonu için 5-fold GKF çalıştır."""
    gkf = GroupKFold(n_splits=N_SPLITS)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_train)
        X_te_sc = scaler.transform(X_test)

        pca = PCA(n_components=1)
        train_pc1 = pca.fit_transform(X_tr_sc).ravel()
        test_pc1 = pca.transform(X_te_sc).ravel()

        train_pats = _prepare_pc1(train_pc1, window_size, alphabet_size)
        test_pats = _prepare_pc1(test_pc1, window_size, alphabet_size)

        train_states = get_unique_states(train_pats)
        counts = count_transitions(train_pats)
        probs = calculate_transition_probabilities(counts)
        density = calculate_transition_density(counts, train_states)

        y_pred = _predict(test_pats, train_states, probs, threshold)
        y_true = y_test[1:1 + len(y_pred)]

        m = calculate_classification_metrics(y_true, y_pred)
        fold_metrics.append({
            "fold": fold + 1,
            "state_count": len(train_states),
            "transition_density": density,
            "accuracy": m["accuracy"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1_score": m["f1_score"],
        })

    return fold_metrics


def main():
    print("SKAB Automata Parametre Analizi (GroupKFold) Başladı")
    print(f"Grid: window_sizes={WINDOW_SIZES}, alphabet_sizes={ALPHABET_SIZES}")
    print(f"Toplam kombinasyon: {len(WINDOW_SIZES) * len(ALPHABET_SIZES)} × {N_SPLITS} fold\n")

    df = pd.read_csv(RAW_SKAB_PATH, sep=";", encoding="cp1254")
    feature_cols = [c for c in df.columns if c not in EXCLUDED_COLS]
    X_df = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    X_df = X_df.fillna(X_df.mean())
    X = X_df.values
    y = df["anomaly"].values
    groups = df["source_file"].values

    all_results = []
    fold_level_results = []

    for window_size in WINDOW_SIZES:
        for alphabet_size in ALPHABET_SIZES:
            print(f"  window={window_size}, alphabet={alphabet_size} ", end="", flush=True)

            fold_metrics = run_single_combo(
                df, X, y, groups, window_size, alphabet_size,
                AUTOMATA_PROBABILITY_THRESHOLD
            )

            for fm in fold_metrics:
                fold_level_results.append({
                    "window_size": window_size,
                    "alphabet_size": alphabet_size,
                    **fm,
                })

            fm_df = pd.DataFrame(fold_metrics)
            summary = {
                "window_size": window_size,
                "alphabet_size": alphabet_size,
                "state_count_mean": fm_df["state_count"].mean(),
                "transition_density_mean": fm_df["transition_density"].mean(),
                "accuracy_mean": fm_df["accuracy"].mean(),
                "accuracy_std": fm_df["accuracy"].std(),
                "precision_mean": fm_df["precision"].mean(),
                "precision_std": fm_df["precision"].std(),
                "recall_mean": fm_df["recall"].mean(),
                "recall_std": fm_df["recall"].std(),
                "f1_score_mean": fm_df["f1_score"].mean(),
                "f1_score_std": fm_df["f1_score"].std(),
            }
            all_results.append(summary)

            print(
                f"F1: {summary['f1_score_mean']:.4f} ± {summary['f1_score_std']:.4f}  "
                f"States: {summary['state_count_mean']:.0f}"
            )

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame(all_results)
    out_path = METRICS_DIR / "skab_automata_parameter_analysis_gkf.csv"
    results_df.to_csv(out_path, index=False)

    fold_df = pd.DataFrame(fold_level_results)
    fold_path = METRICS_DIR / "skab_automata_parameter_analysis_gkf_folds.csv"
    fold_df.to_csv(fold_path, index=False)

    best = results_df.sort_values("f1_score_mean", ascending=False).iloc[0]
    print(
        f"\nEn iyi kombinasyon: window={int(best['window_size'])}, "
        f"alphabet={int(best['alphabet_size'])}, "
        f"F1={best['f1_score_mean']:.4f} +/- {best['f1_score_std']:.4f}"
    )
    print(f"\nSonuclar kaydedildi:\n  {out_path}\n  {fold_path}")
    print("\nSKAB Automata Parametre Analizi (GroupKFold) Tamamlandi")


if __name__ == "__main__":
    main()
