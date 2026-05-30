import json
import pandas as pd

from config.config import (
    SKAB_TRAIN_PC1,
    SKAB_TEST_PC1,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_ALPHABET_SIZE,
    AUTOMATA_PROBABILITY_THRESHOLD,
    STATE_DIAGRAM_THRESHOLD,
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
    plot_state_diagram,
    plot_roc_curve,
    plot_pr_curve,
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
    """
    PDF örneğiyle uyumlu 3-state path probability hesabı:
      P(prev → mapped_cur) × P(mapped_cur → mapped_next)

    Döndürür: (predictions, explanations)
      - predictions[j] ↔ test_patterns[j+1] (i=1'den başlar)
      - explanations[j]: her adıma ait tam açıklama sözlüğü
    """
    predictions = []
    explanations = []

    # i=1'den başlarız: prev=i-1, current=i, next=i+1
    for i in range(1, len(test_patterns) - 1):

        prev_pattern = test_patterns[i - 1]
        current_pattern = test_patterns[i]
        next_pattern = test_patterns[i + 1]

        status = "seen"
        mapped_prev = prev_pattern
        mapped_current = current_pattern
        mapped_next = next_pattern
        nearest_distance = None

        if prev_pattern not in train_states:
            mapped_prev, _ = find_nearest_pattern(prev_pattern, train_states)
            status = "unseen"

        if current_pattern not in train_states:
            mapped_current, dist = find_nearest_pattern(current_pattern, train_states)
            nearest_distance = dist
            status = "unseen"

        if next_pattern not in train_states:
            mapped_next, _ = find_nearest_pattern(next_pattern, train_states)
            status = "unseen"

        # Bireysel geçiş olasılıkları
        prob_prev_cur = (
            transition_probabilities.get(mapped_prev, {}).get(mapped_current, 0.0)
        )
        prob_cur_next = (
            transition_probabilities.get(mapped_current, {}).get(mapped_next, 0.0)
        )
        # 3-state path probability = P(prev→cur) × P(cur→next)
        probability = prob_prev_cur * prob_cur_next

        prediction = 1 if probability < threshold else 0
        predictions.append(prediction)

        explanations.append({
            "time_step": i,
            "state": prev_pattern,
            "pattern": current_pattern,
            "mapped_prev": mapped_prev,
            "mapped_current": mapped_current,
            "mapped_next": mapped_next,
            "status": status,
            "mapped_to": mapped_current if status == "unseen" else None,
            "levenshtein_distance": nearest_distance,
            "transition_prev_cur": f"{mapped_prev} -> {mapped_current}",
            "prob_prev_cur": prob_prev_cur,
            "transition_cur_next": f"{mapped_current} -> {mapped_next}",
            "prob_cur_next": prob_cur_next,
            "probability": probability,
            "confidence_score": probability,
            "decision": "anomaly" if prediction == 1 else "normal",
        })

    return predictions, explanations


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
        threshold=STATE_DIAGRAM_THRESHOLD,
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

    # i=1'den başladığı için label hizalaması [1:1+n]
    y_test = y_test_original[1:1 + len(y_pred)]

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

    # ROC ve PR eğrisi: anomali skoru = 1 - path_probability
    y_scores = [1.0 - exp["probability"] for exp in explanations]

    roc_auc = plot_roc_curve(
        y_true=y_test,
        y_scores=y_scores,
        save_path=FIGURES_DIR / "skab_automata_roc_curve.png",
        title="SKAB Automata ROC Curve"
    )

    ap = plot_pr_curve(
        y_true=y_test,
        y_scores=y_scores,
        save_path=FIGURES_DIR / "skab_automata_pr_curve.png",
        title="SKAB Automata Precision-Recall Curve"
    )

    print(f"\nROC AUC: {roc_auc:.4f}  |  Average Precision: {ap:.4f}")
    print("ROC ve PR eğrileri kaydedildi.")

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
            "f1_score": metrics["f1_score"],
            "roc_auc": roc_auc,
            "average_precision": ap,
        }
    )

    explanations_df = pd.DataFrame(explanations)

    explanation_output_path = LOGS_DIR / "skab_automata_explanations.csv"

    explanations_df.to_csv(
        explanation_output_path,
        index=False,
        encoding="utf-8"
    )

    json_output_path = LOGS_DIR / "skab_automata_explanations.json"
    json_records = []
    for exp in explanations:
        json_records.append({
            "time_step": exp["time_step"],
            "state": exp["state"],
            "pattern": exp["pattern"],
            "status": exp["status"],
            "mapped_to": exp["mapped_to"],
            "nearest_distance": exp["levenshtein_distance"],
            "transitions": [
                {"transition": exp["transition_prev_cur"],
                 "probability": round(exp["prob_prev_cur"], 6)},
                {"transition": exp["transition_cur_next"],
                 "probability": round(exp["prob_cur_next"], 6)},
            ],
            "probability": round(exp["probability"], 6),
            "confidence_score": round(exp["confidence_score"], 6),
            "decision": exp["decision"]
        })
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_records, f, indent=2, ensure_ascii=False)

    print("\nAçıklanabilirlik çıktıları kaydedildi:")
    print(f"  CSV: {explanation_output_path}")
    print(f"  JSON: {json_output_path}")

    print("\nİlk 5 açıklama örneği (JSON formatı):")
    for item in json_records[:5]:
        print(json.dumps(item, indent=2))

    print("\nSKAB Automata Pipeline Tamamlandı")


if __name__ == "__main__":
    main()