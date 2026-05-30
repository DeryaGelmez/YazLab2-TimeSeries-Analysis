import json
import pandas as pd

from config.config import (
    BATADAL_TRAIN_PC1,
    BATADAL_VAL_PC1,
    BATADAL_TEST_PC1,
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
    """
    predictions = []
    explanations = []

    for i in range(1, len(test_patterns) - 1):

        prev_pattern = test_patterns[i - 1]
        current_pattern = test_patterns[i]
        next_pattern = test_patterns[i + 1]

        status = "seen"
        mapped_prev = prev_pattern
        mapped_current = current_pattern
        mapped_next = next_pattern

        if prev_pattern not in train_states:
            mapped_prev, _ = find_nearest_pattern(prev_pattern, train_states)
            status = "unseen"

        if current_pattern not in train_states:
            mapped_current, _ = find_nearest_pattern(current_pattern, train_states)
            status = "unseen"

        if next_pattern not in train_states:
            mapped_next, _ = find_nearest_pattern(next_pattern, train_states)
            status = "unseen"

        prob_prev_cur = (
            transition_probabilities.get(mapped_prev, {}).get(mapped_current, 0.0)
        )
        prob_cur_next = (
            transition_probabilities.get(mapped_current, {}).get(mapped_next, 0.0)
        )
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
    print("BATADAL Automata Pipeline Başladı")

    train_df = pd.read_csv(BATADAL_TRAIN_PC1)
    val_df = pd.read_csv(BATADAL_VAL_PC1)
    test_df = pd.read_csv(BATADAL_TEST_PC1)

    train_pc1 = train_df["PC1"].values
    val_pc1 = val_df["PC1"].values
    test_pc1 = test_df["PC1"].values

    y_val_original = val_df["anomaly"].values
    y_test_original = test_df["anomaly"].values

    print("Train PC1 uzunluğu:", len(train_pc1))
    print("Validation PC1 uzunluğu:", len(val_pc1))
    print("Test PC1 uzunluğu:", len(test_pc1))

    train_patterns = prepare_automata_sequence(
        train_pc1,
        window_size=DEFAULT_WINDOW_SIZE,
        alphabet_size=DEFAULT_ALPHABET_SIZE
    )

    val_patterns = prepare_automata_sequence(
        val_pc1,
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
        save_path=FIGURES_DIR / "batadal_automata_transition_heatmap.png",
        title="BATADAL Automata Transition Probability Heatmap"
    )

    print("\nTransition probability heatmap kaydedildi.")

    plot_state_diagram(
        transition_probabilities=transition_probabilities,
        save_path=FIGURES_DIR / "batadal_automata_state_diagram.png",
        threshold=0.3,
        title="BATADAL Automata State Diagram"
    )

    print("\nState diagram kaydedildi.")

    print("Train pattern sayısı:", len(train_patterns))
    print("Validation pattern sayısı:", len(val_patterns))
    print("Test pattern sayısı:", len(test_patterns))
    print("State sayısı:", len(train_states))
    print("Transition density:", transition_density)

    y_val_pred, val_explanations = predict_with_automata(
        val_patterns,
        train_states,
        transition_probabilities,
        threshold=AUTOMATA_PROBABILITY_THRESHOLD
    )

    y_val = y_val_original[1:1 + len(y_val_pred)]

    val_metrics = calculate_classification_metrics(
        y_val,
        y_val_pred
    )

    print("\nValidation Performans Sonuçları:")
    print(val_metrics)


    y_pred, explanations = predict_with_automata(
        test_patterns,
        train_states,
        transition_probabilities,
        threshold=AUTOMATA_PROBABILITY_THRESHOLD
    )

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
    save_path=FIGURES_DIR / "batadal_automata_confusion_matrix.png",
    title="BATADAL Automata Confusion Matrix"
    )

    print("\nConfusion matrix görseli kaydedildi.")

    y_scores = [1.0 - exp["probability"] for exp in explanations]

    roc_auc = plot_roc_curve(
        y_true=y_test,
        y_scores=y_scores,
        save_path=FIGURES_DIR / "batadal_automata_roc_curve.png",
        title="BATADAL Automata ROC Curve"
    )

    ap = plot_pr_curve(
        y_true=y_test,
        y_scores=y_scores,
        save_path=FIGURES_DIR / "batadal_automata_pr_curve.png",
        title="BATADAL Automata Precision-Recall Curve"
    )

    print(f"\nROC AUC: {roc_auc:.4f}  |  Average Precision: {ap:.4f}")
    print("ROC ve PR eğrileri kaydedildi.")

    log_experiment_result(
        LOGS_DIR / "automata_batadal_results.csv",
        {
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1_score": val_metrics["f1_score"],
            "test_accuracy": metrics["accuracy"],
            "test_precision": metrics["precision"],
            "test_recall": metrics["recall"],
            "test_f1_score": metrics["f1_score"],
            "roc_auc": roc_auc,
            "average_precision": ap,
        }
    )

    explanations_df = pd.DataFrame(explanations)

    explanation_output_path = LOGS_DIR / "batadal_automata_explanations.csv"

    explanations_df.to_csv(
        explanation_output_path,
        index=False,
        encoding="utf-8"
    )

    json_output_path = LOGS_DIR / "batadal_automata_explanations.json"
    json_records = []
    for exp in explanations:
        json_records.append({
            "time_step": exp["time_step"],
            "state": exp["state"],
            "pattern": exp["pattern"],
            "status": exp["status"],
            "mapped_to": exp["mapped_to"],
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

    print("\nBATADAL Automata Pipeline Tamamlandı")


if __name__ == "__main__":
    main()