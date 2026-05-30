"""
İstatistiksel test modülü.

PDF gereksinimi:
  "Uygun durumlarda Wilcoxon veya McNemar testi uygulanarak model farklarının
   istatistiksel anlamlılığı tartışılmalıdır."

- Wilcoxon signed-rank testi: 5 seed üzerinde elde edilen F1 skorlarını
  karşılaştırmak için kullanılır (paired test, n=5 veri noktası).

- McNemar testi: iki modelin prediction vektörlerini karşılaştırmak için
  kullanılır (örn. Automata vs DL, sample bazlı).
"""
import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.contingency_tables import mcnemar


def run_wilcoxon_pairwise(
    results_df: pd.DataFrame,
    model_col: str = "model",
    score_col: str = "f1_score",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Model çiftleri arasında Wilcoxon signed-rank testi uygular.

    Parametreler
    ------------
    results_df : DataFrame
        Her satırın bir seed/fold sonucu olduğu tablo.
        model_col sütunu model adını, score_col sütunu F1 skorunu içermelidir.
    model_col : str
        Model adını içeren sütun.
    score_col : str
        Karşılaştırılacak metrik sütunu (varsayılan: f1_score).
    alpha : float
        Anlamlılık eşiği.

    Döndürür
    --------
    DataFrame
        Model çiftleri, istatistik, p-değeri ve sonuç.
    """
    models = results_df[model_col].unique()
    records = []

    for m1, m2 in combinations(models, 2):
        scores1 = results_df[results_df[model_col] == m1][score_col].values
        scores2 = results_df[results_df[model_col] == m2][score_col].values

        min_len = min(len(scores1), len(scores2))
        scores1 = scores1[:min_len]
        scores2 = scores2[:min_len]

        diff = scores1 - scores2
        if np.all(diff == 0):
            stat, p_value = np.nan, 1.0
        else:
            try:
                stat, p_value = wilcoxon(scores1, scores2)
            except ValueError:
                stat, p_value = np.nan, np.nan

        records.append({
            "model_1": m1,
            "model_2": m2,
            "metric": score_col,
            f"{m1}_mean": scores1.mean(),
            f"{m2}_mean": scores2.mean(),
            "wilcoxon_statistic": stat,
            "p_value": p_value,
            "significant": p_value < alpha if not np.isnan(p_value) else False,
            "alpha": alpha,
        })

    return pd.DataFrame(records)


def run_mcnemar(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    model_a_name: str = "Model A",
    model_b_name: str = "Model B",
    alpha: float = 0.05,
    exact: bool = False,
) -> dict:
    """
    McNemar testi: iki modelin prediction'larını sample bazlı karşılaştırır.

    Parametreler
    ------------
    y_true : ndarray
        Gerçek etiketler.
    y_pred_a, y_pred_b : ndarray
        İki modelin tahminleri.
    exact : bool
        True → exact binomial test; False → chi-square approximation.

    Döndürür
    --------
    dict
        Contingency table, istatistik, p-değeri, sonuç.
    """
    y_true = np.asarray(y_true)
    y_pred_a = np.asarray(y_pred_a)
    y_pred_b = np.asarray(y_pred_b)

    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # 2x2 contingency table
    # [a_correct & b_correct,  a_correct & b_wrong ]
    # [a_wrong   & b_correct,  a_wrong   & b_wrong ]
    n00 = int(np.sum(correct_a & correct_b))
    n01 = int(np.sum(correct_a & ~correct_b))
    n10 = int(np.sum(~correct_a & correct_b))
    n11 = int(np.sum(~correct_a & ~correct_b))

    table = np.array([[n00, n01], [n10, n11]])

    result = mcnemar(table, exact=exact, correction=True)
    stat = result.statistic
    p_value = result.pvalue

    return {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "n_both_correct": n00,
        "n_a_correct_b_wrong": n01,
        "n_a_wrong_b_correct": n10,
        "n_both_wrong": n11,
        "mcnemar_statistic": stat,
        "p_value": p_value,
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "interpretation": (
            f"{model_a_name} ve {model_b_name} arasindaki fark "
            f"{'istatistiksel olarak anlamlidir' if p_value < alpha else 'anlamli degildir'} "
            f"(p={p_value:.4f}, alpha={alpha})."
        ),
    }


def run_wilcoxon_from_raw_results(
    raw_results_path: Path,
    dataset: str,
    scenario: str = "original",
    score_col: str = "f1",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    DL raw_results.csv dosyasından Wilcoxon testi uygulayan yardımcı fonksiyon.
    """
    df = pd.read_csv(raw_results_path)

    if "dataset" in df.columns:
        df = df[df["dataset"].str.lower() == dataset.lower()]
    if "scenario" in df.columns:
        df = df[df["scenario"].str.lower() == scenario.lower()]

    score_col_actual = score_col
    if score_col not in df.columns:
        candidates = [c for c in df.columns if "f1" in c.lower()]
        if candidates:
            score_col_actual = candidates[0]

    model_col = "model" if "model" in df.columns else df.columns[0]

    return run_wilcoxon_pairwise(
        df,
        model_col=model_col,
        score_col=score_col_actual,
        alpha=alpha,
    )
