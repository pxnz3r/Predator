"""
significance.py - Are the differences between models real or just luck?

The test set has 2,466 sessions and only 382 buyers. A different random
test set would give slightly different scores, so a gap of 0.02 F1 may
be noise. Two standard tools tell us:

1. Bootstrap confidence intervals. We draw 1,000 new "test sets" of the
   same size by sampling the real test sessions WITH replacement, score
   the model on each, and take the middle 95% of those scores. If two
   models' intervals overlap a lot, the data cannot separate them well.

2. McNemar's test for two models on the same sessions. Only the
   sessions where the models disagree matter:
       b = sessions model A gets right and model B gets wrong
       c = sessions model B gets right and model A gets wrong
   If the models were equally good, b and c would be about equal. The
   test statistic (|b - c| - 1)^2 / (b + c) follows a chi-squared
   distribution with 1 degree of freedom; the p-value is the chance of a
   difference at least this big if the models were really equal.
"""

import math
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score

from predator.data import RANDOM_STATE
from predator.evaluate import OUTPUTS_FOLDER, get_purchase_probabilities

BOOTSTRAP_ROUNDS = 1000
CONFIDENCE_LEVEL = 0.95


def bootstrap_confidence_intervals(test_target, probabilities, threshold=0.5):
    """
    Bootstrap 95% confidence intervals for F1 and ROC-AUC.

    Inputs:
        test_target: the real labels (0/1).
        probabilities: the model's purchase probabilities.
        threshold: the cutoff used for F1.
    Returns:
        A dictionary with "f1_low", "f1_high", "roc_auc_low", "roc_auc_high".
    """
    true_target = np.asarray(test_target)
    probabilities = np.asarray(probabilities)
    predictions = (probabilities >= threshold).astype(int)
    random_generator = np.random.default_rng(RANDOM_STATE)

    f1_scores = []
    roc_auc_scores = []
    for _ in range(BOOTSTRAP_ROUNDS):
        rows = random_generator.integers(0, len(true_target), size=len(true_target))
        f1_scores.append(f1_score(true_target[rows], predictions[rows], zero_division=0))
        roc_auc_scores.append(roc_auc_score(true_target[rows], probabilities[rows]))

    lower = (1 - CONFIDENCE_LEVEL) / 2 * 100
    upper = (1 + CONFIDENCE_LEVEL) / 2 * 100
    return {
        "f1_low": float(np.percentile(f1_scores, lower)),
        "f1_high": float(np.percentile(f1_scores, upper)),
        "roc_auc_low": float(np.percentile(roc_auc_scores, lower)),
        "roc_auc_high": float(np.percentile(roc_auc_scores, upper)),
    }


def mcnemar_test(test_target, predictions_a, predictions_b):
    """
    McNemar's test (with continuity correction) for two models' predictions.

    For a chi-squared distribution with 1 degree of freedom, the chance of
    a value at least x is erfc(sqrt(x / 2)), so the standard library is enough.

    Inputs:
        test_target: the real labels.
        predictions_a, predictions_b: the 0/1 predictions of the two models.
    Returns:
        A dictionary with "a_right_b_wrong", "b_right_a_wrong", "statistic"
        and "p_value".
    """
    true_target = np.asarray(test_target)
    a_right = np.asarray(predictions_a) == true_target
    b_right = np.asarray(predictions_b) == true_target
    a_right_b_wrong = int((a_right & ~b_right).sum())
    b_right_a_wrong = int((b_right & ~a_right).sum())

    disagreements = a_right_b_wrong + b_right_a_wrong
    if disagreements == 0:
        return {"a_right_b_wrong": 0, "b_right_a_wrong": 0, "statistic": 0.0, "p_value": 1.0}

    statistic = (abs(a_right_b_wrong - b_right_a_wrong) - 1) ** 2 / disagreements
    p_value = math.erfc(math.sqrt(statistic / 2))
    return {
        "a_right_b_wrong": a_right_b_wrong,
        "b_right_a_wrong": b_right_a_wrong,
        "statistic": float(statistic),
        "p_value": float(p_value),
    }


def run_significance_analysis(models, comparison_table, test_features, test_target, pairs_to_compare):
    """
    Add bootstrap intervals to the comparison table and run McNemar's test
    for each pair of models.

    Inputs:
        models: dictionary model name -> fitted pipeline.
        comparison_table: the DataFrame from evaluate.build_comparison_table().
        test_features, test_target: the test data.
        pairs_to_compare: list of (model_a, model_b) name pairs.
    Returns:
        (intervals_table, mcnemar_table), saved to outputs/confidence_intervals.csv
        and outputs/mcnemar_tests.csv.
    """
    print(f"\nBootstrap confidence intervals ({BOOTSTRAP_ROUNDS} resamples of the test set)...")
    probabilities = {}
    rows = []
    for model_name, pipeline in models.items():
        probabilities[model_name] = get_purchase_probabilities(pipeline, test_features)
        intervals = bootstrap_confidence_intervals(test_target, probabilities[model_name])
        scores = comparison_table[comparison_table["Model"] == model_name].iloc[0]
        rows.append(
            {
                "Model": model_name,
                "F1": scores["F1"],
                "F1 95% low": intervals["f1_low"],
                "F1 95% high": intervals["f1_high"],
                "ROC-AUC": scores["ROC-AUC"],
                "ROC-AUC 95% low": intervals["roc_auc_low"],
                "ROC-AUC 95% high": intervals["roc_auc_high"],
            }
        )
    intervals_table = pd.DataFrame(rows).round(3)
    intervals_table.to_csv(os.path.join(OUTPUTS_FOLDER, "confidence_intervals.csv"), index=False)
    print(intervals_table.to_string(index=False))

    print("\nMcNemar's test (threshold 0.5): do two models make different mistakes?")
    rows = []
    for model_a, model_b in pairs_to_compare:
        result = mcnemar_test(
            test_target,
            (probabilities[model_a] >= 0.5).astype(int),
            (probabilities[model_b] >= 0.5).astype(int),
        )
        rows.append(
            {
                "Model A": model_a,
                "Model B": model_b,
                "A right, B wrong": result["a_right_b_wrong"],
                "B right, A wrong": result["b_right_a_wrong"],
                "Chi-squared": round(result["statistic"], 3),
                "p-value": round(result["p_value"], 4),
                "Significant at 5%": result["p_value"] < 0.05,
            }
        )
    mcnemar_table = pd.DataFrame(rows)
    mcnemar_table.to_csv(os.path.join(OUTPUTS_FOLDER, "mcnemar_tests.csv"), index=False)
    print(mcnemar_table.to_string(index=False))
    return intervals_table, mcnemar_table
