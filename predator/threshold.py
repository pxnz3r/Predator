"""
threshold.py - Choosing the decision threshold with a business goal in mind.

A classifier gives a probability, e.g. "this session has a 0.37 chance of
buying". The decision threshold turns it into a yes/no: flag the session
as a buyer if the probability is at least the threshold. The default 0.5
is just a convention. A lower threshold flags more sessions (higher
recall, lower precision); a higher one flags fewer (the opposite).

The scenario: the shop shows a discount offer to every session the model
flags as a likely buyer. We pick the threshold that earns the most profit
under the assumptions below, and compare it with 0.5 and with the
threshold that gives the best F1 score.

To avoid tuning on the test set, thresholds are chosen on the
out-of-fold training predictions (see train.out_of_fold_probabilities)
and only then applied to the test set.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

from predator.data import save_figure
from predator.evaluate import OUTPUTS_FOLDER, get_purchase_probabilities
from predator.models import MODEL_COLORS

# ---------------------------------------------------------------------------
# BUSINESS ASSUMPTIONS
# These numbers are ILLUSTRATIVE ASSUMPTIONS, not facts from the dataset.
# Change them to match a real shop and re-run main.py; the best threshold
# will move accordingly.
# ---------------------------------------------------------------------------

# Profit (not revenue) the shop makes on one average order, in dollars.
AVERAGE_ORDER_PROFIT = 60.0

# What one offer costs the shop (the discount given plus delivery cost),
# in dollars. Paid for every session we target, buyer or not.
OFFER_COST = 6.0

# Share of a targeted buyer's order profit that we credit to the offer:
# the fraction of these sales the offer actually wins or saves (the rest
# would have happened anyway). 0.40 means 40%.
OFFER_CONVERSION_BOOST = 0.40

# Thresholds we try, from 0.01 to 0.99.
THRESHOLDS_TO_TRY = np.round(np.arange(0.01, 1.00, 0.01), 2)


def profit_per_outcome():
    """
    The profit from one session for each of the four possible outcomes.

    - True positive (flagged, really a buyer): profit from the sale the
      offer secures, minus the offer cost.
    - False positive (flagged, not a buyer): the offer cost is wasted.
    - False negative (not flagged, really a buyer): a missed sale. We earn
      nothing extra from the campaign; the lost chance is reported
      separately as "missed profit".
    - True negative (not flagged, not a buyer): nothing happens.

    Returns:
        A dictionary with the profit for "tp", "fp", "fn" and "tn".
    """
    return {
        "tp": AVERAGE_ORDER_PROFIT * OFFER_CONVERSION_BOOST - OFFER_COST,
        "fp": -OFFER_COST,
        "fn": 0.0,
        "tn": 0.0,
    }


def calculate_profit(true_target, probabilities, threshold):
    """
    Work out the campaign profit if we target every session whose
    probability is at least the threshold.

    Inputs:
        true_target: the real labels (0/1).
        probabilities: the model's purchase probabilities.
        threshold: the probability cutoff.
    Returns:
        A dictionary with the total "profit", the counts "tp", "fp", "fn",
        "tn", the number of "targeted" sessions and the "missed_profit"
        from buyers we did not target.
    """
    true_target = np.asarray(true_target)
    predictions = (np.asarray(probabilities) >= threshold).astype(int)

    true_positives = int(((predictions == 1) & (true_target == 1)).sum())
    false_positives = int(((predictions == 1) & (true_target == 0)).sum())
    false_negatives = int(((predictions == 0) & (true_target == 1)).sum())
    true_negatives = int(((predictions == 0) & (true_target == 0)).sum())

    values = profit_per_outcome()
    total_profit = (true_positives * values["tp"] + false_positives * values["fp"]
                    + false_negatives * values["fn"] + true_negatives * values["tn"])
    missed_profit = false_negatives * (AVERAGE_ORDER_PROFIT * OFFER_CONVERSION_BOOST - OFFER_COST)

    return {
        "profit": float(total_profit),
        "tp": true_positives,
        "fp": false_positives,
        "fn": false_negatives,
        "tn": true_negatives,
        "targeted": true_positives + false_positives,
        "missed_profit": float(missed_profit),
    }


def build_threshold_curve(true_target, probabilities):
    """
    Try every threshold in THRESHOLDS_TO_TRY and record profit, precision,
    recall and F1 at each one.

    Inputs:
        true_target: the real labels (0/1).
        probabilities: the model's purchase probabilities.
    Returns:
        A DataFrame with one row per threshold.
    """
    rows = []
    for threshold in THRESHOLDS_TO_TRY:
        predictions = (probabilities >= threshold).astype(int)
        profit_result = calculate_profit(true_target, probabilities, threshold)
        rows.append({
            "threshold": threshold,
            "profit": profit_result["profit"],
            "precision": precision_score(true_target, predictions, zero_division=0),
            "recall": recall_score(true_target, predictions, zero_division=0),
            "f1": f1_score(true_target, predictions, zero_division=0),
        })
    return pd.DataFrame(rows)


def find_best_thresholds(curve):
    """
    Find the threshold with the highest F1 and the one with the highest profit.

    If several thresholds tie, the first (lowest) one is used.

    Input:
        curve: the DataFrame from build_threshold_curve().
    Returns:
        (best_f1_threshold, best_profit_threshold) as floats.
    """
    best_f1_threshold = float(curve.loc[curve["f1"].idxmax(), "threshold"])
    best_profit_threshold = float(curve.loc[curve["profit"].idxmax(), "threshold"])
    return best_f1_threshold, best_profit_threshold


def tune_thresholds(models, training_results, train_target, test_features, test_target):
    """
    Choose the best-F1 and best-profit thresholds for every model and
    compare them with the default 0.5 on the test set.

    Inputs:
        models: dictionary model name -> fitted pipeline.
        training_results: from train.train_all_models() (holds the
            out-of-fold training probabilities).
        train_target: training labels.
        test_features, test_target: the test data.
    Returns:
        (comparison_table, chosen_thresholds, training_curves)
        - comparison_table: DataFrame, also saved to outputs/threshold_comparison.csv
        - chosen_thresholds: model name -> {"default", "best_f1", "best_profit"}
        - training_curves: model name -> curve DataFrame (for plotting)
    """
    print("\nTuning decision thresholds (business angle)...")
    values = profit_per_outcome()
    print("  Assumptions: order profit ${:.0f}, offer cost ${:.0f}, offer conversion boost {:.0%}".format(
        AVERAGE_ORDER_PROFIT, OFFER_COST, OFFER_CONVERSION_BOOST))
    print("  Profit per true positive: ${:.2f}, per false positive: ${:.2f}".format(values["tp"], values["fp"]))

    rows = []
    chosen_thresholds = {}
    training_curves = {}

    for model_name, pipeline in models.items():
        training_probabilities = training_results[model_name]["oof_probabilities"]
        curve = build_threshold_curve(train_target, training_probabilities)
        best_f1_threshold, best_profit_threshold = find_best_thresholds(curve)
        training_curves[model_name] = curve
        chosen_thresholds[model_name] = {
            "default": 0.5,
            "best_f1": best_f1_threshold,
            "best_profit": best_profit_threshold,
        }

        test_probabilities = get_purchase_probabilities(pipeline, test_features)
        for label, threshold in [("Default 0.5", 0.5), ("Best F1", best_f1_threshold),
                                 ("Best profit", best_profit_threshold)]:
            predictions = (test_probabilities >= threshold).astype(int)
            profit_result = calculate_profit(test_target, test_probabilities, threshold)
            rows.append({
                "Model": model_name,
                "Threshold type": label,
                "Threshold": threshold,
                "Precision": precision_score(test_target, predictions, zero_division=0),
                "Recall": recall_score(test_target, predictions, zero_division=0),
                "F1": f1_score(test_target, predictions, zero_division=0),
                "Sessions targeted": profit_result["targeted"],
                "Profit ($)": profit_result["profit"],
            })

    comparison_table = pd.DataFrame(rows).round(3)
    comparison_table.to_csv(os.path.join(OUTPUTS_FOLDER, "threshold_comparison.csv"), index=False)
    print("\nThreshold comparison on the test set ({} sessions):".format(len(test_target)))
    print(comparison_table.to_string(index=False))
    return comparison_table, chosen_thresholds, training_curves


def plot_profit_curves(training_curves, chosen_thresholds):
    """
    Plot profit and F1 against the threshold for every model.

    One small chart per model. The solid line is profit (left axis), the
    dotted line is F1 (right axis). Vertical lines mark the best-profit
    threshold, the best-F1 threshold and the default 0.5.
    The curves come from the out-of-fold training predictions, which is
    where the thresholds were chosen.

    Inputs:
        training_curves: model name -> curve DataFrame.
        chosen_thresholds: model name -> chosen thresholds.
    Returns:
        The path of the saved plot.
    """
    model_names = list(training_curves.keys())
    figure, axes = plt.subplots(1, len(model_names), figsize=(4.6 * len(model_names), 4.4), sharey=True)

    for axis, model_name in zip(axes, model_names):
        curve = training_curves[model_name]
        thresholds = chosen_thresholds[model_name]
        color = MODEL_COLORS[model_name]

        axis.plot(curve["threshold"], curve["profit"], color=color, linewidth=2, label="Profit")
        axis.axhline(0, color="grey", linewidth=0.8)
        axis.axvline(thresholds["best_profit"], color=color, linestyle="--",
                     label="Best profit ({:.2f})".format(thresholds["best_profit"]))
        axis.axvline(thresholds["best_f1"], color="black", linestyle=":",
                     label="Best F1 ({:.2f})".format(thresholds["best_f1"]))
        axis.axvline(0.5, color="grey", linestyle="-.", linewidth=1, label="Default 0.5")
        axis.set_title(model_name, fontsize=11)
        axis.set_xlabel("Threshold")

        f1_axis = axis.twinx()
        f1_axis.plot(curve["threshold"], curve["f1"], color="black", linestyle=":", linewidth=1.2, alpha=0.6)
        f1_axis.set_ylim(0, 1)
        if model_name == model_names[-1]:
            f1_axis.set_ylabel("F1 (dotted line)")
        else:
            f1_axis.set_yticklabels([])
        axis.legend(fontsize=8, loc="lower left")

    axes[0].set_ylabel("Profit on training data ($, out-of-fold)")
    figure.suptitle("Profit and F1 versus decision threshold", fontsize=13)
    figure.tight_layout()
    return save_figure(figure, "profit_vs_threshold.png")
