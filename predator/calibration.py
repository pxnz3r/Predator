"""
calibration.py - Can we trust the probabilities, not just the ranking?

ROC-AUC only checks the ORDER of the probabilities: buyers should get
higher numbers than non-buyers. Calibration checks the numbers
themselves: of all sessions that get a probability of about 0.3, about
30% should really buy.

This matters for the business threshold. The break-even rule in
threshold.break_even_probability() ("send the offer when p is above
cost / (profit x boost)") only works with calibrated probabilities.

Two of our models are miscalibrated on purpose: the Decision Tree uses
class_weight="balanced" and XGBoost uses scale_pos_weight. Both make
buyers count more during training, which pushes every probability up.

We measure calibration with:
- a reliability diagram: predicted probability (x) against the real
  purchase rate (y) in bins; a calibrated model follows the diagonal.
- the Brier score: the mean squared error between the probability and
  the 0/1 outcome (lower is better).

Then we fix it with isotonic regression (CalibratedClassifierCV), which
learns a monotone mapping from the raw probability to the real rate on
cross-validation folds of the training data.
"""

import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score

from predator.data import save_figure
from predator.evaluate import OUTPUTS_FOLDER, get_purchase_probabilities
from predator.models import MODEL_COLORS
from predator.threshold import break_even_probability, calculate_profit
from predator.train import make_folds

NUMBER_OF_BINS = 10


def calibrate_model(pipeline, train_features, train_target):
    """
    Wrap a tuned pipeline in isotonic calibration and fit it.

    CalibratedClassifierCV trains a copy of the pipeline on 4 of the 5
    folds and learns the probability mapping on the fold it did not see,
    five times. Its predictions average the five calibrated copies. Only
    training data is used.

    Inputs:
        pipeline: a fitted pipeline (its settings are copied, not its fit).
        train_features, train_target: the training data.
    Returns:
        The fitted CalibratedClassifierCV.
    """
    calibrated = CalibratedClassifierCV(clone(pipeline), method="isotonic", cv=make_folds())
    calibrated.fit(train_features, train_target)
    return calibrated


def plot_reliability_diagram(probabilities_by_model, test_target, filename, title):
    """
    Draw the reliability diagram for several sets of probabilities.

    Bins hold equal numbers of sessions ("quantile" bins), so every point
    is based on enough sessions.

    Inputs:
        probabilities_by_model: dictionary label -> (probabilities, colour, line style).
        test_target: the real labels.
        filename: the file name in outputs/figures/.
        title: the plot title.
    Returns:
        The path of the saved plot.
    """
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.plot([0, 1], [0, 1], color="grey", linestyle="--", linewidth=1, label="Perfectly calibrated")

    for label, (probabilities, color, line_style) in probabilities_by_model.items():
        real_rate, mean_predicted = calibration_curve(
            test_target, probabilities, n_bins=NUMBER_OF_BINS, strategy="quantile"
        )
        brier = brier_score_loss(test_target, probabilities)
        axis.plot(
            mean_predicted,
            real_rate,
            marker="o",
            color=color,
            linestyle=line_style,
            label=f"{label} (Brier {brier:.3f})",
        )

    axis.set_xlabel("Predicted purchase probability")
    axis.set_ylabel("Real purchase rate in the bin")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.set_title(title, fontsize=12)
    axis.legend(fontsize=8, loc="upper left")
    figure.tight_layout()
    return save_figure(figure, filename)


def run_calibration_analysis(
    models, model_names_to_calibrate, train_features, train_target, test_features, test_target
):
    """
    Measure calibration for every model, calibrate the chosen ones, and
    compare the profit of the break-even rule before and after.

    Inputs:
        models: dictionary model name -> fitted pipeline.
        model_names_to_calibrate: the models to calibrate (the Decision Tree
            and XGBoost: the two whose training reweights buyers).
        train_features, train_target, test_features, test_target: the data.
    Returns:
        A DataFrame with one row per model, also saved to
        outputs/calibration_summary.csv.
    """
    print("\nChecking how well the probabilities are calibrated...")
    cutoff = break_even_probability()
    print(f"  Break-even probability for the business assumptions: {cutoff:.2f}")

    all_models = {}
    for model_name, pipeline in models.items():
        all_models[model_name] = (
            get_purchase_probabilities(pipeline, test_features),
            MODEL_COLORS[model_name],
            "-",
        )
    plot_reliability_diagram(
        all_models, test_target, "calibration_all_models.png", "Reliability diagram (test set)"
    )

    rows = []
    before_and_after = {}
    for model_name, pipeline in models.items():
        raw_probabilities = all_models[model_name][0]
        row = {
            "Model": model_name,
            "Brier (raw)": brier_score_loss(test_target, raw_probabilities),
            "ROC-AUC (raw)": roc_auc_score(test_target, raw_probabilities),
            "Mean probability (raw)": raw_probabilities.mean(),
            "Profit at break-even (raw)": calculate_profit(test_target, raw_probabilities, cutoff)["profit"],
        }

        if model_name in model_names_to_calibrate:
            print("  Calibrating " + model_name + " (isotonic, 5 folds)...")
            calibrated = calibrate_model(pipeline, train_features, train_target)
            calibrated_probabilities = calibrated.predict_proba(test_features)[:, 1]
            row["Brier (calibrated)"] = brier_score_loss(test_target, calibrated_probabilities)
            row["ROC-AUC (calibrated)"] = roc_auc_score(test_target, calibrated_probabilities)
            row["Mean probability (calibrated)"] = calibrated_probabilities.mean()
            row["Profit at break-even (calibrated)"] = calculate_profit(
                test_target, calibrated_probabilities, cutoff
            )["profit"]
            color = MODEL_COLORS[model_name]
            before_and_after[model_name + " raw"] = (raw_probabilities, color, ":")
            before_and_after[model_name + " calibrated"] = (calibrated_probabilities, color, "-")
        rows.append(row)

    plot_reliability_diagram(
        before_and_after,
        test_target,
        "calibration_before_after.png",
        "Before and after isotonic calibration (test set)",
    )

    table = pd.DataFrame(rows).round(3)
    table.to_csv(os.path.join(OUTPUTS_FOLDER, "calibration_summary.csv"), index=False)
    print(f"  Real purchase rate in the test set: {test_target.mean():.3f}")
    print(table.to_string(index=False))
    return table
