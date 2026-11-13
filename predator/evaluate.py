"""
evaluate.py - Scoring the models on the test set and drawing result plots.

The metrics, in plain words (a "positive" is a session that buys):
    - Accuracy: share of all sessions predicted correctly. Misleading here,
      because always saying "no purchase" already gets about 85%.
    - Precision: of the sessions we flagged as buyers, how many really bought.
    - Recall: of the real buyers, how many we found.
    - F1: one number that balances precision and recall (their harmonic mean).
    - ROC-AUC: the chance that a random buyer gets a higher probability than
      a random non-buyer. 0.5 is guessing, 1.0 is perfect. It does not
      depend on the 0.5 decision threshold.
    - Confusion matrix: a 2x2 table of true/false positives and negatives.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from predator.data import save_figure
from predator.models import MODEL_COLORS, MODEL_ROLES

OUTPUTS_FOLDER = "outputs"


def get_purchase_probabilities(pipeline, features):
    """
    Get the predicted probability of a purchase for each session.

    predict_proba() returns two columns: P(no purchase) and P(purchase).
    We keep the second one.

    Inputs:
        pipeline: a fitted pipeline.
        features: DataFrame of sessions.
    Returns:
        A numpy array of probabilities between 0 and 1.
    """
    return pipeline.predict_proba(features)[:, 1]


def evaluate_model(model_name, pipeline, test_features, test_target, threshold=0.5):
    """
    Compute all the test-set metrics for one model.

    A session is predicted as a buyer when its probability is at least the
    threshold (0.5 by default; see threshold.py for choosing a better one).

    Inputs:
        model_name: the model's name (used in the report).
        pipeline: a fitted pipeline.
        test_features, test_target: the test data.
        threshold: the probability cutoff for predicting "purchase".
    Returns:
        A dictionary with the metrics, the confusion matrix and the
        classification report text.
    """
    probabilities = get_purchase_probabilities(pipeline, test_features)
    predictions = (probabilities >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(test_target, predictions),
        "precision": precision_score(test_target, predictions, zero_division=0),
        "recall": recall_score(test_target, predictions),
        "f1": f1_score(test_target, predictions),
        "roc_auc": roc_auc_score(test_target, probabilities),
        "confusion_matrix": confusion_matrix(test_target, predictions),
        "report": classification_report(test_target, predictions,
                                        target_names=["No purchase", "Purchase"], digits=3),
    }
    return metrics


def build_comparison_table(models, training_results, test_features, test_target, file_prefix=""):
    """
    Evaluate every model and put the scores in one table.

    It also prints each model's confusion matrix and classification report,
    and saves the reports to outputs/<file_prefix>classification_reports.txt.

    Inputs:
        models: dictionary model name -> fitted pipeline.
        training_results: the dictionary from train.train_all_models()
            (used for the cross-validated F1 column).
        test_features, test_target: the test data.
        file_prefix: added to both file names, so the ablation run
            (file_prefix="ablation_") does not overwrite the main results.
    Returns:
        A pandas DataFrame with one row per model, also saved to
        outputs/<file_prefix>model_comparison.csv.
    """
    print("\nEvaluating models on the test set (threshold 0.5)...")
    rows = []
    report_text = ""

    for model_name, pipeline in models.items():
        metrics = evaluate_model(model_name, pipeline, test_features, test_target)
        rows.append({
            "Model": model_name,
            "Role": MODEL_ROLES[model_name],
            "CV F1": training_results[model_name]["cv_f1"],
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
            "ROC-AUC": metrics["roc_auc"],
        })

        matrix = metrics["confusion_matrix"]
        section = (
            "=" * 60 + "\n" + model_name + "\n" + "=" * 60 + "\n"
            + "Confusion matrix (rows = actual, columns = predicted):\n"
            + "                 No purchase  Purchase\n"
            + f"  No purchase    {matrix[0][0]:>11}  {matrix[0][1]:>8}\n"
            + f"  Purchase       {matrix[1][0]:>11}  {matrix[1][1]:>8}\n\n"
            + metrics["report"] + "\n"
        )
        print(section)
        report_text = report_text + section

    table = pd.DataFrame(rows).round(3)

    os.makedirs(OUTPUTS_FOLDER, exist_ok=True)
    table.to_csv(os.path.join(OUTPUTS_FOLDER, file_prefix + "model_comparison.csv"), index=False)
    with open(os.path.join(OUTPUTS_FOLDER, file_prefix + "classification_reports.txt"), "w") as file:
        file.write(report_text)

    print("Model comparison (test set):")
    print(table.to_string(index=False))
    return table


def choose_best_model(comparison_table):
    """
    Pick the best model by cross-validated F1 on the training data.

    We do not pick by test-set score: the test set is only for reporting.
    Choosing the winner by test score would quietly tune on the test set.

    Input:
        comparison_table: the DataFrame from build_comparison_table().
    Returns:
        The name of the best model.
    """
    best_row = comparison_table.sort_values("CV F1", ascending=False).iloc[0]
    print("\nBest model by cross-validated F1:", best_row["Model"],
          "(CV F1 = {:.3f})".format(best_row["CV F1"]))
    return best_row["Model"]


def plot_roc_curves(models, test_features, test_target, filename="roc_curves.png", title=None):
    """
    Draw the ROC curve of every model on one plot.

    A ROC curve shows the true positive rate (recall) against the false
    positive rate for every possible threshold. The closer a curve hugs
    the top-left corner, the better the model ranks buyers above
    non-buyers. The dashed diagonal is random guessing.

    Inputs:
        models: dictionary model name -> fitted pipeline.
        test_features, test_target: the test data.
        filename: name of the image file.
        title: optional plot title.
    Returns:
        The path of the saved plot.
    """
    figure, axis = plt.subplots(figsize=(7, 6))
    for model_name, pipeline in models.items():
        probabilities = get_purchase_probabilities(pipeline, test_features)
        # roc_curve tries every threshold and returns the false positive
        # rate and true positive rate at each one.
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_target, probabilities)
        area = roc_auc_score(test_target, probabilities)
        axis.plot(false_positive_rate, true_positive_rate, color=MODEL_COLORS[model_name],
                  linewidth=2, label=f"{model_name} (AUC = {area:.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1, label="Random guess")
    axis.set_xlabel("False positive rate (non-buyers wrongly flagged)")
    axis.set_ylabel("True positive rate (buyers found)")
    axis.set_title(title or "ROC curves on the test set")
    axis.legend(loc="lower right")
    return save_figure(figure, filename)


def plot_confusion_matrices(models, test_features, test_target, threshold=0.5):
    """
    Draw a confusion matrix for every model, side by side.

    Inputs:
        models: dictionary model name -> fitted pipeline.
        test_features, test_target: the test data.
        threshold: the probability cutoff for predicting "purchase".
    Returns:
        The path of the saved plot.
    """
    figure, axes = plt.subplots(1, len(models), figsize=(4.2 * len(models), 4))
    for axis, (model_name, pipeline) in zip(axes, models.items(), strict=True):
        probabilities = get_purchase_probabilities(pipeline, test_features)
        predictions = (probabilities >= threshold).astype(int)
        ConfusionMatrixDisplay.from_predictions(
            test_target, predictions, display_labels=["No buy", "Buy"],
            cmap="Blues", colorbar=False, ax=axis,
        )
        axis.set_title(model_name)
    figure.suptitle(f"Confusion matrices on the test set (threshold {threshold})")
    figure.tight_layout()
    return save_figure(figure, "confusion_matrices.png")


def plot_metric_comparison(comparison_table):
    """
    Grouped bar chart of precision, recall, F1 and ROC-AUC for every model.

    Input:
        comparison_table: the DataFrame from build_comparison_table().
    Returns:
        The path of the saved plot.
    """
    metric_names = ["Precision", "Recall", "F1", "ROC-AUC"]
    number_of_models = len(comparison_table)
    bar_width = 0.8 / number_of_models

    figure, axis = plt.subplots(figsize=(10, 5))
    for index, row in comparison_table.reset_index(drop=True).iterrows():
        positions = [metric_index + index * bar_width for metric_index in range(len(metric_names))]
        values = [row[metric] for metric in metric_names]
        axis.bar(positions, values, width=bar_width, label=row["Model"],
                 color=MODEL_COLORS[row["Model"]])

    axis.set_xticks([metric_index + 0.4 - bar_width / 2 for metric_index in range(len(metric_names))])
    axis.set_xticklabels(metric_names)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Score on the test set")
    axis.set_title("Model comparison (threshold 0.5)")
    axis.legend(loc="upper left", fontsize=9)
    return save_figure(figure, "metric_comparison.png")
