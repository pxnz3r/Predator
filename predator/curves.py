"""
curves.py - Overfitting and underfitting, shown with our own data.

Two diagnostic plots from the course:

1. Learning curve: train the tuned model on 10%, 28%, ... 100% of the
   training data and plot the training score and the cross-validated
   score. A big gap between them means overfitting (high variance): more
   data would help. Two low curves that meet mean underfitting (high
   bias): a more flexible model would help.

2. Validation curve: change ONE setting and keep the others at their
   tuned values. For the Decision Tree we change max_depth: a shallow
   tree underfits, a very deep tree memorises the training data. For
   KNN we change n_neighbors (k): k = 1 memorises every training session,
   a very large k averages everything away. The best value sits where
   the cross-validated score peaks.

All scores are F1 from 5-fold stratified cross-validation on the
training data, the same folds as GridSearchCV.
"""

import matplotlib

matplotlib.use("Agg")
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, validation_curve

from predator.data import save_figure
from predator.evaluate import OUTPUTS_FOLDER
from predator.models import DECISION_TREE, KNN, MODEL_COLORS
from predator.train import make_folds

TRAINING_SIZES = np.linspace(0.1, 1.0, 6)
TREE_DEPTHS_TO_TRY = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]
NEIGHBOURS_TO_TRY = [1, 3, 5, 9, 15, 21, 31, 51, 75, 101]


def draw_score_band(axis, x_values, scores, color, line_style, label):
    """
    Plot the mean score across the 5 folds as a line, with a shaded band
    of plus/minus one standard deviation.

    Inputs:
        axis: the matplotlib axis.
        x_values: the x positions.
        scores: array of shape (number of x values, 5 folds).
        color, line_style, label: how to draw the line.
    """
    means = scores.mean(axis=1)
    deviations = scores.std(axis=1)
    axis.plot(x_values, means, color=color, linestyle=line_style, marker="o", label=label)
    axis.fill_between(x_values, means - deviations, means + deviations, color=color, alpha=0.15)


def plot_learning_curves(models, train_features, train_target):
    """
    Draw the learning curves of the Decision Tree and KNN.

    Inputs:
        models: dictionary model name -> fitted pipeline (the tuned settings are reused).
        train_features, train_target: the training data.
    Returns:
        A DataFrame with the mean scores, also saved to outputs/learning_curves.csv.
    """
    print("\nLearning curves (F1, 5-fold CV)...")
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    rows = []
    for axis, model_name in zip(axes, [DECISION_TREE, KNN], strict=True):
        sizes, train_scores, validation_scores = learning_curve(
            models[model_name],
            train_features,
            train_target,
            train_sizes=TRAINING_SIZES,
            cv=make_folds(),
            scoring="f1",
            n_jobs=-1,
        )
        color = MODEL_COLORS[model_name]
        draw_score_band(axis, sizes, train_scores, color, "--", "Training F1")
        draw_score_band(axis, sizes, validation_scores, color, "-", "Cross-validated F1")
        axis.set_title(model_name, fontsize=11)
        axis.set_xlabel("Number of training sessions")
        axis.legend(fontsize=9, loc="lower right")
        for size, train_score, validation_score in zip(
            sizes, train_scores.mean(axis=1), validation_scores.mean(axis=1), strict=True
        ):
            rows.append(
                {
                    "Model": model_name,
                    "Training sessions": int(size),
                    "Training F1": train_score,
                    "CV F1": validation_score,
                }
            )

    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1.05)
    figure.suptitle("Learning curves: does more data help?", fontsize=13)
    figure.tight_layout()
    save_figure(figure, "learning_curves.png")

    table = pd.DataFrame(rows).round(3)
    table.to_csv(os.path.join(OUTPUTS_FOLDER, "learning_curves.csv"), index=False)
    return table


def plot_validation_curves(models, train_features, train_target):
    """
    Draw validation curves: Decision Tree F1 against max_depth and KNN F1
    against n_neighbors, other settings at their tuned values.

    Inputs:
        models: dictionary model name -> fitted pipeline.
        train_features, train_target: the training data.
    Returns:
        A DataFrame with the mean scores, also saved to outputs/validation_curves.csv.
    """
    print("\nValidation curves (F1, 5-fold CV)...")
    settings = [
        (DECISION_TREE, "model__max_depth", TREE_DEPTHS_TO_TRY, "max_depth"),
        (KNN, "model__n_neighbors", NEIGHBOURS_TO_TRY, "n_neighbors (k)"),
    ]
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    rows = []
    for axis, (model_name, parameter, values, axis_label) in zip(axes, settings, strict=True):
        train_scores, validation_scores = validation_curve(
            models[model_name],
            train_features,
            train_target,
            param_name=parameter,
            param_range=values,
            cv=make_folds(),
            scoring="f1",
            n_jobs=-1,
        )
        color = MODEL_COLORS[model_name]
        draw_score_band(axis, values, train_scores, color, "--", "Training F1")
        draw_score_band(axis, values, validation_scores, color, "-", "Cross-validated F1")

        tuned_value = models[model_name].get_params()[parameter]
        axis.axvline(tuned_value, color="grey", linestyle=":", label=f"Tuned value ({tuned_value})")
        axis.set_title(model_name, fontsize=11)
        axis.set_xlabel(axis_label)
        axis.legend(fontsize=9, loc="lower right")
        for value, train_score, validation_score in zip(
            values, train_scores.mean(axis=1), validation_scores.mean(axis=1), strict=True
        ):
            rows.append(
                {
                    "Model": model_name,
                    "Setting": axis_label,
                    "Value": value,
                    "Training F1": train_score,
                    "CV F1": validation_score,
                }
            )

    axes[0].set_ylabel("F1")
    axes[0].set_ylim(0, 1.05)
    figure.suptitle("Validation curves: from underfitting to overfitting", fontsize=13)
    figure.tight_layout()
    save_figure(figure, "validation_curves.png")

    table = pd.DataFrame(rows).round(3)
    table.to_csv(os.path.join(OUTPUTS_FOLDER, "validation_curves.csv"), index=False)
    return table
