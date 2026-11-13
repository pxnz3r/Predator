"""Tests for predator/threshold.py: the profit calculation and threshold search."""

import numpy as np

from predator import threshold
from predator.threshold import (
    build_threshold_curve,
    calculate_profit,
    find_best_thresholds,
    profit_per_outcome,
)


def test_profit_counts_each_outcome():
    """Hand-made example: 2 true positives, 1 false positive, 1 false negative, 1 true negative."""
    true_target = np.array([1, 1, 0, 1, 0])
    probabilities = np.array([0.9, 0.7, 0.8, 0.2, 0.1])
    result = calculate_profit(true_target, probabilities, threshold=0.5)

    values = profit_per_outcome()
    assert (result["tp"], result["fp"], result["fn"], result["tn"]) == (2, 1, 1, 1)
    assert result["targeted"] == 3
    assert result["profit"] == 2 * values["tp"] + 1 * values["fp"]


def test_true_positive_value_follows_the_assumptions():
    """A true positive earns the secured order profit minus the offer cost."""
    expected = threshold.AVERAGE_ORDER_PROFIT * threshold.OFFER_CONVERSION_BOOST - threshold.OFFER_COST
    assert profit_per_outcome()["tp"] == expected
    assert profit_per_outcome()["fp"] == -threshold.OFFER_COST


def test_best_thresholds_on_a_perfect_model():
    """When buyers all score 0.8 and non-buyers 0.2, a threshold between them is best for F1 and profit."""
    true_target = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    probabilities = np.array([0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2])
    curve = build_threshold_curve(true_target, probabilities)
    best_f1_threshold, best_profit_threshold = find_best_thresholds(curve)
    assert 0.2 < best_f1_threshold <= 0.8
    assert 0.2 < best_profit_threshold <= 0.8
    assert curve["f1"].max() == 1.0
