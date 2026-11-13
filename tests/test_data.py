"""Tests for predator/data.py: the dataset loads and looks right."""

import pytest
from helpers import get_dataset

from predator.data import ALL_FEATURES, TARGET_COLUMN, load_dataset, split_data, validate_dataset


def test_missing_file_gives_friendly_error():
    """A wrong path should raise FileNotFoundError that explains how to fix it."""
    with pytest.raises(FileNotFoundError) as error:
        load_dataset("data/this_file_does_not_exist.csv")
    assert "Download" in str(error.value)


def test_dataset_loads_with_expected_shape():
    """The dataset has 12,330 sessions, 17 features plus the target, and no missing values."""
    dataset = get_dataset()
    assert len(dataset) == 12330
    for column in ALL_FEATURES + [TARGET_COLUMN]:
        assert column in dataset.columns
    report = validate_dataset(dataset)
    assert report["missing_values"] == 0


def test_target_is_binary_and_imbalanced():
    """Revenue is 0/1 and about 15% of sessions are buyers."""
    dataset = get_dataset()
    assert set(dataset[TARGET_COLUMN].unique()) == {0, 1}
    assert 0.14 < dataset[TARGET_COLUMN].mean() < 0.17


def test_split_is_stratified():
    """The 80/20 split keeps the same buyer rate in both parts."""
    dataset = get_dataset()
    train_features, test_features, train_target, test_target = split_data(dataset)
    assert len(test_features) == round(len(dataset) * 0.2)
    assert abs(train_target.mean() - test_target.mean()) < 0.005
