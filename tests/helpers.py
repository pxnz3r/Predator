"""Shared test helpers: load the dataset once, or skip the test if it is not downloaded."""

import pytest
from sklearn.model_selection import train_test_split

from predator.data import find_dataset, load_dataset

# Loaded data is kept here so the CSV is only read once per test run.
loaded_data = {}


def get_dataset():
    """
    Load the real dataset, or skip the calling test if the CSV is missing.

    Skipping (instead of failing) lets the other tests still run on a fresh
    clone where the dataset has not been downloaded yet.

    Returns:
        The full DataFrame.
    """
    if find_dataset() is None:
        pytest.skip("Dataset not found in data/ - download it to run this test")
    if "full" not in loaded_data:
        loaded_data["full"] = load_dataset()
    return loaded_data["full"]


def get_small_sample():
    """
    A stratified sample of 1,000 sessions, so pipeline tests run fast.

    Returns:
        A DataFrame with 1,000 rows and the same buyer rate as the full data.
    """
    if "sample" not in loaded_data:
        dataset = get_dataset()
        sample, rest = train_test_split(dataset, train_size=1000, stratify=dataset["Revenue"], random_state=42)
        loaded_data["sample"] = sample
    return loaded_data["sample"]
