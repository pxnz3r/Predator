"""Shared test helpers: load the dataset once, or skip if it is not downloaded."""

import pytest

from predator.data import find_dataset, load_dataset


@pytest.fixture(scope="session")
def dataset():
    """
    Load the real dataset once for all tests.

    Tests that need it are skipped (not failed) when the CSV is missing,
    so the rest of the tests still run on a fresh clone.

    Returns:
        The full DataFrame.
    """
    if find_dataset() is None:
        pytest.skip("Dataset not found in data/ - download it to run this test")
    return load_dataset()


@pytest.fixture(scope="session")
def small_sample(dataset):
    """
    A stratified sample of 1,000 sessions, so pipeline tests run fast.

    Input:
        dataset: the full DataFrame (fixture).
    Returns:
        A DataFrame with 1,000 rows and the same buyer rate.
    """
    from sklearn.model_selection import train_test_split

    sample, rest = train_test_split(dataset, train_size=1000, stratify=dataset["Revenue"], random_state=42)
    return sample
