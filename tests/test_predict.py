"""Tests for predator/predict.py: a session file gives a valid prediction."""

import pytest

from predator.data import ALL_FEATURES, TARGET_COLUMN
from predator.models import make_model
from predator.predict import load_session_from_json, predict_session, prepare_session
from predator.preprocess import build_pipeline


@pytest.fixture(scope="module")
def small_tree(small_sample):
    """A Decision Tree pipeline trained on the small sample."""
    pipeline = build_pipeline(make_model("Decision Tree", 5.0), scale_numbers=False)
    pipeline.fit(small_sample[ALL_FEATURES], small_sample[TARGET_COLUMN])
    return pipeline


def test_example_file_loads_as_one_row():
    """The example JSON becomes a one-row DataFrame with all 17 features."""
    session = load_session_from_json("examples/session_buyer.json")
    assert session.shape == (1, len(ALL_FEATURES))
    assert list(session.columns) == ALL_FEATURES


def test_prediction_output_is_valid(small_tree):
    """predict_session returns a label, a probability in [0, 1] and a threshold decision."""
    session = load_session_from_json("examples/session_buyer.json")
    result = predict_session(small_tree, session, threshold=0.4)
    assert result["label"] in ["purchase", "no purchase"]
    assert 0 <= result["probability"] <= 1
    assert result["passes_threshold"] == (result["probability"] >= 0.4)


def test_missing_feature_is_reported():
    """A session without PageValues raises a ValueError naming the missing feature."""
    with pytest.raises(ValueError) as error:
        prepare_session({"Month": "Nov"})
    assert "PageValues" in str(error.value)


def test_weekend_text_is_understood():
    """Weekend can be given as the text "true" as well as a JSON boolean."""
    session_values = load_session_from_json("examples/session_buyer.json").iloc[0].to_dict()
    session_values["Weekend"] = "true"
    assert bool(prepare_session(session_values)["Weekend"].iloc[0]) is True
