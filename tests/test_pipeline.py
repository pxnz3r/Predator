"""Tests for predator/preprocess.py and predator/models.py: pipelines fit and predict."""

from predator.data import ALL_FEATURES, TARGET_COLUMN
from predator.models import MODEL_NAMES, make_model, uses_scaling
from predator.preprocess import build_pipeline, get_feature_names, undersample_training_data


def test_every_model_pipeline_fits_and_predicts(small_sample):
    """Each model's pipeline trains on a small sample and returns valid probabilities."""
    features = small_sample[ALL_FEATURES]
    target = small_sample[TARGET_COLUMN]
    for model_name in MODEL_NAMES:
        pipeline = build_pipeline(make_model(model_name, scale_pos_weight=5.0),
                                  scale_numbers=uses_scaling(model_name))
        pipeline.fit(features, target)
        probabilities = pipeline.predict_proba(features)[:, 1]
        assert len(probabilities) == len(features)
        assert probabilities.min() >= 0 and probabilities.max() <= 1


def test_one_hot_encoding_creates_named_columns(small_sample):
    """After preprocessing, categorical columns become one-hot columns like Month_Nov."""
    pipeline = build_pipeline(make_model("Decision Tree", 5.0), scale_numbers=False)
    pipeline.fit(small_sample[ALL_FEATURES], small_sample[TARGET_COLUMN])
    names = get_feature_names(pipeline)
    assert "PageValues" in names
    assert "Month_Nov" in names
    assert len(names) > len(ALL_FEATURES)


def test_unknown_category_does_not_crash(small_sample):
    """A month never seen in training (Jan) is ignored instead of raising an error."""
    pipeline = build_pipeline(make_model("Decision Tree", 5.0), scale_numbers=False)
    pipeline.fit(small_sample[ALL_FEATURES], small_sample[TARGET_COLUMN])
    session = small_sample[ALL_FEATURES].head(1).copy()
    session["Month"] = "Jan"
    assert pipeline.predict(session).shape == (1,)


def test_undersampling_balances_classes(small_sample):
    """Undersampling keeps every buyer and the same number of non-buyers."""
    features, target = undersample_training_data(small_sample[ALL_FEATURES], small_sample[TARGET_COLUMN])
    assert target.sum() == small_sample[TARGET_COLUMN].sum()
    assert target.mean() == 0.5
