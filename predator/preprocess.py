"""
preprocess.py - Turning raw session data into numbers the models can use.

Models only understand numbers, and some models (KNN) also need those
numbers to be on a similar scale. This file builds scikit-learn Pipelines
that do the preprocessing and the model in one object.

Why a Pipeline? It makes data leakage much harder. Data leakage means
information from the test set (or from the validation fold during
cross-validation) sneaks into training. For example, if we computed the
mean and standard deviation for scaling on the WHOLE dataset, the test set
would have influenced training. Inside a Pipeline, the scaler and encoder
are fitted only on the data passed to .fit(), which is always training data.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from predator.data import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE


def build_preprocessor(numerical_features, categorical_features, scale_numbers=True):
    """
    Build a ColumnTransformer that prepares numerical and categorical columns.

    - Categorical columns are one-hot encoded: the column Month becomes
      Month_Feb, Month_Mar, ... with a 1 in the matching column and 0
      elsewhere. This avoids pretending that "Dec" > "Feb".
      handle_unknown="ignore" means a value never seen in training (for
      example "Jan") just gets all zeros instead of crashing.
    - Numerical columns are standardised (mean 0, standard deviation 1)
      when scale_numbers is True. KNN measures distances, so without
      scaling a column measured in thousands of seconds
      (ProductRelated_Duration) would drown out a rate between 0 and 0.2
      (BounceRates).

    Tree models (Decision Tree, XGBoost) split on one column at a time,
    so scaling does not change them. We skip it for trees so the split
    rules stay in real units, e.g. "PageValues <= 0.94".

    Inputs:
        numerical_features: list of numerical column names.
        categorical_features: list of categorical column names.
        scale_numbers: True to standardise numerical columns.
    Returns:
        An unfitted ColumnTransformer.
    """
    if scale_numbers:
        numerical_step = StandardScaler()
    else:
        numerical_step = "passthrough"

    categorical_step = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("numbers", numerical_step, numerical_features),
            ("categories", categorical_step, categorical_features),
        ]
    )
    return preprocessor


def build_pipeline(model, numerical_features=None, categorical_features=None, scale_numbers=True):
    """
    Put the preprocessing and a model together in one Pipeline.

    Calling pipeline.fit(train_features, train_target) first fits the
    preprocessor on the training data and then trains the model on the
    transformed data. pipeline.predict(new_data) applies the exact same
    transformation before predicting.

    Inputs:
        model: an unfitted scikit-learn style classifier.
        numerical_features: numerical columns (defaults to all of them).
        categorical_features: categorical columns (defaults to all of them).
        scale_numbers: passed on to build_preprocessor().
    Returns:
        An unfitted Pipeline with two steps: "preprocess" and "model".
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    preprocessor = build_preprocessor(numerical_features, categorical_features, scale_numbers)
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipeline


def get_feature_names(fitted_pipeline):
    """
    Get readable names for the columns the model actually sees.

    After one-hot encoding there are many more columns than the 17
    original features, for example "VisitorType_New_Visitor".

    Input:
        fitted_pipeline: a Pipeline that has already been fitted.
    Returns:
        A list of column names, in the same order as the model's input.
    """
    raw_names = fitted_pipeline.named_steps["preprocess"].get_feature_names_out()
    # scikit-learn adds prefixes like "numbers__" and "categories__"; drop them.
    clean_names = [name.split("__", 1)[1] for name in raw_names]
    return clean_names


def undersample_training_data(train_features, train_target):
    """
    Randomly undersample the majority class (non-buyers) in the training data.

    We keep every buyer and pick an equal number of non-buyers at random,
    so the result is 50% buyers / 50% non-buyers. This helps KNN, because
    otherwise most of a session's nearest neighbours are non-buyers simply
    because there are so many of them.

    The downside: we throw away most of the non-buyer sessions, so the
    model learns from less data. This is only ever applied to TRAINING
    data. The test set keeps its real 15% buyer rate.

    Inputs:
        train_features: DataFrame of training features.
        train_target: Series of training labels (0/1).
    Returns:
        (undersampled_features, undersampled_target), shuffled.
    """
    buyer_rows = train_target[train_target == 1].index
    non_buyer_rows = train_target[train_target == 0].index

    sampled_non_buyer_rows = pd.Series(non_buyer_rows).sample(n=len(buyer_rows), random_state=RANDOM_STATE)

    kept_rows = list(buyer_rows) + list(sampled_non_buyer_rows)
    undersampled_features = train_features.loc[kept_rows]
    undersampled_target = train_target.loc[kept_rows]

    # Shuffle so buyers and non-buyers are mixed (order should not matter,
    # but it makes the data look like a normal random sample).
    shuffled_order = undersampled_features.sample(frac=1, random_state=RANDOM_STATE).index
    return undersampled_features.loc[shuffled_order], undersampled_target.loc[shuffled_order]
