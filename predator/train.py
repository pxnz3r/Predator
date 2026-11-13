"""
train.py - Tuning the models with cross-validation and saving them.

Cross-validation in one paragraph: we split the TRAINING data into 5 equal
parts (folds). We train on 4 folds and score on the 5th, and repeat so
every fold is used for scoring once. The average of the 5 scores tells us
how well a setting generalises, without ever touching the test set.
"Stratified" folds keep the 15% buyer rate in every fold.

GridSearchCV runs this for every combination in a parameter grid and keeps
the combination with the best average F1 score. We use F1 (not accuracy)
because accuracy is misleading on imbalanced data: always saying
"no purchase" scores about 85% accuracy but finds zero buyers. F1 is the
balance (harmonic mean) of precision and recall on the buyer class.
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from predator.data import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE
from predator.models import (
    KNN_UNDERSAMPLED,
    MODEL_FILE_NAMES,
    MODEL_NAMES,
    get_parameter_grids,
    make_model,
    uses_scaling,
)
from predator.preprocess import build_pipeline, undersample_training_data

MODELS_FOLDER = "models"
NUMBER_OF_FOLDS = 5


def make_folds():
    """
    Create the 5-fold stratified cross-validation splitter used everywhere.

    shuffle=True with a fixed random_state mixes the rows before splitting,
    and gives the same folds on every run.

    Returns:
        A StratifiedKFold object.
    """
    return StratifiedKFold(n_splits=NUMBER_OF_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def calculate_scale_pos_weight(train_target):
    """
    Work out how much more weight each buyer should get in XGBoost.

    Input:
        train_target: Series of training labels (0/1).
    Returns:
        number of non-buyers divided by number of buyers (about 5.5 here).
    """
    number_of_buyers = int((train_target == 1).sum())
    number_of_non_buyers = int((train_target == 0).sum())
    return number_of_non_buyers / number_of_buyers


def tune_model(model_name, train_features, train_target, numerical_features, categorical_features):
    """
    Find the best hyperparameters for one model with GridSearchCV.

    For "KNN (undersampled)" the training data is undersampled first, so
    this KNN learns from a 50/50 balanced set. For Naive Bayes the grid is
    empty, so the model is simply fitted once (it is an untuned baseline).

    Inputs:
        model_name: one of MODEL_NAMES.
        train_features, train_target: the training data.
        numerical_features, categorical_features: which columns to use.
    Returns:
        A dictionary with:
            "pipeline": the best pipeline, refitted on the training data
            "best_params": the winning hyperparameters
            "grid_f1": the best mean F1 inside GridSearchCV (None for Naive Bayes)
    """
    scale_pos_weight = calculate_scale_pos_weight(train_target)
    model = make_model(model_name, scale_pos_weight)
    pipeline = build_pipeline(
        model, numerical_features, categorical_features, scale_numbers=uses_scaling(model_name)
    )
    parameter_grid = get_parameter_grids()[model_name]

    fit_features = train_features
    fit_target = train_target
    if model_name == KNN_UNDERSAMPLED:
        fit_features, fit_target = undersample_training_data(train_features, train_target)
        print("  Undersampled training set:", len(fit_features), "sessions (50% buyers)")

    # Naive Bayes: nothing to tune, just fit it.
    if len(parameter_grid) == 0:
        pipeline.fit(fit_features, fit_target)
        return {"pipeline": pipeline, "best_params": {}, "grid_f1": None}

    number_of_combinations = 1
    for values in parameter_grid.values():
        number_of_combinations = number_of_combinations * len(values)
    print("  Trying", number_of_combinations, "combinations x", NUMBER_OF_FOLDS, "folds")

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1",
        cv=make_folds(),
        n_jobs=-1,
    )
    grid_search.fit(fit_features, fit_target)

    # Remove the "model__" prefix so the parameters are easier to read.
    best_params = {}
    for key, value in grid_search.best_params_.items():
        best_params[key.replace("model__", "")] = value

    return {
        "pipeline": grid_search.best_estimator_,
        "best_params": best_params,
        "grid_f1": float(grid_search.best_score_),
    }


def out_of_fold_probabilities(pipeline, train_features, train_target, undersample=False):
    """
    Get a purchase probability for every training session from a model
    that did NOT see that session during training.

    This is cross-validation written out by hand: for each of the 5 folds,
    train a copy of the pipeline on the other 4 folds and predict the
    held-out fold. We use these "out-of-fold" probabilities to:
      1. compare all models fairly on the real 15% buyer rate (the
         undersampled KNN's GridSearchCV score was measured on balanced
         folds, so it is not comparable), and
      2. choose decision thresholds without looking at the test set.

    When undersample is True, only the 4 training folds are undersampled.
    The held-out fold keeps its real class balance.

    Inputs:
        pipeline: a pipeline with the chosen hyperparameters.
        train_features, train_target: the training data.
        undersample: True for the undersampled KNN.
    Returns:
        A numpy array with one probability per training session.
    """
    probabilities = np.zeros(len(train_features))

    for fold_train_rows, fold_validation_rows in make_folds().split(train_features, train_target):
        fold_train_features = train_features.iloc[fold_train_rows]
        fold_train_target = train_target.iloc[fold_train_rows]
        fold_validation_features = train_features.iloc[fold_validation_rows]

        if undersample:
            fold_train_features, fold_train_target = undersample_training_data(
                fold_train_features, fold_train_target
            )

        # clone() gives an unfitted copy with the same settings.
        fold_model = clone(pipeline)
        fold_model.fit(fold_train_features, fold_train_target)
        probabilities[fold_validation_rows] = fold_model.predict_proba(fold_validation_features)[:, 1]

    return probabilities


def train_all_models(train_features, train_target, numerical_features=None, categorical_features=None):
    """
    Tune, train and cross-validate every model.

    Inputs:
        train_features, train_target: the training data.
        numerical_features, categorical_features: which columns to use
            (defaults to all; the ablation passes a list without PageValues).
    Returns:
        A dictionary: model name -> result dictionary with keys
        "pipeline", "best_params", "grid_f1", "oof_probabilities",
        "cv_f1" (fair 5-fold F1 at threshold 0.5) and "training_seconds".
    """
    if numerical_features is None:
        numerical_features = NUMERICAL_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    results = {}
    for model_name in MODEL_NAMES:
        print("\nTraining " + model_name + "...")
        start_time = time.time()

        result = tune_model(
            model_name, train_features, train_target, numerical_features, categorical_features
        )

        undersample = model_name == KNN_UNDERSAMPLED
        oof_probabilities = out_of_fold_probabilities(
            result["pipeline"], train_features, train_target, undersample
        )
        oof_predictions = (oof_probabilities >= 0.5).astype(int)
        result["oof_probabilities"] = oof_probabilities
        result["cv_f1"] = float(f1_score(train_target, oof_predictions))
        result["training_seconds"] = round(time.time() - start_time, 1)

        if len(result["best_params"]) > 0:
            print("  Best settings:", result["best_params"])
        print("  Cross-validated F1 (real class balance): {:.3f}".format(result["cv_f1"]))
        print("  Took", result["training_seconds"], "seconds")
        results[model_name] = result

    return results


def save_models(results, folder=MODELS_FOLDER, file_prefix=""):
    """
    Save every trained pipeline to disk with joblib.

    The whole pipeline is saved (preprocessing + model), so loading it
    later gives an object that accepts raw session data directly.

    Inputs:
        results: the dictionary returned by train_all_models().
        folder: where to save the files.
        file_prefix: added to each file name (e.g. "no_pagevalues_").
    Returns:
        A dictionary: model name -> saved file path.
    """
    os.makedirs(folder, exist_ok=True)
    saved_paths = {}
    for model_name, result in results.items():
        path = os.path.join(folder, file_prefix + MODEL_FILE_NAMES[model_name])
        save_pipeline(result["pipeline"], path)
        saved_paths[model_name] = path
        print("  Saved", path)
    return saved_paths


def xgboost_file_for(path):
    """
    Where the XGBoost model of a saved pipeline goes.

    Input:
        path: the pipeline file, e.g. "models/xgboost_benchmark.joblib".
    Returns:
        e.g. "models/xgboost_benchmark.json".
    """
    return os.path.splitext(path)[0] + ".json"


def save_pipeline(pipeline, path):
    """
    Save one fitted pipeline with joblib.

    An XGBoost model is saved separately in XGBoost's own JSON format and
    left out of the joblib file. A pickled XGBoost model saved on Linux
    does not load on Windows ("input stream corrupted"); the JSON format
    loads on every platform.

    Inputs:
        pipeline: the fitted pipeline.
        path: the joblib file to write.
    Returns:
        Nothing.
    """
    model = pipeline.named_steps["model"]
    if isinstance(model, XGBClassifier):
        model.save_model(xgboost_file_for(path))
        pipeline = Pipeline(pipeline.steps[:-1] + [("model", "passthrough")])
    joblib.dump(pipeline, path)


def load_pipeline(path):
    """
    Load a pipeline saved by save_pipeline().

    Input:
        path: the joblib file.
    Returns:
        The fitted pipeline, with its XGBoost model loaded back if it has one.
    """
    pipeline = joblib.load(path)
    if pipeline.named_steps["model"] == "passthrough":
        model = XGBClassifier()
        model.load_model(xgboost_file_for(path))
        pipeline.steps[-1] = ("model", model)
    return pipeline


OUT_OF_FOLD_FILE = "oof_probabilities.csv"


def save_out_of_fold_probabilities(training_results, train_target, folder=MODELS_FOLDER):
    """
    Save every model's out-of-fold training probabilities with the real labels.

    The app uses them to find the best-profit threshold again when the
    user changes the business assumptions, without touching the test set.

    Inputs:
        training_results: from train_all_models().
        train_target: the training labels.
        folder: where to save the CSV.
    Returns:
        The path of the saved file.
    """
    table = pd.DataFrame({"Revenue": np.asarray(train_target)})
    for model_name, result in training_results.items():
        table[model_name] = np.round(result["oof_probabilities"], 4)
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, OUT_OF_FOLD_FILE)
    table.to_csv(path, index=False)
    return path


def load_out_of_fold_probabilities(folder=MODELS_FOLDER):
    """
    Load the file saved by save_out_of_fold_probabilities().

    Returns:
        A DataFrame with a "Revenue" column and one probability column per model.
    """
    return pd.read_csv(os.path.join(folder, OUT_OF_FOLD_FILE))


def save_model_info(model_info, folder=MODELS_FOLDER):
    """
    Save extra information about the trained models as JSON.

    This holds things the prediction script and the app need: the best
    model's name, the chosen thresholds, default input values and the
    permutation importance results.

    Inputs:
        model_info: a dictionary that can be written as JSON.
        folder: the models folder.
    Returns:
        The path of the saved JSON file.
    """
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "model_info.json")
    with open(path, "w") as file:
        json.dump(model_info, file, indent=2)
    return path


def load_model_info(folder=MODELS_FOLDER):
    """
    Load the JSON file written by save_model_info().

    Input:
        folder: the models folder.
    Returns:
        The model_info dictionary.
    Raises:
        FileNotFoundError if the file is missing (run main.py first).
    """
    path = os.path.join(folder, "model_info.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path + " not found. Run 'python main.py' first.")
    with open(path) as file:
        return json.load(file)
