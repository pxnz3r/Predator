"""
predict.py - Predict one shopping session from the command line.

Usage (from the project folder, after running main.py):
    python -m predator.predict examples/session_buyer.json
    python -m predator.predict examples/session_browser.json --model "Decision Tree"

The JSON file holds the 17 feature values of one session. The script
prints "purchase" or "no purchase", the purchase probability, and whether
the session passes the business threshold (the best-profit threshold).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from predator.data import ALL_FEATURES, CATEGORICAL_FEATURES, NUMERICAL_FEATURES
from predator.train import MODELS_FOLDER, load_model_info, load_pipeline

# Categorical columns that hold whole numbers (codes) in the dataset.
INTEGER_CODE_FEATURES = ["OperatingSystems", "Browser", "Region", "TrafficType"]


def prepare_session(session_values):
    """
    Turn a dictionary of feature values into a one-row DataFrame with the
    same columns and types as the training data.

    Input:
        session_values: dictionary feature name -> value.
    Returns:
        A one-row pandas DataFrame.
    Raises:
        ValueError if a feature is missing or has the wrong type.
    """
    missing = [feature for feature in ALL_FEATURES if feature not in session_values]
    if len(missing) > 0:
        raise ValueError("These features are missing from the session: " + ", ".join(missing))

    row = {}
    for feature in NUMERICAL_FEATURES:
        try:
            row[feature] = float(session_values[feature])
        except (TypeError, ValueError) as error:
            raise ValueError(feature + " must be a number, got: " + repr(session_values[feature])) from error

    for feature in CATEGORICAL_FEATURES:
        value = session_values[feature]
        if feature in INTEGER_CODE_FEATURES:
            try:
                row[feature] = int(value)
            except (TypeError, ValueError) as error:
                raise ValueError(feature + " must be a whole number code, got: " + repr(value)) from error
        elif feature == "Weekend":
            # Accept true/false as a JSON boolean or as text. Anything else is
            # an error, so a typo like "yes" is not silently read as False.
            if isinstance(value, str) and value.strip().lower() in ("true", "false"):
                value = value.strip().lower() == "true"
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError("Weekend must be true or false, got: " + repr(value))
            row[feature] = bool(value)
        else:
            row[feature] = str(value)

    return pd.DataFrame([row], columns=ALL_FEATURES)


def load_session_from_json(path):
    """
    Read one session's feature values from a JSON file.

    Input:
        path: path to the JSON file.
    Returns:
        A one-row DataFrame ready for the model.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError("Session file not found: " + path)
    with open(path) as file:
        session_values = json.load(file)
    return prepare_session(session_values)


def predict_session(pipeline, session, threshold):
    """
    Predict whether one session will end in a purchase.

    Inputs:
        pipeline: a fitted pipeline.
        session: a one-row DataFrame from prepare_session().
        threshold: the business threshold to compare the probability with.
    Returns:
        A dictionary with:
            "label": "purchase" or "no purchase" (default 0.5 cutoff)
            "probability": the purchase probability (0 to 1)
            "threshold": the business threshold used
            "passes_threshold": True if the probability reaches the threshold
    """
    probability = float(pipeline.predict_proba(session)[0, 1])
    if probability >= 0.5:
        label = "purchase"
    else:
        label = "no purchase"
    return {
        "label": label,
        "probability": probability,
        "threshold": float(threshold),
        "passes_threshold": probability >= threshold,
    }


def load_model_by_name(model_name, model_info, folder=MODELS_FOLDER):
    """
    Load one saved pipeline by its model name.

    Inputs:
        model_name: e.g. "Decision Tree".
        model_info: the dictionary from train.load_model_info().
        folder: the models folder.
    Returns:
        The fitted pipeline.
    """
    if model_name not in model_info["model_files"]:
        raise ValueError(
            "Unknown model '"
            + model_name
            + "'. Choose one of: "
            + ", ".join(model_info["model_files"].keys())
        )
    path = os.path.join(folder, model_info["model_files"][model_name])
    if not os.path.isfile(path):
        raise FileNotFoundError(path + " not found. Run 'python main.py' first.")
    return load_pipeline(path)


def main():
    """
    Command-line entry point: read the arguments, predict, print the result.

    Returns:
        0 on success, 1 on an error (used as the program's exit code).
    """
    parser = argparse.ArgumentParser(description="Predict whether a shopping session ends in a purchase.")
    parser.add_argument("session_file", help="JSON file with the session's 17 feature values")
    parser.add_argument("--model", default=None, help="model name (default: the best saved model)")
    arguments = parser.parse_args()

    try:
        model_info = load_model_info()
        model_name = arguments.model or model_info["best_model"]
        pipeline = load_model_by_name(model_name, model_info)
        session = load_session_from_json(arguments.session_file)
    except (FileNotFoundError, ValueError) as error:
        print("Error:", error)
        return 1

    threshold = model_info["thresholds"][model_name]["best_profit"]
    result = predict_session(pipeline, session, threshold)

    print("Model:                 " + model_name)
    print("Prediction (0.5 cut):  " + result["label"].upper())
    print("Purchase probability:  {:.1%}".format(result["probability"]))
    if result["passes_threshold"]:
        decision = "YES - send the discount offer"
    else:
        decision = "NO - do not send the offer"
    print("Business threshold:    {:.2f} -> {}".format(result["threshold"], decision))
    return 0


if __name__ == "__main__":
    sys.exit(main())
