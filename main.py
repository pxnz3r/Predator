"""
main.py - Run the whole Predator pipeline with one command:

    python main.py

Steps:
    1. Load and check the data, draw EDA plots
    2. Split into training (80%) and test (20%) sets
    3. Tune and train all models with 5-fold cross-validation
    4. Evaluate on the test set (table, ROC curves, confusion matrices,
       bootstrap confidence intervals, McNemar's test, slices)
    5. Learning and validation curves (overfitting and underfitting)
    6. Explain the models (feature importance, tree plot, permutation
       importance, partial dependence)
    7. Ablation: retrain without PageValues
    8. Probability calibration
    9. Business threshold tuning and sensitivity analysis
    10. Error analysis of the best model
    11. Save the models and the information the app needs
"""

import sys
import time
import warnings

from predator import calibration, curves, data, errors, evaluate, explain, significance, threshold, train
from predator.predict import prepare_session
from predator.models import DECISION_TREE, KNN_UNDERSAMPLED, MODEL_FILE_NAMES, XGBOOST

# Some libraries print warnings that are not useful for a student run.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def print_step(number, title):
    """
    Print a clear header for each pipeline step.

    Inputs:
        number: the step number.
        title: what the step does.
    Returns:
        Nothing.
    """
    print("\n" + "#" * 70)
    print(f"# STEP {number}: {title}")
    print("#" * 70)


def to_plain_value(value):
    """
    Convert numpy numbers (and booleans) to plain Python values so they
    can be written to JSON.

    Input:
        value: a number, boolean or string.
    Returns:
        The same value as a plain Python type.
    """
    if hasattr(value, "item"):
        return value.item()
    return value


def build_model_info(
    best_model_name, chosen_thresholds, train_features, comparison_table, importance_table, training_results
):
    """
    Collect everything predict.py and app.py need into one dictionary.

    Inputs:
        best_model_name: name of the best model.
        chosen_thresholds: model name -> thresholds.
        train_features: training features (for default values and ranges).
        comparison_table: test-set scores.
        importance_table: permutation importance (features x models).
        training_results: from train.train_all_models().
    Returns:
        A dictionary that can be saved as JSON.
    """
    default_values = {}
    feature_ranges = {}
    for column in data.NUMERICAL_FEATURES:
        default_values[column] = float(train_features[column].median())
        feature_ranges[column] = [float(train_features[column].min()), float(train_features[column].max())]

    category_options = {}
    for column in data.CATEGORICAL_FEATURES:
        default_values[column] = to_plain_value(train_features[column].mode()[0])
        category_options[column] = [
            to_plain_value(value) for value in sorted(train_features[column].unique())
        ]

    permutation_importance = {}
    for model_name in importance_table.columns:
        permutation_importance[model_name] = {
            feature: round(float(value), 5) for feature, value in importance_table[model_name].items()
        }

    best_params = {}
    for model_name, result in training_results.items():
        best_params[model_name] = {key: to_plain_value(value) for key, value in result["best_params"].items()}

    return {
        "best_model": best_model_name,
        "model_files": MODEL_FILE_NAMES,
        "thresholds": chosen_thresholds,
        "business_assumptions": {
            "AVERAGE_ORDER_PROFIT": threshold.AVERAGE_ORDER_PROFIT,
            "OFFER_COST": threshold.OFFER_COST,
            "OFFER_CONVERSION_BOOST": threshold.OFFER_CONVERSION_BOOST,
        },
        "numerical_features": data.NUMERICAL_FEATURES,
        "categorical_features": data.CATEGORICAL_FEATURES,
        "default_values": default_values,
        "feature_ranges": feature_ranges,
        "category_options": category_options,
        "best_params": best_params,
        "permutation_importance": permutation_importance,
        "test_metrics": comparison_table.to_dict(orient="records"),
    }


def main():
    """
    Run every step of the pipeline in order.

    Returns:
        0 if everything worked, 1 if the dataset is missing.
    """
    start_time = time.time()

    print_step(1, "Load and explore the data")
    try:
        dataset = data.load_dataset()
    except FileNotFoundError as error:
        print(error)
        return 1
    data.validate_dataset(dataset)
    data.run_eda(dataset)

    print_step(2, "Split into training and test sets")
    train_features, test_features, train_target, test_target = data.split_data(dataset)

    print_step(3, "Tune and train the models (GridSearchCV, 5-fold, F1)")
    training_results = train.train_all_models(train_features, train_target)
    models = {name: result["pipeline"] for name, result in training_results.items()}

    print_step(4, "Evaluate on the test set")
    comparison_table = evaluate.build_comparison_table(models, training_results, test_features, test_target)
    best_model_name = evaluate.choose_best_model(comparison_table)
    evaluate.plot_roc_curves(models, test_features, test_target)
    evaluate.plot_confusion_matrices(models, test_features, test_target)
    evaluate.plot_metric_comparison(comparison_table)
    significance.run_significance_analysis(
        models,
        comparison_table,
        test_features,
        test_target,
        pairs_to_compare=[(DECISION_TREE, XGBOOST), (DECISION_TREE, KNN_UNDERSAMPLED)],
    )
    evaluate.evaluate_slices(models, [DECISION_TREE, XGBOOST], test_features, test_target)

    print_step(5, "Learning and validation curves")
    curves.plot_learning_curves(models, train_features, train_target)
    curves.plot_validation_curves(models, train_features, train_target)

    print_step(6, "Explain the models")
    explain.plot_tree_feature_importance(models[DECISION_TREE])
    explain.plot_top_of_tree(models[DECISION_TREE], max_depth=3)
    importance_table = explain.permutation_importance_all_models(models, test_features, test_target)
    explain.write_tree_vs_xgboost_comparison(comparison_table, models, importance_table)
    explain.plot_partial_dependence(models, [DECISION_TREE, XGBOOST], train_features)

    print_step(7, "Ablation: retrain every model without PageValues")
    explain.run_ablation(
        train_features, train_target, test_features, test_target, comparison_table, importance_table
    )

    print_step(8, "Probability calibration")
    calibration.run_calibration_analysis(
        models, [DECISION_TREE, XGBOOST], train_features, train_target, test_features, test_target
    )

    print_step(9, "Business threshold tuning and sensitivity analysis")
    threshold_table, chosen_thresholds, training_curves = threshold.tune_thresholds(
        models, training_results, train_target, test_features, test_target
    )
    threshold.plot_profit_curves(training_curves, chosen_thresholds)
    threshold.run_sensitivity_analysis(training_results, train_target, [DECISION_TREE, XGBOOST])

    print_step(10, "Error analysis of the best model")
    best_threshold = chosen_thresholds[best_model_name]["best_profit"]
    errors.run_error_analysis(
        models[best_model_name], best_model_name, test_features, test_target, best_threshold
    )

    print_step(11, "Save the models")
    train.save_models(training_results)
    print("  Saved", train.save_out_of_fold_probabilities(training_results, train_target))
    model_info = build_model_info(
        best_model_name,
        chosen_thresholds,
        train_features,
        comparison_table,
        importance_table,
        training_results,
    )
    print("  Saved", train.save_model_info(model_info))

    print_step(12, "Test the Decision Tree with a new custom input")
    # A new session typed in by hand, as in the lab lecture: a returning
    # visitor in November who viewed 15 product pages with a high PageValue.
    new_data = {
        "Administrative": 1, "Administrative_Duration": 81.0, "Informational": 0,
        "Informational_Duration": 0.0, "ProductRelated": 15, "ProductRelated_Duration": 914.5,
        "BounceRates": 0.0, "ExitRates": 0.014, "PageValues": 56.58, "SpecialDay": 0.0,
        "Month": "Nov", "OperatingSystems": 2, "Browser": 2, "Region": 2, "TrafficType": 11,
        "VisitorType": "Returning_Visitor", "Weekend": True,
    }  # fmt: skip
    new_df = prepare_session(new_data)
    print("  Predicted class (1 = purchase):", models[DECISION_TREE].predict(new_df))

    minutes = (time.time() - start_time) / 60
    print(f"\nDone in {minutes:.1f} minutes. Figures are in outputs/figures/, tables in outputs/.")
    print("Next: streamlit run app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
