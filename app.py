"""
app.py - Streamlit demo for Predator.

Run it from the project folder (after python main.py):
    streamlit run app.py

Page 1 (Predict a session): enter a session's features and see the
purchase probability, the decision at the business threshold and the
main factors behind the prediction.
Page 2 (Model comparison): the comparison tables and key figures.
"""

import json
import os

import pandas as pd
import streamlit as st

from predator.explain import describe_decision_path
from predator.models import DECISION_TREE, MODEL_NAMES
from predator.predict import load_model_by_name, predict_session, prepare_session
from predator.threshold import break_even_probability, find_best_profit_threshold
from predator.train import MODELS_FOLDER, OUT_OF_FOLD_FILE, load_out_of_fold_probabilities

OUTPUTS_FOLDER = "outputs"
FIGURES_FOLDER = os.path.join(OUTPUTS_FOLDER, "figures")
EXAMPLES_FOLDER = "examples"

# Short help text shown next to every input.
FEATURE_HELP = {
    "Administrative": "Number of account/admin pages visited (login, profile...).",
    "Administrative_Duration": "Seconds spent on admin pages.",
    "Informational": "Number of information pages visited (shipping, contact...).",
    "Informational_Duration": "Seconds spent on information pages.",
    "ProductRelated": "Number of product pages visited.",
    "ProductRelated_Duration": "Seconds spent on product pages.",
    "BounceRates": "Average share of visitors who left right after entering the pages seen (0 to 0.2).",
    "ExitRates": "Average share of page views that were the last of a session, for the pages seen (0 to 0.2).",
    "PageValues": "Average value of the pages seen, based on pages that led to purchases before. 0 = none.",
    "SpecialDay": "Closeness to a holiday like Valentine's Day (0 = not close, 1 = the peak day).",
    "Month": "Month of the visit (the dataset has no January or April).",
    "OperatingSystems": "Operating system code.",
    "Browser": "Browser code.",
    "Region": "Region code of the visitor.",
    "TrafficType": "How the visitor arrived (ad, search, direct...), as a code.",
    "VisitorType": "New visitor, returning visitor, or other.",
    "Weekend": "Was the visit on a weekend?",
}

COUNT_FEATURES = ["Administrative", "Informational", "ProductRelated"]


def load_everything():
    """
    Load model_info.json and every saved model.

    Streamlit re-runs this whole script after every click, so we keep the
    loaded models in st.session_state and only read the files the first time.

    Returns:
        (model_info, models, out_of_fold) where models is a dictionary
        name -> pipeline and out_of_fold holds the out-of-fold training
        probabilities used to re-tune the threshold.
    """
    if "models" not in st.session_state:
        with open(os.path.join(MODELS_FOLDER, "model_info.json")) as file:
            model_info = json.load(file)
        models = {}
        for model_name in MODEL_NAMES:
            models[model_name] = load_model_by_name(model_name, model_info)
        st.session_state["model_info"] = model_info
        st.session_state["models"] = models
        st.session_state["out_of_fold"] = load_out_of_fold_probabilities()
    return st.session_state["model_info"], st.session_state["models"], st.session_state["out_of_fold"]


def models_are_available():
    """
    Check that main.py has been run and the saved models exist.

    Returns:
        True if model_info.json, the out-of-fold probabilities and every
        model file are present.
    """
    info_path = os.path.join(MODELS_FOLDER, "model_info.json")
    if not os.path.isfile(info_path) or not os.path.isfile(os.path.join(MODELS_FOLDER, OUT_OF_FOLD_FILE)):
        return False
    with open(info_path) as file:
        model_files = json.load(file)["model_files"]
    for file_name in model_files.values():
        if not os.path.isfile(os.path.join(MODELS_FOLDER, file_name)):
            return False
    return True


def load_starting_values(choice, model_info):
    """
    Get the values the input widgets start with.

    Input:
        choice: "Typical session", "Example buyer" or "Example browser".
        model_info: the saved model information (holds the typical values).
    Returns:
        A dictionary feature name -> value.
    """
    if choice == "Example buyer":
        path = os.path.join(EXAMPLES_FOLDER, "session_buyer.json")
    elif choice == "Example browser":
        path = os.path.join(EXAMPLES_FOLDER, "session_browser.json")
    else:
        return dict(model_info["default_values"])

    with open(path) as file:
        return json.load(file)


def number_input(feature, start_values, model_info, key_prefix):
    """
    Draw one numeric input box for a feature.

    Inputs:
        feature: the feature name.
        start_values: starting values for the inputs.
        model_info: holds the min/max seen in training.
        key_prefix: makes the widget key unique per starting choice.
    Returns:
        The number the user entered.
    """
    lowest, highest = model_info["feature_ranges"][feature]
    if feature in COUNT_FEATURES:
        return st.number_input(
            feature,
            min_value=0,
            max_value=int(highest * 2),
            value=int(start_values[feature]),
            step=1,
            help=FEATURE_HELP[feature],
            key=key_prefix + feature,
        )
    if feature in ["BounceRates", "ExitRates", "SpecialDay"]:
        return st.number_input(
            feature,
            min_value=0.0,
            max_value=float(highest),
            value=float(start_values[feature]),
            step=0.01,
            format="%.3f",
            help=FEATURE_HELP[feature],
            key=key_prefix + feature,
        )
    return st.number_input(
        feature,
        min_value=0.0,
        value=float(start_values[feature]),
        step=10.0,
        help=FEATURE_HELP[feature],
        key=key_prefix + feature,
    )


def choice_input(feature, start_values, model_info, key_prefix):
    """
    Draw one drop-down box for a categorical feature.

    Inputs:
        feature: the feature name.
        start_values: starting values for the inputs.
        model_info: holds the allowed categories.
        key_prefix: makes the widget key unique per starting choice.
    Returns:
        The category the user picked.
    """
    options = model_info["category_options"][feature]
    if feature == "Month":
        month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
        options = [month for month in month_order if month in options]
    start_value = start_values[feature]
    start_index = options.index(start_value) if start_value in options else 0
    return st.selectbox(
        feature, options, index=start_index, help=FEATURE_HELP[feature], key=key_prefix + feature
    )


def show_prediction_page(model_info, models, model_name, threshold):
    """
    Page 1: inputs, prediction, decision and the factors behind it.

    Inputs:
        model_info: the saved model information.
        models: dictionary name -> pipeline.
        model_name: the model picked in the sidebar.
        threshold: the business threshold picked in the sidebar.
    Returns:
        Nothing (draws on the page).
    """
    start_choice = st.radio(
        "Start from",
        ["Typical session", "Example buyer", "Example browser"],
        horizontal=True,
        help="Typical session = median/most common values from the training data. "
        "The examples are real sessions from the dataset.",
    )
    start_values = load_starting_values(start_choice, model_info)
    key_prefix = start_choice + "_"

    session_values = {}
    column_pages, column_behaviour, column_context = st.columns(3)
    with column_pages:
        st.subheader("Pages visited")
        for feature in [
            "Administrative",
            "Administrative_Duration",
            "Informational",
            "Informational_Duration",
            "ProductRelated",
            "ProductRelated_Duration",
        ]:
            session_values[feature] = number_input(feature, start_values, model_info, key_prefix)
    with column_behaviour:
        st.subheader("Behaviour")
        for feature in ["PageValues", "BounceRates", "ExitRates", "SpecialDay"]:
            session_values[feature] = number_input(feature, start_values, model_info, key_prefix)
        session_values["Weekend"] = st.checkbox(
            "Weekend",
            value=bool(start_values["Weekend"]),
            help=FEATURE_HELP["Weekend"],
            key=key_prefix + "Weekend",
        )
    with column_context:
        st.subheader("Visit context")
        for feature in ["Month", "VisitorType", "TrafficType", "Region", "Browser", "OperatingSystems"]:
            session_values[feature] = choice_input(feature, start_values, model_info, key_prefix)

    session = prepare_session(session_values)
    result = predict_session(models[model_name], session, threshold)

    st.divider()
    st.subheader("Prediction")
    column_number, column_decision = st.columns([1, 2])
    with column_number:
        st.metric("Purchase probability", "{:.1%}".format(result["probability"]))
        st.progress(min(max(result["probability"], 0.0), 1.0))
        st.caption("Model: " + model_name + " | default 0.5 cutoff says: " + result["label"])
    with column_decision:
        if result["passes_threshold"]:
            st.success(f"Above the business threshold ({threshold:.2f}): send the discount offer.")
        else:
            st.warning(f"Below the business threshold ({threshold:.2f}): do not send the offer.")
        if model_name == "Naive Bayes (baseline)":
            st.info(
                "Naive Bayes gives extreme, badly calibrated probabilities on this data. "
                "Treat it as a baseline only."
            )

    st.subheader("What drives this model")
    importance = pd.Series(model_info["permutation_importance"][model_name]).sort_values(ascending=False)
    importance_frame = importance.head(8).rename("Drop in ROC-AUC when shuffled").to_frame()
    st.bar_chart(importance_frame, horizontal=True)
    st.caption(
        "Permutation importance on the test set: how much the model's ROC-AUC drops when a "
        "feature's values are shuffled. This describes the model overall, not this one session."
    )

    if model_name == DECISION_TREE:
        st.subheader("Decision path for this session")
        rules = describe_decision_path(models[DECISION_TREE], session)
        for step_number, rule in enumerate(rules, start=1):
            st.markdown(f"{step_number}. {rule}")
        st.caption(
            "The questions the Decision Tree asked, from the root to the leaf that made this prediction."
        )
    else:
        st.caption(
            "Pick the Decision Tree in the sidebar to see the exact rules it followed for this session."
        )


def show_table(title, file_name):
    """
    Show a CSV table from outputs/ if it exists.

    Inputs:
        title: the heading to show.
        file_name: the CSV file name inside outputs/.
    Returns:
        Nothing.
    """
    path = os.path.join(OUTPUTS_FOLDER, file_name)
    if os.path.isfile(path):
        st.subheader(title)
        st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def show_figure(caption, file_name):
    """
    Show an image from outputs/figures/ if it exists.

    Inputs:
        caption: text under the image.
        file_name: the image file name.
    Returns:
        Nothing.
    """
    path = os.path.join(FIGURES_FOLDER, file_name)
    if os.path.isfile(path):
        st.image(path, caption=caption, width="stretch")


def show_comparison_page(model_info):
    """
    Page 2: model comparison tables and key figures.

    Input:
        model_info: the saved model information.
    Returns:
        Nothing.
    """
    st.markdown(
        "Best model by cross-validated F1: **" + model_info["best_model"] + "**. "
        "XGBoost is a benchmark beyond the course; the Decision Tree and KNN are the main models."
    )
    show_table("Test-set scores (threshold 0.5)", "model_comparison.csv")
    show_figure("ROC curves", "roc_curves.png")
    show_figure("Confusion matrices", "confusion_matrices.png")
    show_table("Thresholds: default vs best F1 vs best profit", "threshold_comparison.csv")
    show_figure("Profit and F1 against the threshold", "profit_vs_threshold.png")
    show_figure("Permutation importance for every model", "permutation_importance.png")
    show_figure("Top 3 levels of the Decision Tree", "decision_tree_top_levels.png")
    show_table("Ablation: scores without PageValues", "ablation_metrics.csv")
    show_figure("Ablation scores", "ablation_scores.png")
    show_figure("Error analysis of the best model", "error_analysis.png")

    st.subheader("Deeper analysis")
    show_table("95% bootstrap confidence intervals (test set)", "confidence_intervals.csv")
    show_table("McNemar's test: do two models make different mistakes?", "mcnemar_tests.csv")
    show_figure("Learning curves", "learning_curves.png")
    show_figure("Validation curves", "validation_curves.png")
    show_figure("Partial dependence", "partial_dependence.png")
    show_table("Calibration before and after isotonic calibration", "calibration_summary.csv")
    show_figure("Reliability diagram before and after calibration", "calibration_before_after.png")
    show_figure("Best threshold for other business assumptions", "sensitivity_thresholds.png")
    show_figure("F1 by visitor type and month", "slice_scores.png")


def main():
    """
    Build the whole app: sidebar settings and the two pages.

    Returns:
        Nothing.
    """
    st.set_page_config(page_title="Predator: purchase prediction", layout="wide")
    st.title("Predator: will this session end in a purchase?")

    if not models_are_available():
        st.error(
            "No trained models found in models/. Train them first by running this in the project folder:"
        )
        st.code("python main.py")
        st.stop()

    model_info, models, out_of_fold = load_everything()

    with st.sidebar:
        st.header("Settings")
        best_model = model_info["best_model"]
        labels = [name + (" (best)" if name == best_model else "") for name in MODEL_NAMES]
        picked_label = st.selectbox("Model", labels, index=MODEL_NAMES.index(best_model))
        model_name = MODEL_NAMES[labels.index(picked_label)]

        # The business assumptions start at the values main.py used. Changing
        # them re-tunes the threshold on the out-of-fold training
        # probabilities, exactly like main.py does (the test set is not used).
        assumptions = model_info["business_assumptions"]
        st.subheader("Business assumptions")
        order_profit = st.number_input(
            "Order profit ($)",
            min_value=1.0,
            value=float(assumptions["AVERAGE_ORDER_PROFIT"]),
            step=5.0,
            help="Average profit the shop makes on one order.",
        )
        offer_cost = st.number_input(
            "Offer cost ($)",
            min_value=0.0,
            value=float(assumptions["OFFER_COST"]),
            step=1.0,
            help="What one discount offer costs, whether or not the visitor buys.",
        )
        conversion_boost = st.number_input(
            "Offer conversion boost",
            min_value=0.01,
            max_value=1.0,
            value=float(assumptions["OFFER_CONVERSION_BOOST"]),
            step=0.05,
            help="Share of the order profit the offer secures for a real buyer. Illustrative, like the other two.",
        )

        best_threshold, best_profit = find_best_profit_threshold(
            out_of_fold["Revenue"].to_numpy(),
            out_of_fold[model_name].to_numpy(),
            order_profit,
            offer_cost,
            conversion_boost,
        )
        threshold = st.slider(
            "Business threshold",
            min_value=0.01,
            max_value=1.0,
            value=best_threshold,
            step=0.01,
            # A new key for every model and set of assumptions moves the
            # slider back to the best threshold when one of them changes.
            key=f"threshold-{model_name}-{order_profit}-{offer_cost}-{conversion_boost}",
            help="Send the offer when the purchase probability is at least this value. "
            "Starts at the threshold that maximised profit on the training data for these assumptions.",
        )
        if best_profit <= 0:
            st.caption("No threshold makes a profit with these assumptions: send no offers.")
        st.caption(
            f"Best profit: {best_threshold:.2f} | Best F1: {model_info['thresholds'][model_name]['best_f1']:.2f}"
            f" | Break-even probability: {min(break_even_probability(order_profit, offer_cost, conversion_boost), 1.0):.2f}"
        )

    prediction_tab, comparison_tab = st.tabs(["Predict a session", "Model comparison"])
    with prediction_tab:
        show_prediction_page(model_info, models, model_name, threshold)
    with comparison_tab:
        show_comparison_page(model_info)


main()
