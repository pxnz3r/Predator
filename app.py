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

import altair as alt
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

# Readable label for every dataset column.
FEATURE_LABELS = {
    "Administrative": "Account pages",
    "Administrative_Duration": "Time on account pages (s)",
    "Informational": "Info pages",
    "Informational_Duration": "Time on info pages (s)",
    "ProductRelated": "Product pages",
    "ProductRelated_Duration": "Time on product pages (s)",
    "BounceRates": "Bounce rate",
    "ExitRates": "Exit rate",
    "PageValues": "Page value",
    "SpecialDay": "Closeness to a holiday",
    "Month": "Month",
    "OperatingSystems": "Operating system",
    "Browser": "Browser",
    "Region": "Region",
    "TrafficType": "Traffic source",
    "VisitorType": "Visitor type",
    "Weekend": "Weekend visit",
}

# Help text: what the label can't say, plus the dataset column name.
FEATURE_HELP = {
    "Administrative": "Login, profile and other account pages.",
    "Administrative_Duration": "",
    "Informational": "Shipping, contact and other information pages.",
    "Informational_Duration": "",
    "ProductRelated": "",
    "ProductRelated_Duration": "",
    "BounceRates": "Average share of visitors who left right after entering the pages seen (0 to 0.2).",
    "ExitRates": "Average share of page views that were the last of a session, for the pages seen (0 to 0.2).",
    "PageValues": "Average value of the pages seen, based on pages that led to purchases before. 0 = none.",
    "SpecialDay": "0 = not near a holiday like Valentine's Day, 1 = the peak day.",
    "Month": "The dataset has no January or April.",
    "OperatingSystems": "A code; the dataset doesn't say which system it is.",
    "Browser": "A code; the dataset doesn't say which browser it is.",
    "Region": "A code; the dataset doesn't name the regions.",
    "TrafficType": "How the visitor arrived (ad, search, direct...), as a code.",
    "VisitorType": "",
    "Weekend": "",
}

COUNT_FEATURES = ["Administrative", "Informational", "ProductRelated"]
DURATION_FEATURES = ["Administrative_Duration", "Informational_Duration", "ProductRelated_Duration"]


def feature_help(feature):
    """
    Help text for one input: the explanation and the dataset column name.

    Input:
        feature: the feature name.
    Returns:
        The help text (Markdown).
    """
    column = f"Column: `{feature}`"
    if FEATURE_HELP[feature]:
        return FEATURE_HELP[feature] + "  \n" + column
    return column


def readable_rule(rule):
    """
    Swap the column name at the start of a decision-path rule for its label.

    Input:
        rule: e.g. "PageValues is 0.00, which is <= 0.94".
    Returns:
        e.g. "Page value is 0.00, which is <= 0.94".
    """
    for feature, label in FEATURE_LABELS.items():
        if rule.startswith(feature + " "):
            return label + rule[len(feature) :]
    return rule


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
            FEATURE_LABELS[feature],
            min_value=0,
            max_value=int(highest * 2),
            value=int(start_values[feature]),
            step=1,
            help=feature_help(feature),
            key=key_prefix + feature,
        )
    if feature in ["BounceRates", "ExitRates", "SpecialDay"]:
        return st.number_input(
            FEATURE_LABELS[feature],
            min_value=0.0,
            max_value=float(highest),
            value=float(start_values[feature]),
            step=0.01,
            format="%.3f",
            help=feature_help(feature),
            key=key_prefix + feature,
        )
    if feature in DURATION_FEATURES:
        number_format = "%.0f"
    else:
        number_format = "%.2f"
    return st.number_input(
        FEATURE_LABELS[feature],
        min_value=0.0,
        value=float(start_values[feature]),
        step=10.0,
        format=number_format,
        help=feature_help(feature),
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
        FEATURE_LABELS[feature],
        options,
        index=start_index,
        help=feature_help(feature),
        key=key_prefix + feature,
    )


def probability_chart(probability, threshold):
    """
    One bar from 0 to 100%: the purchase probability, with the business
    threshold drawn as a dashed line across it.

    Inputs:
        probability: the predicted purchase probability.
        threshold: the business threshold.
    Returns:
        An Altair chart.
    """
    scale = alt.Scale(domain=[0, 1])
    axis = alt.Axis(format="%", title=None, tickCount=5, grid=False)
    track = (
        alt.Chart(pd.DataFrame({"end": [1.0]}))
        .mark_bar(opacity=0.15, cornerRadius=4)
        .encode(x=alt.X("end:Q", scale=scale, axis=axis))
    )
    bar = (
        alt.Chart(pd.DataFrame({"probability": [probability]}))
        .mark_bar(cornerRadius=4)
        .encode(x=alt.X("probability:Q", scale=scale, axis=axis))
    )
    line = (
        alt.Chart(pd.DataFrame({"threshold": [threshold]}))
        .mark_rule(strokeWidth=2, strokeDash=[4, 3])
        .encode(x=alt.X("threshold:Q", scale=scale))
    )
    return (track + bar + line).properties(height=24)


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
    column_inputs, column_result = st.columns([3, 2], gap="large")

    with column_inputs:
        start_choice = st.radio(
            "Start from",
            ["Typical session", "Example buyer", "Example browser"],
            horizontal=True,
            help="Typical session = median or most common values from the training data. "
            "The examples are real sessions from the dataset.",
        )
        start_values = load_starting_values(start_choice, model_info)
        key_prefix = start_choice + "_"

        session_values = {}
        column_pages, column_behaviour = st.columns(2)
        with column_pages:
            st.subheader("Pages visited", anchor=False)
            for feature in COUNT_FEATURES + DURATION_FEATURES:
                session_values[feature] = number_input(feature, start_values, model_info, key_prefix)
        with column_behaviour:
            st.subheader("Behaviour", anchor=False)
            for feature in ["PageValues", "BounceRates", "ExitRates", "SpecialDay"]:
                session_values[feature] = number_input(feature, start_values, model_info, key_prefix)
            for feature in ["Month", "VisitorType"]:
                session_values[feature] = choice_input(feature, start_values, model_info, key_prefix)
            session_values["Weekend"] = st.checkbox(
                FEATURE_LABELS["Weekend"],
                value=bool(start_values["Weekend"]),
                help=feature_help("Weekend"),
                key=key_prefix + "Weekend",
            )
        with st.expander("Traffic, region and device codes"):
            column_left, column_right = st.columns(2)
            with column_left:
                for feature in ["TrafficType", "Region"]:
                    session_values[feature] = choice_input(feature, start_values, model_info, key_prefix)
            with column_right:
                for feature in ["Browser", "OperatingSystems"]:
                    session_values[feature] = choice_input(feature, start_values, model_info, key_prefix)

    session = prepare_session(session_values)
    result = predict_session(models[model_name], session, threshold)

    with column_result:
        if result["passes_threshold"]:
            st.header("Send the discount offer", anchor=False)
        else:
            st.header("Don't send the offer", anchor=False)
        st.metric("Purchase probability", "{:.1%}".format(result["probability"]))
        st.altair_chart(probability_chart(result["probability"], threshold), width="stretch")
        st.caption(
            f"Dashed line: the business threshold, {threshold:.2f}. "
            f"At the default 0.5 cutoff the model predicts {result['label']}."
        )
        if model_name == "Naive Bayes (baseline)":
            st.caption(
                "Naive Bayes gives extreme, badly calibrated probabilities on this data. "
                "Treat it as a baseline only."
            )

        if model_name == DECISION_TREE:
            st.subheader("Decision path", anchor=False)
            rules = describe_decision_path(models[DECISION_TREE], session)
            for step_number, rule in enumerate(rules, start=1):
                st.markdown(f"{step_number}. {readable_rule(rule)}")
        else:
            st.caption("Pick the Decision Tree in the sidebar to see the rules it followed for this session.")

    st.divider()
    st.subheader("What drives this model", anchor=False)
    importance = pd.Series(model_info["permutation_importance"][model_name]).sort_values(ascending=False)
    importance = importance.head(8).rename(index=FEATURE_LABELS)
    st.bar_chart(importance.rename("Drop in ROC-AUC").to_frame(), horizontal=True)
    st.caption(
        "How much the model's test ROC-AUC drops when an input is shuffled. "
        "This describes the model overall, not this one session."
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
        st.subheader(title, anchor=False)
        st.dataframe(pd.read_csv(path), hide_index=True, width="stretch")


def show_figure(file_name):
    """
    Show an image from outputs/figures/ if it exists. Every figure has its
    own title, so no caption is added.

    Input:
        file_name: the image file name.
    Returns:
        Nothing.
    """
    path = os.path.join(FIGURES_FOLDER, file_name)
    if os.path.isfile(path):
        st.image(path, width="stretch")


def show_comparison_page(model_info):
    """
    Page 2: model comparison tables and key figures, one tab per question.

    Input:
        model_info: the saved model information.
    Returns:
        Nothing.
    """
    st.markdown(
        "Best model by cross-validated F1: **" + model_info["best_model"] + "**. "
        "XGBoost is a benchmark beyond the course; the Decision Tree and KNN are the main models."
    )
    scores_tab, threshold_tab, drivers_tab, ablation_tab, errors_tab, reliability_tab = st.tabs(
        ["Scores", "Threshold", "What drives predictions", "Without PageValues", "Errors", "Reliability"]
    )
    with scores_tab:
        show_table("Test-set scores (threshold 0.5)", "model_comparison.csv")
        show_figure("roc_curves.png")
        show_figure("confusion_matrices.png")
        show_table("95% bootstrap confidence intervals (test set)", "confidence_intervals.csv")
        show_table("McNemar's test: do two models make different mistakes?", "mcnemar_tests.csv")
    with threshold_tab:
        show_table("Default vs best F1 vs best profit", "threshold_comparison.csv")
        show_figure("profit_vs_threshold.png")
        show_figure("sensitivity_thresholds.png")
    with drivers_tab:
        show_figure("permutation_importance.png")
        show_figure("partial_dependence.png")
        show_figure("decision_tree_top_levels.png")
    with ablation_tab:
        show_table("Scores without PageValues", "ablation_metrics.csv")
        show_figure("ablation_scores.png")
    with errors_tab:
        show_figure("error_analysis.png")
        show_figure("slice_scores.png")
    with reliability_tab:
        show_figure("learning_curves.png")
        show_figure("validation_curves.png")
        show_table("Calibration before and after isotonic calibration", "calibration_summary.csv")
        show_figure("calibration_before_after.png")


def main():
    """
    Build the whole app: sidebar settings and the two pages.

    Returns:
        Nothing.
    """
    st.set_page_config(page_title="Predator: purchase prediction", layout="wide")
    st.title("Predator", anchor=False)

    if not models_are_available():
        st.error(
            "No trained models found in models/. Train them first by running this in the project folder:"
        )
        st.code("python main.py")
        st.stop()

    model_info, models, out_of_fold = load_everything()

    with st.sidebar:
        best_model = model_info["best_model"]
        model_name = st.selectbox(
            "Model",
            MODEL_NAMES,
            index=MODEL_NAMES.index(best_model),
            format_func=lambda name: name + " · best" if name == best_model else name,
        )

        # The business assumptions start at the values main.py used. Changing
        # them re-tunes the threshold on the out-of-fold training
        # probabilities, exactly like main.py does (the test set is not used).
        assumptions = model_info["business_assumptions"]
        st.subheader("Business assumptions", anchor=False)
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
        break_even = min(break_even_probability(order_profit, offer_cost, conversion_boost), 1.0)
        st.caption(
            f"Best for profit: {best_threshold:.2f}  \n"
            f"Best for F1: {model_info['thresholds'][model_name]['best_f1']:.2f}  \n"
            f"Break-even probability: {break_even:.2f}"
        )

    prediction_tab, comparison_tab = st.tabs(["Predict a session", "Model comparison"])
    with prediction_tab:
        show_prediction_page(model_info, models, model_name, threshold)
    with comparison_tab:
        show_comparison_page(model_info)


main()
