"""
explain.py - Which features drive the predictions?

Three ways of answering that question:
1. Decision Tree feature importance: how much each feature reduced the
   impurity (Gini/entropy) across all the splits that used it.
   Only works for tree models.
2. Permutation importance: shuffle one column of the test set and measure
   how much the model's score drops. If the score drops a lot, the model
   relied on that column. Works for ANY model, so we use it to compare
   all of them on the same 17 original features.
3. Ablation: remove a feature completely (PageValues), retrain every
   model and see how much worse they get.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

from predator.data import CATEGORICAL_FEATURES, NUMERICAL_FEATURES, RANDOM_STATE, save_figure
from predator.evaluate import OUTPUTS_FOLDER, build_comparison_table, plot_roc_curves
from predator.models import DECISION_TREE, MODEL_COLORS, XGBOOST
from predator.preprocess import get_feature_names
from predator.train import train_all_models

ABLATION_FEATURE = "PageValues"


def plot_tree_feature_importance(tree_pipeline, top_n=15):
    """
    Bar chart of the Decision Tree's built-in feature importance.

    Each value is the total impurity decrease the feature produced, as a
    share of all decreases (the values add up to 1). One-hot columns are
    shown separately, e.g. "Month_Nov".

    Inputs:
        tree_pipeline: the fitted Decision Tree pipeline.
        top_n: how many features to show.
    Returns:
        A DataFrame of all features and their importance, sorted, also saved
        to outputs/tree_feature_importance.csv.
    """
    feature_names = get_feature_names(tree_pipeline)
    importances = tree_pipeline.named_steps["model"].feature_importances_

    importance_table = pd.DataFrame({"feature": feature_names, "importance": importances})
    importance_table = importance_table.sort_values("importance", ascending=False)
    importance_table.to_csv(os.path.join(OUTPUTS_FOLDER, "tree_feature_importance.csv"), index=False)

    top_features = importance_table.head(top_n).iloc[::-1]  # reversed so the biggest is on top
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.barh(top_features["feature"], top_features["importance"], color=MODEL_COLORS[DECISION_TREE])
    axis.set_xlabel("Share of total impurity decrease")
    axis.set_title("Decision Tree feature importance (top {})".format(top_n))
    save_figure(figure, "tree_feature_importance.png")
    return importance_table


def plot_top_of_tree(tree_pipeline, max_depth=3):
    """
    Draw the first few levels of the trained Decision Tree.

    The real tree is deeper, but the top levels hold the most important
    questions and are readable. Each box shows the split rule, the
    impurity, how many training sessions reached it, and the majority class.
    Because of class_weight="balanced", the "value" numbers are weighted.

    Inputs:
        tree_pipeline: the fitted Decision Tree pipeline.
        max_depth: how many levels to draw.
    Returns:
        The path of the saved plot.
    """
    tree_model = tree_pipeline.named_steps["model"]
    feature_names = get_feature_names(tree_pipeline)

    figure, axis = plt.subplots(figsize=(30, 11))
    plot_tree(
        tree_model,
        max_depth=max_depth,
        feature_names=feature_names,
        class_names=["No purchase", "Purchase"],
        filled=True,
        rounded=True,
        impurity=True,
        proportion=False,
        precision=2,
        fontsize=10,
        ax=axis,
    )
    axis.set_title("Top {} levels of the trained Decision Tree (full depth: {}, leaves: {})".format(
        max_depth, tree_model.get_depth(), tree_model.get_n_leaves()), fontsize=14)
    return save_figure(figure, "decision_tree_top_levels.png")


def compute_permutation_importance(pipeline, test_features, test_target, repeats=5):
    """
    Permutation importance of each ORIGINAL feature for one model.

    We shuffle one column of the test set at a time (breaking its link to
    the target) and measure the drop in ROC-AUC. We shuffle the raw
    columns before preprocessing, so a categorical feature like Month is
    shuffled as one feature, not as ten one-hot columns. This lets us
    compare every model on the same 17 features.

    ROC-AUC is used as the score because it does not depend on the 0.5
    threshold, which suits some models better than others.

    Inputs:
        pipeline: a fitted pipeline.
        test_features, test_target: the test data.
        repeats: how many times each column is shuffled (results are averaged).
    Returns:
        A pandas Series: feature name -> mean drop in ROC-AUC.
    """
    result = permutation_importance(
        pipeline, test_features, test_target,
        scoring="roc_auc", n_repeats=repeats, random_state=RANDOM_STATE, n_jobs=-1,
    )
    return pd.Series(result.importances_mean, index=test_features.columns)


def permutation_importance_all_models(models, test_features, test_target, filename_prefix=""):
    """
    Compute permutation importance for every model and compare them.

    Inputs:
        models: dictionary model name -> fitted pipeline.
        test_features, test_target: the test data.
        filename_prefix: added to output file names (used by the ablation).
    Returns:
        A DataFrame with features as rows and models as columns, sorted
        by the average importance. Saved as a CSV and a heatmap.
    """
    print("\nComputing permutation importance (this takes a minute)...")
    importance_columns = {}
    for model_name, pipeline in models.items():
        print("  " + model_name)
        importance_columns[model_name] = compute_permutation_importance(pipeline, test_features, test_target)

    importance_table = pd.DataFrame(importance_columns)
    importance_table["average"] = importance_table.mean(axis=1)
    importance_table = importance_table.sort_values("average", ascending=False)
    importance_table = importance_table.drop(columns="average")

    importance_table.round(4).to_csv(
        os.path.join(OUTPUTS_FOLDER, filename_prefix + "permutation_importance.csv"),
        index_label="feature",
    )

    figure, axis = plt.subplots(figsize=(10, 8))
    sns.heatmap(importance_table, annot=True, fmt=".3f", cmap="Greens", cbar_kws={"label": "Drop in ROC-AUC"}, ax=axis)
    title = "Permutation importance: drop in test ROC-AUC when a feature is shuffled"
    if filename_prefix != "":
        title = title + "\n(models retrained without " + ABLATION_FEATURE + ")"
    axis.set_title(title)
    axis.set_ylabel("")
    axis.tick_params(axis="x", rotation=20)
    save_figure(figure, filename_prefix + "permutation_importance.png")
    return importance_table


def rank_features(importance_table):
    """
    Turn an importance table into ranks (1 = most important) per model.

    Input:
        importance_table: features x models DataFrame of importances.
    Returns:
        A DataFrame of the same shape with integer ranks.
    """
    return importance_table.rank(ascending=False, method="min").astype(int)


def run_ablation(train_features, train_target, test_features, test_target,
                 full_comparison_table, full_importance_table):
    """
    Ablation experiment: retrain every model WITHOUT PageValues.

    PageValues is the average value of the pages a visitor saw before
    reaching the page, based on past purchases on the site. It is the
    strongest single signal, so we want to know: how much do the models
    depend on it, and what do they use instead when it is gone?

    Steps:
        1. Drop PageValues from the training and test data.
        2. Tune and train all models again (same grids, same folds).
        3. Compare test scores and feature rankings with the full models.

    Inputs:
        train_features, train_target, test_features, test_target: the data.
        full_comparison_table: test scores of the full models.
        full_importance_table: permutation importance of the full models.
    Returns:
        A dictionary with "metrics" (score comparison), "importance"
        (importance without PageValues) and "top_features" (rank comparison).
    """
    print("\n" + "=" * 60)
    print("ABLATION: retraining every model without " + ABLATION_FEATURE)
    print("=" * 60)

    reduced_numerical = [column for column in NUMERICAL_FEATURES if column != ABLATION_FEATURE]
    reduced_train_features = train_features.drop(columns=ABLATION_FEATURE)
    reduced_test_features = test_features.drop(columns=ABLATION_FEATURE)

    reduced_results = train_all_models(reduced_train_features, train_target,
                                       reduced_numerical, CATEGORICAL_FEATURES)
    reduced_models = {name: result["pipeline"] for name, result in reduced_results.items()}

    reduced_table = build_comparison_table(reduced_models, reduced_results,
                                           reduced_test_features, test_target)
    # build_comparison_table() saves to model_comparison.csv, so put the
    # full table back and save the reduced one under its own name.
    full_comparison_table.to_csv(os.path.join(OUTPUTS_FOLDER, "model_comparison.csv"), index=False)
    reduced_table.to_csv(os.path.join(OUTPUTS_FOLDER, "ablation_model_comparison.csv"), index=False)

    plot_roc_curves(reduced_models, reduced_test_features, test_target,
                    filename="ablation_roc_curves.png",
                    title="ROC curves without " + ABLATION_FEATURE)

    # Side-by-side score comparison.
    metrics_table = pd.DataFrame({
        "Model": full_comparison_table["Model"],
        "F1 (all features)": full_comparison_table["F1"].values,
        "F1 (no PageValues)": reduced_table["F1"].values,
        "ROC-AUC (all features)": full_comparison_table["ROC-AUC"].values,
        "ROC-AUC (no PageValues)": reduced_table["ROC-AUC"].values,
    })
    metrics_table["F1 change"] = metrics_table["F1 (no PageValues)"] - metrics_table["F1 (all features)"]
    metrics_table["ROC-AUC change"] = (metrics_table["ROC-AUC (no PageValues)"]
                                       - metrics_table["ROC-AUC (all features)"])
    metrics_table = metrics_table.round(3)
    metrics_table.to_csv(os.path.join(OUTPUTS_FOLDER, "ablation_metrics.csv"), index=False)
    print("\nAblation results (test set):")
    print(metrics_table.to_string(index=False))
    plot_ablation_scores(metrics_table)

    reduced_importance = permutation_importance_all_models(reduced_models, reduced_test_features,
                                                           test_target, filename_prefix="ablation_")

    # Top 5 features of each model, with and without PageValues.
    full_ranks = rank_features(full_importance_table)
    rows = []
    for model_name in full_importance_table.columns:
        top_full = list(full_ranks[model_name].sort_values().index[:5])
        top_reduced = list(reduced_importance[model_name].sort_values(ascending=False).index[:5])
        rows.append({
            "Model": model_name,
            "Top 5 with all features": ", ".join(top_full),
            "Top 5 without PageValues": ", ".join(top_reduced),
        })
    top_features_table = pd.DataFrame(rows)
    top_features_table.to_csv(os.path.join(OUTPUTS_FOLDER, "ablation_top_features.csv"), index=False)
    print("\nFeature rankings with and without PageValues:")
    print(top_features_table.to_string(index=False))

    return {"metrics": metrics_table, "importance": reduced_importance, "top_features": top_features_table}


def plot_ablation_scores(metrics_table):
    """
    Bar chart of F1 and ROC-AUC with and without PageValues for each model.

    Input:
        metrics_table: the "metrics" table built in run_ablation().
    Returns:
        The path of the saved plot.
    """
    figure, axes = plt.subplots(1, 2, figsize=(14, 5))
    positions = list(range(len(metrics_table)))
    bar_width = 0.38

    for axis, metric in zip(axes, ["F1", "ROC-AUC"]):
        with_values = metrics_table[metric + " (all features)"]
        without_values = metrics_table[metric + " (no PageValues)"]
        axis.bar([p - bar_width / 2 for p in positions], with_values, width=bar_width,
                 label="All features", color="#264653")
        axis.bar([p + bar_width / 2 for p in positions], without_values, width=bar_width,
                 label="Without PageValues", color="#E9C46A")
        axis.set_xticks(positions)
        axis.set_xticklabels(metrics_table["Model"], rotation=20, ha="right")
        axis.set_ylim(0, 1)
        axis.set_title(metric + " on the test set")
        axis.legend()

    figure.suptitle("Ablation: what happens when PageValues is removed")
    figure.tight_layout()
    return save_figure(figure, "ablation_scores.png")


def write_tree_vs_xgboost_comparison(comparison_table, models, importance_table):
    """
    Write a short comparison of the Decision Tree and XGBoost: performance
    versus interpretability. Numbers come from this run.

    Inputs:
        comparison_table: test-set scores of all models.
        models: dictionary model name -> fitted pipeline.
        importance_table: permutation importance of all models.
    Returns:
        The text (markdown), also saved to outputs/tree_vs_xgboost.md.
    """
    tree_row = comparison_table[comparison_table["Model"] == DECISION_TREE].iloc[0]
    boost_row = comparison_table[comparison_table["Model"] == XGBOOST].iloc[0]

    tree_model = models[DECISION_TREE].named_steps["model"]
    boost_model = models[XGBOOST].named_steps["model"]

    tree_top = list(importance_table[DECISION_TREE].sort_values(ascending=False).index[:3])
    boost_top = list(importance_table[XGBOOST].sort_values(ascending=False).index[:3])

    text = (
        "# Decision Tree vs XGBoost (benchmark): performance vs interpretability\n\n"
        "| | Decision Tree | XGBoost (benchmark) |\n"
        "|---|---|---|\n"
        "| Test F1 | {tree_f1:.3f} | {boost_f1:.3f} |\n"
        "| Test ROC-AUC | {tree_auc:.3f} | {boost_auc:.3f} |\n"
        "| Precision / Recall | {tree_p:.3f} / {tree_r:.3f} | {boost_p:.3f} / {boost_r:.3f} |\n"
        "| Size | 1 tree, depth {depth}, {leaves} leaves | {trees} trees, depth up to {boost_depth} |\n"
        "| Top 3 features (permutation) | {tree_top} | {boost_top} |\n\n"
        "**Performance.** XGBoost scores {auc_gap:+.3f} ROC-AUC and {f1_gap:+.3f} F1 compared with the "
        "Decision Tree. Boosting combines many small trees, each one correcting the errors of the "
        "previous ones, so it ranks sessions more smoothly than a single tree, whose probabilities "
        "jump in steps (one value per leaf).\n\n"
        "**Interpretability.** The Decision Tree can be read as a flowchart: every prediction follows "
        "one path of yes/no questions, which we can print for any session (the Streamlit app does "
        "this). XGBoost adds up {trees} trees, so no single path explains a prediction; we need "
        "outside tools such as permutation importance to see what it relies on.\n\n"
        "**Agreement.** Both models rely on the same main signal: the top features are "
        "{tree_top} for the tree and {boost_top} for XGBoost.\n\n"
        "**Verdict.** If the small gain in score matters more than being able to explain each "
        "decision, XGBoost is the stronger benchmark. For a shop team that needs to understand and "
        "trust the rules, the Decision Tree gives most of the performance with full transparency.\n"
    ).format(
        tree_f1=tree_row["F1"], boost_f1=boost_row["F1"],
        tree_auc=tree_row["ROC-AUC"], boost_auc=boost_row["ROC-AUC"],
        tree_p=tree_row["Precision"], tree_r=tree_row["Recall"],
        boost_p=boost_row["Precision"], boost_r=boost_row["Recall"],
        depth=tree_model.get_depth(), leaves=tree_model.get_n_leaves(),
        trees=boost_model.n_estimators, boost_depth=boost_model.max_depth,
        tree_top=", ".join(tree_top), boost_top=", ".join(boost_top),
        auc_gap=boost_row["ROC-AUC"] - tree_row["ROC-AUC"], f1_gap=boost_row["F1"] - tree_row["F1"],
    )

    with open(os.path.join(OUTPUTS_FOLDER, "tree_vs_xgboost.md"), "w") as file:
        file.write(text)
    print("\n" + text)
    return text


def describe_decision_path(tree_pipeline, session):
    """
    List the yes/no questions one session answered on its way down the
    Decision Tree, in plain words.

    Example output line: "PageValues is 204.01, which is > 0.05"
    For one-hot columns (threshold 0.5) it says "Month is Nov" or
    "Month is not Nov" instead of comparing 0/1 numbers.

    Inputs:
        tree_pipeline: the fitted Decision Tree pipeline.
        session: a one-row DataFrame of raw feature values.
    Returns:
        A list of strings, one per question, from the root to the leaf.
    """
    tree_model = tree_pipeline.named_steps["model"]
    feature_names = get_feature_names(tree_pipeline)
    transformed_session = tree_pipeline.named_steps["preprocess"].transform(session)

    # decision_path() marks every node the session visits; indices[...]
    # gives those node numbers in order from the root.
    node_indicator = tree_model.decision_path(transformed_session)
    visited_nodes = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    rules = []
    for node in visited_nodes:
        feature_index = tree_model.tree_.feature[node]
        if feature_index < 0:
            continue  # a leaf: no question is asked here
        feature_name = feature_names[feature_index]
        split_value = tree_model.tree_.threshold[node]
        session_value = transformed_session[0, feature_index]
        goes_left = session_value <= split_value

        # Is this a one-hot column like "Month_Nov"?
        original_feature = None
        for categorical_feature in CATEGORICAL_FEATURES:
            if feature_name.startswith(categorical_feature + "_"):
                original_feature = categorical_feature
        if original_feature is not None:
            category = feature_name[len(original_feature) + 1:]
            if goes_left:
                rules.append("{} is not {}".format(original_feature, category))
            else:
                rules.append("{} is {}".format(original_feature, category))
        else:
            if goes_left:
                rules.append("{} is {:.2f}, which is <= {:.2f}".format(feature_name, session_value, split_value))
            else:
                rules.append("{} is {:.2f}, which is > {:.2f}".format(feature_name, session_value, split_value))
    return rules
