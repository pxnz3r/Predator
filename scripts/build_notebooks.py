"""
build_notebooks.py - Build both notebooks from one story.

    python scripts/build_notebooks.py

Creates:
    notebooks/predator_walkthrough.ipynb  (imports the predator package)
    kaggle/predator-kaggle.ipynb          (self-contained: the package code is pasted in)

Both notebooks share the same markdown and the same "run" cells. The only
difference is how the functions get defined: the walkthrough imports them,
the Kaggle notebook includes each module's source code inline (Kaggle runs a
notebook as a single file and cannot import our package). Building both from
the package source keeps them in sync with the code.
"""

import ast
import os

import nbformat

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GITHUB_LINK = "https://github.com/pxnz3r/Predator"
KAGGLE_LINK = "https://www.kaggle.com/code/YOUR_KAGGLE_USERNAME/predator-purchase-prediction"


def read_module_without_package_imports(module_name):
    """
    Read predator/<module_name>.py and remove the "from predator... import"
    statements, so the code can run on its own inside one notebook.

    Input:
        module_name: e.g. "data".
    Returns:
        The module source code as a string.
    """
    path = os.path.join(PROJECT_FOLDER, "predator", module_name + ".py")
    with open(path) as file:
        source = file.read()

    lines = source.splitlines()
    lines_to_remove = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("predator"):
            for line_number in range(node.lineno, node.end_lineno + 1):
                lines_to_remove.add(line_number)

    kept_lines = [line for number, line in enumerate(lines, start=1) if number not in lines_to_remove]
    return "\n".join(kept_lines).strip() + "\n"


# ---------------------------------------------------------------------------
# The story. Each section has markdown, optionally a module whose code is
# needed, the imports for the walkthrough version, and the code to run.
# ---------------------------------------------------------------------------

SETUP_CODE = '''import os
import warnings

import pandas as pd
from IPython.display import Image, Markdown, display

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


def show(path, width=900):
    """Display a saved figure inside the notebook."""
    display(Image(filename=path, width=width))'''

SECTIONS = [
    {
        "markdown": """# Predator: predicting online purchases and explaining why

**Question:** will an online shopping session end in a purchase, and which behaviours drive that prediction?

We use the **Online Shoppers Purchasing Intention** dataset (UCI / Kaggle): 12,330 browsing sessions from an online store, each described by 17 features (pages visited, time spent, Google Analytics metrics such as bounce rate, exit rate and page value, plus month, visitor type and technical details). The target `Revenue` says whether the session ended in a purchase; only about 15% did.

**Models**
- Main models: **Decision Tree** and **K-Nearest Neighbours** (KNN, trained with and without undersampling)
- Baseline: **Gaussian Naive Bayes**
- Benchmark beyond the main models: **XGBoost**

**What this notebook covers:** data checks and EDA, a leak-free preprocessing pipeline, tuning with 5-fold stratified cross-validation, test-set evaluation with confidence intervals and significance tests, learning and validation curves, feature importance and partial dependence, an ablation without `PageValues` (and whether it leaks the answer), a profit-based decision threshold with a sensitivity analysis, probability calibration, and an error analysis of the sessions the best model gets wrong.

{repo_note}""",
        "imports": "",
        "code": SETUP_CODE,
    },
    {
        "markdown": """## 1. Loading and checking the data

`find_dataset()` looks in a local `data/` folder first and then searches `/kaggle/input/`, so the same code runs on a laptop and on Kaggle. We then check for missing values, duplicates and negative numbers.""",
        "module": "data",
        "imports": "from predator.data import find_dataset, load_dataset, validate_dataset, split_data, run_eda",
        "code": """dataset = load_dataset()
quality_report = validate_dataset(dataset)
dataset.head()""",
    },
    {
        "markdown": """## 2. Exploratory data analysis

Four questions: how imbalanced is the target, how do the numerical features differ between buyers and non-buyers, which categories buy more often, and which features move together?""",
        "imports": "",
        "code": """figure_paths = run_eda(dataset)
for path in figure_paths:
    show(path)""",
    },
    {
        "markdown": """**What the EDA shows**

- Only about 15% of sessions buy. A model that always says "no purchase" is 85% accurate and useless, so we judge models with **precision, recall, F1 and ROC-AUC** instead of accuracy.
- **PageValues** separates the classes most clearly: most non-buyers have a page value of 0, most buyers do not. It is also the numerical feature most correlated with `Revenue`.
- **BounceRates and ExitRates** are strongly correlated with each other, and both are higher for non-buyers.
- **November** has the highest purchase rate, and new visitors buy more often than returning ones.
- The duration features are very skewed (many zeros, a long tail), which matters for distance-based KNN.""",
        "imports": "",
        "code": None,
    },
    {
        "markdown": """## 3. Train/test split and preprocessing

We split 80/20, **stratified** so both parts keep the 15% buyer rate, with `random_state=42` everywhere for reproducibility. The test set is put aside until the final evaluation.

Preprocessing lives inside a scikit-learn `Pipeline` with a `ColumnTransformer`:
- categorical columns (Month, VisitorType, codes such as Browser) are **one-hot encoded**;
- numerical columns are **standardised** for KNN and Naive Bayes (KNN compares distances, so a duration in thousands of seconds would otherwise drown out a rate between 0 and 0.2). Tree models do not need scaling, so we keep their numbers in real units, which keeps the tree's split rules readable.

Because the scaler and encoder are fitted inside the pipeline, they only ever see training data, even inside each cross-validation fold. That prevents **data leakage**.""",
        "module": "preprocess",
        "imports": "from predator.preprocess import build_pipeline, get_feature_names, undersample_training_data",
        "code": """train_features, test_features, train_target, test_target = split_data(dataset)""",
    },
    {
        "markdown": """## 4. The models and their hyperparameter grids

- **Decision Tree**: asks yes/no questions, choosing at each step the split with the highest information gain (criterion = "entropy", as in the lab lecture). `class_weight="balanced"` makes errors on the rare buyer class count more.
- **KNN**: classifies a session by a vote of its k most similar training sessions (Euclidean or Manhattan distance). It has no class weights, so we also train a version on **randomly undersampled** training data (all buyers plus an equal number of non-buyers).
- **Gaussian Naive Bayes**: applies Bayes' theorem assuming features are independent and bell-shaped within each class. Neither assumption holds here, which makes it a useful untuned baseline.
- **XGBoost (benchmark)**: gradient boosting; each new small tree is trained to fix the errors of the trees before it. `scale_pos_weight` (non-buyers / buyers) handles the imbalance.""",
        "module": "models",
        "imports": "from predator.models import MODEL_NAMES, DECISION_TREE, XGBOOST, get_parameter_grids",
        "code": """for model_name, grid in get_parameter_grids().items():
    print(model_name, "->", grid if grid else "no tuning (baseline)")""",
    },
    {
        "markdown": """## 5. Tuning with 5-fold stratified cross-validation

`GridSearchCV` tries every combination in each grid, training on 4 folds and scoring F1 on the 5th, and keeps the best average. Afterwards we compute **out-of-fold** probabilities for every training session (each one predicted by a model that did not see it). These give a fair cross-validated F1 for every model on the real class balance, including the undersampled KNN, and they are what we later use to choose decision thresholds without touching the test set.""",
        "module": "train",
        "imports": "from predator.train import train_all_models, save_models",
        "code": """training_results = train_all_models(train_features, train_target)
models = {name: result["pipeline"] for name, result in training_results.items()}""",
    },
    {
        "markdown": """## 6. Evaluation on the test set

Every model is scored once on the untouched test set at the default 0.5 threshold. The best model is chosen by **cross-validated F1** (choosing by test score would quietly tune on the test set).""",
        "module": "evaluate",
        "imports": "from predator.evaluate import build_comparison_table, choose_best_model, plot_roc_curves, plot_confusion_matrices, plot_metric_comparison",
        "code": """comparison_table = build_comparison_table(models, training_results, test_features, test_target)
best_model_name = choose_best_model(comparison_table)
show(plot_roc_curves(models, test_features, test_target), width=600)
show(plot_confusion_matrices(models, test_features, test_target))
show(plot_metric_comparison(comparison_table), width=750)
comparison_table""",
    },
    {
        "markdown": """**Reading the results**

- **XGBoost (benchmark)** and the **Decision Tree** are clearly ahead, with test ROC-AUC around 0.92 to 0.93. The tree gets most of the benchmark's performance.
- **KNN** struggles: with dozens of one-hot columns and skewed durations, "nearest" sessions are not very informative. Undersampling raises its recall a lot at the cost of precision, and improves its F1 and ROC-AUC.
- **Naive Bayes** flags almost every session as a buyer: its independence assumption breaks on correlated, skewed features, and its probabilities end up pushed to 0 or 1.""",
        "imports": "",
        "code": None,
    },
    {
        "markdown": """## 7. How sure are we? Confidence intervals, McNemar's test and slices

The test set has only 382 buyers, so every score carries some luck. Three checks:

- **Bootstrap 95% confidence intervals:** resample the test set with replacement 1,000 times, re-score each model, keep the middle 95%.
- **McNemar's test:** for two models on the same sessions, count the sessions only one of them gets right. If the models were equally good, the two counts would be about equal.
- **Slices:** scores per visitor type and month. One overall F1 can hide a group the model serves badly.""",
        "module": "significance",
        "imports": "from predator.significance import run_significance_analysis\nfrom predator.evaluate import evaluate_slices\nfrom predator.models import KNN, KNN_UNDERSAMPLED",
        "code": """intervals_table, mcnemar_table = run_significance_analysis(
    models, comparison_table, test_features, test_target,
    pairs_to_compare=[(DECISION_TREE, XGBOOST), (DECISION_TREE, KNN_UNDERSAMPLED)])
slice_table = evaluate_slices(models, [DECISION_TREE, XGBOOST], test_features, test_target)
show("outputs/figures/slice_scores.png")
display(intervals_table, mcnemar_table)""",
    },
    {
        "markdown": """**Reading the checks**

- The Decision Tree's F1 interval (0.59 to 0.66) overlaps XGBoost's (0.61 to 0.68), but McNemar's test still finds XGBoost **significantly better** (87 sessions only XGBoost gets right against 52, p = 0.004). Overlapping intervals are not the same as "no difference": both models are scored on the same sessions, and McNemar uses that pairing.
- The Decision Tree beats the undersampled KNN by a wide margin (p < 0.001).
- **Returning visitors are the hard group:** the tree's F1 is 0.59 for them against 0.77 for new visitors, because many returning visitors browse valuable pages without buying (precision 0.47). Months with only a handful of buyers (February has 1) give noisy scores.

## 8. Overfitting and underfitting: learning and validation curves

A **learning curve** trains the tuned model on growing parts of the training data. A **validation curve** changes one setting (tree depth, number of neighbours) and keeps the rest. Both use the same 5 folds and F1 as the grid search.""",
        "module": "curves",
        "imports": "from predator.curves import plot_learning_curves, plot_validation_curves",
        "code": """learning_table = plot_learning_curves(models, train_features, train_target)
validation_table = plot_validation_curves(models, train_features, train_target)
show("outputs/figures/learning_curves.png")
show("outputs/figures/validation_curves.png")""",
    },
    {
        "markdown": """- The **Decision Tree** has a small gap between training F1 (0.68) and cross-validated F1 (0.64), so its depth limit and minimum leaf size keep overfitting in check, and its cross-validated F1 levels off after about 5,000 sessions.
- **Surprise in the validation curve:** a depth-1 tree, the single rule `PageValues > 0.05`, has the highest cross-validated F1 (0.667, against 0.643 at the tuned depth 7). The grid started at depth 3, so the search never saw it. The deeper tree is still the better model for this project: a one-split tree gives only two probabilities, so it cannot rank sessions (ROC-AUC) or support a profit threshold. Above depth 7, F1 falls as the tree memorises noise. The result is another sign of how much the task rests on PageValues.
- **KNN** scores a perfect training F1 at every size, because distance weighting gives each training session full weight on itself: a textbook case of high variance. Its cross-validated F1 is still rising at the full training size (0.38 to 0.49), so more data would help it. The validation curve peaks at k = 3 and declines as larger neighbourhoods outvote the minority class.""",
        "imports": "",
        "code": None,
    },
    {
        "markdown": """## 9. Which features drive the predictions?

Three views: the Decision Tree's own impurity-based importance, a picture of its top levels, and **permutation importance** for every model. Permutation importance shuffles one original column of the test set and measures how much ROC-AUC drops, so it works for any model and compares them on the same 17 features.""",
        "module": "explain",
        "imports": "from predator.explain import plot_tree_feature_importance, plot_top_of_tree, permutation_importance_all_models, write_tree_vs_xgboost_comparison, run_ablation",
        "code": """tree_importance = plot_tree_feature_importance(models[DECISION_TREE])
show("outputs/figures/tree_feature_importance.png", width=650)
show(plot_top_of_tree(models[DECISION_TREE], max_depth=3))
importance_table = permutation_importance_all_models(models, test_features, test_target)
show("outputs/figures/permutation_importance.png", width=750)
importance_table.round(3)""",
    },
    {
        "markdown": """The tree's very first question is about **PageValues**, and permutation importance agrees across models: shuffling PageValues costs far more ROC-AUC than any other feature, with **Month** (November) second. Behaviour features like ExitRates and product-page time matter much less.

### Decision Tree vs XGBoost: performance versus interpretability""",
        "imports": "",
        "code": """display(Markdown(write_tree_vs_xgboost_comparison(comparison_table, models, importance_table)))""",
    },
    {
        "markdown": """### Partial dependence: in which direction does a feature push?

Permutation importance says how much a feature matters. **Partial dependence** shows how: set the feature to each value for 2,000 training sessions, predict, and average. The Decision Tree jumps from about 0.15 to 0.8 as soon as PageValues is above zero; XGBoost climbs more smoothly and levels off around 0.85 by PageValues 25. November raises the probability for both models, and a higher exit rate lowers it.""",
        "imports": "from predator.explain import plot_partial_dependence",
        "code": """show(plot_partial_dependence(models, [DECISION_TREE, XGBOOST], train_features))""",
    },
    {
        "markdown": """## 10. Ablation: what if PageValues did not exist?

PageValues is the average value of the pages a session visited, where a page's value comes from the purchases that followed views of it. If those values were computed over the same period, a buying session's own purchase raised its PageValues: a possible **target leak**. The dataset does not say which period was used, so we measure the dependence instead: drop PageValues, retrain every model with the same grids and folds, and compare scores and feature rankings. The scores without it are the honest lower bound for a deployment that can only use page values from earlier periods.""",
        "imports": "",
        "code": """ablation = run_ablation(train_features, train_target, test_features, test_target,
                        comparison_table, importance_table)
show("outputs/figures/ablation_scores.png")
show("outputs/figures/ablation_permutation_importance.png", width=750)
ablation["metrics"]""",
    },
    {
        "markdown": """Without PageValues the Decision Tree, KNN and XGBoost lose roughly a third or more of their F1 (Naive Bayes barely changes, because it was already flagging almost every session), and the rankings reshuffle: the models fall back on **Month**, **ExitRates/BounceRates** and **product-page activity**. Those behaviour signals carry some information, but none replaces PageValues.

## 11. Choosing the decision threshold with a business goal

The default 0.5 cutoff is arbitrary. Suppose the shop sends a discount offer to every session the model flags. Using the illustrative assumptions at the top of the next cell (order profit, offer cost, and the share of a buyer's profit the offer secures):

- **true positive**: secured profit minus the offer cost
- **false positive**: the offer cost is wasted
- **false negative**: a missed sale (no extra profit)

We sweep thresholds from 0.01 to 0.99 on the out-of-fold training predictions, pick the best-F1 and best-profit thresholds, and compare them with 0.5 on the test set.""",
        "module": "threshold",
        "imports": "from predator.threshold import tune_thresholds, plot_profit_curves",
        "code": """threshold_table, chosen_thresholds, training_curves = tune_thresholds(
    models, training_results, train_target, test_features, test_target)
show(plot_profit_curves(training_curves, chosen_thresholds))
threshold_table""",
    },
    {
        "markdown": """The best threshold depends on the model's probability calibration as much as on the business numbers. The class-weighted Decision Tree, for example, pushes probabilities upwards, so its best-profit threshold sits well above 0.5. Naive Bayes loses money at every threshold, another sign its probabilities cannot be trusted.

### How sensitive is the best threshold to the assumptions?

The offer cost and the conversion boost are guesses, so we repeat the search for 36 combinations of them (still on out-of-fold training data). A threshold of 1.00 means no session is worth an offer.""",
        "imports": "from predator.threshold import run_sensitivity_analysis",
        "code": """sensitivity_table = run_sensitivity_analysis(training_results, train_target, [DECISION_TREE, XGBOOST])
show("outputs/figures/sensitivity_thresholds.png")""",
    },
    {
        "markdown": """Cheaper offers and stronger boosts push the best threshold down (target more sessions); with a 10% boost and an offer cost of $6 or more, no threshold makes a profit. The ordering is stable, so the conclusion "use a cutoff above 0.5 for the tree" holds over a wide range of assumptions, while the exact number does not.

## 12. Probability calibration: can we trust the numbers?

ROC-AUC only checks the **order** of the probabilities. **Calibration** checks the numbers: of the sessions given about 0.3, about 30% should buy. If they are calibrated, the best threshold follows from the assumptions alone: send the offer when p is above offer cost / (order profit x conversion boost), which is **0.25** here.

The **reliability diagram** plots predicted probability against the real purchase rate (the diagonal is perfect), and the **Brier score** is the mean squared error of the probabilities. We then calibrate the Decision Tree and XGBoost with isotonic regression on the training folds.""",
        "module": "calibration",
        "imports": "from predator.calibration import run_calibration_analysis",
        "code": """calibration_table = run_calibration_analysis(
    models, [DECISION_TREE, XGBOOST], train_features, train_target, test_features, test_target)
show("outputs/figures/calibration_all_models.png", width=600)
show("outputs/figures/calibration_before_after.png", width=600)
calibration_table""",
    },
    {
        "markdown": """The Decision Tree's raw probabilities average 0.28 although only 15.5% of sessions buy: the balanced class weights inflate them. After calibration they average 0.15, the Brier score drops from 0.109 to 0.072, and ROC-AUC is unchanged. With calibrated probabilities the simple break-even rule (p > 0.25) earns **$4,086** on the test set, more than the tuned 0.70 threshold on the raw probabilities ($3,870). XGBoost improves the same way (Brier 0.094 to 0.071, $4,020). Calibration turns the threshold from a tuned number into one you can explain.

## 13. Error analysis: who does the best model get wrong?

We label every test session as a caught buyer, missed buyer, false alarm or correct non-buyer (using the best model at its best-profit threshold) and compare the groups.""",
        "module": "errors",
        "imports": "from predator.errors import run_error_analysis",
        "code": """best_threshold = chosen_thresholds[best_model_name]["best_profit"]
labelled_sessions = run_error_analysis(models[best_model_name], best_model_name,
                                       test_features, test_target, best_threshold)
show("outputs/figures/error_analysis.png")
show("outputs/figures/error_probability_histogram.png", width=650)
display(Markdown(open("outputs/error_analysis.md").read()))""",
    },
    {
        "markdown": """## 14. Conclusions

- **PageValues is the deciding signal.** Every model ranks it first, the Decision Tree splits on it at the root, and removing it costs every model a large share of its F1.
- **The Decision Tree is the best main model.** It comes close to the XGBoost benchmark while staying fully readable, which matters when a shop team has to trust the rules.
- **XGBoost is the strongest benchmark**, a little ahead on F1 and ROC-AUC, at the price of interpretability.
- **KNN and Naive Bayes do not fit this data well.** KNN suffers from many one-hot dimensions and skewed features (undersampling helps); Naive Bayes' independence assumption breaks and its probabilities are unusable.
- **The gap to XGBoost is small but real.** The confidence intervals overlap, yet McNemar's test (p = 0.004) shows XGBoost makes significantly fewer mistakes on the same sessions.
- **PageValues may leak part of the answer.** The scores without it are the honest lower bound for a live system that only knows page values from earlier periods.
- **The threshold is a business decision.** With a profit model, the best cutoff differs from 0.5 and from the best-F1 cutoff, and it moves with the assumptions; calibrated probabilities make it explainable (p > cost / (profit x boost)) and earned the most profit.
- **Missed buyers are mostly sessions with PageValues = 0**, which look like ordinary browsing; false alarms are mostly returning visitors who reached valuable pages and left. Fixing them needs more data about the visitor, not more tuning.

{repo_note}""",
        "imports": "",
        "code": None,
    },
]

WALKTHROUGH_NOTE = """> This walkthrough imports the code from the `predator/` package. Run `python main.py` for the same pipeline in one command, and `streamlit run app.py` for the demo app."""

KAGGLE_NOTE = f"""> **Full project on GitHub:** [{GITHUB_LINK}]({GITHUB_LINK}) (the `predator` package, tests, documentation and a **Streamlit demo app**). This notebook is a self-contained version: all the code is included below."""

WALKTHROUGH_SETUP = """# Run from the project folder so data/ and outputs/ are found.
if os.path.basename(os.getcwd()) == "notebooks":
    os.chdir("..")
print("Working folder:", os.getcwd())"""


def build_notebook(for_kaggle):
    """
    Build one notebook from SECTIONS.

    Input:
        for_kaggle: True for the self-contained Kaggle version.
    Returns:
        An nbformat notebook object.
    """
    notebook = nbformat.v4.new_notebook()
    cells = []
    note = KAGGLE_NOTE if for_kaggle else WALKTHROUGH_NOTE

    for section in SECTIONS:
        cells.append(nbformat.v4.new_markdown_cell(section["markdown"].replace("{repo_note}", note)))

        if for_kaggle and section.get("module"):
            module_name = section["module"]
            cells.append(
                nbformat.v4.new_markdown_cell(
                    f"<details><summary>The code below is <code>predator/{module_name}.py</code> from the GitHub repo.</summary>"
                    "Each function has a docstring and comments explaining the method.</details>"
                )
            )
            cells.append(nbformat.v4.new_code_cell(read_module_without_package_imports(module_name)))

        code_parts = []
        if not for_kaggle and section["imports"]:
            code_parts.append(section["imports"])
        if section["code"]:
            code_parts.append(section["code"])
        if not for_kaggle and section["code"] == SETUP_CODE:
            code_parts.append(WALKTHROUGH_SETUP)
        if code_parts:
            cells.append(nbformat.v4.new_code_cell("\n\n".join(code_parts)))

    notebook.cells = cells
    notebook.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    notebook.metadata["language_info"] = {"name": "python"}
    return notebook


def main():
    """
    Write both notebooks to disk.

    Returns:
        Nothing.
    """
    walkthrough_path = os.path.join(PROJECT_FOLDER, "notebooks", "predator_walkthrough.ipynb")
    kaggle_path = os.path.join(PROJECT_FOLDER, "kaggle", "predator-kaggle.ipynb")
    os.makedirs(os.path.dirname(walkthrough_path), exist_ok=True)
    os.makedirs(os.path.dirname(kaggle_path), exist_ok=True)

    nbformat.write(build_notebook(for_kaggle=False), walkthrough_path)
    nbformat.write(build_notebook(for_kaggle=True), kaggle_path)
    print("Wrote", walkthrough_path)
    print("Wrote", kaggle_path)


if __name__ == "__main__":
    main()
