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
GITHUB_LINK = "https://github.com/YOUR_GITHUB_USERNAME/Predator"
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

**What this notebook covers:** data checks and EDA, a leak-free preprocessing pipeline, tuning with 5-fold stratified cross-validation, test-set evaluation, feature importance, an ablation without `PageValues`, a profit-based decision threshold, and an error analysis of the sessions the best model gets wrong.

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

- **Decision Tree**: asks yes/no questions, choosing at each step the split that lowers Gini impurity or entropy the most. `class_weight="balanced"` makes errors on the rare buyer class count more.
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
        "markdown": """## 7. Which features drive the predictions?

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
        "markdown": """## 8. Ablation: what if PageValues did not exist?

PageValues is computed from earlier purchases on the site, so it is close to a "historical purchase" signal. To see how much the models depend on it, we drop it, retrain every model with the same grids and folds, and compare scores and feature rankings.""",
        "imports": "",
        "code": """ablation = run_ablation(train_features, train_target, test_features, test_target,
                        comparison_table, importance_table)
show("outputs/figures/ablation_scores.png")
show("outputs/figures/ablation_permutation_importance.png", width=750)
ablation["metrics"]""",
    },
    {
        "markdown": """Without PageValues the Decision Tree, KNN and XGBoost lose roughly a third or more of their F1 (Naive Bayes barely changes, because it was already flagging almost every session), and the rankings reshuffle: the models fall back on **Month**, **ExitRates/BounceRates** and **product-page activity**. Those behaviour signals carry some information, but none replaces PageValues.

## 9. Choosing the decision threshold with a business goal

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
        "markdown": """The best threshold depends on the model's probability calibration as much as on the business numbers. The class-weighted Decision Tree, for example, pushes probabilities upwards, so its best-profit threshold sits well above 0.5. Naive Bayes loses money at every threshold, another sign its probabilities cannot be trusted. Change the assumptions and re-run to see the thresholds move.

## 10. Error analysis: who does the best model get wrong?

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
        "markdown": """## 11. Conclusions

- **PageValues is the deciding signal.** Every model ranks it first, the Decision Tree splits on it at the root, and removing it costs every model a large share of its F1.
- **The Decision Tree is the best main model.** It comes close to the XGBoost benchmark while staying fully readable, which matters when a shop team has to trust the rules.
- **XGBoost is the strongest benchmark**, a little ahead on F1 and ROC-AUC, at the price of interpretability.
- **KNN and Naive Bayes do not fit this data well.** KNN suffers from many one-hot dimensions and skewed features (undersampling helps); Naive Bayes' independence assumption breaks and its probabilities are unusable.
- **The threshold is a business decision.** With a profit model, the best cutoff differs from 0.5 and from the best-F1 cutoff, and it depends on the assumptions.
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
