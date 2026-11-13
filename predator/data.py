"""
data.py - Loading, checking and exploring the dataset.

This file has three jobs:
1. Find the CSV file, both on a laptop (data/ folder) and on Kaggle (/kaggle/input/).
2. Load it and run simple data quality checks (missing values, duplicates, types).
3. Draw the exploratory data analysis (EDA) plots and split the data into
   a training set and a test set.
"""

import os

import matplotlib

# "Agg" lets matplotlib save plots to files without opening a window.
# This keeps main.py working on servers and on Kaggle.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Settings used across the whole project
# ---------------------------------------------------------------------------

DATASET_FILENAME = "online_shoppers_intention.csv"
LOCAL_DATA_FOLDER = "data"
KAGGLE_INPUT_FOLDER = "/kaggle/input"
FIGURES_FOLDER = os.path.join("outputs", "figures")

# The same random_state everywhere makes every run give the same results.
RANDOM_STATE = 42
TEST_SIZE = 0.20

TARGET_COLUMN = "Revenue"

# Numbers we can measure on a scale (counts, seconds, rates).
NUMERICAL_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
]

# Labels or codes. OperatingSystems = 2 is not "twice" OperatingSystems = 1,
# so these columns are treated as categories even when they hold numbers.
CATEGORICAL_FEATURES = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

# Colours used for the two classes in every plot.
NO_PURCHASE_COLOR = "#9AA5B1"
PURCHASE_COLOR = "#2A9D8F"


def find_dataset(filename=DATASET_FILENAME):
    """
    Look for the dataset file in the places it is likely to be.

    First it checks the local data/ folder (when running on your own computer).
    If it is not there, it searches every folder under /kaggle/input/
    (when running as a Kaggle notebook, where datasets are mounted there).

    Input:
        filename: the name of the CSV file to look for.
    Returns:
        The full path to the file as a string, or None if it was not found.
    """
    # 1. Local copy, for example data/online_shoppers_intention.csv
    local_path = os.path.join(LOCAL_DATA_FOLDER, filename)
    if os.path.isfile(local_path):
        return local_path

    # 2. Kaggle copy. The exact sub-folder depends on which dataset was
    #    attached, so we walk through all folders under /kaggle/input/.
    if os.path.isdir(KAGGLE_INPUT_FOLDER):
        for folder, _subfolders, files in os.walk(KAGGLE_INPUT_FOLDER):
            if filename in files:
                return os.path.join(folder, filename)

    return None


def load_dataset(path=None):
    """
    Load the Online Shoppers dataset into a pandas DataFrame.

    If no path is given, find_dataset() is used to locate the file.
    The target column Revenue is turned from True/False into 1/0 so the
    models and metrics can work with it directly.

    Input:
        path: optional path to the CSV file.
    Returns:
        A pandas DataFrame with one row per shopping session.
    Raises:
        FileNotFoundError with a friendly explanation if the file is missing.
    """
    if path is None:
        path = find_dataset()

    if path is None or not os.path.isfile(path):
        message = (
            "\n\nCould not find the dataset file '" + DATASET_FILENAME + "'.\n\n"
            "How to fix it:\n"
            "  1. Download 'Online Shoppers Purchasing Intention Dataset' from Kaggle\n"
            "     (https://www.kaggle.com/datasets/henrysue/online-shoppers-intention)\n"
            "     or from the UCI Machine Learning Repository\n"
            "     (https://archive.ics.uci.edu/dataset/468).\n"
            "  2. Put the CSV file here: " + os.path.join(LOCAL_DATA_FOLDER, DATASET_FILENAME) + "\n"
            "  3. Run the command again.\n"
        )
        raise FileNotFoundError(message)

    print("Loading dataset from:", path)
    data = pd.read_csv(path)

    # pandas reads TRUE/FALSE as booleans. We keep Weekend as True/False
    # (it is a category) but turn the target into 0 = no purchase, 1 = purchase.
    data[TARGET_COLUMN] = data[TARGET_COLUMN].astype(int)
    data["Weekend"] = data["Weekend"].astype(bool)
    return data


def validate_dataset(data):
    """
    Run basic data quality checks and print what we find.

    Checks: expected columns are present, number of rows, missing values,
    duplicate rows, and the share of sessions that ended in a purchase.

    Input:
        data: the DataFrame returned by load_dataset().
    Returns:
        A dictionary with the results of each check.
    Raises:
        ValueError if a required column is missing, because nothing
        after this point can work without it.
    """
    print("\nChecking data quality...")

    expected_columns = ALL_FEATURES + [TARGET_COLUMN]
    missing_columns = [column for column in expected_columns if column not in data.columns]
    if len(missing_columns) > 0:
        raise ValueError("The dataset is missing these columns: " + ", ".join(missing_columns))

    number_of_rows = len(data)
    missing_values = int(data.isna().sum().sum())
    duplicate_rows = int(data.duplicated().sum())
    purchase_rate = float(data[TARGET_COLUMN].mean())

    print("  Rows (sessions):", number_of_rows)
    print("  Feature columns:", len(ALL_FEATURES))
    print("  Missing values:", missing_values)
    print("  Duplicate rows:", duplicate_rows)
    print(f"  Sessions that ended in a purchase: {purchase_rate:.1%}")

    # We keep duplicate rows. In this dataset they are mostly very short
    # sessions (one page, then leave), which really do repeat in real life.
    # Removing them would make the data less like real traffic.
    if duplicate_rows > 0:
        print("  Note: duplicate rows are kept, they are mostly identical one-page visits.")

    # Negative counts or durations would mean broken data.
    negative_numbers = int((data[NUMERICAL_FEATURES] < 0).sum().sum())
    print("  Negative values in numerical columns:", negative_numbers)

    report = {
        "rows": number_of_rows,
        "missing_values": missing_values,
        "duplicate_rows": duplicate_rows,
        "negative_values": negative_numbers,
        "purchase_rate": purchase_rate,
    }
    return report


def split_data(data):
    """
    Split the data into a training set (80%) and a test set (20%).

    The split is stratified: both sets keep the same share of buyers
    (about 15%). Without this, the small buyer class could end up
    over- or under-represented in the test set by chance.

    The test set is put aside and only used at the very end, so the
    final scores tell us how the models do on sessions they never saw.

    Input:
        data: the full DataFrame.
    Returns:
        train_features, test_features, train_target, test_target
    """
    features = data[ALL_FEATURES]
    target = data[TARGET_COLUMN]

    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    print("\nSplit the data (stratified 80/20):")
    print("  Training sessions:", len(train_features), f"(buyers: {train_target.mean():.1%})")
    print("  Test sessions:    ", len(test_features), f"(buyers: {test_target.mean():.1%})")
    return train_features, test_features, train_target, test_target


# ---------------------------------------------------------------------------
# Exploratory data analysis (EDA) plots
# ---------------------------------------------------------------------------


def save_figure(figure, filename):
    """
    Save a matplotlib figure into outputs/figures/ and close it.

    Closing the figure frees memory, which matters because the pipeline
    makes many plots in a row.

    Input:
        figure: the matplotlib figure to save.
        filename: the file name, for example "class_balance.png".
    Returns:
        The full path of the saved image.
    """
    os.makedirs(FIGURES_FOLDER, exist_ok=True)
    path = os.path.join(FIGURES_FOLDER, filename)
    figure.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_class_balance(data):
    """
    Bar chart of how many sessions did and did not end in a purchase.

    This shows the class imbalance: only about 15% of sessions are buyers.
    A model that always says "no purchase" would already be about 85%
    accurate, which is why accuracy alone is a misleading score here.

    Input:
        data: the full DataFrame.
    Returns:
        The path of the saved plot.
    """
    counts = data[TARGET_COLUMN].value_counts().sort_index()
    labels = ["No purchase", "Purchase"]

    figure, axis = plt.subplots(figsize=(6, 4))
    bars = axis.bar(labels, counts.values, color=[NO_PURCHASE_COLOR, PURCHASE_COLOR])
    for bar, count in zip(bars, counts.values, strict=True):
        share = count / counts.sum()
        axis.text(
            bar.get_x() + bar.get_width() / 2, count, f"{count:,}\n({share:.1%})", ha="center", va="bottom"
        )
    axis.set_title("Class balance: most sessions do not buy")
    axis.set_ylabel("Number of sessions")
    axis.set_ylim(0, counts.max() * 1.2)
    return save_figure(figure, "class_balance.png")


def plot_numerical_distributions(data):
    """
    Histograms of every numerical feature, split by purchase vs no purchase.

    Most of these features are heavily skewed (many zeros, a few very large
    values), so the x-axis is cut at the 99th percentile to keep the shape
    readable.

    Input:
        data: the full DataFrame.
    Returns:
        The path of the saved plot.
    """
    figure, axes = plt.subplots(2, 5, figsize=(20, 7))
    axes = axes.flatten()

    for index, column in enumerate(NUMERICAL_FEATURES):
        axis = axes[index]
        upper_limit = data[column].quantile(0.99)
        clipped = data[data[column] <= upper_limit]
        sns.histplot(
            data=clipped,
            x=column,
            hue=TARGET_COLUMN,
            bins=30,
            stat="density",
            common_norm=False,
            element="step",
            ax=axis,
            palette={0: NO_PURCHASE_COLOR, 1: PURCHASE_COLOR},
        )
        axis.set_title(column)
        axis.set_xlabel("")
        legend = axis.get_legend()
        if legend is not None:
            legend.set_title("")
            for text, label in zip(legend.texts, ["No purchase", "Purchase"], strict=True):
                text.set_text(label)

    figure.suptitle("Numerical features: buyers vs non-buyers (density, top 1% cut)", fontsize=14)
    figure.tight_layout()
    return save_figure(figure, "numerical_distributions.png")


def plot_categorical_purchase_rates(data):
    """
    For each categorical feature, show the purchase rate for every value.

    For example: what share of November sessions ended in a purchase,
    compared with February sessions. The dashed line is the overall rate.

    Input:
        data: the full DataFrame.
    Returns:
        The path of the saved plot.
    """
    overall_rate = data[TARGET_COLUMN].mean()
    month_order = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    figure, axes = plt.subplots(2, 4, figsize=(20, 8))
    axes = axes.flatten()

    for index, column in enumerate(CATEGORICAL_FEATURES):
        axis = axes[index]
        rates = data.groupby(column)[TARGET_COLUMN].mean()
        if column == "Month":
            rates = rates.reindex([month for month in month_order if month in rates.index])
        axis.bar([str(value) for value in rates.index], rates.values, color=PURCHASE_COLOR)
        axis.axhline(overall_rate, color="black", linestyle="--", linewidth=1)
        axis.set_title(column)
        axis.set_ylabel("Purchase rate")
        axis.tick_params(axis="x", rotation=45)

    # 7 categorical features, 8 boxes: hide the empty one.
    axes[-1].axis("off")
    figure.suptitle("Purchase rate by category (dashed line = overall rate)", fontsize=14)
    figure.tight_layout()
    return save_figure(figure, "categorical_purchase_rates.png")


def plot_correlation_heatmap(data):
    """
    Heatmap of the correlation between the numerical features and the target.

    Correlation goes from -1 to +1. Values near +1 or -1 mean two columns
    move together. It helps us spot features that carry the same
    information (for example BounceRates and ExitRates) and features that
    are related to buying (PageValues stands out).

    Input:
        data: the full DataFrame.
    Returns:
        The path of the saved plot.
    """
    columns = NUMERICAL_FEATURES + [TARGET_COLUMN]
    correlation = data[columns].corr()

    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=axis,
    )
    axis.set_title("Correlation between numerical features and Revenue")
    return save_figure(figure, "correlation_heatmap.png")


def run_eda(data):
    """
    Run all the EDA steps and save every plot to outputs/figures/.

    Input:
        data: the full DataFrame.
    Returns:
        A list with the paths of the saved plots.
    """
    print("\nDrawing EDA plots...")
    saved_paths = [
        plot_class_balance(data),
        plot_numerical_distributions(data),
        plot_categorical_purchase_rates(data),
        plot_correlation_heatmap(data),
    ]
    for path in saved_paths:
        print("  Saved", path)
    return saved_paths
