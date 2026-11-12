"""
models.py - The four models and the settings we try for each one.

Main models (from the AI Lab course):
    - Decision Tree
    - K-Nearest Neighbours (KNN), trained two ways: on all training data,
      and on randomly undersampled (balanced) training data
Baseline:
    - Gaussian Naive Bayes (no tuning, it shows the score to beat)
Benchmark beyond the course:
    - XGBoost, to see how far a stronger off-the-shelf model gets

Each model comes with a "parameter grid": the hyperparameter values that
GridSearchCV will try. Hyperparameters are settings we choose before
training (like the tree depth), unlike parameters the model learns itself.
The names start with "model__" because the model sits inside a Pipeline
step called "model".
"""

from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from predator.data import RANDOM_STATE

# Model names used everywhere (tables, plots, saved files, the app).
DECISION_TREE = "Decision Tree"
KNN = "KNN"
KNN_UNDERSAMPLED = "KNN (undersampled)"
NAIVE_BAYES = "Naive Bayes (baseline)"
XGBOOST = "XGBoost (benchmark)"

MODEL_NAMES = [DECISION_TREE, KNN, KNN_UNDERSAMPLED, NAIVE_BAYES, XGBOOST]

# Short file-friendly names for saving models with joblib.
MODEL_FILE_NAMES = {
    DECISION_TREE: "decision_tree.joblib",
    KNN: "knn.joblib",
    KNN_UNDERSAMPLED: "knn_undersampled.joblib",
    NAIVE_BAYES: "naive_bayes.joblib",
    XGBOOST: "xgboost_benchmark.joblib",
}

# One colour per model, so every plot uses the same colour for a model.
MODEL_COLORS = {
    DECISION_TREE: "#2A9D8F",
    KNN: "#E76F51",
    KNN_UNDERSAMPLED: "#F4A261",
    NAIVE_BAYES: "#8D99AE",
    XGBOOST: "#264653",
}


def make_decision_tree():
    """
    Create an unfitted Decision Tree classifier.

    How it works: the tree asks yes/no questions about the features
    ("Is PageValues <= 0.94?"). At every step it picks the question that
    best separates buyers from non-buyers. "Best" is measured with an
    impurity score:
        - Gini impurity: the chance that two sessions picked at random from
          the node have different labels. 0 means the node is pure.
        - Entropy: how "surprised" we are on average by a label in the
          node, from information theory. Also 0 when the node is pure.
    The split that lowers the impurity the most wins.

    class_weight="balanced" makes mistakes on the rare buyer class count
    more (about 5.5 times more here), so the tree does not just learn to
    say "no purchase" for everything.

    Returns:
        A DecisionTreeClassifier.
    """
    return DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE)


def make_knn():
    """
    Create an unfitted K-Nearest Neighbours classifier.

    How it works: KNN does not really "train". It remembers all training
    sessions. To classify a new session it finds the k most similar
    training sessions (its nearest neighbours) and lets them vote.
    Similarity is measured with a distance:
        - Euclidean: straight-line distance, sqrt(sum of squared differences)
        - Manhattan: sum of absolute differences (like walking city blocks)
    Because it is all about distances, features must be scaled first.
    weights="distance" lets closer neighbours have a bigger vote.

    KNN has no class_weight option, which is why we also try it on
    undersampled training data.

    Returns:
        A KNeighborsClassifier.
    """
    return KNeighborsClassifier()


def make_naive_bayes():
    """
    Create an unfitted Gaussian Naive Bayes classifier.

    How it works: it uses Bayes' theorem to compute
    P(purchase | features) from P(features | purchase) and P(purchase).
    The "naive" assumption is that all features are independent of each
    other once you know the class, and "Gaussian" means each numerical
    feature is assumed to follow a bell curve within each class.
    Neither is true here (ProductRelated and ProductRelated_Duration are
    strongly linked, and most features are very skewed), so we use it as a
    simple, fast baseline that the other models should beat.

    Returns:
        A GaussianNB classifier.
    """
    return GaussianNB()


def make_xgboost(scale_pos_weight):
    """
    Create an unfitted XGBoost classifier (benchmark beyond the course).

    How it works: gradient boosting builds many small decision trees one
    after another. Each new tree is trained to fix the mistakes (the
    remaining errors) of all the trees before it, and the final prediction
    adds up all the trees. learning_rate controls how big a step each new
    tree takes.

    scale_pos_weight tells XGBoost to weigh buyer sessions more, set to
    (number of non-buyers / number of buyers) in the training data. It is
    the XGBoost version of class_weight="balanced".

    Input:
        scale_pos_weight: weight for the positive (buyer) class.
    Returns:
        An XGBClassifier.
    """
    return XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def get_parameter_grids():
    """
    The hyperparameter values GridSearchCV will try for each model.

    GridSearchCV tries every combination, so the grids are kept small
    enough to finish in a few minutes on a laptop.

    Returns:
        A dictionary: model name -> parameter grid (dictionary of lists).
        Naive Bayes has an empty grid because it is an untuned baseline.
    """
    decision_tree_grid = {
        "model__max_depth": [3, 5, 7, 10, 15, None],
        "model__min_samples_split": [2, 10, 20, 50],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__criterion": ["gini", "entropy"],
    }

    # Odd values of k (1, 3, 5, ..., 31) avoid ties in a two-class vote.
    knn_grid = {
        "model__n_neighbors": list(range(1, 32, 2)),
        "model__weights": ["uniform", "distance"],
        "model__metric": ["euclidean", "manhattan"],
    }

    xgboost_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [3, 4, 5, 6],
        "model__learning_rate": [0.03, 0.1, 0.3],
    }

    grids = {
        DECISION_TREE: decision_tree_grid,
        KNN: knn_grid,
        KNN_UNDERSAMPLED: knn_grid,
        NAIVE_BAYES: {},
        XGBOOST: xgboost_grid,
    }
    return grids


def uses_scaling(model_name):
    """
    Say whether a model needs its numerical features scaled.

    Input:
        model_name: one of MODEL_NAMES.
    Returns:
        True for distance/probability models (KNN, Naive Bayes),
        False for tree models (Decision Tree, XGBoost).
    """
    tree_models = [DECISION_TREE, XGBOOST]
    return model_name not in tree_models


def make_model(model_name, scale_pos_weight):
    """
    Create a fresh, unfitted model from its name.

    Inputs:
        model_name: one of MODEL_NAMES.
        scale_pos_weight: only used by XGBoost.
    Returns:
        The unfitted model object.
    """
    if model_name == DECISION_TREE:
        return make_decision_tree()
    if model_name == KNN or model_name == KNN_UNDERSAMPLED:
        return make_knn()
    if model_name == NAIVE_BAYES:
        return make_naive_bayes()
    if model_name == XGBOOST:
        return make_xgboost(scale_pos_weight)
    raise ValueError("Unknown model name: " + model_name)
