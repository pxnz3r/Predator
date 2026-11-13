# Methodology

The full pipeline, step by step, and why each choice was made. `main.py` runs these steps in this order.

## 1. Data loading and quality checks

**What:** `find_dataset()` looks for `online_shoppers_intention.csv` in `data/`, then anywhere under `/kaggle/input/`. `load_dataset()` reads it and turns `Revenue` into 0/1. `validate_dataset()` checks the columns, missing values, duplicates and negative numbers.

**Findings:** 12,330 sessions, 17 features, no missing values, no negative values, 125 duplicate rows, 15.5% buyers.

**Why keep the duplicates:** they are almost all one-page visits that bounced (all counts 0, bounce and exit rate 0.2). Identical short visits really do repeat on a website, so removing them would make the data less like real traffic.

## 2. Exploratory data analysis

Four plots in `outputs/figures/`:

| Plot | What it shows |
|---|---|
| `class_balance.png` | 84.5% non-buyers, 15.5% buyers |
| `numerical_distributions.png` | Buyers have higher PageValues and lower exit/bounce rates; durations are very skewed |
| `categorical_purchase_rates.png` | November (25.4%) and October (20.9%) buy most, February (1.6%) least; new visitors buy more often (24.9%) than returning ones (13.9%) |
| `correlation_heatmap.png` | PageValues has the highest correlation with Revenue (0.49); BounceRates and ExitRates are almost the same signal (0.91) |

The most striking fact: 56% of sessions with PageValues > 0 end in a purchase, against 3.9% of sessions with PageValues = 0.

## 3. Train/test split

Stratified 80/20 split with `random_state=42`: 9,864 training sessions and 2,466 test sessions, both with 15.5% buyers. Stratifying matters because the buyer class is small; a random split could leave the test set with noticeably more or fewer buyers by chance.

The test set is used once, at the end. Model selection, hyperparameter tuning and threshold choice all happen on the training data.

## 4. Preprocessing

Built with a scikit-learn `ColumnTransformer` inside a `Pipeline` (`predator/preprocess.py`):

- **Categorical features** (Month, OperatingSystems, Browser, Region, TrafficType, VisitorType, Weekend) are one-hot encoded. The codes in OperatingSystems, Browser, Region and TrafficType are labels, not quantities, so they must not be treated as numbers. `handle_unknown="ignore"` turns an unseen value (for example a session in January, which the dataset never contains) into all zeros instead of an error.
- **Numerical features** are standardised (mean 0, standard deviation 1) for KNN and Naive Bayes. KNN compares distances, and without scaling ProductRelated_Duration (thousands of seconds) would outweigh BounceRates (0 to 0.2). The Decision Tree and XGBoost split on one feature at a time, so scaling cannot change them; we leave their numbers in real units so the tree's rules stay readable ("PageValues <= 0.05").

**Data leakage:** because preprocessing is a step of the pipeline, `fit()` only ever sees training data. Inside cross-validation, the scaler is refitted on the 4 training folds each time, so the validation fold never influences it.

## 5. Class imbalance

| Model | Technique |
|---|---|
| Decision Tree | `class_weight="balanced"`: each buyer counts about 5.5 times as much as a non-buyer when choosing splits |
| XGBoost | `scale_pos_weight` = non-buyers / buyers in the training set (about 5.5) |
| KNN | Compared with and without random undersampling of the training set (all 1,526 buyers plus 1,526 random non-buyers) |
| Naive Bayes | None (untuned baseline) |

Undersampling is done with plain pandas on training data only, never on the test set. When we cross-validate the undersampled KNN, only the training folds are undersampled.

## 6. Tuning

`GridSearchCV` with 5-fold stratified cross-validation (`shuffle=True, random_state=42`) and `scoring="f1"`.

| Model | Grid | Combinations | Best |
|---|---|---|---|
| Decision Tree | max_depth {3, 5, 7, 10, 15, None}, min_samples_split {2, 10, 20, 50}, min_samples_leaf {1, 5, 10, 20}, criterion {gini, entropy} | 192 | entropy, depth 7, split 50, leaf 20 |
| KNN | n_neighbors {1, 3, ..., 31}, weights {uniform, distance}, metric {euclidean, manhattan} | 64 | k = 3, distance, euclidean |
| KNN (undersampled) | same | 64 | k = 27, distance, manhattan |
| XGBoost | n_estimators {100, 200, 300}, max_depth {3, 4, 5, 6}, learning_rate {0.03, 0.1, 0.3} | 36 | 100 trees, depth 6, rate 0.1 |
| Naive Bayes | untuned | 1 | |

**Why F1:** accuracy is misleading at 15% positives (always predicting "no" scores 84.5%). F1 balances precision and recall on the buyer class.

**Fair comparison:** the undersampled KNN's grid-search score is measured on balanced folds, which inflates F1. So after tuning we compute **out-of-fold probabilities** for every model with a hand-written 5-fold loop on the real class balance (`train.out_of_fold_probabilities`). The F1 of those predictions ("CV F1") is what we compare models on.

## 7. Evaluation

Each model is scored once on the test set: precision, recall, F1, ROC-AUC, confusion matrix and classification report (`outputs/model_comparison.csv`, `outputs/classification_reports.txt`, `roc_curves.png`, `confusion_matrices.png`).

**Best model:** chosen by CV F1, not test F1, so the test set stays an honest estimate. XGBoost (benchmark) wins; the Decision Tree is the best main model.

## 8. Feature contribution analysis

- **Tree importance** (`tree_feature_importance.png`): impurity decrease per one-hot column.
- **Tree plot** (`decision_tree_top_levels.png`): the top 3 levels of the trained depth-7 tree.
- **Permutation importance** (`permutation_importance.png`): shuffle one raw column of the test set, measure the drop in ROC-AUC, 5 repeats. Shuffling raw columns (before one-hot encoding) keeps the 17 original features comparable across all models. ROC-AUC is used because it does not depend on the threshold, which would favour some models.
- **Ablation**: drop PageValues, retune and retrain every model with the same grids and folds, compare scores and the top 5 features.
- **Decision Tree vs XGBoost**: `outputs/tree_vs_xgboost.md`.

## 9. Business threshold

Thresholds from 0.01 to 0.99 are tried on the out-of-fold training probabilities; the best-F1 and best-profit thresholds are then applied to the test set and compared with 0.5. Details in [business_threshold.md](business_threshold.md).

## 10. Error analysis

The best model's test predictions at its best-profit threshold are split into caught buyers, missed buyers, false alarms and correct non-buyers, and the groups are compared. Details in [error_analysis.md](error_analysis.md).

## 11. Saving

Each fitted pipeline (preprocessing plus model) is saved with joblib to `models/`, so it accepts raw session data directly. `models/model_info.json` stores the best model's name, the chosen thresholds, the business assumptions, default input values and the permutation importance, for `predict.py` and `app.py`.

## Choices we rejected

- **SMOTE or other synthetic oversampling:** the brief rules out synthetic data, and it would need the `imbalanced-learn` dependency.
- **Dropping duplicates:** see step 1.
- **Log-transforming skewed features:** it would help KNN and Naive Bayes a little, but the brief asks for a standard scaling pipeline and the effect would blur the comparison.
- **Selecting the best model or threshold on the test set:** that leaks test information into the choice.
