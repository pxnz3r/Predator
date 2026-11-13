# Model card: Predator purchase prediction

Follows the structure of [Model Cards for Model Reporting](https://arxiv.org/abs/1810.03993) (Mitchell et al., 2019).

## Model details

- **Recommended model:** Decision Tree (entropy, max_depth 7, min_samples_split 50, min_samples_leaf 20, balanced class weights), inside a scikit-learn pipeline with one-hot encoding. File: `models/decision_tree.joblib`.
- **Benchmark:** XGBoost (100 trees, max_depth 6, learning rate 0.1, `scale_pos_weight`), the best model by cross-validated F1. File: `models/xgboost_benchmark.joblib`.
- **Also trained:** KNN with and without undersampling, Gaussian Naive Bayes (baseline).
- **Output:** the probability that a browsing session ends in a purchase, and a yes/no decision at a business threshold.
- **Version:** 1.0.0, trained with the versions in `uv.lock` (Python 3.15, scikit-learn 1.9.1, XGBoost 3.4.1), `random_state=42`.
- **License:** MIT (code), CC BY 4.0 (data).

## Intended use

- Teaching: a complete, explained supervised-learning study for an Artificial Intelligence Lab course.
- Demonstrating how a shop could decide which sessions get a discount offer, with illustrative profit assumptions.

**Out of scope:** real marketing decisions without retraining on the shop's own recent data; any decision about individual people (the data has no personal attributes, and the model should not be extended to them); stores, years or traffic sources different from the training data.

## Training data

[Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468) (Sakar et al., 2018): 12,330 sessions from one online store over one year, 17 features, 15.5% purchases. Stratified 80/20 split: 9,864 training sessions, 2,466 test sessions. No personal data; Region, Browser, OperatingSystems and TrafficType are anonymised integer codes.

## Metrics

F1 for the purchase class (the course's tuning metric), ROC-AUC, precision and recall; profit under the business assumptions; Brier score for calibration. Models and thresholds are chosen on 5-fold cross-validation of the training data; the test set is used once.

## Results (test set)

| Model | F1 (95% interval) | ROC-AUC (95% interval) | Brier (raw / calibrated) |
|---|---|---|---|
| Decision Tree | 0.627 (0.590 to 0.661) | 0.922 (0.908 to 0.935) | 0.109 / 0.072 |
| XGBoost (benchmark) | 0.645 (0.609 to 0.679) | 0.929 (0.917 to 0.941) | 0.094 / 0.071 |

McNemar's test: XGBoost makes significantly fewer mistakes than the Decision Tree (p = 0.004). Full numbers: [results.md](results.md).

## Performance across groups

Decision Tree at threshold 0.5 (`outputs/slice_analysis.csv`):

| Group | Buyers | Precision | Recall | F1 |
|---|---|---|---|---|
| New visitors | 89 | 0.758 | 0.775 | 0.767 |
| Returning visitors | 290 | 0.465 | 0.814 | 0.592 |
| November | 174 | 0.470 | 0.770 | 0.584 |
| Region 8 | 8 | 0.333 | 0.500 | 0.400 |

Returning visitors get many more false alarms. Groups with few buyers (February: 1, region 5: 6, region 8: 8) have unreliable scores; do not draw conclusions about them.

## Limitations and risks

- **PageValues may leak the label.** It summarises the value of the pages a session visited, computed from the purchases that followed those pages. If it was computed over the same period, buyers' own purchases are in it. Without it the Decision Tree drops to F1 0.384 and ROC-AUC 0.721; expect live performance between that and the full scores unless page values come from earlier periods only.
- **One store, one year, 2018-era analytics.** Seasonality (November, May) is learned from a single year.
- **Raw probabilities are inflated** by class weighting (mean 0.28 for the tree against a real rate of 0.155). Use the calibrated model, or the tuned threshold, never 0.5 on raw probabilities for business decisions.
- **The profit assumptions are illustrative.** The best threshold moves from 0.13 to "send no offers" across plausible assumptions (`sensitivity_thresholds.png`).
- **Offers change behaviour.** The model is trained on sessions without the campaign; once offers are shown, the data it was trained on no longer describes the visitors.

## Ethical considerations

The data holds no names, demographics or contact details, so the model cannot discriminate on them directly. Region and technical codes could act as proxies if the shop's regions map to demographic groups; check slice performance before using it to give some visitors better prices than others.
