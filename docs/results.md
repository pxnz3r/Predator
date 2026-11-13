# Results

All numbers come from `python main.py` with `random_state=42`: 9,864 training sessions, 2,466 test sessions, 15.5% buyers in both. Tables are in `outputs/`, figures in `outputs/figures/`.

## Model comparison (test set, threshold 0.5)

| Model | Role | CV F1 | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Decision Tree | Main (course) | 0.642 | 0.851 | 0.512 | 0.806 | 0.627 | 0.922 |
| KNN | Main (course) | 0.493 | 0.861 | 0.576 | 0.395 | 0.469 | 0.734 |
| KNN (undersampled) | Main (course) | 0.521 | 0.789 | 0.401 | 0.728 | 0.517 | 0.843 |
| Naive Bayes | Baseline | 0.299 | 0.273 | 0.173 | 0.974 | 0.293 | 0.733 |
| **XGBoost** | **Benchmark (beyond course)** | **0.673** | 0.865 | 0.545 | 0.791 | **0.645** | **0.929** |

CV F1 is the 5-fold cross-validated F1 on the training data at the real class balance, and is what the best model is chosen by.

![ROC curves](../outputs/figures/roc_curves.png)

![Confusion matrices](../outputs/figures/confusion_matrices.png)

**Takeaways**

- **XGBoost (benchmark)** is the best model; the **Decision Tree** is the best main model and reaches 97% of XGBoost's F1 and 99% of its ROC-AUC.
- Plain **KNN** has the highest accuracy after XGBoost (0.861) but finds only 40% of buyers: a clear example of accuracy hiding a weak model on imbalanced data.
- **Undersampling** helps KNN: recall 0.40 to 0.73, F1 0.469 to 0.517, ROC-AUC 0.734 to 0.843.
- **Naive Bayes** flags 87% of sessions as buyers and has an accuracy of 0.273, worse than always saying "no".

## How sure are the scores?

95% bootstrap confidence intervals (1,000 resamples of the test set):

| Model | F1 | F1 95% interval | ROC-AUC | ROC-AUC 95% interval |
|---|---|---|---|---|
| Decision Tree | 0.627 | 0.590 to 0.661 | 0.922 | 0.908 to 0.935 |
| KNN | 0.469 | 0.422 to 0.516 | 0.734 | 0.704 to 0.761 |
| KNN (undersampled) | 0.517 | 0.481 to 0.554 | 0.843 | 0.822 to 0.863 |
| Naive Bayes | 0.293 | 0.270 to 0.317 | 0.733 | 0.710 to 0.757 |
| XGBoost | 0.645 | 0.609 to 0.679 | 0.929 | 0.917 to 0.941 |

McNemar's test at threshold 0.5 (sessions only one model gets right):

| Model A | Model B | Only A right | Only B right | p-value |
|---|---|---|---|---|
| Decision Tree | XGBoost | 52 | 87 | 0.004 |
| Decision Tree | KNN (undersampled) | 257 | 104 | < 0.001 |

The Decision Tree's and XGBoost's intervals overlap, but McNemar's test, which uses the fact that both are scored on the same sessions, shows XGBoost makes significantly fewer mistakes. The gap is small and real.

## Slices

![F1 by visitor type and month](../outputs/figures/slice_scores.png)

| Decision Tree, threshold 0.5 | Sessions | Buyers | Precision | Recall | F1 |
|---|---|---|---|---|---|
| New visitors | 347 | 89 | 0.758 | 0.775 | 0.767 |
| Returning visitors | 2,108 | 290 | 0.465 | 0.814 | 0.592 |
| November | 638 | 174 | 0.470 | 0.770 | 0.584 |
| March | 369 | 33 | 0.733 | 1.000 | 0.846 |

Returning visitors are the hard group: many of them reach valuable pages without buying, so precision drops. Slices with few buyers (February has 1, region 5 has 6) give noisy scores. All slices: `outputs/slice_analysis.csv`.

## Learning and validation curves

![Learning curves](../outputs/figures/learning_curves.png)

![Validation curves](../outputs/figures/validation_curves.png)

- **Decision Tree:** training F1 0.68 against cross-validated F1 0.64 at full size, a small gap; the curve levels off after about 5,000 sessions.
- **KNN:** training F1 is 1.0 at every size (distance weighting lets every session vote for itself), a clear case of high variance. Its cross-validated F1 is still rising at full size (0.38 to 0.49), and the validation curve peaks at k = 3.
- **A one-split tree has the best F1.** At depth 1, the rule `PageValues > 0.05` reaches a cross-validated F1 of 0.667, above the tuned depth 7 (0.643). The specified grid starts at depth 3. The deeper tree is kept because a stump gives only two probabilities, so it cannot rank sessions or support a profit threshold. Above depth 7 the score falls (0.605 at depth 20).

## Best hyperparameters

| Model | Best settings |
|---|---|
| Decision Tree | criterion = entropy, max_depth = 7, min_samples_split = 50, min_samples_leaf = 20 |
| KNN | n_neighbors = 3, weights = distance, metric = euclidean |
| KNN (undersampled) | n_neighbors = 27, weights = distance, metric = manhattan |
| XGBoost | n_estimators = 100, max_depth = 6, learning_rate = 0.1 |

## Feature contributions

Permutation importance (drop in test ROC-AUC when a feature is shuffled):

| Feature | Decision Tree | KNN | KNN (undersampled) | Naive Bayes | XGBoost |
|---|---|---|---|---|---|
| PageValues | 0.318 | 0.156 | 0.136 | 0.055 | 0.300 |
| Month | 0.065 | 0.008 | 0.022 | 0.024 | 0.073 |
| ExitRates | 0.003 | 0.005 | 0.007 | 0.031 | 0.016 |

![Permutation importance](../outputs/figures/permutation_importance.png)

![Decision Tree top levels](../outputs/figures/decision_tree_top_levels.png)

PageValues is the most important feature for every model, by a wide margin. Month (mostly November) is second for the tree-based models. The Decision Tree's first split is `PageValues <= 0.05`.

![Partial dependence](../outputs/figures/partial_dependence.png)

Partial dependence shows the direction: the Decision Tree's average probability jumps from about 0.15 to 0.8 as soon as PageValues is above zero, while XGBoost climbs more smoothly to about 0.85. November raises the probability for both models; a higher exit rate lowers it.

## Decision Tree vs XGBoost

| | Decision Tree | XGBoost (benchmark) |
|---|---|---|
| Test F1 / ROC-AUC | 0.627 / 0.922 | 0.645 / 0.929 |
| Size | 1 tree, depth 7, 63 leaves | 100 trees, depth up to 6 |
| Explaining one prediction | Print its path of questions | Needs outside tools |

XGBoost is slightly more accurate because boosting combines many trees that correct each other's errors and gives smoother probabilities. The Decision Tree can be read as a flowchart, and the app prints the exact path any session takes. For a shop team that must understand and trust the rules, the small loss in score is worth the transparency. Full text: `outputs/tree_vs_xgboost.md`.

## Ablation: without PageValues

| Model | F1 all features | F1 without | Change | ROC-AUC all | ROC-AUC without | Change |
|---|---|---|---|---|---|---|
| Decision Tree | 0.627 | 0.384 | -0.243 | 0.922 | 0.721 | -0.201 |
| KNN | 0.469 | 0.261 | -0.208 | 0.734 | 0.562 | -0.172 |
| KNN (undersampled) | 0.517 | 0.356 | -0.161 | 0.843 | 0.723 | -0.120 |
| Naive Bayes | 0.293 | 0.292 | -0.001 | 0.733 | 0.678 | -0.055 |
| XGBoost | 0.645 | 0.414 | -0.231 | 0.929 | 0.772 | -0.157 |

![Ablation scores](../outputs/figures/ablation_scores.png)

**Is PageValues leaking the answer?** A page's value comes from the purchases that followed views of it. If the values were computed over the same period as the sessions, a buyer's own purchase raised its PageValues. The dataset does not say, so the scores without PageValues are the honest lower bound for a live system ([methodology.md](methodology.md#is-pagevalues-leaking-the-answer)).

Without PageValues the Decision Tree, KNN and XGBoost lose a third or more of their F1. Naive Bayes barely changes because it was already flagging almost everyone. The new top features are behaviour signals: **ExitRates**, **ProductRelated_Duration**, **ProductRelated** and **Month** for the trees, and **Month**, **VisitorType** and **TrafficType** for KNN (`outputs/ablation_top_features.csv`). They carry real information (XGBoost still reaches ROC-AUC 0.772) but cannot replace the page-value signal.

## Business threshold

With an $18 gain per true positive and a $6 loss per false positive (illustrative assumptions):

- The **Decision Tree** earns most at threshold **0.70**: $3,870 on the test set against $3,786 at 0.5, while targeting 447 sessions instead of 601.
- **XGBoost** earns $3,924 at 0.5 (its training-data optimum) and $3,996 at its best-F1 threshold of 0.62; the profit curve is flat near the top.
- **Naive Bayes** loses money at every threshold.

![Profit versus threshold](../outputs/figures/profit_vs_threshold.png)

**Sensitivity:** across 36 combinations of offer cost ($2 to $12) and conversion boost (10% to 60%), the Decision Tree's best threshold ranges from 0.13 to 0.93, and with a 10% boost and an offer cost of $6 or more no threshold is profitable. Cheaper offers and stronger boosts always push the threshold down.

![Best threshold for other assumptions](../outputs/figures/sensitivity_thresholds.png)

## Calibration

| Model | Brier raw | Brier calibrated | Mean probability raw | Mean probability calibrated | Test profit at p > 0.25, raw | Calibrated |
|---|---|---|---|---|---|---|
| Decision Tree | 0.109 | 0.072 | 0.284 | 0.151 | $2,580 | **$4,086** |
| XGBoost | 0.094 | 0.071 | 0.257 | 0.155 | $3,330 | **$4,020** |

The real purchase rate is 0.155. Balanced class weights and `scale_pos_weight` inflate the raw probabilities; isotonic calibration on the training folds fixes that without changing ROC-AUC. With calibrated probabilities, the break-even rule p > offer cost / (order profit x boost) = 0.25 earns more than the tuned thresholds on raw probabilities ($3,870 for the tree, $3,924 for XGBoost).

![Before and after calibration](../outputs/figures/calibration_before_after.png)

Details: [business_threshold.md](business_threshold.md).

## Error analysis

The best model misses 80 of 382 test buyers. 80% of them have PageValues = 0 and they spend less time on product pages; November is over-represented. The 252 false alarms look like buyers (high PageValues, long product-page time) and 91% are returning visitors. Details: [error_analysis.md](error_analysis.md).

## Conclusions

1. **PageValues drives purchase prediction.** It is the top feature for every model, the root of the Decision Tree, and removing it costs a third or more of F1.
2. **The Decision Tree is the recommended course model**: close to the XGBoost benchmark and fully interpretable.
3. **XGBoost is the strongest benchmark**, by a small margin that McNemar's test confirms (p = 0.004).
4. **KNN needs balancing and suffers from many dimensions; Naive Bayes' assumptions do not hold here.**
5. **The decision threshold is a business choice**; with a profit model it differs from 0.5 and from the best-F1 threshold. Calibrating the probabilities makes it explainable (p > cost / (profit x boost)) and earns the most.
6. **The remaining errors are sessions whose behaviour contradicts their outcome**, which more visitor data could fix and more tuning cannot.
7. **PageValues may leak part of the answer**; the ablation scores are the honest lower bound for a live system.
