# Decision Tree vs XGBoost (benchmark): performance vs interpretability

| | Decision Tree | XGBoost (benchmark) |
|---|---|---|
| Test F1 | 0.627 | 0.645 |
| Test ROC-AUC | 0.922 | 0.929 |
| Precision / Recall | 0.512 / 0.806 | 0.545 / 0.791 |
| Size | 1 tree, depth 7, 63 leaves | 100 trees, depth up to 6 |
| Top 3 features (permutation) | PageValues, Month, ProductRelated | PageValues, Month, ExitRates |

**Performance.** XGBoost scores +0.007 ROC-AUC and +0.018 F1 compared with the Decision Tree. Boosting combines many small trees, each one correcting the errors of the previous ones, so it ranks sessions more smoothly than a single tree, whose probabilities jump in steps (one value per leaf).

**Interpretability.** The Decision Tree can be read as a flowchart: every prediction follows one path of yes/no questions, which we can print for any session (the Streamlit app does this). XGBoost adds up 100 trees, so no single path explains a prediction; we need outside tools such as permutation importance to see what it relies on.

**Agreement.** Both models rely on the same main signal: the top features are PageValues, Month, ProductRelated for the tree and PageValues, Month, ExitRates for XGBoost.

**Verdict.** If the small gain in score matters more than being able to explain each decision, XGBoost is the stronger benchmark. For a shop team that needs to understand and trust the rules, the Decision Tree gives most of the performance with full transparency.
