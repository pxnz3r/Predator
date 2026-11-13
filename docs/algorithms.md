# The algorithms, explained

A beginner-friendly look at the four models, and how each one did on this problem.

## Decision Tree (main model)

**Idea:** a flowchart of yes/no questions. "Is PageValues above 0.05? If yes, is BounceRates 0? ..." Each leaf at the bottom gives a prediction.

**How it learns:** starting with all training sessions, it tries every feature and every cut point and picks the question that separates buyers from non-buyers best. "Best" means the biggest drop in impurity:

- **Gini impurity** = 1 - (share of buyers)^2 - (share of non-buyers)^2. A node with only one class has Gini 0; a 50/50 node has Gini 0.5.
- **Entropy** = -sum(p * log2(p)) over both classes. 0 for a pure node, 1 for 50/50. It measures how uncertain we are about a label in the node.

It repeats this on each side until a stopping rule (max_depth, min_samples_split, min_samples_leaf) says stop. Those rules prevent **overfitting**: a tree that is allowed to grow until every leaf is pure memorises the training data and does worse on new sessions.

**On this problem:** tuned to entropy, depth 7, 63 leaves. Test F1 0.627, ROC-AUC 0.922. Its first question is PageValues, and the next ones use November, BounceRates and product-page time.

| Strengths | Weaknesses |
|---|---|
| Fully readable: every prediction follows one printed path | Probabilities come in steps (one value per leaf) |
| No scaling needed, handles mixed feature types | A single tree is unstable: small data changes can change the splits |
| `class_weight` handles imbalance directly | `class_weight="balanced"` pushes probabilities up, so 0.5 is not its best cutoff |

## K-Nearest Neighbours (main model)

**Idea:** "you are like your neighbours." To classify a session, find the k most similar training sessions and let them vote.

**How it measures similarity:** with a distance between feature vectors.
- **Euclidean:** straight-line distance, sqrt((a1-b1)^2 + (a2-b2)^2 + ...).
- **Manhattan:** sum of absolute differences, |a1-b1| + |a2-b2| + ...

`weights="distance"` gives closer neighbours a bigger vote. KNN does no real training; it stores the data, which makes prediction slower than training.

**Why scaling matters:** distances add up the differences in every feature. Unscaled, a 600-second difference in ProductRelated_Duration would swamp a 0.1 difference in ExitRates, even if the exit rate is far more telling. Standardising puts every feature on the same footing.

**On this problem:** the weakest main model. Plain KNN (k = 3) has test F1 0.469 and ROC-AUC 0.734; it misses 60% of buyers because most neighbours of any session are non-buyers. Trained on undersampled data (k = 27), recall jumps from 0.40 to 0.73 and ROC-AUC to 0.843, at the cost of precision.

| Strengths | Weaknesses |
|---|---|
| Simple, no assumptions about the data's shape | About 75 dimensions after one-hot encoding: in high dimensions "nearest" means little (curse of dimensionality) |
| Naturally captures local patterns | Every feature counts equally, so the many weak one-hot columns dilute PageValues |
| | No class weights; needs undersampling to find buyers |

## Gaussian Naive Bayes (baseline)

**Idea:** Bayes' theorem: P(buy | features) is proportional to P(features | buy) x P(buy).

**The naive part:** it assumes every feature is independent of the others once you know the class, so P(features | buy) is just the product of each feature's probability. **Gaussian** means each numerical feature is assumed to follow a bell curve inside each class.

**On this problem:** both assumptions fail. ProductRelated and ProductRelated_Duration move together, BounceRates and ExitRates are 0.91 correlated, and most features are piles of zeros with long tails, not bell curves. Multiplying many correlated pieces of evidence makes it extremely overconfident: it flags 87% of test sessions as buyers (recall 0.97, precision 0.17, F1 0.293), and its probabilities are so close to 0 or 1 that no threshold fixes it.

| Strengths | Weaknesses |
|---|---|
| Very fast, no tuning | Independence assumption badly broken here |
| A useful floor: any serious model must beat it | Probabilities are badly calibrated |

## XGBoost (benchmark, beyond the course)

**Idea:** gradient boosting. Instead of one big tree, build many small trees **one after another**. Each new tree is trained on what the previous trees still get wrong (the gradient of the loss), and the final score adds up all trees. `learning_rate` shrinks each tree's contribution so no single tree dominates.

**On this problem:** 100 trees of depth 6, learning rate 0.1, `scale_pos_weight` about 5.5. The best model: test F1 0.645, ROC-AUC 0.929, CV F1 0.673.

| Strengths | Weaknesses |
|---|---|
| Usually the most accurate on tabular data | 100 trees cannot be read like a flowchart |
| Built-in handling of imbalance and regularisation | More hyperparameters to tune |
| Smoother probabilities than a single tree | Needs outside tools (permutation importance) to explain |

## Summary

| Model | Role | Test F1 | Test ROC-AUC | One-line verdict |
|---|---|---|---|---|
| Decision Tree | Main | 0.627 | 0.922 | Best main model, nearly matches the benchmark, fully readable |
| KNN | Main | 0.469 | 0.734 | Hurt by imbalance and many dimensions |
| KNN (undersampled) | Main | 0.517 | 0.843 | Undersampling clearly helps KNN |
| Naive Bayes | Baseline | 0.293 | 0.733 | Assumptions broken, overconfident |
| XGBoost | Benchmark | 0.645 | 0.929 | Strongest, least interpretable |
