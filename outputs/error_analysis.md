# Error analysis: XGBoost (benchmark) (threshold 0.50)

| Group | Sessions | Mean PageValues | Share with PageValues = 0 | Median product time (s) | Mean ExitRates | Mean probability |
|---|---|---|---|---|---|---|
| Caught buyer (TP) | 302 | 31.6 | 7% | 1148 | 0.019 | 0.86 |
| Missed buyer (FN) | 80 | 1.0 | 80% | 781 | 0.033 | 0.29 |
| False alarm (FP) | 252 | 16.0 | 25% | 1662 | 0.022 | 0.72 |
| Correct non-buyer (TN) | 1832 | 0.2 | 97% | 424 | 0.049 | 0.09 |

## What the missed buyers have in common

- **They look like browsers, not buyers.** 80% of missed buyers have PageValues = 0, against 7% of caught buyers. PageValues is the model's strongest signal, so a buyer who never passed through a page linked to earlier purchases gets a low probability (mean 0.29).
- **Less engagement.** Their median time on product pages is 781 s, against 1148 s for caught buyers, and their mean exit rate is higher (0.033 vs 0.019).
- **Visitor type is not the reason.** 22% of missed buyers are new visitors, about the same as caught buyers (24%).
- **Timing.** Nov is over-represented: 57% of missed buyers against 42% of caught buyers.

## What the false alarms have in common

- They look like buyers: mean PageValues 16.0 (caught buyers: 31.6) and only 25% have PageValues = 0. They reached valuable pages and then left without paying, which the features cannot tell apart from a purchase.
- 91% of false alarms are returning visitors, against 76% of caught buyers: people who come back, browse valuable pages and still do not buy.
- Dec is over-represented: 13% of false alarms against 8% of caught buyers.

## Why the model struggles

The dataset records how a session behaved, not why. A buyer who already knew what to buy can arrive, buy quickly and leave, without the page-value trail that most buyers leave. And a visitor who filled a cart and abandoned it leaves exactly that trail. With PageValues as the dominant feature, the model's mistakes are mostly the sessions where PageValues points the wrong way. More data about the visitor (past orders, cart contents) would be needed to fix them; tuning the model further cannot.
