"""
errors.py - Error analysis: which sessions does the best model get wrong, and why?

We split the test sessions into four groups using the best model's
predictions at its chosen threshold:
    - Caught buyers (true positives)
    - Missed buyers (false negatives): bought, but the model said no
    - False alarms (false positives): did not buy, but the model said yes
    - Correct non-buyers (true negatives)
Then we compare the groups' feature averages and distributions to see
what the mistakes have in common.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from predator.data import NUMERICAL_FEATURES, save_figure
from predator.evaluate import OUTPUTS_FOLDER, get_purchase_probabilities

CAUGHT_BUYER = "Caught buyer (TP)"
MISSED_BUYER = "Missed buyer (FN)"
FALSE_ALARM = "False alarm (FP)"
CORRECT_NON_BUYER = "Correct non-buyer (TN)"
GROUP_ORDER = [CAUGHT_BUYER, MISSED_BUYER, FALSE_ALARM, CORRECT_NON_BUYER]
GROUP_COLORS = {
    CAUGHT_BUYER: "#2A9D8F",
    MISSED_BUYER: "#E76F51",
    FALSE_ALARM: "#E9C46A",
    CORRECT_NON_BUYER: "#9AA5B1",
}
MONTH_ORDER = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def label_prediction_groups(pipeline, test_features, test_target, threshold):
    """
    Put every test session into one of the four outcome groups.

    Inputs:
        pipeline: the fitted best model.
        test_features, test_target: the test data.
        threshold: the decision threshold used for the best model.
    Returns:
        A copy of test_features with extra columns: "actual",
        "probability", "predicted" and "group".
    """
    sessions = test_features.copy()
    sessions["actual"] = test_target.values
    sessions["probability"] = get_purchase_probabilities(pipeline, test_features)
    sessions["predicted"] = (sessions["probability"] >= threshold).astype(int)

    groups = []
    for actual, predicted in zip(sessions["actual"], sessions["predicted"], strict=True):
        if actual == 1 and predicted == 1:
            groups.append(CAUGHT_BUYER)
        elif actual == 1 and predicted == 0:
            groups.append(MISSED_BUYER)
        elif actual == 0 and predicted == 1:
            groups.append(FALSE_ALARM)
        else:
            groups.append(CORRECT_NON_BUYER)
    sessions["group"] = groups
    return sessions


def summarise_numerical_features(sessions):
    """
    Average of every numerical feature for each group, plus a few extras.

    Input:
        sessions: the DataFrame from label_prediction_groups().
    Returns:
        A DataFrame with one row per group, saved to
        outputs/error_numerical_summary.csv.
    """
    summary = sessions.groupby("group")[NUMERICAL_FEATURES + ["probability"]].mean()
    summary["sessions"] = sessions.groupby("group").size()
    summary["share with PageValues = 0"] = sessions.groupby("group")["PageValues"].apply(
        lambda values: (values == 0).mean()
    )
    summary["median ProductRelated_Duration"] = sessions.groupby("group")["ProductRelated_Duration"].median()
    summary = summary.reindex(GROUP_ORDER).round(3)
    summary.to_csv(os.path.join(OUTPUTS_FOLDER, "error_numerical_summary.csv"), index_label="group")
    return summary


def summarise_categorical_feature(sessions, column):
    """
    Share of each category inside every group (each row adds up to 100%).

    For example, for VisitorType: what share of missed buyers were new
    visitors, compared with caught buyers.

    Inputs:
        sessions: the DataFrame from label_prediction_groups().
        column: a categorical column name.
    Returns:
        A DataFrame: groups as rows, categories as columns, values are shares.
    """
    shares = pd.crosstab(sessions["group"], sessions[column], normalize="index")
    shares = shares.reindex(GROUP_ORDER)
    if column == "Month":
        shares = shares[[month for month in MONTH_ORDER if month in shares.columns]]
    return shares.round(3)


def plot_error_analysis(sessions, model_name):
    """
    Four plots comparing the groups: PageValues, ProductRelated_Duration,
    VisitorType and Month.

    PageValues and durations are very skewed, so the box plots use a
    log scale (log(1 + value)) to keep the zeros and the big values visible.

    Inputs:
        sessions: the DataFrame from label_prediction_groups().
        model_name: shown in the title.
    Returns:
        The path of the saved plot.
    """
    figure, axes = plt.subplots(2, 2, figsize=(16, 11))
    colors = [GROUP_COLORS[group] for group in GROUP_ORDER]
    short_labels = [group.replace(" (", "\n(") for group in GROUP_ORDER]

    # 1 and 2: box plots of two numerical features.
    for axis, column in zip([axes[0][0], axes[0][1]], ["PageValues", "ProductRelated_Duration"], strict=True):
        values_per_group = [np.log1p(sessions.loc[sessions["group"] == group, column]) for group in GROUP_ORDER]
        boxes = axis.boxplot(values_per_group, patch_artist=True, showfliers=False)
        axis.set_xticks(range(1, len(GROUP_ORDER) + 1))
        axis.set_xticklabels(short_labels)
        for box, color in zip(boxes["boxes"], colors, strict=True):
            box.set_facecolor(color)
        axis.set_title(column + " by outcome group")
        axis.set_ylabel("log(1 + " + column + ")")

    # 3: VisitorType share per group.
    visitor_shares = summarise_categorical_feature(sessions, "VisitorType")
    visitor_shares.plot(kind="bar", ax=axes[1][0], color=["#264653", "#E9C46A", "#E76F51"], rot=0)
    axes[1][0].set_xticklabels(short_labels)
    axes[1][0].set_title("Visitor type within each group")
    axes[1][0].set_ylabel("Share of the group")
    axes[1][0].set_xlabel("")
    axes[1][0].legend(title="VisitorType", loc="upper left")

    # 4: Month share for the three groups that matter most.
    month_shares = summarise_categorical_feature(sessions, "Month")
    for group in [CAUGHT_BUYER, MISSED_BUYER, FALSE_ALARM]:
        axes[1][1].plot(month_shares.columns, month_shares.loc[group], marker="o",
                        color=GROUP_COLORS[group], linewidth=2, label=group)
    axes[1][1].set_title("Month within each group")
    axes[1][1].set_ylabel("Share of the group")
    axes[1][1].legend()

    figure.suptitle("Error analysis of " + model_name + " on the test set", fontsize=14)
    figure.tight_layout()
    return save_figure(figure, "error_analysis.png")


def plot_probability_histogram(sessions, threshold, model_name):
    """
    Histogram of predicted probabilities for real buyers and non-buyers.

    Missed buyers are the real buyers left of the threshold line; false
    alarms are the non-buyers right of it.

    Inputs:
        sessions: the DataFrame from label_prediction_groups().
        threshold: the decision threshold.
        model_name: shown in the title.
    Returns:
        The path of the saved plot.
    """
    figure, axis = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)
    axis.hist(sessions.loc[sessions["actual"] == 0, "probability"], bins=bins, alpha=0.7,
              color=GROUP_COLORS[CORRECT_NON_BUYER], label="Did not buy")
    axis.hist(sessions.loc[sessions["actual"] == 1, "probability"], bins=bins, alpha=0.7,
              color=GROUP_COLORS[CAUGHT_BUYER], label="Bought")
    axis.axvline(threshold, color="black", linestyle="--", label="Threshold {:.2f}".format(threshold))
    axis.set_yscale("log")
    axis.set_xlabel("Predicted purchase probability")
    axis.set_ylabel("Number of sessions (log scale)")
    axis.set_title("Predicted probabilities of " + model_name)
    axis.legend()
    return save_figure(figure, "error_probability_histogram.png")


def write_error_findings(sessions, numerical_summary, model_name, threshold):
    """
    Write the main error-analysis findings in plain English, using the
    numbers from this run.

    Inputs:
        sessions: the DataFrame from label_prediction_groups().
        numerical_summary: the DataFrame from summarise_numerical_features().
        model_name: the best model's name.
        threshold: its decision threshold.
    Returns:
        The findings text (markdown), also saved to outputs/error_analysis.md.
    """
    visitor_shares = summarise_categorical_feature(sessions, "VisitorType")
    month_shares = summarise_categorical_feature(sessions, "Month")

    caught = numerical_summary.loc[CAUGHT_BUYER]
    missed = numerical_summary.loc[MISSED_BUYER]
    false_alarm = numerical_summary.loc[FALSE_ALARM]

    # The busiest month is always the most common one in every group, so
    # instead we look for the month that is MORE common among the mistakes
    # than among the caught buyers.
    missed_month_gap = month_shares.loc[MISSED_BUYER] - month_shares.loc[CAUGHT_BUYER]
    missed_month = missed_month_gap.idxmax()
    false_alarm_month_gap = month_shares.loc[FALSE_ALARM] - month_shares.loc[CAUGHT_BUYER]
    false_alarm_month = false_alarm_month_gap.idxmax()

    missed_new = visitor_shares.loc[MISSED_BUYER].get("New_Visitor", 0.0)
    caught_new = visitor_shares.loc[CAUGHT_BUYER].get("New_Visitor", 0.0)
    if abs(missed_new - caught_new) < 0.05:
        visitor_sentence = ("**Visitor type is not the reason.** {:.0%} of missed buyers are new visitors, "
                            "about the same as caught buyers ({:.0%}).").format(missed_new, caught_new)
    else:
        visitor_sentence = ("**Visitor type.** {:.0%} of missed buyers are new visitors, against {:.0%} "
                            "of caught buyers.").format(missed_new, caught_new)

    text = (
        "# Error analysis: " + model_name + " (threshold {:.2f})\n\n".format(threshold)
        + "| Group | Sessions | Mean PageValues | Share with PageValues = 0 | Median product time (s) "
          "| Mean ExitRates | Mean probability |\n"
        + "|---|---|---|---|---|---|---|\n"
    )
    for group in GROUP_ORDER:
        row = numerical_summary.loc[group]
        text = text + "| {} | {:.0f} | {:.1f} | {:.0%} | {:.0f} | {:.3f} | {:.2f} |\n".format(
            group, row["sessions"], row["PageValues"], row["share with PageValues = 0"],
            row["median ProductRelated_Duration"], row["ExitRates"], row["probability"])

    text = text + (
        "\n## What the missed buyers have in common\n\n"
        "- **They look like browsers, not buyers.** {missed_zero:.0%} of missed buyers have "
        "PageValues = 0, against {caught_zero:.0%} of caught buyers. PageValues is the model's "
        "strongest signal, so a buyer who never passed through a page linked to earlier purchases "
        "gets a low probability (mean {missed_prob:.2f}).\n"
        "- **Less engagement.** Their median time on product pages is {missed_time:.0f} s, against "
        "{caught_time:.0f} s for caught buyers, and their mean exit rate is higher "
        "({missed_exit:.3f} vs {caught_exit:.3f}).\n"
        "- {visitor_sentence}\n"
        "- **Timing.** {missed_month} is over-represented: {missed_month_share:.0%} of missed buyers "
        "against {missed_month_caught:.0%} of caught buyers.\n\n"
        "## What the false alarms have in common\n\n"
        "- They look like buyers: mean PageValues {fa_pv:.1f} (caught buyers: {caught_pv:.1f}) "
        "and only {fa_zero:.0%} have PageValues = 0. They reached valuable pages and then left "
        "without paying, which the features cannot tell apart from a purchase.\n"
        "- {fa_returning:.0%} of false alarms are returning visitors, against {caught_returning:.0%} "
        "of caught buyers: people who come back, browse valuable pages and still do not buy.\n"
        "- {fa_month} is over-represented: {fa_month_share:.0%} of false alarms against "
        "{fa_month_caught:.0%} of caught buyers.\n\n"
        "## Why the model struggles\n\n"
        "The dataset records how a session behaved, not why. A buyer who already knew what to buy "
        "can arrive, buy quickly and leave, without the page-value trail that most buyers leave. "
        "And a visitor who filled a cart and abandoned it leaves exactly that trail. With "
        "PageValues as the dominant feature, the model's mistakes are mostly the sessions where "
        "PageValues points the wrong way. More data about the visitor (past orders, cart "
        "contents) would be needed to fix them; tuning the model further cannot.\n"
    ).format(
        missed_zero=missed["share with PageValues = 0"], caught_zero=caught["share with PageValues = 0"],
        missed_prob=missed["probability"],
        missed_time=missed["median ProductRelated_Duration"], caught_time=caught["median ProductRelated_Duration"],
        missed_exit=missed["ExitRates"], caught_exit=caught["ExitRates"],
        visitor_sentence=visitor_sentence,
        missed_month=missed_month,
        missed_month_share=month_shares.loc[MISSED_BUYER, missed_month],
        missed_month_caught=month_shares.loc[CAUGHT_BUYER, missed_month],
        fa_pv=false_alarm["PageValues"], caught_pv=caught["PageValues"],
        fa_zero=false_alarm["share with PageValues = 0"], fa_month=false_alarm_month,
        fa_month_share=month_shares.loc[FALSE_ALARM, false_alarm_month],
        fa_month_caught=month_shares.loc[CAUGHT_BUYER, false_alarm_month],
        fa_returning=visitor_shares.loc[FALSE_ALARM].get("Returning_Visitor", 0.0),
        caught_returning=visitor_shares.loc[CAUGHT_BUYER].get("Returning_Visitor", 0.0),
    )

    with open(os.path.join(OUTPUTS_FOLDER, "error_analysis.md"), "w") as file:
        file.write(text)
    print("\n" + text)
    return text


def run_error_analysis(pipeline, model_name, test_features, test_target, threshold):
    """
    Run the whole error analysis for the best model.

    Inputs:
        pipeline: the fitted best model.
        model_name: its name.
        test_features, test_target: the test data.
        threshold: its decision threshold.
    Returns:
        The labelled sessions DataFrame.
    """
    print("\nError analysis of " + model_name + " at threshold {:.2f}...".format(threshold))
    sessions = label_prediction_groups(pipeline, test_features, test_target, threshold)

    numerical_summary = summarise_numerical_features(sessions)
    print("\nAverage feature values per group:")
    print(numerical_summary[["sessions", "PageValues", "share with PageValues = 0",
                             "ProductRelated_Duration", "ExitRates", "BounceRates", "probability"]].to_string())

    category_tables = []
    for column in ["VisitorType", "Month", "Weekend"]:
        shares = summarise_categorical_feature(sessions, column)
        shares.columns = [column + " = " + str(value) for value in shares.columns]
        category_tables.append(shares)
    category_summary = pd.concat(category_tables, axis=1)
    category_summary.to_csv(os.path.join(OUTPUTS_FOLDER, "error_categorical_summary.csv"), index_label="group")

    plot_error_analysis(sessions, model_name)
    plot_probability_histogram(sessions, threshold, model_name)
    write_error_findings(sessions, numerical_summary, model_name, threshold)
    return sessions
