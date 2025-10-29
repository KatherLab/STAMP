"""Calculate statistics for deployments on categorical targets."""

from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import scipy.stats as st
from sklearn import metrics

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


_score_labels = ["roc_auc_score", "average_precision_score", "p_value", "count"]


def _categorical(preds_df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """Calculates some stats for categorical prediction tables.

    This will calculate the number of items, the AUROC, AUPRC and p value
    for a prediction file.
    """
    categories = preds_df[target_label].unique()
    y_true = preds_df[target_label]
    y_pred = preds_df[[f"{target_label}_{cat}" for cat in categories]].map(float).values

    stats_df = pd.DataFrame(index=categories)

    # class counts
    stats_df["count"] = y_true.value_counts()

    # roc_auc
    stats_df["roc_auc_score"] = [
        metrics.roc_auc_score(y_true == cat, y_pred[:, i])  # pyright: ignore[reportCallIssue,reportArgumentType]
        for i, cat in enumerate(categories)
    ]

    # average_precision
    stats_df["average_precision_score"] = [
        metrics.average_precision_score(y_true == cat, y_pred[:, i])  # pyright: ignore[reportCallIssue,reportArgumentType]
        for i, cat in enumerate(categories)
    ]

    # p values
    p_values = []
    for i, cat in enumerate(categories):
        pos_scores = y_pred[:, i][y_true == cat]  # pyright: ignore[reportCallIssue,reportArgumentType]
        neg_scores = y_pred[:, i][y_true != cat]  # pyright: ignore[reportCallIssue,reportArgumentType]
        p_values.append(st.ttest_ind(pos_scores, neg_scores).pvalue)  # pyright: ignore[reportGeneralTypeIssues, reportAttributeAccessIssue]
    stats_df["p_value"] = p_values

    assert set(_score_labels) & set(stats_df.columns) == set(_score_labels)

    return stats_df


def _aggregate_categorical_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = {}
    for cat, data in df.groupby("level_1"):
        scores_df = data[["roc_auc_score", "average_precision_score"]]
        means, sems = scores_df.mean(), scores_df.sem()
        lower, upper = st.t.interval(0.95, df=len(scores_df) - 1, loc=means, scale=sems)
        cat_stats_df = (
            pd.DataFrame.from_dict({"mean": means, "95%_low": lower, "95%_high": upper})
            .transpose()
            .unstack()
        )
        cat_stats_df[("count", "sum")] = data["count"].sum()
        stats[cat] = cat_stats_df

    return pd.DataFrame.from_dict(stats, orient="index")


def categorical_aggregated_(
    *, preds_csvs: Sequence[Path], outpath: Path, ground_truth_label: str
) -> None:
    """Calculate statistics for categorical deployments.

    Args:
        preds_csvs:  CSV files containing predictions.
        outpath:  Path to save the results to.
        target_label:  Label to compute the predictions for.

    This will apply `categorical` to all of the given `preds_csvs` and
    calculate the mean and 95% confidence interval for all the scores as
    well as sum the total instane count for each class.
    """
    preds_dfs = {
        Path(p).parent.name: _categorical(
            pd.read_csv(p, dtype=str).dropna(subset=[ground_truth_label]),
            ground_truth_label,
        )
        for p in preds_csvs
    }
    preds_df = pd.concat(preds_dfs).sort_index()
    preds_df.to_csv(outpath / f"{ground_truth_label}_categorical-stats_individual.csv")
    stats_df = _aggregate_categorical_stats(preds_df.reset_index())
    stats_df.to_csv(outpath / f"{ground_truth_label}_categorical-stats_aggregated.csv")
