from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn import metrics

_score_labels_regression = ["l1", "cc", "cc_p_value", "r2", "binarized_auc", "count"]


def _regression(
    preds_df: pd.DataFrame, target_label: str, pred_label: str
) -> pd.DataFrame:
    """Calculate regression + stratification metrics."""
    y_true = preds_df[target_label].astype(float).to_numpy()
    y_pred = preds_df[pred_label].astype(float).to_numpy()

    # standard regression metrics
    l1 = metrics.mean_absolute_error(y_true, y_pred)
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        r, pval = np.nan, np.nan
    else:
        r, pval = st.pearsonr(y_true, y_pred)
    r2 = metrics.r2_score(y_true, y_pred)

    # binarization at median
    median = np.median(y_true)
    y_true_bin = (y_true >= median).astype(int)
    try:
        bin_auc = metrics.roc_auc_score(y_true_bin, y_pred)
    except ValueError:
        # all y_true_bin are the same (degenerate case)
        bin_auc = np.nan

    stats_df = pd.DataFrame(
        {
            "l1": [l1],
            "cc": [r],
            "cc_p_value": [pval],
            "r2": [r2],
            "binarized_auc": [bin_auc],
            "count": [len(y_true)],
        },
        index=[pred_label],
    )

    assert set(_score_labels_regression) & set(stats_df.columns) == set(
        _score_labels_regression
    )
    return stats_df


def regression_aggregated_(
    *,
    preds_csvs: Sequence[Path],
    outpath: Path,
    ground_truth_label: str,
    pred_label: str,
) -> None:
    """Calculate regression stats (L1, CC) across multiple predictions."""
    preds_dfs = {
        Path(p).parent.name: _regression(
            pd.read_csv(p).dropna(subset=[ground_truth_label]),
            target_label=ground_truth_label,
            pred_label=pred_label,
        )
        for p in preds_csvs
    }
    preds_df = pd.concat(preds_dfs).sort_index()
    preds_df.to_csv(outpath / f"{ground_truth_label}_regression-stats_individual.csv")

    preds_df.to_csv(outpath / f"{ground_truth_label}_regression-stats_aggregated.csv")
