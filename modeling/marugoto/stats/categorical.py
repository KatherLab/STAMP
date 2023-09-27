#!/usr/bin/env python3
"""Calculate statistics for deployments on categorical targets."""

from pathlib import Path
import pandas as pd
from sklearn import metrics
import scipy.stats as st


__author__ = 'Marko van Treeck'
__copyright__ = 'Copyright 2022, Kather Lab'
__license__ = 'MIT'
__version__ = '0.1.0'
__maintainer__ = 'Marko van Treeck'
__email__ = 'mvantreeck@ukaachen.de'


__all__ = ['categorical', 'aggregate_categorical_stats',
           'categorical_aggregated_']


score_labels = ['roc_auc_score', 'average_precision_score', 'p_value', 'count']


def categorical(preds_df: pd.DataFrame, target_label: str) -> pd.DataFrame:
    """Calculates some stats for categorical prediction tables.

    This will calculate the number of items, the AUROC, AUPRC and p value
    for a prediction file.
    """
    categories = preds_df[target_label].unique()
    y_true = preds_df[target_label]
    y_pred = preds_df[[f'{target_label}_{cat}' for cat in categories]].applymap(float).values

    stats_df = pd.DataFrame(index=categories)

    # class counts
    stats_df['count'] = pd.value_counts(y_true)

    # roc_auc
    stats_df['roc_auc_score'] = [
        metrics.roc_auc_score(y_true == cat, y_pred[:, i])
        for i, cat in enumerate(categories)]

    # average_precision
    stats_df['average_precision_score'] = [
        metrics.average_precision_score(y_true == cat, y_pred[:, i])
        for i, cat in enumerate(categories)]

    # p values
    p_values = []
    for i, cat in enumerate(categories):
        pos_scores = y_pred[:, i][y_true == cat]
        neg_scores = y_pred[:, i][y_true != cat]
        p_values.append(st.ttest_ind(pos_scores, neg_scores).pvalue)
    stats_df['p_value'] = p_values

    assert set(score_labels) & set(stats_df.columns) == set(score_labels)

    return stats_df


def aggregate_categorical_stats(df) -> pd.DataFrame:
    stats = {}
    for cat, data in df.groupby('level_1'):
        scores_df = data[score_labels]
        means, sems = scores_df.mean(), scores_df.sem()
        l, h = st.t.interval(alpha=.95, df=len(
            scores_df)-1, loc=means, scale=sems)
        cat_stats_df = pd.DataFrame.from_dict(
            {'mean': means, '95% conf': (h-l)/2}).transpose().unstack()
        cat_stats_df[('count', 'sum')] = data['count'].sum()
        stats[cat] = cat_stats_df

    return pd.DataFrame.from_dict(stats, orient='index')


def categorical_aggregated_(*preds_csvs: str, outpath: str, target_label: str) -> None:
    """Calculate statistics for categorical deployments.

    Args:
        preds_csvs:  CSV files containing predictions.
        outpath:  Path to save the results to.
        target_label:  Label to compute the predictions for.

    This will apply `categorical` to all of the given `preds_csvs` and
    calculate the mean and 95% confidence interval for all the scores as
    well as sum the total instane count for each class.
    """
    outpath = Path(outpath)
    preds_dfs = {
        Path(p).parent.name: categorical(
            pd.read_csv(p, dtype=str), target_label)
        for p in preds_csvs}
    preds_df = pd.concat(preds_dfs).sort_index()
    preds_df.to_csv(outpath/f'{target_label}-categorical-stats-individual.csv')
    stats_df = aggregate_categorical_stats(preds_df.reset_index())
    stats_df.to_csv(outpath/f'{target_label}-categorical-stats-aggregated.csv')


if __name__ == '__main__':
    from fire import Fire
    Fire(categorical_aggregated_)
