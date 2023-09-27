# %%
from collections import namedtuple
from typing import Iterable, Sequence, Optional, Tuple

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score


all = ['plot_precision_recall_curve', 'plot_precision_recall_curves', 'plot_precision_recall_curves_']


def plot_precision_recall_curve(
    ax: plt.Axes,
    y_true: Sequence[int],
    y_pred: Sequence[float],
    *,
    title: Optional[str] = None,
) -> int:
    """Plots a single precision-recall curve.

    Args:
        ax:  Axis to plot to.
        y_true:  A sequence of ground truths.
        y_pred:  A sequence of predictions.
        title:  Title of the plot.

    Returns:
        The area under the curve.
    """
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    ax.plot(recall, precision)

    style_auc(ax, baseline=y_true.sum()/len(y_true))

    auc = average_precision_score(y_true, y_pred)
    if title:
        ax.set_title(f'{title}\n(AUC = {auc:0.2f})')
    else:
        ax.set_title(f'AUC = {auc:0.2f}')

    return auc


def style_auc(ax, baseline: float):
    ax.plot([0, 1], [0, 1], 'r--', alpha=0)
    ax.set_aspect('equal')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.plot([0, 1], [baseline, baseline], 'r--')


TPA = namedtuple('TPA', ['true', 'pred', 'auc'])


def plot_precision_recall_curves(
    ax: plt.Axes,
    y_trues: Sequence[Sequence[int]],
    y_preds: Sequence[Sequence[float]],
    *,
    title: Optional[str] = None
) -> Tuple[float, float]:
    """Plots a family of precision-recall curves.

    Args:
        ax:  Axis to plot to.
        y_trues:  Sequence of ground truth lists.
        y_preds:  Sequence of prediction lists.
        title:  Title of the plot.

    Returns:
        The 95% confidence interval of the area under the curve.
    """
    # sort trues, preds, AUCs by AUC
    tpas = [TPA(t, p, average_precision_score(t, p))
            for t, p in zip(y_trues, y_preds)]  # , strict=True)]
    tpas = sorted(tpas, key=lambda x: x.auc, reverse=True)

    # plot precision_recalls
    for t, p, auc in tpas:
        precision, recall, _ = precision_recall_curve(t, p)
        ax.plot(recall, precision, label=f'AUC = {auc:0.2f}')

    # style plot
    all_samples = np.concatenate(y_trues)
    baseline = all_samples.sum()/len(all_samples)
    style_auc(ax, baseline=baseline)
    ax.legend()

    # calculate confidence intervals and print title
    aucs = [x.auc for x in tpas]
    l, h = st.t.interval(
        0.95, len(aucs)-1, loc=np.mean(aucs), scale=st.sem(aucs))
    conf_range = (h-l)/2
    auc_str = f'AUC = ${np.mean(aucs):0.2f} \pm {conf_range:0.2f}$'

    if title:
        ax.set_title(f'{title}\n({auc_str})')
    else:
        ax.set_title(auc_str)

    return l, h


def plot_precision_recall_curves_(
        *pred_csvs: Iterable[str], target_label: str, true_label: str, outpath
) -> None:
    """Creates precision-recall curves.

    Args:
        pred_csvs:  A list of prediction CSV files.
        target_label:  The target label to calculate the precision-recall for.
        true_label:  The positive class for the precision-recall.
        outpath:  Path to save the `.svg` to.
    """
    import pandas as pd
    from pathlib import Path
    pred_dfs = [pd.read_csv(p, dtype=str) for p in pred_csvs]

    y_trues = [df[target_label] == true_label for df in pred_dfs]
    y_preds = [pd.to_numeric(df[f'{target_label}_{true_label}']) for df in pred_dfs]
    title = f'{target_label} = {true_label}'
    fig, ax = plt.subplots()
    if len(pred_dfs) == 1:
        plot_precision_recall_curve(ax, y_trues[0], y_preds[0], title=title)
    else:
        plot_precision_recall_curves(ax, y_trues, y_preds, title=title)

    fig.savefig(Path(outpath)/f'prc-{target_label}={true_label}.svg')
    plt.close(fig)


if __name__ == '__main__':
    from fire import Fire
    Fire(plot_precision_recall_curves_)
