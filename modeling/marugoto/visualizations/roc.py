# %%
from collections import namedtuple
from typing import Iterable, Sequence, Optional, Tuple

import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


all = ['plot_roc_curve', 'plot_roc_curves', 'plot_roc_curves_']


def plot_roc_curve(
    ax: plt.Axes,
    y_true: Sequence[int],
    y_pred: Sequence[float],
    *,
    title: Optional[str] = None,
) -> int:
    """Plots a single ROC curve.

    Args:
        ax:  Axis to plot to.
        y_true:  A sequence of ground truths.
        y_pred:  A sequence of predictions.
        title:  Title of the plot.

    Returns:
        The area under the curve.
    """
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    ax.plot(fpr, tpr)

    style_auc(ax)

    auc = roc_auc_score(y_true, y_pred)
    if title:
        ax.set_title(f'{title}\n(AUC = {auc:0.2f})')
    else:
        ax.set_title(f'AUC = {auc:0.2f}')

    return auc


def style_auc(ax):
    ax.plot([0, 1], [0, 1], 'r--')
    ax.set_aspect('equal')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')


TPA = namedtuple('TPA', ['true', 'pred', 'auc'])


def plot_roc_curves(
    ax: plt.Axes,
    y_trues: Sequence[Sequence[int]],
    y_preds: Sequence[Sequence[float]],
    *,
    title: Optional[str] = None
) -> Tuple[float, float]:
    """Plots a family of ROC curves.

    Args:
        ax:  Axis to plot to.
        y_trues:  Sequence of ground truth lists.
        y_preds:  Sequence of prediction lists.
        title:  Title of the plot.

    Returns:
        The 95% confidence interval of the area under the curve.
    """
    # sort trues, preds, AUCs by AUC
    tpas = [TPA(t, p, roc_auc_score(t, p))
            for t, p in zip(y_trues, y_preds)]  # , strict=True)]
    tpas = sorted(tpas, key=lambda x: x.auc, reverse=True)

    # plot rocs
    for t, p, auc in tpas:
        fpr, tpr, _ = roc_curve(t, p)
        ax.plot(fpr, tpr, label=f'AUC = {auc:0.2f}')

    # style plot
    style_auc(ax)
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


def plot_roc_curves_(
        *pred_csvs: Iterable[str], target_label: str, true_label: str, outpath
) -> None:
    """Creates ROC curves.

    Args:
        pred_csvs:  A list of prediction CSV files.
        target_label:  The target label to calculate the ROC for.
        true_label:  The positive class for the ROC.
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
        plot_roc_curve(ax, y_trues[0], y_preds[0], title=title)
    else:
        plot_roc_curves(ax, y_trues, y_preds, title=title)
    fig.savefig(Path(outpath)/f'roc-{target_label}={true_label}.svg')
    plt.close(fig)


if __name__ == '__main__':
    from fire import Fire
    Fire(plot_roc_curves_)
