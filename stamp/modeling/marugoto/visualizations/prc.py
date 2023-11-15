# %%
from collections import namedtuple
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from collections import namedtuple
from typing import Optional, Sequence, Tuple
import numpy.typing as npt
from tqdm import trange
from sklearn.metrics import precision_recall_curve, auc, average_precision_score
from scipy.interpolate import interp1d


all = ['plot_precision_recall_curve', 'plot_precision_recall_curves', 'plot_precision_recall_curves_']


def style_prc(ax, baseline: float):
    ax.plot([0, 1], [0, 1], "r--", alpha=0)
    ax.set_aspect("equal")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.plot([0, 1], [baseline, baseline], "r--")


def plot_bootstrapped_pr_curve(
    ax: plt.Axes,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    title: Optional[str] = None,
    n_bootstrap_samples: Optional[int] = None,
) -> Tuple[float, Optional[float]]:
    """Plots a precision-recall curve with bootstrap interval.

    Args:
        ax: The axes to plot onto.
        y_true: The ground truths.
        y_pred: The predicted probabilities or scores.
        title: A title for the plot.
        n_bootstrap_samples: Number of bootstrap samples for confidence intervals.

    Returns:
        A tuple containing the AUPRC and the confidence interval (if calculated).
    """
    assert len(y_true) == len(y_pred), "Length of y_true and y_pred does not match."
    conf_range = None
    l = None
    h = None

    if n_bootstrap_samples:
        rng = np.random.default_rng()
        interp_recall = np.linspace(0, 1, num=1000)
        interp_prcs = np.full((n_bootstrap_samples, len(interp_recall)), np.nan)
        bootstrap_auprcs = []  # Initialize to collect AUPRC values for bootstrapped samples

        for i in trange(n_bootstrap_samples, 
                        desc="Bootstrapping PRC curves", leave=False):
            sample_idxs = rng.choice(len(y_true), len(y_true), replace=True)
            sample_y_true = y_true[sample_idxs]
            sample_y_pred = y_pred[sample_idxs]
            
            if len(np.unique(sample_y_true)) != 2 or not (0 in sample_y_true and 1 in sample_y_true):
                continue
            
            precision, recall, _ = precision_recall_curve(sample_y_true, sample_y_pred)
            # Create an interpolation function with decreasing values
            interp_func = interp1d(recall[::-1], precision[::-1], 
                                   kind='linear', fill_value=np.nan, 
                                   bounds_error=False)
            interp_prc = interp_func(interp_recall)
            interp_prcs[i] = interp_prc
            bootstrapped_auprc = auc(interp_recall, interp_prc)
            bootstrap_auprcs.append(bootstrapped_auprc)

    # Calculate the confidence intervals for each threshold
    lower = np.nanpercentile(interp_prcs, 2.5, axis=0)
    upper = np.nanpercentile(interp_prcs, 97.5, axis=0)
    h=np.quantile(bootstrap_auprcs, 0.975)
    l=np.quantile(bootstrap_auprcs, 0.025)
    conf_range = (
        np.quantile(bootstrap_auprcs, 0.975) - np.quantile(bootstrap_auprcs, 0.025)
        ) / 2

    # Calculate the standard AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    auprc_value = auc(recall, precision)

    # Plot the AUPRC curve with confidence intervals
    ax.plot(recall, precision, label=f'PRC = {auprc_value:.2f}')
    ax.fill_between(interp_recall, lower, upper, alpha=0.5)

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(title + f'\nAUPRC = {auprc_value:.2f} [{l:.2f}-{h:.2f}]')
    ax.legend()

    return auprc_value, conf_range


def plot_single_decorated_prc_curve(
    ax: plt.Axes,
    y_true: npt.NDArray[np.bool_],
    y_pred: npt.NDArray[np.float_],
    *,
    title: Optional[str] = None,
    n_bootstrap_samples: Optional[int] = None
) -> plt.Axes:
    """Plots a single ROC curve.

    Args:
        ax:  Axis to plot to.
        y_true:  A sequence of ground truths.
        y_pred:  A sequence of predictions.
        title:  Title of the plot.
    """
    plot_bootstrapped_pr_curve(
        ax,
        y_true,
        y_pred,
        title=title,
        n_bootstrap_samples=n_bootstrap_samples
    )
    style_prc(ax, baseline=y_true.sum() / len(y_true))


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

    style_prc(ax, baseline=y_true.sum()/len(y_true))

    prc = average_precision_score(y_true, y_pred)
    if title:
        ax.set_title(f'{title}\n(PRC = {prc:0.2f})')
    else:
        ax.set_title(f'PRC = {prc:0.2f}')

    return prc


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
    for t, p, prc in tpas:
        precision, recall, _ = precision_recall_curve(t, p)
        ax.plot(recall, precision, label=f'PRC = {prc:0.2f}')

    # style plot
    all_samples = np.concatenate(y_trues)
    baseline = all_samples.sum()/len(all_samples)
    style_prc(ax, baseline=baseline)
    ax.legend()

    # calculate confidence intervals and print title
    aucs = [x.auc for x in tpas]
    l, h = st.t.interval(
        0.95, len(aucs)-1, loc=np.mean(aucs), scale=st.sem(aucs))
    
    # limit conf bounds to [0,1] in case of low sample numbers
    l = max(0, l)
    h = min(1, h)
    
    # conf_range = (h-l)/2
    auc_str = f'PRC = {np.mean(aucs):0.2f} [{l:0.2f}-{h:0.2f}]'

    if title:
        ax.set_title(f'{title}\n{auc_str}')
    else:
        ax.set_title(auc_str)

    return l, h


def plot_precision_recall_curves_(
        ax, pred_csvs, target_label: str, true_label: str, outpath
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
    if len(pred_dfs) == 1:
        plot_precision_recall_curve(ax, y_trues[0], y_preds[0], title=title)
    else:
        plot_precision_recall_curves(ax, y_trues, y_preds, title=title)

    # fig.savefig(Path(outpath)/f'AUPRC_{target_label}={true_label}.svg')
    # plt.close(fig)


if __name__ == '__main__':
    from fire import Fire
    Fire(plot_precision_recall_curves_)
