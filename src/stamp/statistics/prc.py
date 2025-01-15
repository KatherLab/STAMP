# %%
from collections.abc import Sequence
from typing import NamedTuple, TypeAlias

import numpy as np
import numpy.typing as npt
import scipy.stats as st
from jaxtyping import Bool, Float
from matplotlib.axes import Axes
from scipy.interpolate import interp1d
from sklearn.metrics import (
    auc,
    average_precision_score,
    precision_recall_curve,
)
from tqdm import trange

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2024 Marko van Treeck"
__license__ = "MIT"


_Auprc: TypeAlias = float
_Auprc95CILower: TypeAlias = float
_Auprc95CIUpper: TypeAlias = float


def _plot_bootstrapped_pr_curve(
    *,
    ax: Axes,
    y_true: Bool[np.ndarray, "sample"],  # noqa: F821
    y_score: Float[np.ndarray, "sample"],  # noqa: F821
    n_bootstrap_samples: int,
) -> tuple[_Auprc, _Auprc95CILower, _Auprc95CIUpper]:
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
    rng = np.random.default_rng()
    interp_recall = np.linspace(0, 1, num=1000)
    interp_prcs = np.full((n_bootstrap_samples, len(interp_recall)), np.nan)
    bootstrap_auprcs = []  # Initialize to collect AUPRC values for bootstrapped samples

    for i in trange(n_bootstrap_samples, desc="Bootstrapping PRC curves", leave=False):
        sample_idxs = rng.choice(len(y_true), len(y_true), replace=True)
        sample_y_true = y_true[sample_idxs]
        sample_y_pred = y_score[sample_idxs]

        if len(np.unique(sample_y_true)) != 2 or not (
            0 in sample_y_true and 1 in sample_y_true
        ):
            continue

        precision, recall, _ = precision_recall_curve(sample_y_true, sample_y_pred)
        # Create an interpolation function with decreasing values
        interp_func = interp1d(
            recall[::-1],
            precision[::-1],
            kind="linear",
            fill_value=np.nan,
            bounds_error=False,
        )
        interp_prc = interp_func(interp_recall)
        interp_prcs[i] = interp_prc
        bootstrapped_auprc = auc(interp_recall, interp_prc)
        bootstrap_auprcs.append(bootstrapped_auprc)

    # Calculate the confidence intervals for each threshold
    prc_lower: Float[np.ndarray, "fpr"]  # noqa: F821
    prc_upper: Float[np.ndarray, "fpr"]  # noqa: F821
    prc_lower, prc_upper = np.quantile(interp_prcs, [0.025, 0.975], axis=0)
    ax.fill_between(interp_recall, prc_lower, prc_upper, alpha=0.5)

    auprc_lower: _Auprc95CILower
    auprc_upper: _Auprc95CIUpper
    auprc_lower, auprc_upper = np.quantile(bootstrap_auprcs, [0.025, 0.975])

    # Calculate the standard AUPRC
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    auprc = float(auc(recall, precision))

    # Plot PRC itself
    ax.plot(recall, precision, label=f"PRC = {auprc:.2f}")

    return auprc, auprc_lower, auprc_upper


def plot_single_decorated_precision_recall_curve(
    *,
    ax: Axes,
    y_true: Bool[np.ndarray, "sample"],  # noqa: F821
    y_score: Float[np.ndarray, "sample"],  # noqa: F821
    title: str,
    n_bootstrap_samples: int | None,
) -> None:
    """Plots a single ROC curve.

    Args:
        ax:  Axis to plot to.
        y_true:  A sequence of ground truths.
        y_pred:  A sequence of predictions.
        title:  Title of the plot.
    """
    if n_bootstrap_samples is not None:
        auprc, lower, upper = _plot_bootstrapped_pr_curve(
            ax=ax,
            y_true=y_true,
            y_score=y_score,
            n_bootstrap_samples=n_bootstrap_samples,
        )
        ax.set_title(f"{title}\nAUPRC = {auprc:.2f} [{lower:.2f}-{upper:.2f}]")
    else:
        raise NotImplementedError()

    ax.plot([0, 1], [0, 1], "r--", alpha=0)
    ax.set_aspect("equal")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    baseline = y_true.sum() / len(y_true)
    ax.plot([0, 1], [baseline, baseline], "r--")


class _TPA(NamedTuple):
    trues: Bool[np.ndarray, "sample"]  # noqa: F821
    scores: Float[np.ndarray, "sample"]  # noqa: F821
    auc: float


def plot_multiple_decorated_precision_recall_curves(
    *,
    ax: Axes,
    y_trues: Sequence[npt.NDArray[np.bool_]],
    y_scores: Sequence[npt.NDArray[np.float64]],
    title: str | None = None,
) -> tuple[float, float]:
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
    tpas = [
        _TPA(t, p, float(average_precision_score(t, p)))
        for t, p in zip(y_trues, y_scores)
    ]
    tpas = sorted(tpas, key=lambda x: x.auc, reverse=True)

    # plot precision_recalls
    for t, p, prc in tpas:
        precision, recall, _ = precision_recall_curve(t, p)
        ax.plot(recall, precision, label=f"PRC = {prc:0.2f}")

    # style plot
    all_samples = np.concatenate(y_trues)
    ax.plot([0, 1], [0, 1], "r--", alpha=0)
    ax.set_aspect("equal")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    baseline = all_samples.sum() / len(all_samples)
    ax.plot([0, 1], [baseline, baseline], "r--")
    ax.legend()

    # calculate confidence intervals and print title
    aucs = [x.auc for x in tpas]
    lower, upper = st.t.interval(
        0.95, len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs)
    )

    # limit conf bounds to [0,1] in case of low sample numbers
    lower = max(0, lower)
    upper = min(1, upper)

    # conf_range = (h-l)/2
    auc_str = f"PRC = {np.mean(aucs):0.2f} [{lower:0.2f}-{upper:0.2f}]"

    if title:
        ax.set_title(f"{title}\n{auc_str}")
    else:
        ax.set_title(auc_str)

    return lower, upper
