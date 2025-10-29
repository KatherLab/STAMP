from collections.abc import Sequence
from typing import NamedTuple, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import scipy.stats as st
from jaxtyping import Bool, Float
from matplotlib.axes import Axes
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import trange

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2024 Marko van Treeck"
__license__ = "MIT"


_Auc: TypeAlias = float
_Auc95CILower: TypeAlias = float
_Auc95CIUpper: TypeAlias = float


def plot_single_decorated_roc_curve(
    *,
    ax: Axes,
    y_true: Bool[np.ndarray, "sample"],  # noqa: F821
    y_score: Float[np.ndarray, "sample"],  # noqa: F821
    title: str,
    n_bootstrap_samples: int | None,
    threshold_cmap: Colormap | None,
) -> None:
    """Plots a single ROC curve.

    Args:
        ax:  Axis to plot to.
        y_true:  A sequence of ground truths.
        y_pred:  A sequence of predictions.
        title:  Title of the plot.
    """
    if n_bootstrap_samples is not None:
        auc, lower, upper = _plot_bootstrapped_roc_curve(
            ax=ax,
            y_true=y_true,
            y_score=y_score,
            n_bootstrap_samples=n_bootstrap_samples,
            threshold_cmap=threshold_cmap,
        )
        ax.set_title(f"{title}\nAUROC = {auc:.2f} [{lower:.2f}-{upper:.2f}]")
    else:
        fpr, tpr, thresh = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)

        _plot_curve(
            ax=ax,
            x=fpr,
            y=tpr,
            thresh=np.clip(thresh, 0.0, 1.0),
            label=f"AUC = {auc:0.2f}",
            threshold_cmap=threshold_cmap,
        )

        ax.set_title(f"{title}\nAUROC = {auc:.2f}")

    ax.plot([0, 1], [0, 1], "r--")
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")


def _auc_str(auc: float, lower: float, upper: float) -> str:
    return f"AUC = {auc:0.2f} [{lower:0.2f}-{upper:0.2f}]"


class _TPA(NamedTuple):
    trues: Bool[np.ndarray, "sample"]  # noqa: F821
    scores: Float[np.ndarray, "sample"]  # noqa: F821
    auc: float


def plot_multiple_decorated_roc_curves(
    ax: Axes,
    y_trues: Sequence[npt.NDArray[np.bool_]],
    y_scores: Sequence[npt.NDArray[np.float64]],
    *,
    title: str | None = None,
    n_bootstrap_samples: int | None = None,
) -> None:
    """Plots a family of ROC curves.

    Args:
        ax:  Axis to plot to.
        y_trues:  Sequence of ground truth lists.
        y_scores:  Sequence of prediction lists.
        title:  Title of the plot.
    """
    # sort trues, preds, AUCs by AUC
    tpas = [_TPA(t, p, float(roc_auc_score(t, p))) for t, p in zip(y_trues, y_scores)]
    tpas = sorted(tpas, key=lambda x: x.auc, reverse=True)

    lower, upper = None, None
    # plot rocs
    if n_bootstrap_samples is not None:
        for t, p, auc in tpas:
            _, lower, upper = _plot_bootstrapped_roc_curve(
                ax=ax,
                y_true=t,
                y_score=p,
                n_bootstrap_samples=n_bootstrap_samples,
                threshold_cmap=None,
            )
    else:
        for t, p, auc in tpas:
            fpr, tpr, thresh = roc_curve(t, p)

            _plot_curve(
                ax=ax,
                x=fpr,
                y=tpr,
                thresh=np.clip(thresh, 0.0, 1.0),
                label=f"AUC = {auc:0.2f}",
                threshold_cmap=None,
            )

    # style plot
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")

    # calculate confidence intervals and print title
    aucs = [x.auc for x in tpas]
    mean_auc = np.mean(aucs).item()

    if n_bootstrap_samples is None:
        sem_val = st.sem(aucs)
        if len(aucs) < 2 or not np.isfinite(sem_val) or sem_val == 0.0:
            # Not enough or invalid variance → CI collapses to mean
            lower, upper = cast(
                tuple[_Auc95CILower, _Auc95CIUpper],
                (mean_auc, mean_auc),
            )
        else:
            lower, upper = cast(
                tuple[_Auc95CILower, _Auc95CIUpper],
                st.t.interval(0.95, len(aucs) - 1, loc=mean_auc, scale=sem_val),
            )
    assert lower is not None
    assert upper is not None

    # limit conf bounds to [0,1] in case of low sample numbers
    lower, upper = max(0.0, lower), min(1.0, upper)

    if title:
        ax.set_title(f"{title}\n{_auc_str(mean_auc, lower, upper)}")
    else:
        ax.set_title(_auc_str(mean_auc, lower, upper))


def _plot_bootstrapped_roc_curve(
    *,
    ax: Axes,
    y_true: Bool[np.ndarray, "sample"],  # noqa: F821
    y_score: Float[np.ndarray, "sample"],  # noqa: F821
    n_bootstrap_samples: int,
    threshold_cmap: Colormap | None,
) -> tuple[_Auc, _Auc95CILower, _Auc95CIUpper]:
    """Plots a roc curve with bootstrap interval.

    Args:
        ax:  The axes to plot onto.

        y_true:  The ground truths.
        y_score:  The predictions corresponding to the ground truths.
    """
    # draw some confidence intervals based on bootstrapping
    # sample repeatedly (with replacement) from our data points,
    # interpolate along the resulting ROC curves
    # and then sample the bottom 0.025 / top 0.975 quantile point
    # for each sampled fpr-position
    rng = np.random.default_rng()
    interp_rocs = []
    interp_fpr = np.linspace(0, 1, num=1000)
    bootstrap_aucs: list[float] = []
    for _ in trange(n_bootstrap_samples, desc="Bootstrapping ROC curves", leave=False):
        sample_idxs = rng.choice(len(y_true), len(y_true))
        sample_y_true = y_true[sample_idxs]
        sample_y_score = y_score[sample_idxs]
        if len(np.unique(sample_y_true)) != 2:
            continue
        fpr, tpr, thresh = roc_curve(sample_y_true, sample_y_score)
        interp_rocs.append(np.interp(interp_fpr, fpr, tpr))
        bootstrap_aucs.append(float(roc_auc_score(sample_y_true, sample_y_score)))

    roc_lower, roc_upper = cast(
        tuple[
            Float[np.ndarray, "fpr"],  # noqa: F821
            Float[np.ndarray, "fpr"],  # noqa: F821
        ],
        np.quantile(interp_rocs, [0.025, 0.975], axis=0),
    )
    ax.fill_between(interp_fpr, roc_lower, roc_upper, alpha=0.5)

    auc_lower: _Auc95CILower
    auc_upper: _Auc95CIUpper
    auc_lower, auc_upper = np.quantile(bootstrap_aucs, [0.025, 0.975])

    fpr, tpr, thresh = roc_curve(y_true, y_score)
    auc = float(roc_auc_score(y_true, y_score))

    _plot_curve(
        ax=ax,
        x=fpr,
        y=tpr,
        thresh=np.clip(thresh, 0.0, 1.0),
        label=f"AUC = {auc:0.2f}",
        threshold_cmap=threshold_cmap,
    )
    return auc, auc_lower, auc_upper


def _plot_curve(
    *,
    ax: Axes,
    x: Float[np.ndarray, "sample"],  # noqa: F821
    y: Float[np.ndarray, "sample"],  # noqa: F821
    thresh: Float[np.ndarray, "sample"],  # noqa: F821
    label: str | None,
    threshold_cmap: Colormap | None,
) -> None:
    if threshold_cmap is not None:
        points = np.array([x, y]).transpose().reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(list(segments), cmap=threshold_cmap, label=label)
        lc.set_array(thresh)
        ax.add_collection(lc)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.plot(x, y, label=label)
