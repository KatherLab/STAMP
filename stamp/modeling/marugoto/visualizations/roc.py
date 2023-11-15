import logging
from collections import namedtuple
from typing import List, Mapping, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
import scipy.stats as st
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Colormap
from sklearn.metrics import roc_auc_score, roc_curve
from tqdm import trange

all = [
    "plot_roc_curve",
    "plot_roc_curves",
    "plot_rocs_for_subtypes",
]


def plot_single_decorated_roc_curve(
    ax: plt.Axes,
    y_true: npt.NDArray[np.bool_],
    y_pred: npt.NDArray[np.float_],
    *,
    title: Optional[str] = None,
    n_bootstrap_samples: Optional[int] = None,
    threshold_cmap: Optional[Colormap] = None,
) -> plt.Axes:
    """Plots a single ROC curve.

    Args:
        ax:  Axis to plot to.
        y_true:  A sequence of ground truths.
        y_pred:  A sequence of predictions.
        title:  Title of the plot.
    """
    auc, l, h = plot_bootstrapped_roc_curve(
        ax,
        y_true,
        y_pred,
        label=title,
        n_bootstrap_samples=n_bootstrap_samples,
        threshold_cmap=threshold_cmap,
    )
    style_auc(ax)
    ax.set_title(title+f'\nAUROC = {auc:.2f} [{l:.2f}-{h:.2f}]')


def auc_str(auc: float, l: Optional[float], h: Optional[float]) -> str:
    return f"AUC = {auc:0.2f} [{l:0.2f}-{h:0.2f}]"


def style_auc(ax: plt.Axes) -> None:
    ax.plot([0, 1], [0, 1], "r--")
    ax.set_aspect("equal")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")


TPA = namedtuple("TPA", ["true", "pred", "auc"])


def plot_multiple_decorated_roc_curves(
    ax: plt.Axes,
    y_trues: Sequence[npt.NDArray[np.bool_]],
    y_scores: Sequence[npt.NDArray[np.float_]],
    *,
    title: Optional[str] = None,
    n_bootstrap_samples: Optional[int] = None,
):
    """Plots a family of ROC curves.

    Args:
        ax:  Axis to plot to.
        y_trues:  Sequence of ground truth lists.
        y_scores:  Sequence of prediction lists.
        title:  Title of the plot.
    """
    # sort trues, preds, AUCs by AUC
    tpas = [TPA(t, p, roc_auc_score(t, p)) for t, p in zip(y_trues, y_scores)]
    tpas = sorted(tpas, key=lambda x: x.auc, reverse=True)

    # plot rocs
    for t, p, auc in tpas:
        auc, l, h = plot_bootstrapped_roc_curve(
            ax, t, p, label="AUC = {ci}", n_bootstrap_samples=n_bootstrap_samples
        )

    # style plot
    style_auc(ax)

    # calculate confidence intervals and print title
    aucs = [x.auc for x in tpas]
    mean_auc=np.mean(aucs)
    if not n_bootstrap_samples:    
        l, h = st.t.interval(0.95, len(aucs) - 1, loc=np.mean(aucs), scale=st.sem(aucs))

    # limit conf bounds to [0,1] in case of low sample numbers
    l = max(0, l)
    h = min(1, h)

    if title:
        ax.set_title(f"{title}\n {auc_str(mean_auc, l, h)}")
    else:
        ax.set_title(auc_str(mean_auc, l, h))


def split_preds_into_groups(
    preds_df: pd.DataFrame,
    *,
    clini_df: pd.DataFrame,
    target_label: str,
    true_label: str,
    subgroup_label: str,
) -> Mapping[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float_]]]:
    """Splits predictions into a mapping `subgroup_name -> (y_true, y_pred)."""
    groups = {}
    for subgroup, subgroup_patients in clini_df.PATIENT.groupby(
        clini_df[subgroup_label]
    ):
        subgroup_preds = preds_df[preds_df.PATIENT.isin(subgroup_patients)]
        y_true = subgroup_preds[target_label] == true_label
        y_pred = pd.to_numeric(subgroup_preds[f"{target_label}_{true_label}"])
        groups[subgroup] = (y_true.values, y_pred.values)

    return groups


def plot_decorated_rocs_for_subtypes(
    ax: plt.Axes,
    groups: Mapping[str, Tuple[npt.NDArray[np.bool_], npt.NDArray[np.float_]]],
    *,
    target_label: str,
    true_label: str,
    subgroup_label: str,
    subgroups: Optional[Sequence[str]] = None,
    n_bootstrap_samples: Optional[int] = None,
) -> None:
    """Plots a ROC for multiple groups."""
    tpas: List[Tuple[str, TPA]] = []
    for subgroup, (y_true, y_pred) in groups.items():
        if subgroups and subgroup not in subgroups:
            continue

        if len(np.unique(y_true)) <= 1:
            logging.warn(
                f"subgroup {subgroup} does only have samples of one class... skipping"
            )
            continue

        tpas.append((subgroup, TPA(y_true, y_pred, roc_auc_score(y_true, y_pred))))

    # sort trues, preds, AUCs by AUC
    tpas = sorted(tpas, key=lambda x: x[1].auc, reverse=True)

    # plot rocs
    for subgroup, (t, s, _) in tpas:
        plot_bootstrapped_roc_curve(
            ax=ax,
            y_true=t,
            y_score=s,
            label=f"{target_label} for {subgroup} (AUC = {{ci}})",
            n_bootstrap_samples=n_bootstrap_samples,
        )

    # style plot
    style_auc(ax)
    ax.legend(loc="lower right")
    ax.set_title(f"{target_label} = {true_label} Subgrouped by {subgroup_label}")


def plot_bootstrapped_roc_curve(
    ax: plt.Axes,
    y_true: npt.NDArray[np.bool_],
    y_score: npt.NDArray[np.float_],
    label: Optional[str],
    n_bootstrap_samples: Optional[int] = None,
    threshold_cmap: Optional[Colormap] = None,
):
    """Plots a roc curve with bootstrap interval.

    Args:
        ax:  The axes to plot onto.
        y_true:  The ground truths.
        y_score:  The predictions corresponding to the ground truths.
        label:  A label to attach to the curve.
            The string `{ci}` will be replaced with the AUC
            and the range of the confidence interval.
    """
    assert len(y_true) == len(y_score), "length of truths and scores does not match."
    conf_range = None
    l = None
    h = None
    if n_bootstrap_samples:
        # draw some confidence intervals based on bootstrapping
        # sample repeatedly (with replacement) from our data points,
        # interpolate along the resulting ROC curves
        # and then sample the bottom 0.025 / top 0.975 quantile point
        # for each sampled fpr-position
        rng = np.random.default_rng()
        interp_rocs = []
        interp_fpr = np.linspace(0, 1, num=1000)
        bootstrap_aucs = []
        for _ in trange(
            n_bootstrap_samples, desc="Bootstrapping ROC curves", leave=False
        ):
            sample_idxs = rng.choice(len(y_true), len(y_true))
            sample_y_true = y_true[sample_idxs]
            sample_y_score = y_score[sample_idxs]
            if len(np.unique(sample_y_true)) != 2:
                continue
            fpr, tpr, thresh = roc_curve(sample_y_true, sample_y_score)
            interp_rocs.append(np.interp(interp_fpr, fpr, tpr))
            bootstrap_aucs.append(roc_auc_score(sample_y_true, sample_y_score))

        lower = np.quantile(interp_rocs, 0.025, axis=0)
        upper = np.quantile(interp_rocs, 0.975, axis=0)
        ax.fill_between(interp_fpr, lower, upper, alpha=0.5)
        h=np.quantile(bootstrap_aucs, 0.975)
        l=np.quantile(bootstrap_aucs, 0.025)

    fpr, tpr, thresh = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    # ci_str = f"${auc:0.2f} [{l:0.2f}-{h:0.2f}]$" if conf_range else f"${auc:0.2f}$"
    # ax.plot(fpr, tpr, label=label.format(ci=ci_str) if label else "")
    plot_curve(
        ax,
        fpr,
        tpr,
        np.clip(thresh, 0, 1),
        label=f"AUC = {auc:0.2f}",
        threshold_cmap=threshold_cmap
    )
    return auc, l, h


def plot_curve(
    ax: plt.Axes,
    x: npt.NDArray[np.float_],
    y: npt.NDArray[np.float_],
    thresh: npt.NDArray[np.float_],
    *,
    label: Optional[str],
    threshold_cmap: Optional[Colormap] = None,
) -> None:
    if threshold_cmap is not None:
        points = np.array([x, y]).transpose().reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=threshold_cmap, label=label)
        lc.set_array(thresh)
        ax.add_collection(lc)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
    else:
        ax.plot(x, y, label=label)