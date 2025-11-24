"""Survival statistics: C-index, KM curves, log-rank p-value."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.plotting import add_at_risk_counts
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index


def _comparable_pairs_count(times: np.ndarray, events: np.ndarray) -> int:
    """Number of comparable (event,censored) pairs."""
    t_i = times[:, None]
    t_j = times[None, :]
    e_i = events[:, None]
    return int(((t_i < t_j) & (e_i == 1)).sum())


def _cindex(
    time: np.ndarray,
    event: np.ndarray,
    risk: np.ndarray,  # will be flipped in function
) -> tuple[float, int]:
    """Compute C-index using Lifelines convention:
    higher risk → shorter survival (worse outcome).
    """
    c_index = float(concordance_index(time, -risk, event))
    n_pairs = _comparable_pairs_count(time, event)
    return c_index, n_pairs


def _survival_stats_for_csv(
    df: pd.DataFrame,
    *,
    time_label: str,
    status_label: str,
    risk_label: str | None = None,
    cut_off: float | None = None,  # will be flipped in function
) -> pd.Series:
    """Compute C-index and log-rank p for one CSV."""
    if risk_label is None:
        risk_label = "pred_score"

    # --- Clean NaNs and invalid events before computing stats ---
    df = df.dropna(subset=[time_label, status_label, risk_label]).copy()
    df = df[df[status_label].isin([0, 1])]
    if len(df) == 0:
        raise ValueError("No valid rows after dropping NaN or invalid survival data.")

    time = np.asarray(df[time_label], dtype=float)
    event = np.asarray(df[status_label], dtype=int)
    risk = np.asarray(df[risk_label], dtype=float)

    # --- Concordance index ---
    c_index, n_pairs = _cindex(time, event, risk)

    # --- Log-rank test (median split) ---
    median_risk = float(-cut_off) if cut_off is not None else float(np.nanmedian(risk))
    low_mask = risk >= median_risk
    high_mask = risk < median_risk
    if low_mask.sum() > 0 and high_mask.sum() > 0:
        res = logrank_test(
            time[low_mask],
            time[high_mask],
            event_observed_A=event[low_mask],
            event_observed_B=event[high_mask],
        )
        p_logrank = float(res.p_value)
    else:
        p_logrank = np.nan

    return pd.Series(
        {
            "c_index": c_index,
            "logrank_p": p_logrank,
            "count": int(len(df)),
            "events": int(event.sum()),
            "censored": int((event == 0).sum()),
            "comparable_pairs": n_pairs,
            "threshold": median_risk,
        }
    )


def _plot_km(
    df: pd.DataFrame,
    *,
    fold_name: str,
    time_label: str,
    status_label: str,
    risk_label: str | None = None,
    cut_off: float | None = None,
    outdir: Path,
) -> None:
    """Kaplan–Meier curve (median split) with log-rank p and C-index annotation."""
    if risk_label is None:
        risk_label = "pred_score"

    # --- Clean NaNs and invalid entries ---
    df = df.replace(["NaN", "nan", "None", "Inf", "inf"], np.nan)
    df = df.dropna(subset=[time_label, status_label, risk_label]).copy()
    df = df[df[status_label].isin([0, 1])]

    if len(df) == 0:
        raise ValueError(f"No valid rows to plot for {fold_name}.")

    time = np.asarray(df[time_label], dtype=float)
    event = np.asarray(df[status_label], dtype=int)
    risk = np.asarray(df[risk_label], dtype=float)

    # --- split groups ---
    median_risk = float(cut_off) if cut_off is not None else np.nanmedian(risk)
    low_mask = risk >= median_risk
    high_mask = risk < median_risk

    low_df = df[low_mask]
    high_df = df[high_mask]

    kmf_low = KaplanMeierFitter()
    kmf_high = KaplanMeierFitter()

    fig, ax = plt.subplots(figsize=(8, 6))
    if len(low_df) > 0:
        kmf_low.fit(
            low_df[time_label], event_observed=low_df[status_label], label="Low risk"
        )
        kmf_low.plot_survival_function(ax=ax, ci_show=False, color="blue")
    if len(high_df) > 0:
        kmf_high.fit(
            high_df[time_label], event_observed=high_df[status_label], label="High risk"
        )
        kmf_high.plot_survival_function(ax=ax, ci_show=False, color="red")

    add_at_risk_counts(kmf_low, kmf_high, ax=ax)

    # --- log-rank and c-index ---
    res = logrank_test(
        low_df[time_label],
        high_df[time_label],
        event_observed_A=low_df[status_label],
        event_observed_B=high_df[status_label],
    )
    logrank_p = float(res.p_value)
    c_used, used, *_ = _cindex(time, event, risk)

    ax.text(
        0.6,
        0.08,
        f"Log-rank p = {logrank_p:.4e}\nC-index = {c_used:.3f}\nCut-off = {median_risk:.3f}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"),
    )

    ax.set_title(
        f"{fold_name} – Kaplan–Meier Survival Curve", fontsize=13, weight="bold"
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Survival probability")
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_ylim(0, 1)
    plt.tight_layout()

    (outdir / "plots").mkdir(parents=True, exist_ok=True)
    outpath = outdir / "plots" / f"fold_{fold_name}_km_curve.svg"
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
