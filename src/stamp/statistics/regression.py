"""Calculate statistics for deployments on regression targets."""

from collections.abc import Sequence
from pathlib import Path
from typing import Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck"
__license__ = "MIT"


_score_labels = [
    "r2_score",
    "pearson_r",
    "pearson_p",
    "mae",
    "rmse",
    "count",
]


def _regression(preds_df: pd.DataFrame, target_label: str) -> pd.Series:
    """Compute regression metrics for one prediction table."""
    y_true = np.asarray(preds_df[target_label], dtype=float)
    y_pred = np.asarray(preds_df["pred"], dtype=float)

    r2 = float(r2_score(y_true, y_pred))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        pearson_r, pearson_p = np.nan, np.nan
    else:
        r_result = st.pearsonr(y_true, y_pred)
        r_result = cast(Tuple[float, float], r_result)
        pearson_r: float = float(r_result[0])
        pearson_p: float = float(r_result[1])
    return pd.Series(
        {
            "r2_score": r2,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "mae": mae,
            "rmse": rmse,
            "count": int(len(y_true)),
        }
    )


def regression_aggregated_(
    *,
    preds_csvs: Sequence[Path],
    outpath: Path,
    ground_truth_label: str,
) -> None:
    """Calculate regression statistics and generate per-fold plots.

    Args:
        preds_csvs:  CSV files containing columns [ground_truth_label, "pred"]
        outpath:  Path to save outputs to.
        ground_truth_label:  Column name of ground truth.
    """
    stats = {}
    for fold, p in enumerate(preds_csvs):
        df = pd.read_csv(p)
        df = df.dropna(subset=[ground_truth_label, "pred"])
        fold_name = Path(p).stem

        # compute and store stats
        stats[fold_name] = _regression(df, ground_truth_label)

        # plot
        fig, ax = plt.subplots(figsize=(3.2, 3.2), dpi=300)
        y_true = df[ground_truth_label].astype(float)
        y_pred = df["pred"].astype(float)

        # regression line
        slope, intercept, r_value, p_value, std_err = st.linregress(y_true, y_pred)
        x_vals = np.linspace(y_true.min(), y_true.max(), 100)
        y_line = intercept + slope * x_vals  # type: ignore
        ax.scatter(y_true, y_pred, color="black", s=15)
        ax.plot(x_vals, y_line, color="royalblue", linewidth=1.5)
        ax.fill_between(
            x_vals,
            y_line - std_err,
            y_line + std_err,
            color="royalblue",
            alpha=0.2,
        )

        ax.set_xlabel(f"{ground_truth_label}")
        ax.set_ylabel("Prediction")
        ax.set_title(f"{fold_name}")

        # annotate stats
        ax.text(
            0.05,
            0.95,
            (
                rf"$R^2$={stats[fold_name]['r2_score']:.2f} | "
                rf"Pearson R={stats[fold_name]['pearson_r']:.2f}"
                "\n"
                rf"$p$={stats[fold_name]['pearson_p']:.1e}"
            ),
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=8,
        )

        fig.tight_layout()
        (outpath / "plots").mkdir(parents=True, exist_ok=True)
        fig.savefig(outpath / "plots" / f"fold_{fold_name}_scatter.svg")
        plt.close(fig)

    # Save individual stats and aggregate
    stats_df = pd.DataFrame(stats).transpose()
    stats_df.to_csv(outpath / f"{ground_truth_label}_regression-stats_individual.csv")

    mean = stats_df.mean(numeric_only=True)
    sem = stats_df.sem(numeric_only=True)
    lower, upper = st.t.interval(0.95, len(stats_df) - 1, loc=mean, scale=sem)
    agg = pd.DataFrame({"mean": mean, "95%_low": lower, "95%_high": upper})
    agg.to_csv(outpath / f"{ground_truth_label}_regression-stats_aggregated.csv")
