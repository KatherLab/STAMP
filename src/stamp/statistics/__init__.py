from collections.abc import Sequence
from pathlib import Path
from typing import NewType

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict

from stamp.statistics.categorical import categorical_aggregated_
from stamp.statistics.prc import (
    plot_multiple_decorated_precision_recall_curves,
    plot_single_decorated_precision_recall_curve,
)
from stamp.statistics.regression import regression_aggregated_
from stamp.statistics.roc import (
    plot_multiple_decorated_roc_curves,
    plot_single_decorated_roc_curve,
)
from stamp.types import PandasLabel

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2024 Marko van Treeck"
__license__ = "MIT"


def _read_table(file: Path, **kwargs) -> pd.DataFrame:
    """Loads a dataframe from a file."""
    if isinstance(file, Path) and file.suffix == ".xlsx":
        return pd.read_excel(file, **kwargs)
    else:
        return pd.read_csv(file, **kwargs)


class StatsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_dir: Path

    pred_csvs: list[Path]
    ground_truth_label: PandasLabel
    true_class: str | None = None
    pred_label: str | None = None


_Inches = NewType("_Inches", float)


def compute_stats_(
    *,
    output_dir: Path,
    pred_csvs: Sequence[Path],
    ground_truth_label: PandasLabel,
    true_class: str | None = None,  # None means regression,
    pred_label: str | None = None,
) -> None:
    n_bootstrap_samples = 1000
    figure_width = _Inches(3.8)
    roc_curve_figure_aspect_ratio = 1.08
    threshold_cmap = None

    if true_class is not None:
        # === Classification branch ===
        preds_dfs = [
            _read_table(
                p,
                usecols=[ground_truth_label, f"{ground_truth_label}_{true_class}"],
                dtype={
                    ground_truth_label: str,
                    f"{ground_truth_label}_{true_class}": float,
                },
            )
            for p in pred_csvs
        ]

        y_trues = [np.array(df[ground_truth_label] == true_class) for df in preds_dfs]
        y_preds = [
            np.array(df[f"{ground_truth_label}_{true_class}"].values)
            for df in preds_dfs
        ]

        fig, ax = plt.subplots(
            figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
            dpi=300,
        )

        if len(preds_dfs) == 1:
            plot_single_decorated_roc_curve(
                ax=ax,
                y_true=y_trues[0],
                y_score=y_preds[0],
                title=f"{ground_truth_label} = {true_class}",
                n_bootstrap_samples=n_bootstrap_samples,
                threshold_cmap=threshold_cmap,
            )

        else:
            plot_multiple_decorated_roc_curves(
                ax=ax,
                y_trues=y_trues,
                y_scores=y_preds,
                title=f"{ground_truth_label} = {true_class}",
                n_bootstrap_samples=None,
            )

        fig.tight_layout()
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)

        fig.savefig(output_dir / f"roc-curve_{ground_truth_label}={true_class}.svg")
        plt.close(fig)

        fig, ax = plt.subplots(
            figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
            dpi=300,
        )
        if len(preds_dfs) == 1:
            plot_single_decorated_precision_recall_curve(
                ax=ax,
                y_true=y_trues[0],
                y_score=y_preds[0],
                title=f"{ground_truth_label} = {true_class}",
                n_bootstrap_samples=n_bootstrap_samples,
            )

        else:
            plot_multiple_decorated_precision_recall_curves(
                ax=ax,
                y_trues=y_trues,
                y_scores=y_preds,
                title=f"{ground_truth_label} = {true_class}",
            )

        fig.tight_layout()
        fig.savefig(output_dir / f"pr-curve_{ground_truth_label}={true_class}.svg")
        plt.close(fig)

        categorical_aggregated_(
            preds_csvs=pred_csvs,
            ground_truth_label=ground_truth_label,
            outpath=output_dir,
        )

    else:
        # === Regression branch ===
        if pred_label is None:
            raise ValueError("pred_label must be set for regression mode")

        preds_dfs = [
            pd.read_csv(p, usecols=[ground_truth_label, pred_label], dtype=float)
            for p in pred_csvs
        ]

        y_trues = [df[ground_truth_label].to_numpy() for df in preds_dfs]
        y_preds = [df[pred_label].to_numpy() for df in preds_dfs]

        # binarize at median of all ground truth values
        all_true = np.concatenate(y_trues)
        median = np.median(all_true)

        y_trues_bin = [(y >= median).astype(bool) for y in y_trues]

        # --- ROC ---
        fig, ax = plt.subplots(
            figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
            dpi=300,
        )
        if len(preds_dfs) == 1:
            plot_single_decorated_roc_curve(
                ax=ax,
                y_true=y_trues_bin[0],
                y_score=y_preds[0],
                title=f"{ground_truth_label} (median split)",
                n_bootstrap_samples=n_bootstrap_samples,
                threshold_cmap=threshold_cmap,
            )
        else:
            plot_multiple_decorated_roc_curves(
                ax=ax,
                y_trues=y_trues_bin,
                y_scores=y_preds,
                title=f"{ground_truth_label} (median split)",
            )
        fig.tight_layout()
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"roc-curve_{ground_truth_label}_median-split.svg")
        plt.close(fig)

        # --- PR ---
        fig, ax = plt.subplots(
            figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
            dpi=300,
        )
        if len(preds_dfs) == 1:
            plot_single_decorated_precision_recall_curve(
                ax=ax,
                y_true=y_trues_bin[0],
                y_score=y_preds[0],
                title=f"{ground_truth_label} (median split)",
                n_bootstrap_samples=n_bootstrap_samples,
            )
        else:
            plot_multiple_decorated_precision_recall_curves(
                ax=ax,
                y_trues=y_trues_bin,
                y_scores=y_preds,
                title=f"{ground_truth_label} (median split)",
            )
        fig.tight_layout()
        fig.savefig(output_dir / f"pr-curve_{ground_truth_label}_median-split.svg")
        plt.close(fig)

        # Then run regression_aggregated_ for numeric stats
        regression_aggregated_(
            preds_csvs=pred_csvs,
            ground_truth_label=ground_truth_label,
            pred_label=pred_label,
            outpath=output_dir,
        )
