"""Statistics utilities (wrappers) for classification, regression and survival.

This module provides a small, stable wrapper `compute_stats_` that dispatches
to the task-specific statistic implementations found in the submodules.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import NewType

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, Field

from stamp.statistics.categorical import (
    categorical_aggregated_,
    categorical_aggregated_multitarget_,
)
from stamp.statistics.prc import (
    plot_multiple_decorated_precision_recall_curves,
    plot_single_decorated_precision_recall_curve,
)
from stamp.statistics.regression import regression_aggregated_
from stamp.statistics.roc import (
    plot_multiple_decorated_roc_curves,
    plot_single_decorated_roc_curve,
)
from stamp.statistics.survival import _plot_km, _survival_stats_for_csv
from stamp.types import PandasLabel, Task

__all__ = ["StatsConfig", "compute_stats_"]


__author__ = "Marko van Treeck, Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2022-2024 Marko van Treeck, Minh Duc Nguyen"
__license__ = "MIT"


def _read_table(file: Path, **kwargs) -> pd.DataFrame:
    """Load a dataframe from CSV or XLSX file path.

    This small helper centralizes file IO formatting and keeps callers simple.
    """
    if isinstance(file, Path) and file.suffix == ".xlsx":
        return pd.read_excel(file, **kwargs)
    return pd.read_csv(file, **kwargs)


class StatsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    task: Task = Field(default="classification")
    output_dir: Path
    pred_csvs: list[Path]
    ground_truth_label: PandasLabel | Sequence[PandasLabel] | None = None
    true_class: str | None = None
    time_label: str | None = None
    status_label: str | None = None


_Inches = NewType("_Inches", float)


def _compute_multitarget_classification_stats(
    *,
    output_dir: Path,
    pred_csvs: Sequence[Path],
    target_labels: Sequence[str],
) -> None:
    """Compute statistics and plots for multi-target classification.

    For each target, creates ROC and PRC curves for each class,
    similar to single-target classification.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_bootstrap_samples = 1000
    figure_width = _Inches(3.8)
    roc_curve_figure_aspect_ratio = 1.08

    # Validate all target labels exist in CSV
    first_df = _read_table(pred_csvs[0], nrows=0)
    missing_targets = [t for t in target_labels if t not in first_df.columns]
    if missing_targets:
        raise ValueError(
            f"Target labels not found in CSV: {missing_targets}. Available columns: {list(first_df.columns)}"
        )

    # Process each target
    for target_label in target_labels:
        # Load data for this target
        preds_dfs = []
        for p in pred_csvs:
            df = _read_table(p, dtype=str)
            # Only keep rows where this target has ground truth
            df_clean = df.dropna(subset=[target_label])
            if len(df_clean) > 0:
                preds_dfs.append(df_clean)

        if not preds_dfs:
            continue

        # Get unique classes for this target
        classes = sorted(preds_dfs[0][target_label].unique())

        # Create plots for each class in this target
        for true_class in classes:
            # Extract ground truth and predictions for this class
            y_trues = []
            y_preds = []

            for df in preds_dfs:
                prob_col = f"{target_label}_{true_class}"
                if prob_col not in df.columns:
                    continue

                y_trues.append(np.array(df[target_label] == true_class))
                y_preds.append(np.array(df[prob_col].astype(float).values))

            if not y_trues:
                continue

            # Plot ROC curve
            fig, ax = plt.subplots(
                figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
                dpi=300,
            )

            if len(preds_dfs) == 1:
                plot_single_decorated_roc_curve(
                    ax=ax,
                    y_true=y_trues[0],
                    y_score=y_preds[0],
                    title=f"{target_label} = {true_class}",
                    n_bootstrap_samples=n_bootstrap_samples,
                    threshold_cmap=None,
                )
            else:
                plot_multiple_decorated_roc_curves(
                    ax=ax,
                    y_trues=y_trues,
                    y_scores=y_preds,
                    title=f"{target_label} = {true_class}",
                    n_bootstrap_samples=None,
                )

            fig.tight_layout()
            fig.savefig(output_dir / f"roc-curve_{target_label}={true_class}.svg")
            plt.close(fig)

            # Plot PRC curve
            fig, ax = plt.subplots(
                figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
                dpi=300,
            )

            if len(preds_dfs) == 1:
                plot_single_decorated_precision_recall_curve(
                    ax=ax,
                    y_true=y_trues[0],
                    y_score=y_preds[0],
                    title=f"{target_label} = {true_class}",
                    n_bootstrap_samples=n_bootstrap_samples,
                )
            else:
                plot_multiple_decorated_precision_recall_curves(
                    ax=ax,
                    y_trues=y_trues,
                    y_scores=y_preds,
                    title=f"{target_label} = {true_class}",
                )

            fig.tight_layout()
            fig.savefig(output_dir / f"pr-curve_{target_label}={true_class}.svg")
            plt.close(fig)

    # Compute aggregated statistics for all targets
    categorical_aggregated_multitarget_(
        preds_csvs=pred_csvs,
        outpath=output_dir,
        target_labels=target_labels,
    )


def compute_stats_(
    *,
    task: Task,
    output_dir: Path,
    pred_csvs: Sequence[Path],
    ground_truth_label: PandasLabel | Sequence[PandasLabel] | None = None,
    true_class: str | None = None,
    time_label: str | None = None,
    status_label: str | None = None,
) -> None:
    """Compute and save statistics for the provided task and prediction CSVs.

    This wrapper keeps the external API stable while delegating the detailed
    computations and plotting to the submodules under `stamp.statistics.*`.
    """
    match task:
        case "classification":
            # Check if multi-target based on ground_truth_label type
            is_multitarget = (
                isinstance(ground_truth_label, (list, tuple))
                and len(ground_truth_label) > 1
            )

            if is_multitarget:
                # Multi-target classification
                if not isinstance(ground_truth_label, (list, tuple)):
                    raise ValueError(
                        "ground_truth_label must be a list or tuple for multi-target classification"
                    )
                _compute_multitarget_classification_stats(
                    output_dir=output_dir,
                    pred_csvs=pred_csvs,
                    target_labels=list(ground_truth_label),
                )
            else:
                # Single-target classification (original behavior)
                if true_class is None or ground_truth_label is None:
                    raise ValueError(
                        "both true_class and ground_truth_label are required in statistic configuration"
                    )
                if not isinstance(ground_truth_label, str):
                    raise ValueError(
                        "ground_truth_label must be a string for single-target classification"
                    )

                preds_dfs = [
                    df
                    for p in pred_csvs
                    if len(
                        df := _read_table(
                            p,
                            usecols=[
                                ground_truth_label,
                                f"{ground_truth_label}_{true_class}",
                            ],
                            dtype={
                                ground_truth_label: str,
                                f"{ground_truth_label}_{true_class}": float,
                            },
                        ).dropna(subset=[ground_truth_label])
                    )
                    > 0
                ]
                if not preds_dfs:
                    raise ValueError(
                        "No classification rows with ground truth available for plotting."
                    )

                y_trues = [
                    np.array(df[ground_truth_label] == true_class) for df in preds_dfs
                ]
                y_preds = [
                    np.array(df[f"{ground_truth_label}_{true_class}"].values)
                    for df in preds_dfs
                ]
                n_bootstrap_samples = 1000
                figure_width = _Inches(3.8)
                threshold_cmap = None

                roc_curve_figure_aspect_ratio = 1.08
                fig, ax = plt.subplots(
                    figsize=(
                        figure_width,
                        figure_width * roc_curve_figure_aspect_ratio,
                    ),
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
                output_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(
                    output_dir / f"roc-curve_{ground_truth_label}={true_class}.svg"
                )
                plt.close(fig)

                fig, ax = plt.subplots(
                    figsize=(
                        figure_width,
                        figure_width * roc_curve_figure_aspect_ratio,
                    ),
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
                fig.savefig(
                    output_dir / f"pr-curve_{ground_truth_label}={true_class}.svg"
                )
                plt.close(fig)

                categorical_aggregated_(
                    preds_csvs=pred_csvs,
                    ground_truth_label=ground_truth_label,
                    outpath=output_dir,
                )

        case "regression":
            if ground_truth_label is None:
                raise ValueError(
                    "no ground_truth_label configuration supplied in statistic"
                )
            if not isinstance(ground_truth_label, str):
                raise ValueError(
                    "ground_truth_label must be a string for regression (multi-target regression not yet supported)"
                )
            regression_aggregated_(
                preds_csvs=pred_csvs,
                ground_truth_label=ground_truth_label,
                outpath=output_dir,
            )

        case "survival":
            if time_label is None or status_label is None:
                raise ValueError(
                    "both time_label and status_label are required in statistic configuration"
                )
            output_dir.mkdir(parents=True, exist_ok=True)

            per_fold: dict[str, pd.Series] = {}

            for p in pred_csvs:
                df = pd.read_csv(p)

                cut_off = (
                    float(df.columns[-1].split("=")[1])
                    if "cut_off" in df.columns[-1]
                    else None
                )

                fold_name = Path(p).parent.name
                pred_name = Path(p).stem
                key = f"{fold_name}_{pred_name}"

                stats = _survival_stats_for_csv(
                    df,
                    time_label=time_label,
                    status_label=status_label,
                    cut_off=cut_off,
                )
                per_fold[key] = stats

                _plot_km(
                    df,
                    fold_name=key,  # use same naming for plots
                    time_label=time_label,
                    status_label=status_label,
                    outdir=output_dir,
                    cut_off=cut_off,
                )

            # Save individual and aggregated CSVs
            stats_df = pd.DataFrame(per_fold).transpose()
            stats_df.index.name = "fold_name"  # label the index column
            stats_df.to_csv(output_dir / "survival-stats_individual.csv", index=True)
