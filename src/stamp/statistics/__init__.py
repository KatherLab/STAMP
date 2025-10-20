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
from stamp.statistics.survival import (
    _plot_km,
    _survival_stats_for_csv,
)
from stamp.types import PandasLabel, Task

__author__ = "Marko van Treeck, Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2022-2024 Marko van Treeck, Minh Duc Nguyen"
__license__ = "MIT"


def _read_table(file: Path, **kwargs) -> pd.DataFrame:
    """Loads a dataframe from a file."""
    if isinstance(file, Path) and file.suffix == ".xlsx":
        return pd.read_excel(file, **kwargs)
    else:
        return pd.read_csv(file, **kwargs)


class StatsConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")
    output_dir: Path
    pred_csvs: list[Path]
    ground_truth_label: PandasLabel | None = None
    true_class: str | None = None
    time_label: str | None = None
    status_label: str | None = None
    risk_label: str | None = None


_Inches = NewType("_Inches", float)


def compute_stats_(
    *,
    task: Task,
    output_dir: Path,
    pred_csvs: Sequence[Path],
    ground_truth_label: PandasLabel | None = None,
    true_class: str | None = None,
    time_label: str | None = None,
    status_label: str | None = None,
    risk_label: str | None = None,
) -> None:
    match task:
        case "classification":
            if true_class is None or ground_truth_label is None:
                raise ValueError(
                    "both true_class and ground_truth_label are required in statistic configuration"
                )

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

        case "regression":
            if ground_truth_label is None:
                raise ValueError(
                    "no ground_truth_label configuration supplied in statistic"
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
                fold_name = Path(p).parent.name
                pred_name = Path(p).stem
                key = f"{fold_name}_{pred_name}"

                stats = _survival_stats_for_csv(
                    df, time_label=time_label, status_label=status_label
                )
                per_fold[key] = stats

                _plot_km(
                    df,
                    fold_name=key,  # use same naming for plots
                    time_label=time_label,
                    status_label=status_label,
                    outdir=output_dir,
                )

            # ------------------------------------------------------------------ #
            # Save individual and aggregated CSVs
            # ------------------------------------------------------------------ #
            stats_df = pd.DataFrame(per_fold).transpose()
            stats_df.index.name = "fold_name"  # label the index column
            stats_df.to_csv(output_dir / "survival-stats_individual.csv", index=True)

            # agg_df = _aggregate_with_ci(stats_df)
            # agg_df.to_csv(output_dir / "survival-stats_aggregated.csv", index=True)
