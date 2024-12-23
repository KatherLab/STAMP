from pathlib import Path
from typing import NewType, Sequence

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from pydantic import BaseModel

from stamp.modeling.data import PandasLabel
from stamp.statistics.categorical import categorical_aggregated_
from stamp.statistics.prc import (
    plot_precision_recall_curves_,
    plot_single_decorated_prc_curve,
)
from stamp.statistics.roc import (
    plot_multiple_decorated_roc_curves,
    plot_single_decorated_roc_curve,
)

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
    output_dir: Path

    pred_csvs: list[Path]
    ground_truth_label: PandasLabel
    true_class: str


_Inches = NewType("_Inches", float)


def compute_stats_(
    *,
    output_dir: Path,
    pred_csvs: Sequence[Path],
    ground_truth_label: PandasLabel,
    true_class: str,
) -> None:
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
        np.array(df[f"{ground_truth_label}_{true_class}"].values) for df in preds_dfs
    ]
    n_bootstrap_samples = 1000
    figure_width = _Inches(3.8)
    threshold_cmap = plt.get_cmap()

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
    stats_dir = output_dir / "statistics"
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(stats_dir / f"AUROC_{ground_truth_label}={true_class}.svg")
    plt.close(fig)

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
        dpi=300,
    )
    if len(preds_dfs) == 1:
        plot_single_decorated_prc_curve(
            ax,
            y_trues[0],
            y_preds[0],
            title=f"{ground_truth_label} = {true_class}",
            n_bootstrap_samples=n_bootstrap_samples,
        )

    else:
        plot_precision_recall_curves_(
            ax,
            pred_csvs,
            target_label=ground_truth_label,
            true_label=true_class,
            outpath=stats_dir,
        )

    fig.tight_layout()
    fig.savefig(stats_dir / f"AUPRC_{ground_truth_label}={true_class}.svg")
    plt.close(fig)

    categorical_aggregated_(
        preds_csvs=pred_csvs, target_label=ground_truth_label, outpath=stats_dir
    )
