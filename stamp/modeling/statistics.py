import sys
import argparse
from pathlib import Path
import os
from typing import Sequence
import pandas as pd
from matplotlib import pyplot as plt

from .marugoto.stats.categorical import categorical_aggregated_
from .marugoto.visualizations.roc import plot_multiple_decorated_roc_curves, plot_single_decorated_roc_curve
from .marugoto.visualizations.prc import plot_precision_recall_curves_, plot_single_decorated_prc_curve

def add_roc_curve_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "pred_csvs",
        metavar="PREDS_CSV",
        nargs="*",
        type=Path,
        help="Predictions to create ROC curves for.",
        default=[sys.stdin],
    )
    parser.add_argument(
        "--target-label",
        metavar="LABEL",
        required=True,
        type=str,
        help="The target label to calculate the ROC/PRC for.",
    )
    parser.add_argument(
        "--true-class",
        metavar="CLASS",
        required=True,
        type=str,
        help="The class to consider as positive for the ROC/PRC.",
    )
    parser.add_argument(
        "-o",
        "--outpath",
        metavar="PATH",
        required=True,
        type=Path,
        help=(
            "Path to save the statistics to."
        ),
    )

    parser.add_argument(
        "--n-bootstrap-samples",
        metavar="N",
        type=int,
        required=False,
        help="Number of bootstrapping samples to take for confidence interval generation.",
        default=1000
    )

    parser.add_argument(
        "--figure-width",
        metavar="INCHES",
        type=float,
        required=False,
        help="Width of the figure in inches.",
        default=3.8,
    )
    
    parser.add_argument(
        "--threshold-cmap",
        metavar="COLORMAP",
        type=plt.get_cmap,
        required=False,
        help="Draw Curve with threshold color.",
    )

    return parser


def read_table(file) -> pd.DataFrame:
    """Loads a dataframe from a file."""
    if isinstance(file, Path) and file.suffix == ".xlsx":
        return pd.read_excel(file)
    else:
        return pd.read_csv(file)

def compute_stats(pred_csvs: Sequence[Path], target_label: str, true_class: str, output_dir: Path):
    # read all the patient preds
    # and transform their true / preds columns into np arrays
    preds_dfs = [
        pd.read_csv(p, dtype={f"{target_label}": str, "pred": str})
        for p in pred_csvs
    ]
    y_trues = [df[target_label] == true_class for df in preds_dfs]
    y_preds = [
        pd.to_numeric(df[f"{target_label}_{true_class}"]) for df in preds_dfs
    ]
    n_bootstrap_samples = 1000
    figure_width = 3.8 # inches
    threshold_cmap= plt.get_cmap()
    
    roc_curve_figure_aspect_ratio = 1.08
    fig, ax = plt.subplots(
        figsize=(figure_width, figure_width * roc_curve_figure_aspect_ratio),
        dpi=300,
    )

    if len(preds_dfs) == 1:
        plot_single_decorated_roc_curve(
                ax,
                y_trues[0],
                y_preds[0],
                title=f"{target_label} = {true_class}",
                n_bootstrap_samples=n_bootstrap_samples,
                threshold_cmap=threshold_cmap,
            )

    else:
        plot_multiple_decorated_roc_curves(
            ax,
            y_trues,
            y_preds,
            title=f"{target_label} = {true_class}",
            n_bootstrap_samples=None,
        )

    fig.tight_layout()
    stats_dir=(output_dir/"model_statistics")
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(stats_dir/f"AUROC_{target_label}={true_class}.svg")
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
                title=f"{target_label} = {true_class}",
                n_bootstrap_samples=n_bootstrap_samples
        )

    else:
        plot_precision_recall_curves_(
            ax,
            pred_csvs,
            target_label=target_label,
            true_label=true_class,
            outpath=stats_dir
        )

    fig.tight_layout()
    fig.savefig(stats_dir/f"AUPRC_{target_label}={true_class}.svg")
    plt.close(fig)

    categorical_aggregated_(pred_csvs,
                            target_label=target_label,
                            outpath=stats_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ROC Curve.")
    args = add_roc_curve_args(parser).parse_args()
    compute_stats(pred_csvs=args.pred_csvs,
                  target_label=args.target_label,
                  true_class=args.true_class,
                  output_dir=args.outpath,
                  n_bootstrap_samples=args.n_bootstrap_samples,
                  figure_width=args.figure_width,
                  threshold_cmap=args.threshold_cmap)