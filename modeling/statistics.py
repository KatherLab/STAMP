import sys
import argparse
from pathlib import Path
import os

import pandas as pd
from matplotlib import pyplot as plt
from marugoto.stats.categorical import categorical_aggregated_
from marugoto.visualizations.roc import plot_multiple_decorated_roc_curves, plot_single_decorated_roc_curve
from marugoto.visualizations.prc import plot_precision_recall_curves_

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a ROC Curve.")
    args = (add_roc_curve_args(parser)).parse_args()

    # read all the patient preds
    # and transform their true / preds columns into np arrays
    preds_dfs = [
        pd.read_csv(p, dtype={f"{args.target_label}": str, "pred": str})
        for p in args.pred_csvs
    ]
    y_trues = [df[args.target_label] == args.true_class for df in preds_dfs]
    y_preds = [
        pd.to_numeric(df[f"{args.target_label}_{args.true_class}"]) for df in preds_dfs
    ]

    roc_curve_figure_aspect_ratio = 1.08
    fig, ax = plt.subplots(
        figsize=(args.figure_width, args.figure_width * roc_curve_figure_aspect_ratio),
        dpi=300,
    )

    if len(preds_dfs) == 1:
        plot_single_decorated_roc_curve(
                ax,
                y_trues[0],
                y_preds[0],
                title=f"{args.target_label} = {args.true_class}",
                n_bootstrap_samples=args.n_bootstrap_samples,
                threshold_cmap=args.threshold_cmap,
            )

    else:
        plot_multiple_decorated_roc_curves(
            ax,
            y_trues,
            y_preds,
            title=f"{args.target_label} = {args.true_class}",
            n_bootstrap_samples=None,
        )

    fig.tight_layout()
    stats_dir=(args.outpath/"model_statistics")
    if not stats_dir.exists():
        stats_dir.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(stats_dir/f"AUROC_{args.target_label}={args.true_class}.svg")
    plt.close(fig)

    plot_precision_recall_curves_(args.pred_csvs,
                                  target_label=args.target_label,
                                  true_label=args.true_class,
                                  outpath=stats_dir)
    
    categorical_aggregated_(args.pred_csvs,
                            target_label=args.target_label,
                            outpath=stats_dir)