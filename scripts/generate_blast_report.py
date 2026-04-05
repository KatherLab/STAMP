#!/usr/bin/env python3
"""Generate a self-contained HTML report for BLAST-related STAMP experiments.

Covers three experiments derived from the BLAST_PERCENT column:
  1. BLAST_PERCENT — regression (continuous 0-91%)
  2. BLAST_SEVERITY — 3-class classification (low/intermediate/high)
  3. HIGH_BLAST — binary classification (yes/no, threshold ≥20%)

The report includes per-experiment: summary metrics, ROC/PR or scatter plots,
per-fold stats tables, and a representative heatmap gallery (sampled to keep
file size manageable).

Usage:
    python scripts/generate_blast_report.py [--max-slides N]

    --max-slides: max heatmap slides per experiment per class (default: 10)
"""

import argparse
import base64
import random
from datetime import datetime
from pathlib import Path

import pandas as pd

# === Configuration ===
DATA_ROOT = Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen")
SLIDE_TABLE = Path("/home/jeff/Projects/STAMP/tables/stamp_slide.csv")
CLINI_TABLE = Path("/home/jeff/Projects/STAMP/tables/stamp_clini.csv")

EXPERIMENTS = {
    "blast_percent": {
        "name": "BLAST_PERCENT",
        "display_name": "Blast Percentage (Regression)",
        "base_dir": DATA_ROOT / "stamp_aml_blast_percent_uni2",
        "task": "regression",
        "ground_truth_label": "BLAST_PERCENT",
        "description": (
            "Continuous regression predicting the percentage of blast cells (0–91%) "
            "in AML bone marrow aspirates. The model directly outputs a numeric prediction. "
            "The distribution is heavily right-skewed (median=3%, mean=14.6%)."
        ),
        "class_balance": "Continuous: range 0–91%, median=3%, mean=14.6%",
        "crossval_type": "5-fold KFold",
    },
    "blast_severity": {
        "name": "BLAST_SEVERITY",
        "display_name": "Blast Severity (3-class)",
        "base_dir": DATA_ROOT / "stamp_aml_blast_severity_uni2",
        "task": "classification",
        "ground_truth_label": "BLAST_SEVERITY",
        "categories": ["high", "intermediate", "low"],
        "true_class": "high",
        "description": (
            "Three-class classification of blast severity: low (<5% blasts, n=336), "
            "intermediate (5–19%, n=55), and high (≥20%, n=98). Thresholds based on "
            "clinically meaningful blast count cutoffs in AML."
        ),
        "class_balance": "low=336, intermediate=55, high=98",
        "crossval_type": "5-fold StratifiedKFold",
    },
    "high_blast": {
        "name": "HIGH_BLAST",
        "display_name": "High Blast (Binary)",
        "base_dir": DATA_ROOT / "stamp_aml_high_blast_uni2",
        "task": "classification",
        "ground_truth_label": "HIGH_BLAST",
        "categories": ["yes", "no"],
        "true_class": "yes",
        "description": (
            "Binary classification predicting whether blast percentage is ≥20% (high blast). "
            "Clinically relevant threshold: ≥20% blasts indicates active/refractory disease. "
            "Positive class ('yes') = ≥20% blasts."
        ),
        "class_balance": "yes=98, no=391",
        "crossval_type": "5-fold StratifiedKFold",
    },
}


def embed_image(path: Path, mime: str | None = None) -> str:
    """Return a data-URI string for embedding an image in HTML."""
    if not path.exists():
        return ""
    if mime is None:
        suffix = path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
            ".gif": "image/gif",
        }.get(suffix, "image/png")
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{b64}"


def embed_svg(path: Path) -> str:
    """Read an SVG file and return its content for inline embedding."""
    if not path.exists():
        return "<p><em>SVG not found</em></p>"
    return path.read_text()


def load_clini_and_slide():
    """Load clinical and slide tables."""
    clini_df = pd.read_csv(CLINI_TABLE)
    slide_df = pd.read_csv(SLIDE_TABLE)
    fname_to_sample = dict(zip(slide_df["FILENAME"], slide_df["SAMPLE_ID"]))
    return clini_df, slide_df, fname_to_sample


# =============================================================================
# Regression experiment helpers
# =============================================================================


def load_regression_stats(exp):
    """Load regression statistics."""
    stats_dir = exp["base_dir"] / "statistics"
    name = exp["name"]
    agg_path = stats_dir / f"{name}_regression-stats_aggregated.csv"
    agg = pd.read_csv(agg_path, index_col=0)
    return {
        "r2": float(agg.loc["r2_score", "mean"]),
        "pearson_r": float(agg.loc["pearson_r", "mean"]),
        "pearson_p": float(agg.loc["pearson_p", "mean"]),
        "mae": float(agg.loc["mae", "mean"]),
        "rmse": float(agg.loc["rmse", "mean"]),
        "count": int(float(agg.loc["count", "mean"])),
    }


def make_regression_section(exp):
    """Generate the HTML section for a regression experiment."""
    stats = load_regression_stats(exp)
    stats_dir = exp["base_dir"] / "statistics"
    scatter_svg_path = stats_dir / "plots" / "fold_patient-preds_scatter.svg"

    html = f"""
<div class="experiment-section" id="exp-{exp['name'].lower()}">
<h2>{exp['display_name']}</h2>

<p>{exp['description']}</p>

<div class="summary-box">
  <div class="summary-item"><div class="label">Task</div><div class="value">Regression</div></div>
  <div class="summary-item"><div class="label">Target</div><div class="value">{exp['name']}</div></div>
  <div class="summary-item"><div class="label">Distribution</div><div class="value">{exp['class_balance']}</div></div>
  <div class="summary-item"><div class="label">Crossval</div><div class="value">{exp['crossval_type']}</div></div>
  <div class="summary-item"><div class="label">Samples</div><div class="value">489</div></div>
  <div class="summary-item"><div class="label">Model</div><div class="value">ViT MIL (UNI2)</div></div>
</div>

<h3>Performance Metrics</h3>
<div class="metrics-grid metrics-grid-3">
  <div class="metric-card">
    <div class="metric-value">{stats['pearson_r']:.3f}</div>
    <div class="metric-label">Pearson r</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{stats['mae']:.1f}%</div>
    <div class="metric-label">Mean Absolute Error</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{stats['rmse']:.1f}%</div>
    <div class="metric-label">Root Mean Squared Error</div>
  </div>
</div>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-value">{stats['r2']:.3f}</div>
    <div class="metric-label">R&sup2; Score</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{stats['pearson_p']:.2e}</div>
    <div class="metric-label">Pearson p-value</div>
  </div>
</div>

<h3>Predicted vs. Actual Scatter Plot</h3>
<div class="svg-container">
{embed_svg(scatter_svg_path)}
</div>

<p class="caption">Each point represents one patient (test-set prediction from the respective crossval fold).
The dashed line is the identity line (perfect prediction). Pearson r = {stats['pearson_r']:.3f}
indicates moderate-strong correlation. Note the R&sup2; = {stats['r2']:.3f} is near zero due to the
heavily skewed distribution (many samples near 0%, few at high %).</p>
"""
    return html


# =============================================================================
# Classification experiment helpers
# =============================================================================


def load_classification_stats(exp):
    """Load classification statistics (works for both binary and multi-class)."""
    stats_dir = exp["base_dir"] / "statistics"
    name = exp["name"]
    agg = pd.read_csv(stats_dir / f"{name}_categorical-stats_aggregated.csv")
    ind = pd.read_csv(stats_dir / f"{name}_categorical-stats_individual.csv")
    return agg, ind


def make_classification_section(exp):
    """Generate the HTML section for a classification experiment."""
    agg, ind = load_classification_stats(exp)
    stats_dir = exp["base_dir"] / "statistics"
    true_class = exp["true_class"]
    name = exp["name"]
    categories = exp["categories"]

    # Parse aggregated stats
    # agg columns: unnamed_class, roc_auc_score (mean), roc_auc_score (95%_low), roc_auc_score (95%_high),
    #              average_precision_score (mean, low, high), f1_score (mean, low, high), count
    # Skip header row (row 0 is the sub-header)
    class_stats = {}
    for _, row in agg.iterrows():
        cls = str(row.iloc[0])
        if cls in ("", "nan") or cls.startswith("Unnamed"):
            continue
        class_stats[cls] = {
            "auroc_mean": float(row.iloc[1]),
            "auroc_low": float(row.iloc[2]),
            "auroc_high": float(row.iloc[3]),
            "auprc_mean": float(row.iloc[4]),
            "auprc_low": float(row.iloc[5]),
            "auprc_high": float(row.iloc[6]),
            "f1_mean": float(row.iloc[7]),
            "f1_low": float(row.iloc[8]),
            "f1_high": float(row.iloc[9]),
            "count": int(float(row.iloc[10])),
        }

    # Build metric cards
    primary = class_stats.get(true_class, {})
    primary_auroc = primary.get("auroc_mean", 0)
    primary_auroc_ci = f"{primary.get('auroc_low', 0):.3f}–{primary.get('auroc_high', 0):.3f}"
    primary_auprc = primary.get("auprc_mean", 0)

    html = f"""
<div class="experiment-section" id="exp-{name.lower()}">
<h2>{exp['display_name']}</h2>

<p>{exp['description']}</p>

<div class="summary-box">
  <div class="summary-item"><div class="label">Task</div><div class="value">Classification ({len(categories)}-class)</div></div>
  <div class="summary-item"><div class="label">Target</div><div class="value">{name}</div></div>
  <div class="summary-item"><div class="label">Class Balance</div><div class="value">{exp['class_balance']}</div></div>
  <div class="summary-item"><div class="label">Crossval</div><div class="value">{exp['crossval_type']}</div></div>
  <div class="summary-item"><div class="label">Samples</div><div class="value">489</div></div>
  <div class="summary-item"><div class="label">Model</div><div class="value">ViT MIL (UNI2)</div></div>
</div>

<h3>Performance Metrics</h3>
"""

    # Per-class metric cards
    html += '<div class="per-class-metrics">\n'
    for cls in categories:
        cs = class_stats.get(cls, {})
        badge_cls = "badge-yes" if cls == true_class else "badge-no"
        html += f"""
<div class="class-metric-block">
  <h4><span class="badge {badge_cls}">{cls}</span> (n={cs.get('count', '?')})</h4>
  <div class="metrics-grid metrics-grid-3">
    <div class="metric-card">
      <div class="metric-value">{cs.get('auroc_mean', 0):.3f}</div>
      <div class="metric-label">AUROC ({cs.get('auroc_low', 0):.3f}–{cs.get('auroc_high', 0):.3f})</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{cs.get('auprc_mean', 0):.3f}</div>
      <div class="metric-label">AUPRC ({cs.get('auprc_low', 0):.3f}–{cs.get('auprc_high', 0):.3f})</div>
    </div>
    <div class="metric-card">
      <div class="metric-value">{cs.get('f1_mean', 0):.3f}</div>
      <div class="metric-label">F1 ({cs.get('f1_low', 0):.3f}–{cs.get('f1_high', 0):.3f})</div>
    </div>
  </div>
</div>
"""
    html += "</div>\n"

    # ROC and PR curves
    roc_path = stats_dir / f"roc-curve_{name}={true_class}.svg"
    pr_path = stats_dir / f"pr-curve_{name}={true_class}.svg"

    if roc_path.exists():
        html += f"""
<h3>ROC Curve ({true_class} class, 5-fold)</h3>
<div class="svg-container">
{embed_svg(roc_path)}
</div>
"""

    if pr_path.exists():
        html += f"""
<h3>Precision-Recall Curve ({true_class} class, 5-fold)</h3>
<div class="svg-container">
{embed_svg(pr_path)}
</div>
"""

    # Per-fold stats table
    html += "<h3>Per-Fold Statistics</h3>\n"
    html += make_classification_table(ind, categories)

    return html


def make_classification_table(ind: pd.DataFrame, categories: list[str]) -> str:
    """Generate an HTML table from per-fold individual stats."""
    html = "<table>\n<tr>"
    html += "<th>Fold</th><th>Class</th><th>Count</th><th>AUROC</th><th>AUPRC</th><th>F1</th><th>p-value</th>"
    html += "</tr>\n"

    for _, row in ind.iterrows():
        fold = row.iloc[0]
        cls = str(row.iloc[1])
        count = int(row.iloc[2])
        auroc = float(row.iloc[3])
        ap = float(row.iloc[4])
        f1 = float(row.iloc[5])
        pval = float(row.iloc[6])

        # Color code rows by class
        if cls == categories[0]:
            row_style = 'style="background: #f0fff0;"'
        elif len(categories) > 2 and cls == categories[1]:
            row_style = 'style="background: #fff8f0;"'
        else:
            row_style = ""

        html += f"<tr {row_style}><td>{fold}</td><td><span class='badge'>{cls}</span></td>"
        html += f"<td>{count}</td><td>{auroc:.4f}</td><td>{ap:.4f}</td><td>{f1:.4f}</td>"
        html += f"<td>{pval:.2e}</td></tr>\n"

    html += "</table>\n"
    return html


# =============================================================================
# Heatmap gallery helpers
# =============================================================================


def collect_heatmap_slides(exp, clini_df, fname_to_sample, max_per_class=10):
    """Collect heatmap slide info, sampling to keep report size manageable."""
    heatmap_dir = exp["base_dir"] / "heatmaps"
    crossval_dir = exp["base_dir"] / "crossval"
    gt_label = exp["ground_truth_label"]
    task = exp["task"]

    sample_to_gt = dict(zip(clini_df["SAMPLE_ID"], clini_df[gt_label]))

    # Load all predictions
    all_preds = {}
    for split_i in range(5):
        pred_file = crossval_dir / f"split-{split_i}" / "patient-preds.csv"
        if not pred_file.exists():
            continue
        preds = pd.read_csv(pred_file)
        for _, row in preds.iterrows():
            pred_info = {"split": split_i}
            if task == "regression":
                pred_info["pred"] = row["pred"]
                pred_info["gt_value"] = row[gt_label]
            else:
                pred_info["pred"] = row["pred"]
                pred_info["gt"] = row[gt_label]
                # Collect per-class probabilities
                pred_info["probs"] = {}
                for col in preds.columns:
                    if col.startswith(f"{gt_label}_"):
                        cls = col.replace(f"{gt_label}_", "")
                        pred_info["probs"][cls] = row[col]
            all_preds[row["SAMPLE_ID"]] = pred_info

    slides = []
    total_available = 0
    for split_dir in sorted(heatmap_dir.glob("split-*")):
        split_i = int(split_dir.name.split("-")[1])
        for slide_dir in sorted(split_dir.iterdir()):
            if not slide_dir.is_dir():
                continue
            stem = slide_dir.name
            h5_name = stem + ".h5"
            sample_id = fname_to_sample.get(h5_name, "unknown")
            info = all_preds.get(sample_id, {})

            overview_path = slide_dir / "plots" / f"overview-{stem}.png"
            if not overview_path.exists():
                continue

            total_available += 1

            # Collect top/bottom tiles
            tiles_dir = slide_dir / "tiles"
            top_tiles = (
                sorted(
                    [
                        t
                        for ext in ("*.png", "*.jpg", "*.jpeg")
                        for t in tiles_dir.glob(f"top_*{ext}")
                    ]
                )[:4]
                if tiles_dir.exists()
                else []
            )
            bottom_tiles = (
                sorted(
                    [
                        t
                        for ext in ("*.png", "*.jpg", "*.jpeg")
                        for t in tiles_dir.glob(f"bottom_*{ext}")
                    ]
                )[:4]
                if tiles_dir.exists()
                else []
            )

            slide_info = {
                "stem": stem,
                "sample_id": sample_id,
                "split": split_i,
                "overview_path": overview_path,
                "top_tiles": top_tiles,
                "bottom_tiles": bottom_tiles,
            }

            if task == "regression":
                gt_val = info.get("gt_value", sample_to_gt.get(sample_id, "?"))
                slide_info["gt_value"] = gt_val
                slide_info["pred_value"] = info.get("pred", "?")
                # Bin for grouping
                try:
                    gt_num = float(gt_val)
                    if gt_num < 5:
                        slide_info["group"] = "low (<5%)"
                    elif gt_num < 20:
                        slide_info["group"] = "intermediate (5-19%)"
                    else:
                        slide_info["group"] = "high (≥20%)"
                except (ValueError, TypeError):
                    slide_info["group"] = "unknown"
            else:
                slide_info["gt"] = info.get("gt", sample_to_gt.get(sample_id, "?"))
                slide_info["pred"] = info.get("pred", "?")
                slide_info["probs"] = info.get("probs", {})
                slide_info["correct"] = slide_info["gt"] == slide_info["pred"]
                slide_info["group"] = str(slide_info["gt"])

            slides.append(slide_info)

    # Sample slides per group to limit report size
    if max_per_class > 0:
        groups = {}
        for s in slides:
            groups.setdefault(s["group"], []).append(s)

        sampled = []
        for group_name, group_slides in sorted(groups.items()):
            if len(group_slides) <= max_per_class:
                sampled.extend(group_slides)
            else:
                # Deterministic sample: mix of correct/incorrect for classification
                random.seed(42)
                sampled.extend(random.sample(group_slides, max_per_class))

        slides = sampled

    return slides, total_available


def make_slide_card_regression(slide: dict) -> str:
    """Generate HTML card for a regression heatmap slide."""
    gt_val = slide.get("gt_value", "?")
    pred_val = slide.get("pred_value", "?")
    try:
        error = abs(float(pred_val) - float(gt_val))
        error_str = f"{error:.1f}%"
    except (ValueError, TypeError):
        error_str = "?"

    overview_uri = embed_image(slide["overview_path"])

    html = f"""
<div class="slide-card">
  <div class="slide-header">
    <h4 style="font-size: 1em; margin: 0;">{slide['sample_id']}</h4>
    <div class="slide-meta">
      <span class="badge">GT: {gt_val}%</span>
      <span class="badge">Pred: {float(pred_val):.1f}%</span>
      <span class="badge badge-{'correct' if error_str != '?' and float(error_str.replace('%','')) < 10 else 'wrong'}">Error: {error_str}</span>
      <span class="badge">Split {slide['split']}</span>
    </div>
  </div>
  <p style="font-size: 0.8em; color: #999; margin-bottom: 8px;">{slide['stem']}</p>
  <img class="overview-img" src="{overview_uri}" alt="Heatmap overview for {slide['sample_id']}" loading="lazy">
"""

    if slide["top_tiles"]:
        html += '  <h4 style="margin-top: 12px; font-size: 0.9em;">Top Attended Tiles</h4>\n'
        html += '  <div class="tiles-row">\n'
        for tile_path in slide["top_tiles"]:
            uri = embed_image(tile_path)
            html += f'    <img src="{uri}" alt="top tile" loading="lazy">\n'
        html += "  </div>\n"

    if slide["bottom_tiles"]:
        html += '  <h4 style="margin-top: 12px; font-size: 0.9em;">Least Attended Tiles</h4>\n'
        html += '  <div class="tiles-row">\n'
        for tile_path in slide["bottom_tiles"]:
            uri = embed_image(tile_path)
            html += f'    <img src="{uri}" alt="bottom tile" loading="lazy">\n'
        html += "  </div>\n"

    html += "</div>\n"
    return html


def make_slide_card_classification(slide: dict) -> str:
    """Generate HTML card for a classification heatmap slide."""
    correct_badge = (
        '<span class="badge badge-correct">Correct</span>'
        if slide.get("correct")
        else '<span class="badge badge-wrong">Incorrect</span>'
    )
    gt = slide.get("gt", "?")
    pred = slide.get("pred", "?")
    probs = slide.get("probs", {})

    prob_badges = " ".join(
        f'<span class="badge">P({cls})={prob:.3f}</span>' for cls, prob in sorted(probs.items())
    )

    overview_uri = embed_image(slide["overview_path"])

    html = f"""
<div class="slide-card">
  <div class="slide-header">
    <h4 style="font-size: 1em; margin: 0;">{slide['sample_id']}</h4>
    <div class="slide-meta">
      <span class="badge badge-yes">GT: {gt}</span>
      <span class="badge">Pred: {pred}</span>
      {prob_badges}
      <span class="badge">Split {slide['split']}</span>
      {correct_badge}
    </div>
  </div>
  <p style="font-size: 0.8em; color: #999; margin-bottom: 8px;">{slide['stem']}</p>
  <img class="overview-img" src="{overview_uri}" alt="Heatmap overview for {slide['sample_id']}" loading="lazy">
"""

    if slide["top_tiles"]:
        html += '  <h4 style="margin-top: 12px; font-size: 0.9em;">Top Attended Tiles</h4>\n'
        html += '  <div class="tiles-row">\n'
        for tile_path in slide["top_tiles"]:
            uri = embed_image(tile_path)
            html += f'    <img src="{uri}" alt="top tile" loading="lazy">\n'
        html += "  </div>\n"

    if slide["bottom_tiles"]:
        html += '  <h4 style="margin-top: 12px; font-size: 0.9em;">Least Attended Tiles</h4>\n'
        html += '  <div class="tiles-row">\n'
        for tile_path in slide["bottom_tiles"]:
            uri = embed_image(tile_path)
            html += f'    <img src="{uri}" alt="bottom tile" loading="lazy">\n'
        html += "  </div>\n"

    html += "</div>\n"
    return html


def make_heatmap_gallery(exp, slides, total_available) -> str:
    """Generate heatmap gallery section for an experiment."""
    task = exp["task"]
    gt_label = exp["ground_truth_label"]

    html = f"""
<h3>Heatmap Gallery — Attention Maps</h3>
<div class="note">
<strong>Note:</strong> Showing {len(slides)} representative slides out of {total_available} available
(all 5 crossval splits). Each heatmap uses the model from the split where that slide was in the
<em>test set</em>, ensuring no data leakage. Top and bottom 8 most/least attended tiles are extracted.
"""

    if task == "regression":
        html += """<br><br>
<strong>Warm</strong> regions indicate areas the model attends to most when predicting blast percentage.
The attention map uses a single channel (regression) rather than per-class gradients.
"""
    else:
        html += """<br><br>
<strong>Red/warm</strong> regions indicate high attention for that category prediction.
<strong>Blue/cool</strong> regions indicate low attention.
"""

    html += "</div>\n"

    # Group slides
    groups = {}
    for s in slides:
        groups.setdefault(s["group"], []).append(s)

    # Define group ordering and colors
    if task == "regression":
        group_order = ["low (<5%)", "intermediate (5-19%)", "high (≥20%)"]
        group_colors = {
            "low (<5%)": ("#28a745", "#d4edda"),
            "intermediate (5-19%)": ("#fd7e14", "#fff3cd"),
            "high (≥20%)": ("#dc3545", "#f8d7da"),
        }
    elif gt_label == "BLAST_SEVERITY":
        group_order = ["high", "intermediate", "low"]
        group_colors = {
            "high": ("#dc3545", "#f8d7da"),
            "intermediate": ("#fd7e14", "#fff3cd"),
            "low": ("#28a745", "#d4edda"),
        }
    elif gt_label == "HIGH_BLAST":
        group_order = ["yes", "no"]
        group_colors = {
            "yes": ("#dc3545", "#f8d7da"),
            "no": ("#28a745", "#d4edda"),
        }
    else:
        group_order = sorted(groups.keys())
        group_colors = {}

    for group_name in group_order:
        group_slides = groups.get(group_name, [])
        if not group_slides:
            continue

        border_color, bg_color = group_colors.get(group_name, ("#333", "#f0f0f0"))
        html += f"""
<div class="group-header" style="border-color: {border_color}; background: {bg_color};">
  Ground Truth: {group_name} — {len(group_slides)} slides shown
</div>
"""
        for slide in sorted(group_slides, key=lambda s: (s["sample_id"], s["stem"])):
            if task == "regression":
                html += make_slide_card_regression(slide)
            else:
                html += make_slide_card_classification(slide)

    return html


# =============================================================================
# Main report generation
# =============================================================================

CSS = """
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    line-height: 1.6; color: #333; max-width: 1400px;
    margin: 0 auto; padding: 20px; background: #f8f9fa;
  }
  h1 { font-size: 2em; color: #1a1a2e; border-bottom: 3px solid #0077b6;
       padding-bottom: 10px; margin-bottom: 20px; }
  h2 { font-size: 1.5em; color: #0077b6; margin-top: 40px; margin-bottom: 15px;
       border-bottom: 1px solid #ddd; padding-bottom: 8px; }
  h3 { font-size: 1.2em; color: #333; margin-top: 20px; margin-bottom: 10px; }
  .experiment-section {
    background: white; border-radius: 8px; padding: 25px;
    margin-bottom: 30px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  .metrics-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 15px; margin: 15px 0; }
  .metrics-grid-3 { grid-template-columns: 1fr 1fr 1fr; }
  .metric-card { background: #f0f7ff; border-radius: 8px; padding: 15px; text-align: center; }
  .metric-value { font-size: 1.8em; font-weight: bold; color: #0077b6; }
  .metric-label { font-size: 0.85em; color: #666; margin-top: 4px; }
  .svg-container { text-align: center; margin: 15px 0; overflow-x: auto; }
  .svg-container svg { max-width: 100%; height: auto; }
  table { border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }
  th, td { border: 1px solid #ddd; padding: 6px 10px; text-align: center; }
  th { background: #0077b6; color: white; font-weight: 600; }
  tr:nth-child(even) { background: #f8f9fa; }
  tr:hover { background: #e8f4fd; }
  .slide-card {
    background: white; border-radius: 8px; padding: 15px;
    margin-bottom: 15px; box-shadow: 0 1px 3px rgba(0,0,0,0.1);
  }
  .slide-header { display: flex; justify-content: space-between; align-items: center;
                   margin-bottom: 8px; flex-wrap: wrap; gap: 8px; }
  .slide-meta { display: flex; gap: 8px; flex-wrap: wrap; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 10px;
           font-size: 0.8em; font-weight: 600; background: #e9ecef; color: #333; }
  .badge-yes { background: #d4edda; color: #155724; }
  .badge-no { background: #f8d7da; color: #721c24; }
  .badge-correct { background: #cce5ff; color: #004085; }
  .badge-wrong { background: #fff3cd; color: #856404; }
  .overview-img { max-width: 100%; border-radius: 4px; margin: 8px 0; }
  .tiles-row { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
  .tiles-row img { width: 110px; height: 110px; object-fit: cover;
                   border-radius: 4px; border: 1px solid #ddd; }
  .group-header { font-size: 1.2em; color: #333; margin: 25px 0 12px;
                  padding: 10px 15px; border-left: 4px solid; border-radius: 4px; }
  .note { background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px;
          padding: 15px; margin: 15px 0; font-size: 0.95em; }
  .note strong { color: #856404; }
  .summary-box { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                 gap: 12px; margin: 15px 0; }
  .summary-item { background: #f0f7ff; padding: 10px; border-radius: 6px; }
  .summary-item .label { font-size: 0.82em; color: #666; }
  .summary-item .value { font-size: 1em; font-weight: 600; color: #333; }
  .per-class-metrics { margin: 15px 0; }
  .class-metric-block { margin-bottom: 15px; padding: 10px; background: #fafafa;
                        border-radius: 6px; }
  .class-metric-block h4 { margin-bottom: 8px; }
  .toc { background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px;
         box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
  .toc ul { list-style: none; padding-left: 0; }
  .toc li { margin: 8px 0; }
  .toc a { color: #0077b6; text-decoration: none; font-weight: 500; }
  .toc a:hover { text-decoration: underline; }
  .caption { font-size: 0.9em; color: #666; font-style: italic; margin: 8px 0 15px; }
  .footer { text-align: center; color: #999; font-size: 0.85em; margin-top: 40px;
            padding-top: 20px; border-top: 1px solid #ddd; }
  .comparison-table { margin: 20px 0; }
  .comparison-table td { font-weight: 500; }
  .comparison-table td:first-child { text-align: left; font-weight: 600; }
  @media (max-width: 768px) {
    .metrics-grid, .metrics-grid-3 { grid-template-columns: 1fr; }
    .summary-box { grid-template-columns: 1fr 1fr; }
  }
</style>
"""


def generate_report(max_per_class: int = 10):
    """Generate the combined HTML report."""
    clini_df, slide_df, fname_to_sample = load_clini_and_slide()

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    date_short = datetime.now().strftime("%Y-%m-%d")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AML Blast Analysis — STAMP Crossval Report</title>
{CSS}
</head>
<body>

<h1>AML Blast Analysis — STAMP Crossvalidation Report</h1>
<p style="color: #666; margin-bottom: 20px;">
  Generated {now} &mdash; STAMP Framework, UNI2 features, ViT MIL model
</p>

<div class="toc">
<h3>Table of Contents</h3>
<ul>
  <li><a href="#overview">Overview &amp; Cross-Experiment Comparison</a></li>
  <li><a href="#exp-blast_percent">1. BLAST_PERCENT — Regression</a></li>
  <li><a href="#exp-blast_severity">2. BLAST_SEVERITY — 3-class Classification</a></li>
  <li><a href="#exp-high_blast">3. HIGH_BLAST — Binary Classification</a></li>
  <li><a href="#discussion">Discussion</a></li>
</ul>
</div>

<div class="experiment-section" id="overview">
<h2>Overview &amp; Cross-Experiment Comparison</h2>

<p>
Three prediction targets derived from the <strong>BLAST_PERCENT</strong> column in AML bone marrow aspirates.
All experiments use the same dataset (489 samples, 69 biological patients), feature extractor (UNI2, 3072-dim),
and MIL architecture (ViT, 2-layer, 8-head, 512-dim, max_epochs=32, patience=16).
</p>

<table class="comparison-table">
<tr>
  <th>Experiment</th>
  <th>Task</th>
  <th>Classes</th>
  <th>Primary Metric</th>
  <th>Result</th>
</tr>
"""

    # Collect summary metrics for the comparison table
    # BLAST_PERCENT
    bp_stats = load_regression_stats(EXPERIMENTS["blast_percent"])
    html += f"""
<tr>
  <td>BLAST_PERCENT</td>
  <td>Regression</td>
  <td>Continuous (0–91%)</td>
  <td>Pearson r</td>
  <td><strong>{bp_stats['pearson_r']:.3f}</strong> (MAE={bp_stats['mae']:.1f}%)</td>
</tr>
"""

    # BLAST_SEVERITY
    bs_agg, _ = load_classification_stats(EXPERIMENTS["blast_severity"])
    bs_classes = {}
    for _, row in bs_agg.iterrows():
        cls = str(row.iloc[0])
        if cls not in ("", "nan"):
            bs_classes[cls] = float(row.iloc[1])
    html += f"""
<tr>
  <td>BLAST_SEVERITY</td>
  <td>3-class Classification</td>
  <td>high / intermediate / low</td>
  <td>AUROC (per-class)</td>
  <td><strong>high={bs_classes.get('high', 0):.3f}</strong>, int={bs_classes.get('intermediate', 0):.3f}, low={bs_classes.get('low', 0):.3f}</td>
</tr>
"""

    # HIGH_BLAST
    hb_agg, _ = load_classification_stats(EXPERIMENTS["high_blast"])
    hb_auroc = 0.0
    hb_auprc = 0.0
    for _, row in hb_agg.iterrows():
        cls = str(row.iloc[0])
        if cls == "yes":
            hb_auroc = float(row.iloc[1])
            hb_auprc = float(row.iloc[4])
    html += f"""
<tr>
  <td>HIGH_BLAST</td>
  <td>Binary Classification</td>
  <td>yes (&ge;20%) / no</td>
  <td>AUROC</td>
  <td><strong>{hb_auroc:.3f}</strong> (AUPRC={hb_auprc:.3f})</td>
</tr>
"""

    html += """
</table>

<p style="margin-top: 15px;">
<strong>Key finding:</strong> The binary HIGH_BLAST classifier achieves the best discriminative performance
(AUROC=0.929), suggesting the model can reliably identify high-blast cases. The 3-class BLAST_SEVERITY
model shows strong performance for high (0.912) and low (0.864) classes but struggles with the
intermediate class (0.783), consistent with the small sample size (n=55) and the inherent difficulty
of distinguishing 5–19% blast ranges. The regression model shows moderate correlation (r=0.680) but
high RMSE (24.5%) driven by the skewed distribution.
</p>
</div>
"""

    # === Experiment 1: BLAST_PERCENT ===
    exp_bp = EXPERIMENTS["blast_percent"]
    html += make_regression_section(exp_bp)
    slides_bp, total_bp = collect_heatmap_slides(
        exp_bp, clini_df, fname_to_sample, max_per_class=max_per_class
    )
    html += make_heatmap_gallery(exp_bp, slides_bp, total_bp)
    html += "</div>\n"

    # === Experiment 2: BLAST_SEVERITY ===
    exp_bs = EXPERIMENTS["blast_severity"]
    html += make_classification_section(exp_bs)
    slides_bs, total_bs = collect_heatmap_slides(
        exp_bs, clini_df, fname_to_sample, max_per_class=max_per_class
    )
    html += make_heatmap_gallery(exp_bs, slides_bs, total_bs)
    html += "</div>\n"

    # === Experiment 3: HIGH_BLAST ===
    exp_hb = EXPERIMENTS["high_blast"]
    html += make_classification_section(exp_hb)
    slides_hb, total_hb = collect_heatmap_slides(
        exp_hb, clini_df, fname_to_sample, max_per_class=max_per_class
    )
    html += make_heatmap_gallery(exp_hb, slides_hb, total_hb)
    html += "</div>\n"

    # === Discussion ===
    html += f"""
<div class="experiment-section" id="discussion">
<h2>Discussion</h2>

<ul style="list-style: disc; padding-left: 25px; line-height: 2;">
<li><strong>HIGH_BLAST is the strongest classifier:</strong> AUROC of 0.929 (95% CI: 0.877–0.981)
for identifying &ge;20% blast cases. This clinically relevant threshold separates active disease from
remission/low-burden states, and the model achieves this with high confidence across all 5 folds.</li>

<li><strong>BLAST_SEVERITY captures the ordinal structure:</strong> The 3-class model's AUROC gradient
(high=0.912 &gt; low=0.864 &gt; intermediate=0.783) mirrors the biological separability — extreme cases
(very low or very high blasts) have more distinctive morphological features than borderline cases.
The intermediate class (5–19%) is small (n=55) and morphologically heterogeneous.</li>

<li><strong>Regression faces distribution challenges:</strong> While Pearson r=0.680 indicates the model
captures the rank ordering of blast percentages, the negative R&sup2; (−0.016) and high RMSE (24.5%)
reflect the difficulty of precise point estimation on a heavily right-skewed distribution
(median=3%, IQR ~0–12%). The model tends to regress toward the mean, underestimating high-blast cases
and overestimating low-blast cases.</li>

<li><strong>Biological relevance of attention patterns:</strong> Compare heatmaps across the three experiments
for the same slides. In HIGH_BLAST, high-attention (red) regions should correspond to blast-dense areas.
In BLAST_SEVERITY, the per-class heatmaps (high/intermediate/low) should show complementary attention patterns.
The regression heatmaps use attention rollout rather than class-specific GradCAM.</li>

<li><strong>Clinical utility:</strong> For treatment response monitoring, the binary HIGH_BLAST classifier
is most directly actionable — it answers "does this patient have clinically significant blast burden?"
The BLAST_SEVERITY model adds nuance by distinguishing intermediate from extreme cases. The regression
model provides a continuous estimate useful for trend monitoring across timepoints.</li>

<li><strong>Heatmap coverage:</strong> All {total_bp + total_bs + total_hb} slide heatmaps were generated
across the three experiments ({total_bp} + {total_bs} + {total_hb} per experiment), with representative
samples shown in this report. Full heatmap galleries are available in the output directories.</li>

<li><strong>Limitations:</strong> (1) All three targets are derived from the same BLAST_PERCENT column,
so the experiments are not independent. (2) The dataset has 489 samples from only 69 patients — multiple
biopsies per patient introduce correlation that crossvalidation at the sample level does not fully address.
(3) The feature extractor (UNI2) was pretrained on general pathology, not specifically on bone marrow
morphology.</li>
</ul>
</div>

<div class="footer">
Generated by STAMP — Spatial Tissue Analysis and Modeling Pipeline<br>
Report generated on {date_short} | UNI2 features | ViT MIL model | 5-fold crossvalidation<br>
Dataset: 489 AML bone marrow aspirate samples, 69 biological patients
</div>

</body>
</html>"""

    return html


def main():
    parser = argparse.ArgumentParser(description="Generate combined blast analysis report")
    parser.add_argument(
        "--max-slides",
        type=int,
        default=10,
        help="Max heatmap slides per class per experiment (default: 10). Use 0 for all slides.",
    )
    args = parser.parse_args()

    report_dir = DATA_ROOT / "blast_analysis_report"
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Generating combined blast analysis report...")
    print(f"  Max slides per class: {args.max_slides}")

    html = generate_report(max_per_class=args.max_slides)

    report_path = report_dir / "aml_blast_analysis_report.html"
    report_path.write_text(html)

    size_mb = report_path.stat().st_size / 1024 / 1024
    print(f"\nReport generated: {report_path}")
    print(f"Report size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
