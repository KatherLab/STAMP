#!/usr/bin/env python3
"""Generate a self-contained HTML report combining STAMP crossval performance
metrics with GradCAM heatmap visualizations.

The report includes:
  1. Experiment summary (dataset, features, model, crossval setup)
  2. Performance metrics (ROC curve, PR curve, per-fold stats table)
  3. Heatmap gallery grouped by ground-truth class
  4. Discussion pointers

Usage:
    python scripts/generate_report.py
"""

import base64
import csv
from datetime import datetime
from pathlib import Path

import pandas as pd

# === Configuration ===
BASE_DIR = Path(
    "/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_response_uni2"
)
STATS_DIR = BASE_DIR / "statistics"
HEATMAP_DIR = BASE_DIR / "heatmaps"
CROSSVAL_DIR = BASE_DIR / "crossval"
REPORT_DIR = BASE_DIR / "report"
SLIDE_TABLE = Path("/home/jeff/Projects/STAMP/tables/stamp_slide.csv")
CLINI_TABLE = Path("/home/jeff/Projects/STAMP/tables/stamp_clini.csv")


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


def load_stats():
    """Load the per-fold and aggregated statistics CSVs."""
    individual = pd.read_csv(STATS_DIR / "RESPONSE_CR_categorical-stats_individual.csv")
    aggregated = pd.read_csv(STATS_DIR / "RESPONSE_CR_categorical-stats_aggregated.csv")
    return individual, aggregated


def collect_heatmap_slides():
    """Collect all heatmap slide info with predictions and ground truth."""
    slide_df = pd.read_csv(SLIDE_TABLE)
    clini_df = pd.read_csv(CLINI_TABLE)

    fname_to_sample = dict(zip(slide_df["FILENAME"], slide_df["SAMPLE_ID"]))
    sample_response = dict(zip(clini_df["SAMPLE_ID"], clini_df["RESPONSE_CR"]))

    # Load all predictions
    all_preds = {}
    for split_i in range(5):
        pred_file = CROSSVAL_DIR / f"split-{split_i}" / "patient-preds.csv"
        if not pred_file.exists():
            continue
        preds = pd.read_csv(pred_file)
        for _, row in preds.iterrows():
            all_preds[row["SAMPLE_ID"]] = {
                "split": split_i,
                "pred": row["pred"],
                "prob_yes": row["RESPONSE_CR_yes"],
                "prob_no": row["RESPONSE_CR_no"],
                "gt": row["RESPONSE_CR"],
            }

    slides = []
    for split_dir in sorted(HEATMAP_DIR.glob("split-*")):
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
                continue  # Skip slides with no overview (e.g., 1-tile failures)

            # Collect top/bottom tiles
            tiles_dir = slide_dir / "tiles"
            top_tiles = sorted(
                [t for ext in ("*.png", "*.jpg", "*.jpeg") for t in tiles_dir.glob(f"top_*{ext}")]
            )[:4] if tiles_dir.exists() else []
            bottom_tiles = sorted(
                [t for ext in ("*.png", "*.jpg", "*.jpeg") for t in tiles_dir.glob(f"bottom_*{ext}")]
            )[:4] if tiles_dir.exists() else []

            slides.append(
                {
                    "stem": stem,
                    "sample_id": sample_id,
                    "split": split_i,
                    "gt": info.get("gt", "?"),
                    "pred": info.get("pred", "?"),
                    "prob_yes": info.get("prob_yes", 0),
                    "prob_no": info.get("prob_no", 0),
                    "overview_path": overview_path,
                    "top_tiles": top_tiles,
                    "bottom_tiles": bottom_tiles,
                    "correct": info.get("gt") == info.get("pred"),
                }
            )

    return slides


def generate_html():
    individual, aggregated = load_stats()
    slides = collect_heatmap_slides()

    # Group slides by ground truth
    yes_slides = [s for s in slides if s["gt"] == "yes"]
    no_slides = [s for s in slides if s["gt"] == "no"]

    # Aggregate AUROC for summary
    agg_yes = aggregated[aggregated.iloc[:, 0] == "yes"].iloc[0] if len(aggregated) > 0 else None

    # Extract mean AUROC from aggregated CSV
    # The columns are: class, mean, 95%_low, 95%_high (for roc_auc_score)
    mean_auroc = float(aggregated.iloc[1, 1])  # yes class, roc_auc_score mean
    auroc_low = float(aggregated.iloc[1, 2])
    auroc_high = float(aggregated.iloc[1, 3])
    mean_auprc = float(aggregated.iloc[1, 4])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AML Response Crossval Report — STAMP</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; line-height: 1.6; color: #333; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
  h1 {{ font-size: 2em; color: #1a1a2e; border-bottom: 3px solid #0077b6; padding-bottom: 10px; margin-bottom: 20px; }}
  h2 {{ font-size: 1.5em; color: #0077b6; margin-top: 40px; margin-bottom: 15px; border-bottom: 1px solid #ddd; padding-bottom: 8px; }}
  h3 {{ font-size: 1.2em; color: #333; margin-top: 20px; margin-bottom: 10px; }}
  .section {{ background: white; border-radius: 8px; padding: 25px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .metrics-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  .metric-card {{ background: #f0f7ff; border-radius: 8px; padding: 15px; text-align: center; }}
  .metric-value {{ font-size: 2em; font-weight: bold; color: #0077b6; }}
  .metric-label {{ font-size: 0.9em; color: #666; }}
  .svg-container {{ text-align: center; margin: 15px 0; }}
  .svg-container svg {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 0.9em; }}
  th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: center; }}
  th {{ background: #0077b6; color: white; font-weight: 600; }}
  tr:nth-child(even) {{ background: #f8f9fa; }}
  tr:hover {{ background: #e8f4fd; }}
  .slide-card {{ background: white; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
  .slide-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }}
  .slide-meta {{ display: flex; gap: 15px; flex-wrap: wrap; }}
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 12px; font-size: 0.85em; font-weight: 600; }}
  .badge-yes {{ background: #d4edda; color: #155724; }}
  .badge-no {{ background: #f8d7da; color: #721c24; }}
  .badge-correct {{ background: #cce5ff; color: #004085; }}
  .badge-wrong {{ background: #fff3cd; color: #856404; }}
  .overview-img {{ max-width: 100%; border-radius: 4px; margin: 10px 0; }}
  .tiles-row {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 10px; }}
  .tiles-row img {{ width: 120px; height: 120px; object-fit: cover; border-radius: 4px; border: 1px solid #ddd; }}
  .group-header {{ font-size: 1.3em; color: #333; margin: 30px 0 15px; padding: 10px 15px; border-left: 4px solid; border-radius: 4px; }}
  .group-yes {{ border-color: #28a745; background: #d4edda; }}
  .group-no {{ border-color: #dc3545; background: #f8d7da; }}
  .note {{ background: #fff3cd; border: 1px solid #ffc107; border-radius: 8px; padding: 15px; margin: 15px 0; }}
  .note strong {{ color: #856404; }}
  .summary-box {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 15px 0; }}
  .summary-item {{ background: #f0f7ff; padding: 12px; border-radius: 6px; }}
  .summary-item .label {{ font-size: 0.85em; color: #666; }}
  .summary-item .value {{ font-size: 1.1em; font-weight: 600; color: #333; }}
  .footer {{ text-align: center; color: #999; font-size: 0.85em; margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; }}
</style>
</head>
<body>

<h1>AML Response Prediction — Crossvalidation Report</h1>
<p style="color: #666; margin-bottom: 20px;">Generated {datetime.now().strftime('%Y-%m-%d %H:%M')} &mdash; STAMP Framework</p>

<div class="section">
<h2>1. Experiment Summary</h2>

<div class="summary-box">
  <div class="summary-item"><div class="label">Dataset</div><div class="value">AML Bone Marrow Aspirates</div></div>
  <div class="summary-item"><div class="label">Samples (SAMPLE_ID)</div><div class="value">489</div></div>
  <div class="summary-item"><div class="label">Biological Patients</div><div class="value">69</div></div>
  <div class="summary-item"><div class="label">Target</div><div class="value">RESPONSE_CR (blast &lt;5%)</div></div>
  <div class="summary-item"><div class="label">Class Balance</div><div class="value">yes=336 / no=153</div></div>
  <div class="summary-item"><div class="label">Feature Extractor</div><div class="value">UNI2 (3072-dim tiles)</div></div>
  <div class="summary-item"><div class="label">MIL Model</div><div class="value">ViT (2-layer, 8-head, 512-dim)</div></div>
  <div class="summary-item"><div class="label">Crossval</div><div class="value">5-fold StratifiedKFold</div></div>
</div>

<p style="margin-top: 15px;">
<strong>Task:</strong> Binary classification predicting complete remission (CR) response in AML patients from
H&amp;E-stained bone marrow aspirate whole-slide images. Each sample represents a bone marrow biopsy section
(parts A, B, C) from a timepoint (year 19 or 26). The positive class ("yes") indicates blast percentage
below 5% (complete remission), while "no" indicates &ge;5% blasts.
</p>
</div>

<div class="section">
<h2>2. Performance Metrics</h2>

<div class="metrics-grid">
  <div class="metric-card">
    <div class="metric-value">{mean_auroc:.3f}</div>
    <div class="metric-label">Mean AUROC (95% CI: {auroc_low:.3f}&ndash;{auroc_high:.3f})</div>
  </div>
  <div class="metric-card">
    <div class="metric-value">{mean_auprc:.3f}</div>
    <div class="metric-label">Mean AUPRC (yes class)</div>
  </div>
</div>

<h3>ROC Curve (5-fold)</h3>
<div class="svg-container">
{embed_svg(STATS_DIR / "roc-curve_RESPONSE_CR=yes.svg")}
</div>

<h3>Precision-Recall Curve (5-fold)</h3>
<div class="svg-container">
{embed_svg(STATS_DIR / "pr-curve_RESPONSE_CR=yes.svg")}
</div>

<h3>Per-Fold Statistics</h3>
{_make_stats_table(individual)}
</div>

<div class="section">
<h2>3. Heatmap Gallery — GradCAM Attention Maps</h2>

<div class="note">
<strong>Note:</strong> Heatmaps are available for {len(slides)} out of 529 slides (covering
{len(set(s['sample_id'] for s in slides))} samples from
{len(set(s['sample_id'].rsplit('_', 3)[0] + '_' + s['sample_id'].rsplit('_', 3)[1] for s in slides))} biological patients).
Only slides with available WSI (.ndpi) files could be visualized. Each heatmap uses the model
from the crossval split where that slide was in the <em>test set</em>, ensuring no data leakage.
<br><br>
<strong>Red</strong> regions indicate areas where the model attends for the <em>positive</em> (yes/CR) prediction.
<strong>Blue</strong> regions indicate areas supporting the <em>negative</em> (no/non-CR) prediction.
The top and bottom 8 most/least attended tiles are extracted for detailed inspection.
</div>

<div class="group-header group-yes">CR Responders (Ground Truth: YES) &mdash; {len(yes_slides)} slides</div>
"""

    for slide in sorted(yes_slides, key=lambda s: (s["sample_id"], s["stem"])):
        html += _make_slide_card(slide)

    html += f"""
<div class="group-header group-no">Non-Responders (Ground Truth: NO) &mdash; {len(no_slides)} slides</div>
"""

    for slide in sorted(no_slides, key=lambda s: (s["sample_id"], s["stem"])):
        html += _make_slide_card(slide)

    html += """
</div>

<div class="section">
<h2>4. Discussion Points</h2>

<ul style="list-style: disc; padding-left: 25px; line-height: 2;">
<li><strong>Model performance:</strong> Mean AUROC of {auroc:.3f} suggests the model can distinguish CR responders
from non-responders at the sample level. However, note the class imbalance (69% positive) — the AUPRC
and per-fold F1 scores provide more informative assessment.</li>

<li><strong>Biological relevance of attention:</strong> The key question for Dr. Janssen is whether high-attention
(red) regions correspond to blast-rich areas in non-responders and blast-poor areas in responders.
Compare the GradCAM overlays above with expert morphological assessment of the same tissue regions.</li>

<li><strong>Misclassifications to examine:</strong> Slides where the model was wrong (highlighted with
"Incorrect" badges) are particularly informative — check whether the model attends to stroma, adipose tissue,
or other non-diagnostic regions in those cases.</li>

<li><strong>Limited heatmap coverage:</strong> Only {n_wsi} out of 529 slides have WSI files available for
heatmap generation. Additional WSIs would strengthen the interpretability analysis. The current subset
covers 7 biological patients across both classes.</li>

<li><strong>Sample vs. patient-level analysis:</strong> Multiple sections (A, B, C) from the same patient may
show different attention patterns. Comparing within-patient consistency could reveal whether the model
captures biologically stable features or is influenced by sectioning artifacts.</li>
</ul>
</div>

<div class="footer">
Generated by STAMP — Spatial Tissue Analysis and Modeling Pipeline<br>
Report generated on {date} | UNI2 features | ViT MIL model | 5-fold crossvalidation
</div>

</body>
</html>""".format(
        auroc=mean_auroc,
        n_wsi=len(slides),
        date=datetime.now().strftime("%Y-%m-%d"),
    )

    return html


def _make_stats_table(individual: pd.DataFrame) -> str:
    """Generate an HTML table from the per-fold stats DataFrame."""
    # Parse the individual CSV which has columns: fold, class, count, roc_auc, ap, f1, p_value
    html = "<table>\n<tr>"
    html += "<th>Fold</th><th>Class</th><th>Count</th><th>AUROC</th><th>AUPRC</th><th>F1</th><th>p-value</th>"
    html += "</tr>\n"

    for _, row in individual.iterrows():
        fold = row.iloc[0]
        cls = row.iloc[1]
        count = int(row.iloc[2])
        auroc = float(row.iloc[3])
        ap = float(row.iloc[4])
        f1 = float(row.iloc[5])
        pval = float(row.iloc[6])

        badge = "badge-yes" if cls == "yes" else "badge-no"
        html += f'<tr><td>{fold}</td><td><span class="badge {badge}">{cls}</span></td>'
        html += f"<td>{count}</td><td>{auroc:.4f}</td><td>{ap:.4f}</td><td>{f1:.4f}</td>"
        html += f"<td>{pval:.2e}</td></tr>\n"

    html += "</table>\n"
    return html


def _make_slide_card(slide: dict) -> str:
    """Generate HTML card for a single slide with heatmap."""
    correct_badge = (
        '<span class="badge badge-correct">Correct</span>'
        if slide["correct"]
        else '<span class="badge badge-wrong">Incorrect</span>'
    )
    gt_badge = (
        f'<span class="badge badge-yes">GT: {slide["gt"]}</span>'
        if slide["gt"] == "yes"
        else f'<span class="badge badge-no">GT: {slide["gt"]}</span>'
    )

    overview_uri = embed_image(slide["overview_path"])

    html = f"""
<div class="slide-card">
  <div class="slide-header">
    <h3 style="font-size: 1em; margin: 0;">{slide['sample_id']}</h3>
    <div class="slide-meta">
      {gt_badge}
      <span class="badge">Pred: {slide['pred']}</span>
      <span class="badge">P(yes)={slide['prob_yes']:.4f}</span>
      <span class="badge">Split {slide['split']}</span>
      {correct_badge}
    </div>
  </div>
  <p style="font-size: 0.8em; color: #999; margin-bottom: 8px;">{slide['stem']}</p>
  <img class="overview-img" src="{overview_uri}" alt="Heatmap overview for {slide['sample_id']}">
"""

    if slide["top_tiles"]:
        html += '  <h4 style="margin-top: 12px; font-size: 0.9em;">Top Attended Tiles</h4>\n'
        html += '  <div class="tiles-row">\n'
        for tile_path in slide["top_tiles"]:
            uri = embed_image(tile_path)
            html += f'    <img src="{uri}" alt="top tile">\n'
        html += "  </div>\n"

    if slide["bottom_tiles"]:
        html += '  <h4 style="margin-top: 12px; font-size: 0.9em;">Least Attended Tiles</h4>\n'
        html += '  <div class="tiles-row">\n'
        for tile_path in slide["bottom_tiles"]:
            uri = embed_image(tile_path)
            html += f'    <img src="{uri}" alt="bottom tile">\n'
        html += "  </div>\n"

    html += "</div>\n"
    return html


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    html = generate_html()
    report_path = REPORT_DIR / "aml_response_report.html"
    report_path.write_text(html)
    print(f"Report generated: {report_path}")
    print(f"Report size: {report_path.stat().st_size / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
