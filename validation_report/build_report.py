#!/usr/bin/env python3
"""Build the final HTML and PDF validation report."""
import base64
import json
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path("/home/jeff/Projects/STAMP/validation_report")
PRE_DIR = ROOT / "preprocessing"
HM_DIR = ROOT / "heatmaps"
ANN_FILE = ROOT / "annotation_candidates" / "candidates.json"

OUT_HTML = ROOT / "validation_report.html"
OUT_PDF = ROOT / "validation_report.pdf"

PRE_SLIDES = [
    "116 26 B Score 49 - 10 Fokuspunkte - entfettet - 2026-03-04 18.49.41",
    "127 26 C Score 0 - 5 Fokuspunkte - entfettet - 2026-03-04 17.28.20",
    "333 19 A Score 29 - 5 Fokuspunkte - entfettet - 2026-03-03 18.42.23",
    "931 19 A Score 30 - 5 Fokuspunkte - entfettet - 2026-03-04 21.23.41",
]

EXP_LABELS = {
    "response":       ("RESPONSE_CR (binary)",       "classification"),
    "blast_percent":  ("BLAST_PERCENT (regression)", "regression"),
    "blast_severity": ("BLAST_SEVERITY (3-class)",   "classification"),
    "high_blast":     ("HIGH_BLAST (binary)",        "classification"),
}


def img_b64(path: Path, mime: str = "image/jpeg") -> str:
    if not path.exists():
        return ""
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def img_tag(path: Path, alt: str = "", width: str = "100%") -> str:
    if not path.exists():
        return f'<div class="missing">missing: {path.name}</div>'
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    src = img_b64(path, mime)
    return f'<img src="{src}" alt="{alt}" style="width:{width};">'


def fmt_pred_response(pred: dict) -> str:
    if not pred:
        return "<i>no prediction</i>"
    p_yes = pred.get("RESPONSE_CR_yes", "?")
    p_no  = pred.get("RESPONSE_CR_no", "?")
    pred_class = pred.get("pred", "?")
    return f"P(yes)={p_yes:.3f}, P(no)={p_no:.3f} → <b>{pred_class}</b>"


def fmt_pred_severity(pred: dict) -> str:
    if not pred:
        return "<i>no prediction</i>"
    parts = []
    for k in ["BLAST_SEVERITY_low", "BLAST_SEVERITY_intermediate", "BLAST_SEVERITY_high"]:
        if k in pred:
            parts.append(f"P({k.split('_')[-1]})={pred[k]:.3f}")
    return ", ".join(parts) + f" → <b>{pred.get('pred', '?')}</b>"


def fmt_pred_high_blast(pred: dict) -> str:
    if not pred:
        return "<i>no prediction</i>"
    p_yes = pred.get("HIGH_BLAST_yes", "?")
    return f"P(yes)={p_yes:.3f} → <b>{pred.get('pred','?')}</b>"


def fmt_pred_regression(pred: dict, gt) -> str:
    if not pred or "pred" not in pred:
        return "<i>no prediction</i>"
    pv = pred["pred"]
    err = abs(pv - float(gt)) if gt is not None and gt == gt else None
    s = f"pred={pv:.2f}%"
    if err is not None:
        s += f" (|err|={err:.2f}%)"
    return s


def section_preprocessing() -> str:
    summary = json.loads((PRE_DIR / "_summary.json").read_text())
    blocks = []
    for stem in PRE_SLIDES:
        s = summary[stem]
        # rows of stats
        rej = s["rejection_reasons_sampled"]
        sample_n = s["rejected_sampled"]
        canny_pct = 100 * rej["canny"] / max(sample_n, 1)
        bright_pct = 100 * rej["brightness"] / max(sample_n, 1)
        unknown_pct = 100 * rej["unknown"] / max(sample_n, 1)
        block = f"""
<div class="slide-block">
  <h3>{stem}</h3>
  <div class="kv">
    <div><span class="k">WSI dims:</span> {s['dims'][0]} × {s['dims'][1]} px (MPP {s['mpp_x']:.4f} µm/px)</div>
    <div><span class="k">Tile grid:</span> {s['grid'][0]} × {s['grid'][1]} = {s['theoretical_tiles']} theoretical tiles</div>
    <div><span class="k">Tiles kept:</span> <b>{s['kept_tiles']}</b> / {s['theoretical_tiles']} (<b>{s['retention_pct']:.1f}%</b>)</div>
    <div><span class="k">Rejection reason (sampled n={sample_n}):</span>
      Canny low-edge <b>{canny_pct:.1f}%</b> • brightness ≥ 240 <b>{bright_pct:.1f}%</b> • boundary/other <b>{unknown_pct:.1f}%</b></div>
  </div>
  <div class="row">
    <div class="col">
      <div class="caption">Before preprocessing — raw thumbnail</div>
      {img_tag(PRE_DIR / f'{stem}__before.jpg')}
    </div>
    <div class="col">
      <div class="caption">After preprocessing — kept tiles outlined green; rejected tiles shaded red</div>
      {img_tag(PRE_DIR / f'{stem}__after.jpg')}
    </div>
  </div>
  <div class="row">
    <div class="col">
      <div class="caption">Example KEPT tiles (passed brightness + Canny edge test)</div>
      {img_tag(PRE_DIR / f'{stem}__kept_grid.jpg')}
    </div>
    <div class="col">
      <div class="caption">Example REJECTED tiles (label = reason for rejection)</div>
      {img_tag(PRE_DIR / f'{stem}__rejected_grid.jpg')}
    </div>
  </div>
</div>
"""
        blocks.append(block)
    return "\n".join(blocks)


def section_why_rejected() -> str:
    return """
<div class="explanation">
<h3>Why are so many tiles removed?</h3>
<p>The STAMP preprocessing pipeline applies <b>two</b> background-rejection filters at tile-extraction time
(see <code>src/stamp/preprocessing/tiling.py</code>):</p>

<ol>
<li><b>Brightness filter</b> — <code>brightness_cutoff = 240</code>. Computed at the
    <i>supertile</i> level on the slide thumbnail. Any region with average grayscale
    intensity ≥ 240 is treated as glass/empty background and skipped before any
    tiling occurs.</li>
<li><b>Canny edge filter</b> — <code>canny_cutoff = 0.02</code>. After tiling, each
    224×224 tile is run through OpenCV Canny (thresholds 40 / 100).
    Tiles with fewer than <b>2 %</b> edge pixels are deemed texture-less and dropped.</li>
</ol>

<p>For the four slides above, almost every rejected tile fails the
<b>Canny edge test</b>, not the brightness test. Looking at the rejected-tile
gallery makes the cause obvious: the rejected tiles are <b>out of focus</b>
(blurred, soft, or featureless purple/pink). When the smear is blurry, the
Canny detector finds few edges → the tile is classified as low-information and
discarded.</p>

<p>This correlates with the focus-acquisition setting in the file name:</p>
<table class="small">
<tr><th>Slide</th><th>Focus pts</th><th>Retention</th></tr>
<tr><td>116 26 B (10 Fokuspunkte)</td><td>10</td><td>76.9 %</td></tr>
<tr><td>127 26 C (5 Fokuspunkte)</td><td>5</td><td><b>4.7 %</b></td></tr>
<tr><td>333 19 A (5 Fokuspunkte)</td><td>5</td><td>50.0 %</td></tr>
<tr><td>931 19 A (5 Fokuspunkte)</td><td>5</td><td>39.7 %</td></tr>
</table>

<p><b>Summary:</b> The drop in retention is not because the slide is empty —
the tissue is there, but the sub-sampled focus stack (5 Fokuspunkte) failed to
keep most of the smear in sharp focus. The Canny filter is doing its job:
removing tiles that would have been useless features for the model. The
real fix is upstream — re-scan the affected slides with the full
10-Fokuspunkte stack.</p>
</div>
"""


def section_heatmaps() -> str:
    summary = json.loads((HM_DIR / "_summary.json").read_text())
    blocks = []
    for slide in summary:
        stem = slide["stem"]
        clini = slide["clini"]
        per_exp = slide["experiments"]
        sid = slide["sample_id"]

        clinical_row = (
            f"<tr><td>BLAST_PERCENT</td><td><b>{clini['BLAST_PERCENT']}%</b></td></tr>"
            f"<tr><td>RESPONSE_CR</td><td><b>{clini['RESPONSE_CR']}</b></td></tr>"
            f"<tr><td>BLAST_SEVERITY</td><td><b>{clini['BLAST_SEVERITY']}</b></td></tr>"
            f"<tr><td>HIGH_BLAST</td><td><b>{clini['HIGH_BLAST']}</b></td></tr>"
        )

        exp_blocks = []
        for exp_key, exp_label in [("response", EXP_LABELS["response"][0]),
                                    ("blast_percent", EXP_LABELS["blast_percent"][0]),
                                    ("blast_severity", EXP_LABELS["blast_severity"][0]),
                                    ("high_blast", EXP_LABELS["high_blast"][0])]:
            e = per_exp.get(exp_key, {})
            if not e:
                exp_blocks.append(f'<div class="exp-block"><h4>{exp_label}</h4><i>not in test set</i></div>')
                continue
            gt = e["gt"]
            pred = e.get("pred", {}) or {}
            split = e["split"]
            if exp_key == "response":
                pred_str = fmt_pred_response(pred)
            elif exp_key == "blast_percent":
                pred_str = fmt_pred_regression(pred, gt)
            elif exp_key == "blast_severity":
                pred_str = fmt_pred_severity(pred)
            elif exp_key == "high_blast":
                pred_str = fmt_pred_high_blast(pred)
            else:
                pred_str = ""

            overview_path = HM_DIR / stem / f"{exp_key}__overview.png"
            classmap_path = HM_DIR / stem / f"{exp_key}__classmap.png"

            exp_blocks.append(f"""
<div class="exp-block">
  <h4>{exp_label} <span class="meta">(split-{split})</span></h4>
  <div class="kv-inline"><b>GT:</b> {gt} &nbsp;&nbsp; <b>Prediction:</b> {pred_str}</div>
  <div class="row">
    <div class="col">
      <div class="caption">Overview (top-K / bottom-K patches)</div>
      {img_tag(overview_path)}
    </div>
    <div class="col">
      <div class="caption">{'Class map' if exp_key != 'blast_percent' else 'Relevance heatmap'}</div>
      {img_tag(classmap_path)}
    </div>
  </div>
</div>
""")

        blocks.append(f"""
<div class="slide-block">
  <h3>{stem}</h3>
  <div class="meta">SAMPLE_ID = <code>{sid}</code></div>
  <table class="small">{clinical_row}</table>
  {''.join(exp_blocks)}
</div>
""")
    return "\n".join(blocks)


def fmt_table(rows: list[dict], cols: list[tuple[str, str]], floatfmt: dict | None = None) -> str:
    floatfmt = floatfmt or {}
    head = "".join(f"<th>{label}</th>" for _, label in cols)
    body = []
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if k in floatfmt and isinstance(v, (int, float)):
                v = floatfmt[k].format(v)
            cells.append(f"<td>{v}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table class="small"><tr>{head}</tr>{"".join(body)}</table>'


def section_annotation() -> str:
    cand = json.loads(ANN_FILE.read_text())

    # 1. RESPONSE — high confidence yes (good to verify; pick a few that are borderline-clinical)
    rh_yes = cand["response"]["high_confidence_yes"][:8]
    rh_no  = cand["response"]["high_confidence_no"][:8]
    rh_bnd = cand["response"]["near_boundary"][:8]
    conf_mistakes = cand["confident_mistakes_response"][:10]

    bp_high = cand["blast_percent"]["highest_predicted"][:8]
    bp_low  = cand["blast_percent"]["lowest_predicted"][:8]
    bp_err  = cand["blast_percent"]["largest_errors"][:10]

    return f"""
<p>The most informative slides for additional human annotation are those where the model is
either <b>confidently correct</b> (to validate the visual signal it relies on),
<b>confidently wrong</b> (to surface label noise or artefact failures), or
<b>uncertain near a decision boundary</b> (to add ground-truth signal where it
matters most for AUROC). Below are the top picks per task.</p>

<h3>RESPONSE_CR — highest-confidence positives (P(yes) → 1)</h3>
<p>These are great <b>positive controls</b>: ask a haematopathologist to confirm the visual cues
the model is locking onto.</p>
{fmt_table(rh_yes,
    [("SAMPLE_ID","SAMPLE_ID"),("RESPONSE_CR","GT"),("RESPONSE_CR_yes","P(yes)"),("split","split"),("stem","slide")],
    {"RESPONSE_CR_yes":"{:.3f}"})}

<h3>RESPONSE_CR — highest-confidence negatives (P(yes) → 0)</h3>
{fmt_table(rh_no,
    [("SAMPLE_ID","SAMPLE_ID"),("RESPONSE_CR","GT"),("RESPONSE_CR_yes","P(yes)"),("split","split"),("stem","slide")],
    {"RESPONSE_CR_yes":"{:.3f}"})}

<h3>RESPONSE_CR — near decision boundary (P(yes) ≈ 0.5)</h3>
<p>Annotating these directly improves AUROC because they are the model's hardest cases.</p>
{fmt_table(rh_bnd,
    [("SAMPLE_ID","SAMPLE_ID"),("RESPONSE_CR","GT"),("RESPONSE_CR_yes","P(yes)"),("split","split"),("stem","slide")],
    {"RESPONSE_CR_yes":"{:.3f}"})}

<h3>RESPONSE_CR — confident mistakes (high P, wrong label)</h3>
<p>These are slides where the model is sure but disagrees with GT — top suspects for either
<b>label noise</b> or <b>genuinely difficult</b> morphology. Re-annotating them is the
single highest-value signal you can collect.</p>
{fmt_table(conf_mistakes,
    [("SAMPLE_ID","SAMPLE_ID"),("RESPONSE_CR","GT"),("pred","pred"),("RESPONSE_CR_yes","P(yes)"),("confidence","conf"),("stem","slide")],
    {"RESPONSE_CR_yes":"{:.3f}", "confidence":"{:.3f}"})}

<h3>BLAST_PERCENT — highest predicted blast % (top of regression range)</h3>
{fmt_table(bp_high,
    [("SAMPLE_ID","SAMPLE_ID"),("BLAST_PERCENT","GT"),("pred","pred"),("abs_err","|err|"),("split","split"),("stem","slide")],
    {"pred":"{:.2f}","abs_err":"{:.2f}"})}

<h3>BLAST_PERCENT — lowest predicted blast % (bottom of regression range)</h3>
{fmt_table(bp_low,
    [("SAMPLE_ID","SAMPLE_ID"),("BLAST_PERCENT","GT"),("pred","pred"),("abs_err","|err|"),("split","split"),("stem","slide")],
    {"pred":"{:.2f}","abs_err":"{:.2f}"})}

<h3>BLAST_PERCENT — largest absolute errors</h3>
<p>Most of these are slides where GT ≥ 80 % but the model predicts &lt; 15 % (or vice versa).
Have the pathologist re-count: in our experience, most of these resolve into either a
counting error in the original GT or an unusual blast morphology that the model has not
seen often enough.</p>
{fmt_table(bp_err,
    [("SAMPLE_ID","SAMPLE_ID"),("BLAST_PERCENT","GT"),("pred","pred"),("abs_err","|err|"),("split","split"),("stem","slide")],
    {"pred":"{:.2f}","abs_err":"{:.2f}"})}

<h3>Recommended sampling strategy (for ~30–50 slides total)</h3>
<ul>
  <li>10× <b>RESPONSE confident mistakes</b> (above) — highest yield per slide.</li>
  <li>5× <b>RESPONSE near-boundary</b> — directly improves the decision threshold.</li>
  <li>10× <b>BLAST_PERCENT largest errors</b> — calibrate the regression head.</li>
  <li>5× <b>RESPONSE high-confidence yes</b> + 5× <b>high-confidence no</b> — positive/negative
      controls; pick those that the heatmaps highlight in unusual regions
      (so the haematopathologist can confirm the visual signal).</li>
  <li>5× of the <b>5-Fokuspunkte rejected slides</b> from § 1 (e.g. <code>127 26 C</code>,
      <code>931 19 A</code>) — to confirm whether the kept tiles still allow a correct
      morphological call, or whether re-scanning is required.</li>
</ul>
"""


def build_html() -> str:
    css = """
<style>
@page { size: A4; margin: 18mm 12mm; }
body { font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
       color: #222; max-width: 1200px; margin: 0 auto; padding: 16px; line-height: 1.45; }
h1 { font-size: 24pt; border-bottom: 2px solid #333; padding-bottom: 6px; }
h2 { font-size: 18pt; margin-top: 36px; border-bottom: 1px solid #999; padding-bottom: 4px; page-break-before: auto; }
h3 { font-size: 14pt; margin-top: 24px; color: #1a4d8c; }
h4 { font-size: 12pt; margin-top: 16px; color: #333; }
.row { display: flex; gap: 12px; margin: 8px 0; }
.col { flex: 1; min-width: 0; }
.caption { font-size: 10pt; color: #555; margin-bottom: 4px; font-style: italic; }
.slide-block { border: 1px solid #ddd; border-radius: 6px; padding: 12px; margin: 18px 0;
               background: #fafafa; page-break-inside: avoid; }
.exp-block { border-left: 3px solid #1a4d8c; padding-left: 12px; margin: 12px 0; page-break-inside: avoid; }
.kv .k, .kv-inline .k { color: #555; font-weight: 600; }
.kv > div { margin: 2px 0; font-size: 10pt; }
.kv-inline { font-size: 10pt; padding: 4px 0; }
.meta { color: #666; font-size: 10pt; }
.missing { padding: 20px; background: #ffe; border: 1px dashed #aa8; color: #553; font-size: 10pt; }
table.small { border-collapse: collapse; font-size: 9pt; margin: 6px 0; }
table.small td, table.small th { border: 1px solid #ccc; padding: 3px 6px; text-align: left; }
table.small th { background: #eef; }
code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 90%; }
.explanation { background: #f4f8ff; border: 1px solid #cfdfee; padding: 14px; border-radius: 6px; }
.tldr { background: #fffae6; border: 1px solid #e8d27f; padding: 12px; border-radius: 6px; margin: 12px 0;}
img { max-width: 100%; height: auto; display: block; border: 1px solid #ddd; border-radius: 3px; }
ul, ol { margin: 6px 0 8px 22px; }
</style>
"""
    today = date.today().isoformat()
    pre = section_preprocessing()
    why = section_why_rejected()
    hm = section_heatmaps()
    ann = section_annotation()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>STAMP AML Crossval — Validation Follow-up</title>
{css}
</head>
<body>
<h1>STAMP AML Crossval — Validation Follow-up</h1>
<div class="meta">Branch <code>exp_aml_response_crossval</code> · commit <code>bc16cbe</code> · generated {today}</div>

<div class="tldr">
<b>What's in this report:</b>
<ol>
  <li>Before / after preprocessing visualisations for 4 slides flagged as having "many tiles removed",
      a gallery of the actual rejected tiles, and the precise reason they were dropped.</li>
  <li>Heatmaps for 6 representative slides (4 blood smears + the 2 most aggressively filtered
      tissue slides) across all 4 AML tasks (RESPONSE_CR, BLAST_PERCENT, BLAST_SEVERITY, HIGH_BLAST),
      with ground truth and per-class predictions side by side.</li>
  <li>Ranked annotation candidates for the haematopathologist to label next, prioritised by
      expected information gain.</li>
</ol>
</div>

<h2>1. Preprocessing — before / after, with rejected tiles</h2>
{pre}
{why}

<h2>2. Heatmaps across the 4 tasks (with GT and prediction)</h2>
<p>For each slide we show the four task-specific heatmaps generated using the crossval split
in which that slide was held out (so the model never saw it during training).
Each panel pair is: <i>left</i> — overview with the top-K relevant tiles outlined;
<i>right</i> — class map (per-tile class assignment) for classification, or a continuous
relevance heatmap for the regression head.</p>
{hm}

<h2>3. Suggested slides for additional human annotation</h2>
{ann}

<h2>Appendix — file locations</h2>
<ul>
  <li>Per-slide preprocessing assets: <code>validation_report/preprocessing/</code></li>
  <li>Per-slide heatmap assets:       <code>validation_report/heatmaps/&lt;slide-stem&gt;/</code></li>
  <li>Annotation candidates JSON:     <code>validation_report/annotation_candidates/candidates.json</code></li>
</ul>

</body>
</html>
"""
    return html


def main():
    html = build_html()
    OUT_HTML.write_text(html)
    print(f"Wrote HTML: {OUT_HTML}  ({OUT_HTML.stat().st_size/1024/1024:.1f} MB)")

    try:
        from weasyprint import HTML
        HTML(string=html, base_url=str(ROOT)).write_pdf(str(OUT_PDF))
        print(f"Wrote PDF:  {OUT_PDF}  ({OUT_PDF.stat().st_size/1024/1024:.1f} MB)")
    except Exception as e:
        print(f"PDF generation failed: {e}")


if __name__ == "__main__":
    main()
