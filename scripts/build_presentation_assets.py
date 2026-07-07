"""Generate figures + shortlist heatmap examples for the B01 project-review deck.

Produces (in presentations/assets/):
  fig_labels.png     - distribution of the 4 clinical labels
  fig_cohort_cv.png  - cohort funnel + 5-fold cross-validation schematic
  fig_workflow.png   - STAMP pipeline flow chart
  montage_<task>.png - candidate heatmap overviews (for human selection)
  candidates.json    - shortlisted heatmap candidates with metrics + paths

Selection uses the blast-enrichment triage `slide_summary.csv` as an objective
"attention lands on tissue, not glass background" signal
(top_enrichment_proxy_mean high, technical_concern_score / high-background flags low).

Run: .venv/bin/python scripts/build_presentation_assets.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
VR = ROOT / "validation_report"
DATA = Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen")
ASSETS = ROOT / "presentations" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY = "#1A4D8C"
STEEL = "#4E7CB5"
GREY = "#8A94A6"
AMBER = "#C77F1A"
RED = "#B3402F"
GREEN = "#3E7D4F"
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "figure.dpi": 200,
    }
)


# --------------------------------------------------------------------------- #
# 1. Label distributions
# --------------------------------------------------------------------------- #
def fig_labels(clini: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(13.3, 3.4))

    def bars(ax, counts, order, title, colors):
        vals = [int(counts.get(k, 0)) for k in order]
        x = np.arange(len(order))
        ax.bar(x, vals, color=colors, width=0.62, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(order, fontsize=10)
        ax.set_title(title, fontsize=12, color=NAVY, fontweight="bold", pad=8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, max(vals) * 1.18)
        for xi, v in zip(x, vals):
            ax.text(xi, v + max(vals) * 0.02, str(v), ha="center", va="bottom", fontsize=10)

    bars(axes[0], clini["RESPONSE_CR"].value_counts(), ["yes", "no"],
         "RESPONSE_CR", [GREEN, GREY])
    bars(axes[1], clini["HIGH_BLAST"].value_counts(), ["no", "yes"],
         "HIGH_BLAST", [GREY, AMBER])
    bars(axes[2], clini["BLAST_SEVERITY"].value_counts(), ["low", "intermediate", "high"],
         "BLAST_SEVERITY", [STEEL, GREY, RED])

    ax = axes[3]
    vals = pd.to_numeric(clini["BLAST_PERCENT"], errors="coerce").dropna()
    ax.hist(vals, bins=np.arange(0, 101, 5), color=NAVY, edgecolor="white", alpha=0.9)
    ax.set_title("BLAST_PERCENT", fontsize=12, color=NAVY, fontweight="bold", pad=8)
    ax.set_xlabel("clinical blast %", fontsize=10)
    ax.set_ylabel("samples", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.axvline(5, color=RED, lw=1, ls="--")
    ax.axvline(20, color=RED, lw=1, ls="--")

    fig.suptitle(f"Clinical label distribution (n = {len(clini)} samples)",
                 fontsize=13, color="#222", fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_labels.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_labels.png")


# --------------------------------------------------------------------------- #
# 2. Cohort funnel + CV schematic
# --------------------------------------------------------------------------- #
def fig_cohort_cv(n_pat: int, n_samp: int, n_slide: int) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.3, 3.6),
                                   gridspec_kw={"width_ratios": [1, 1.25]})

    # -- funnel --
    labels = [f"{n_pat}\npatients", f"{n_samp}\nsamples", f"{n_slide}\nslide images"]
    widths = [0.5, 0.8, 1.0]
    colors = [NAVY, STEEL, GREY]
    for i, (lab, w, c) in enumerate(zip(labels, widths, colors)):
        y = 2 - i
        axL.add_patch(FancyBboxPatch((0.5 - w / 2, y - 0.36), w, 0.72,
                                     boxstyle="round,pad=0.02,rounding_size=0.05",
                                     facecolor=c, edgecolor="none"))
        axL.text(0.5, y, lab, ha="center", va="center", color="white",
                 fontsize=12, fontweight="bold")
        if i < 2:
            axL.annotate("", xy=(0.5, y - 0.4), xytext=(0.5, y - 0.62),
                         arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.5))
    axL.set_xlim(0, 1)
    axL.set_ylim(-0.2, 2.6)
    axL.axis("off")
    axL.set_title("Cohort", fontsize=12, color=NAVY, fontweight="bold")

    # -- 5-fold CV schematic --
    k = 5
    for f in range(k):
        y = k - 1 - f
        for j in range(k):
            is_test = j == f
            axR.add_patch(plt.Rectangle((j, y), 0.96, 0.8,
                                        facecolor=(AMBER if is_test else STEEL),
                                        edgecolor="white"))
        axR.text(-0.25, y + 0.4, f"fold {f + 1}", ha="right", va="center", fontsize=10)
    axR.set_xlim(-1.6, k + 0.1)
    axR.set_ylim(-0.6, k + 0.2)
    axR.axis("off")
    axR.set_title("5-fold cross-validation", fontsize=12, color=NAVY, fontweight="bold")
    axR.add_patch(plt.Rectangle((0.2, -0.55), 0.4, 0.32, facecolor=STEEL, edgecolor="white"))
    axR.text(0.72, -0.39, "train", va="center", fontsize=9)
    axR.add_patch(plt.Rectangle((2.0, -0.55), 0.4, 0.32, facecolor=AMBER, edgecolor="white"))
    axR.text(2.52, -0.39, "held-out test (every sample predicted once)", va="center", fontsize=9)

    fig.tight_layout()
    fig.savefig(ASSETS / "fig_cohort_cv.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_cohort_cv.png")


# --------------------------------------------------------------------------- #
# 3. Workflow chart
# --------------------------------------------------------------------------- #
def fig_workflow() -> None:
    fig, ax = plt.subplots(figsize=(13.3, 3.0))
    stages = [
        ("Bone-marrow\nsmear WSI", NAVY),
        ("Tiling + QC\n224×224 px\nfocus/bg filter", STEEL),
        ("UNI2\ntile features\n(foundation model)", STEEL),
        ("ViT-MIL\nattention\naggregation", STEEL),
        ("Prediction\n5-fold CV\n(class / value)", NAVY),
    ]
    n = len(stages)
    bw, bh, gap = 2.05, 1.5, 0.55
    x = 0.2
    centers = []
    for label, color in stages:
        ax.add_patch(FancyBboxPatch((x, 0.9), bw, bh,
                                    boxstyle="round,pad=0.03,rounding_size=0.12",
                                    facecolor=color, edgecolor="none"))
        ax.text(x + bw / 2, 0.9 + bh / 2, label, ha="center", va="center",
                color="white", fontsize=11, fontweight="bold")
        centers.append(x + bw / 2)
        x += bw + gap
    for i in range(n - 1):
        ax.add_patch(FancyArrowPatch((centers[i] + bw / 2, 1.65),
                                     (centers[i + 1] - bw / 2, 1.65),
                                     arrowstyle="-|>", mutation_scale=18,
                                     color="#555", lw=1.6))
    # outputs branch
    ax.add_patch(FancyBboxPatch((centers[-1] - bw / 2, -0.55),
                                bw, 1.0,
                                boxstyle="round,pad=0.03,rounding_size=0.12",
                                facecolor=GREY, edgecolor="none"))
    ax.text(centers[-1], -0.05, "Heatmaps +\ntop/bottom tiles", ha="center",
            va="center", color="white", fontsize=10, fontweight="bold")
    ax.add_patch(FancyArrowPatch((centers[-1], 0.88), (centers[-1], 0.47),
                                 arrowstyle="-|>", mutation_scale=16, color="#555", lw=1.6))
    ax.text(centers[0], 2.65, "Weakly supervised: one label per slide, no cell annotations",
            ha="left", fontsize=10, style="italic", color="#555")
    ax.set_xlim(0, x)
    ax.set_ylim(-0.8, 3.0)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_workflow.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_workflow.png")


# --------------------------------------------------------------------------- #
# Heatmap example selection
# --------------------------------------------------------------------------- #
def _overview(exp_dir_name: str, split: str, stem: str) -> Path | None:
    p = DATA / exp_dir_name / "heatmaps" / split / stem / "plots" / f"overview-{stem}.png"
    return p if p.exists() else None


def select_candidates(clini: pd.DataFrame) -> dict:
    ss = pd.read_csv(VR / "blast_enrichment_triage" / "slide_summary.csv")
    ss["top_enrichment_proxy_mean"] = pd.to_numeric(ss["top_enrichment_proxy_mean"], errors="coerce")
    ss["blast_percent_gt"] = pd.to_numeric(ss["blast_percent_gt"], errors="coerce")
    ss["blast_percent_abs_err"] = pd.to_numeric(ss["blast_percent_abs_err"], errors="coerce")

    exp_dir = {
        "high_blast": "stamp_aml_high_blast_uni2",
        "blast_percent": "stamp_aml_blast_percent_uni2",
        "response": "stamp_aml_response_uni2",
    }
    out: dict[str, list] = {}

    def row_to_cand(r, note):
        ov = _overview(exp_dir[r["experiment"]], r["split"], r["stem"])
        if ov is None:
            return None
        return {
            "sample_id": r["sample_id"],
            "stem": r["stem"],
            "split": r["split"],
            "experiment": r["experiment"],
            "blast_pct_gt": None if pd.isna(r["blast_percent_gt"]) else round(float(r["blast_percent_gt"]), 1),
            "enrichment": None if pd.isna(r["top_enrichment_proxy_mean"]) else round(float(r["top_enrichment_proxy_mean"]), 3),
            "tech_concern": int(r["technical_concern_score"]),
            "note": note,
            "overview": str(ov),
        }

    # ---- HIGH_BLAST: correct positives, attention on tissue ----
    hb = ss[ss.experiment == "high_blast"].copy()
    cand = hb[(hb.high_blast_gt == "yes") & (hb.high_blast_pred_label == "yes")
              & (hb.flag_high_background == 0) & (hb.technical_concern_score <= 1)]
    cand = cand.sort_values("top_enrichment_proxy_mean", ascending=False).head(8)
    out["high_blast"] = [c for c in (row_to_cand(r, "GT high / pred high") for _, r in cand.iterrows()) if c]

    # ---- BLAST_PERCENT: correct across the range + a dramatic failure ----
    bp = ss[ss.experiment == "blast_percent"].copy()
    good = bp[(bp.blast_percent_abs_err <= 6) & (bp.flag_high_background == 0)]
    lo = good[good.blast_percent_gt < 5].sort_values("top_enrichment_proxy_mean", ascending=False).head(3)
    hi = good[good.blast_percent_gt >= 20].sort_values("top_enrichment_proxy_mean", ascending=False).head(3)
    fails = bp[bp.is_blast_percent_top10_failure == 1].sort_values("blast_percent_gt", ascending=False).head(3)
    out["blast_percent"] = [c for c in (
        [row_to_cand(r, f"GT {r.blast_percent_gt:.0f}% (low) — correct") for _, r in lo.iterrows()]
        + [row_to_cand(r, f"GT {r.blast_percent_gt:.0f}% (high) — correct") for _, r in hi.iterrows()]
        + [row_to_cand(r, f"GT {r.blast_percent_gt:.0f}% -> pred {r.blast_percent_pred:.0f}% — FAILURE") for _, r in fails.iterrows()]
    ) if c]

    # ---- EXPLAINABILITY: same slide, high_blast vs blast_percent overview ----
    pairs = []
    for sid in [c["sample_id"] for c in out["high_blast"][:5]]:
        hbrow = hb[hb.sample_id == sid]
        bprow = bp[bp.sample_id == sid]
        if len(hbrow) and len(bprow):
            hbc = row_to_cand(hbrow.iloc[0], "HIGH_BLAST attention")
            bpc = row_to_cand(bprow.iloc[0], "BLAST_PERCENT attention")
            if hbc and bpc:
                pairs.append({"sample_id": sid, "high_blast": hbc, "blast_percent": bpc})
    out["explain_pairs"] = pairs

    # ---- RESPONSE: only 23 slides rendered; rank by cellularity (join hb) + correctness ----
    preds = []
    for sp in range(5):
        f = DATA / exp_dir["response"] / "crossval" / f"split-{sp}" / "patient-preds.csv"
        if f.exists():
            d = pd.read_csv(f)
            d["split"] = f"split-{sp}"
            preds.append(d)
    preds = pd.concat(preds, ignore_index=True)
    # sample_id -> cellularity from high_blast slide_summary
    cell = hb.set_index("sample_id")["top_enrichment_proxy_mean"].to_dict()
    bg = hb.set_index("sample_id")["zoom_background_fraction_mean"].to_dict()
    resp_rows = []
    resp_heatmap_dir = DATA / exp_dir["response"] / "heatmaps"
    for ov in resp_heatmap_dir.glob("split-*/*/plots/overview-*.png"):
        stem = ov.parent.parent.name
        split = ov.parent.parent.parent.name
        srow = ss[ss.stem == stem]
        if not len(srow):
            continue
        sid = srow.iloc[0]["sample_id"]
        prow = preds[preds.SAMPLE_ID == sid]
        if not len(prow):
            continue
        pr = prow.iloc[0]
        resp_rows.append({
            "sample_id": sid, "stem": stem, "split": split, "experiment": "response",
            "gt": pr["RESPONSE_CR"], "pred": pr["pred"], "p_yes": round(float(pr["RESPONSE_CR_yes"]), 3),
            "correct": bool(pr["RESPONSE_CR"] == pr["pred"]),
            "enrichment": round(float(cell.get(sid, np.nan)), 3) if sid in cell else None,
            "background": round(float(bg.get(sid, np.nan)), 3) if sid in bg else None,
            "overview": str(ov),
        })
    rdf = pd.DataFrame(resp_rows)
    rdf = rdf[rdf.correct].sort_values("enrichment", ascending=False)
    out["response"] = rdf.head(10).to_dict("records")
    return out


def montage(task: str, cands: list, cols: int = 4) -> None:
    if not cands:
        print(f"no candidates for {task}")
        return
    n = len(cands)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.4, rows * 3.4))
    axes = np.atleast_1d(axes).ravel()
    for ax, c in zip(axes, cands):
        img = plt.imread(c["overview"])
        ax.imshow(img)
        cap = f"{c['sample_id']}\n{c.get('note') or (('GT ' + str(c.get('gt')) + '/pred ' + str(c.get('pred'))))}"
        extra = []
        if c.get("enrichment") is not None:
            extra.append(f"enr {c['enrichment']}")
        if c.get("p_yes") is not None:
            extra.append(f"P(yes) {c['p_yes']}")
        if extra:
            cap += "  [" + ", ".join(extra) + "]"
        ax.set_title(cap, fontsize=8)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle(f"{task} candidates", fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = ASSETS / f"montage_{task}.png"
    fig.savefig(p, bbox_inches="tight", dpi=130)
    plt.close(fig)
    print(f"wrote {p} ({n} candidates)")


# --------------------------------------------------------------------------- #
# Explainability: attention enrichment vs clinical blast band (group_summary)
# --------------------------------------------------------------------------- #
def fig_explain_enrichment() -> None:
    gs = pd.read_csv(VR / "blast_enrichment_triage" / "group_summary.csv")
    gs = gs[gs.group_type == "clinical_blast_band"]
    order = ["low_blast_lt5", "intermediate_blast_5_19", "high_blast_ge20"]
    xlabels = ["low\n(<5%)", "intermediate\n(5–19%)", "high\n(≥20%)"]

    def series(exp):
        d = gs[gs.experiment == exp].set_index("group_name")
        return [float(d.loc[b, "top_enrichment_proxy_mean"]) for b in order]

    hb, bp = series("high_blast"), series("blast_percent")
    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    ax.bar(x - w / 2, hb, w, label="HIGH_BLAST model  (proxy AUROC 0.86)", color=NAVY, edgecolor="white")
    ax.bar(x + w / 2, bp, w, label="BLAST_PERCENT model  (proxy AUROC 0.42 ≈ chance)", color=GREY, edgecolor="white")
    for xi, v in zip(x - w / 2, hb):
        ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=10, color=NAVY)
    for xi, v in zip(x + w / 2, bp):
        ax.text(xi, v + 0.008, f"{v:.2f}", ha="center", fontsize=10, color="#555")
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels)
    ax.set_ylim(0, 0.52)
    ax.set_xlabel("clinical blast burden")
    ax.set_ylabel("top-attention enrichment\n(overlap with cell-rich regions)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper center", fontsize=10, bbox_to_anchor=(0.5, 1.02))
    ax.annotate("rises with blast burden", xy=(2 - w / 2, hb[2] + 0.005),
                xytext=(0.55, 0.20), fontsize=10, color=NAVY, fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=NAVY, lw=1.4))
    ax.annotate("stays flat", xy=(2 + w / 2, bp[2] + 0.005),
                xytext=(2.05, 0.44), fontsize=10, color="#555", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#777", lw=1.4))
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_explain_enrichment.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_explain_enrichment.png")


# --------------------------------------------------------------------------- #
# Compose clean per-experiment example panels: WSI thumbnail (top) + attention
# overlay (bottom), N examples across columns.
# --------------------------------------------------------------------------- #
def _folder(overview_path: str) -> Path:
    return Path(overview_path).parent.parent  # <stem>/plots/overview.png -> <stem>


def compose_examples(task: str, picks: list[dict]) -> None:
    n = len(picks)
    fig, axes = plt.subplots(2, n, figsize=(4.3 * n, 5.6),
                             gridspec_kw={"height_ratios": [1, 1]})
    axes = np.atleast_2d(axes)
    for i, p in enumerate(picks):
        folder = _folder(p["overview"])
        stem = folder.name
        thumb = folder / "raw" / f"thumbnail-{stem}.png"
        if p["kind"] == "classification":
            overlay = folder / "plots" / f"overlay-{stem}-{p['cls']}.png"
        else:
            overlay = folder / "raw" / f"raw-overlay-{stem}.png"
        for row, img_path, sub in ((0, thumb, "WSI"), (1, overlay, "attention overlay")):
            ax = axes[row, i]
            if img_path.exists():
                ax.imshow(plt.imread(img_path))
            else:
                ax.text(0.5, 0.5, f"missing:\n{img_path.name}", ha="center", fontsize=7)
            ax.axis("off")
            if row == 1:
                ax.set_xlabel(sub, fontsize=9, color="#555")
        axes[0, i].set_title(p["caption"], fontsize=11, color=NAVY, fontweight="bold", pad=6)
    fig.tight_layout()
    out = ASSETS / f"exp_{task}.png"
    fig.savefig(out, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"wrote {out.name}")


def compose_all_examples(cands: dict) -> None:
    by_id = {}
    for key in ("response", "high_blast", "blast_percent"):
        for c in cands[key]:
            by_id[(key, c["sample_id"])] = c

    def get(key, sid):
        return by_id[(key, sid)]

    # Exp 1 — RESPONSE: one confident non-responder + one confident responder, tissue visible
    resp = [
        {**get("response", "SAMPLE_416_1012_19_A"), "kind": "classification", "cls": "no",
         "caption": "1012_19_A\nGT non-responder · pred non-responder"},
        {**get("response", "SAMPLE_380_1020_19_A"), "kind": "classification", "cls": "yes",
         "caption": "1020_19_A\nGT responder · pred responder (P=0.88)"},
    ]
    compose_examples("response", resp)

    # Exp 2 — HIGH_BLAST: three correct positives with clear blast clusters
    hb_ids = ["SAMPLE_404_1349_19_A", "SAMPLE_401_960_19_B", "SAMPLE_402_1085_19_B"]
    hb = [{**get("high_blast", sid), "kind": "classification", "cls": "yes",
           "caption": f"{sid.replace('SAMPLE_', '')}\nGT high-blast · pred high"} for sid in hb_ids]
    compose_examples("high_blast", hb)

    # Exp 4 — BLAST_PERCENT: one correct low + two dramatic high-blast under-predictions
    bp_specs = [
        ("SAMPLE_388_583_19_A", "583_19_A\nGT 3% · pred ~1%  (correct)"),
        ("SAMPLE_404_1261_19_A", "1261_19_A\nGT 90% · pred 12%  (FAILURE)"),
        ("SAMPLE_408_1411_19_C", "1411_19_C\nGT 91% · pred 13%  (FAILURE)"),
    ]
    bp = [{**get("blast_percent", sid), "kind": "regression", "caption": cap}
          for sid, cap in bp_specs]
    compose_examples("blast_percent", bp)


def main() -> None:
    clini = pd.read_csv(ROOT / "tables" / "stamp_clini.csv")
    n_pat = clini["PATIENT"].nunique()
    n_samp = len(clini)
    n_slide = sum(1 for _ in open(ROOT / "tables" / "stamp_slide.csv")) - 1

    fig_labels(clini)
    fig_cohort_cv(n_pat, n_samp, n_slide)
    fig_workflow()

    fig_explain_enrichment()

    cands = select_candidates(clini)
    (ASSETS / "candidates.json").write_text(json.dumps(cands, indent=2, default=str))
    print("wrote candidates.json")
    compose_all_examples(cands)


if __name__ == "__main__":
    main()
