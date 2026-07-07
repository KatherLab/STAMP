"""Patient-level re-analysis figures + metrics for the consortium deck.

Reads pooled cross-validated predictions from the ORIGINAL (sample-split) and NEW
(patient-grouped) runs and produces:
  metrics.json                    - single source of truth for the deck's numbers
  fig_sample_vs_patient.png       - AUROC drop from fixing the leakage
  fig_auprc_calibration.png       - PR + reliability for RESPONSE_CR & HIGH_BLAST
  fig_response_beyond_blast.png   - does response signal exceed blast burden?
  fig_confounder.png              - do predictions track scan protocol (batch)?

All metrics are recomputed here from the raw patient-preds so old vs new are
apples-to-apples (the shipped STAMP regression stats used a single fold).

Run AFTER training: .venv/bin/python scripts/build_analysis_figs.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    mean_absolute_error,
    precision_recall_curve,
    r2_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen")
ASSETS = ROOT / "presentations" / "assets"
ASSETS.mkdir(parents=True, exist_ok=True)

NAVY, STEEL, GREY, AMBER, RED, GREEN = "#1A4D8C", "#4E7CB5", "#8A94A6", "#C77F1A", "#B3402F", "#3E7D4F"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                     "axes.edgecolor": "#444", "figure.dpi": 200})

TASKS = ["response", "high_blast", "blast_severity", "blast_percent"]
LABEL = {"response": "RESPONSE_CR", "high_blast": "HIGH_BLAST",
         "blast_severity": "BLAST_SEVERITY", "blast_percent": "BLAST_PERCENT"}
POS = {"response": "yes", "high_blast": "yes"}
KIND = {"response": "clf", "high_blast": "clf", "blast_severity": "multiclf", "blast_percent": "reg"}
NICE = {"response": "RESPONSE_CR", "high_blast": "HIGH_BLAST",
        "blast_severity": "BLAST_SEVERITY", "blast_percent": "BLAST_PERCENT"}


def pooled(exp_dir: Path) -> pd.DataFrame | None:
    files = sorted(exp_dir.glob("crossval/split-*/patient-preds.csv"))
    if not files:
        return None
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def auroc_ci(y: np.ndarray, p: np.ndarray, n_boot: int = 2000):
    auc = roc_auc_score(y, p)
    rng = np.random.default_rng(0)
    idx = np.arange(len(y))
    boots = []
    for _ in range(n_boot):
        s = rng.choice(idx, len(idx), replace=True)
        if len(np.unique(y[s])) == 2:
            boots.append(roc_auc_score(y[s], p[s]))
    lo, hi = (np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan))
    return float(auc), float(lo), float(hi)


def compute_metrics(df: pd.DataFrame, task: str) -> dict:
    label = LABEL[task]
    if KIND[task] == "clf":
        pos = POS[task]
        y = (df[label].astype(str) == pos).to_numpy().astype(int)
        p = df[f"{label}_{pos}"].to_numpy()
        auc, lo, hi = auroc_ci(y, p)
        return {"auroc": auc, "lo": lo, "hi": hi,
                "auprc": float(average_precision_score(y, p)), "n": int(len(y)),
                "pos_rate": float(y.mean())}
    if KIND[task] == "multiclf":
        classes = [c.split(f"{label}_")[1] for c in df.columns if c.startswith(f"{label}_")]
        aucs = {}
        for c in classes:
            y = (df[label].astype(str) == c).to_numpy().astype(int)
            if len(np.unique(y)) == 2:
                aucs[c] = float(roc_auc_score(y, df[f"{label}_{c}"]))
        macro = float(np.mean(list(aucs.values())))
        return {"auroc": macro, "lo": np.nan, "hi": np.nan, "per_class": aucs, "n": int(len(df))}
    # regression
    y = pd.to_numeric(df[label], errors="coerce").to_numpy()
    p = pd.to_numeric(df["pred"], errors="coerce").to_numpy()
    m = ~np.isnan(y) & ~np.isnan(p)
    y, p = y[m], p[m]
    return {"pearson": float(np.corrcoef(y, p)[0, 1]), "r2": float(r2_score(y, p)),
            "mae": float(mean_absolute_error(y, p)), "n": int(len(y))}


# --------------------------------------------------------------------------- #
def fig_sample_vs_patient(metrics: dict) -> None:
    tasks = ["response", "high_blast", "blast_severity"]
    labels = ["RESPONSE_CR", "HIGH_BLAST", "BLAST_SEVERITY\n(macro)"]
    x = np.arange(len(tasks))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for off, key, color, name in [(-w / 2, "sample", GREY, "sample-split (leaky)"),
                                  (w / 2, "patient", NAVY, "patient-grouped (honest)")]:
        vals = [metrics[key][t]["auroc"] for t in tasks]
        err = [[metrics[key][t]["auroc"] - (metrics[key][t].get("lo") or np.nan) for t in tasks],
               [(metrics[key][t].get("hi") or np.nan) - metrics[key][t]["auroc"] for t in tasks]]
        err = np.nan_to_num(np.abs(np.array(err)), nan=0.0)  # 0 = no bar (e.g. macro severity)
        ax.bar(x + off, vals, w, color=color, label=name,
               yerr=err, capsize=4, edgecolor="white")
        for xi, v, ue in zip(x + off, vals, err[1]):
            ax.text(xi, v + ue + 0.02, f"{v:.2f}", ha="center", fontsize=10,
                    color=color if color != GREY else "#555")
    ax.axhline(0.5, ls="--", color="#bbb", lw=1)
    ax.text(len(tasks) - 0.5, 0.51, "chance", color="#999", fontsize=9, ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("cross-validated AUROC")
    ax.set_ylim(0.4, 1.0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.22))
    r_s = metrics["sample"]["blast_percent"]["pearson"]
    r_p = metrics["patient"]["blast_percent"]["pearson"]
    ax.set_title(f"Honest evaluation: patient-grouped vs sample-split CV\n"
                 f"(BLAST_PERCENT Pearson r  {r_s:.2f} → {r_p:.2f})",
                 color=NAVY, fontweight="bold", fontsize=12)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_sample_vs_patient.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_sample_vs_patient.png")


def fig_auprc_calibration(new_preds: dict) -> None:
    tasks = ["response", "high_blast"]
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2))
    for r, task in enumerate(tasks):
        label, pos = LABEL[task], POS[task]
        df = new_preds[task]
        y = (df[label].astype(str) == pos).to_numpy().astype(int)
        p = df[f"{label}_{pos}"].to_numpy()
        # PR curve
        prec, rec, _ = precision_recall_curve(y, p)
        ap = average_precision_score(y, p)
        ax = axes[r, 0]
        ax.plot(rec, prec, color=NAVY, lw=2)
        ax.axhline(y.mean(), ls="--", color=GREY, lw=1, label=f"baseline {y.mean():.2f}")
        ax.set_xlabel("recall")
        ax.set_ylabel("precision")
        ax.set_title(f"{NICE[task]} — PR (AP {ap:.2f})", color=NAVY, fontweight="bold", fontsize=11)
        ax.set_ylim(0, 1.02)
        ax.legend(frameon=False, fontsize=9, loc="lower left")
        ax.spines[["top", "right"]].set_visible(False)
        # calibration
        frac, mean_pred = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax = axes[r, 1]
        ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1)
        ax.plot(mean_pred, frac, "o-", color=NAVY, lw=2)
        ax.set_xlabel("predicted probability")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"{NICE[task]} — calibration", color=NAVY, fontweight="bold", fontsize=11)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Patient-grouped models: precision-recall & calibration",
                 fontsize=13, fontweight="bold", color="#222")
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_auprc_calibration.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_auprc_calibration.png")


def fig_response_beyond_blast(new_preds: dict, clini: pd.DataFrame) -> dict:
    df = new_preds["response"].merge(
        clini[["SAMPLE_ID", "BLAST_PERCENT"]], on="SAMPLE_ID", how="left")
    df["blast"] = pd.to_numeric(df["BLAST_PERCENT"], errors="coerce")
    df["y"] = (df["RESPONSE_CR"].astype(str) == "yes").astype(int)
    df = df.dropna(subset=["blast"])
    corr = float(np.corrcoef(df.blast, df.y)[0, 1])

    bands = {"low\n(<5%)": df.blast < 5, "intermediate\n(5–19%)": (df.blast >= 5) & (df.blast < 20),
             "high\n(≥20%)": df.blast >= 20}
    resp_rate, ns, single = {}, {}, True
    for name, mask in bands.items():
        sub = df[mask]
        resp_rate[name] = float(sub.y.mean()) if len(sub) else float("nan")
        ns[name] = int(len(sub))
        if 0 < sub.y.mean() < 1:
            single = False
    # is RESPONSE_CR exactly the <5% blast (morphologic-CR) rule?
    equiv = bool(((df.blast < 5).astype(int) == df.y).all())

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    # A: blast% by response — shows the (near-)perfect separation
    data = [df[df.y == 1].blast, df[df.y == 0].blast]
    bp = axL.boxplot(data, tick_labels=["responder", "non-responder"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [GREEN, GREY]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    axL.axhline(5, ls="--", color=RED, lw=1)
    axL.text(2.4, 6, "5% (CR threshold)", color=RED, fontsize=9, ha="right", va="bottom")
    axL.set_ylabel("clinical blast %")
    axL.set_title(f"Blast burden by response\n(point-biserial r = {corr:+.2f})",
                  color=NAVY, fontweight="bold", fontsize=11)
    axL.spines[["top", "right"]].set_visible(False)
    # B: responder fraction by blast band — the definitional confound
    names = list(resp_rate.keys())
    xb = np.arange(len(names))
    axR.bar(xb, [resp_rate[k] for k in names], color=[GREEN, GREY, GREY], edgecolor="white", width=0.6)
    for xi, k in zip(xb, names):
        axR.text(xi, resp_rate[k] + 0.02, f"{resp_rate[k]:.0%}\n(n={ns[k]})", ha="center", fontsize=9)
    axR.set_xticks(xb)
    axR.set_xticklabels(names, fontsize=9)
    axR.set_ylim(0, 1.12)
    axR.set_ylabel("fraction labelled 'responder'")
    axR.set_title("Responder fraction by blast band", color=NAVY, fontweight="bold", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_response_beyond_blast.png", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote fig_response_beyond_blast.png (equiv <5%: {equiv})")
    return {"blast_response_corr": corr, "resp_rate_by_band": resp_rate,
            "n_by_band": ns, "response_equals_lt5_blast": equiv, "strata_single_class": single}


def _parse_protocol(fname: str) -> tuple[str, str]:
    m = re.search(r"(\d+)\s*Fokuspunkte", fname)
    focus = m.group(1) if m else "?"
    defat = "nicht_entfettet" if "nicht entfettet" in fname else ("entfettet" if "entfettet" in fname else "?")
    return focus, defat


def fig_confounder(new_preds: dict, slide: pd.DataFrame) -> dict:
    # sample-level dominant protocol
    prot = {}
    for sid, grp in slide.groupby("SAMPLE_ID"):
        foci, defs = set(), set()
        for fn in grp["FILENAME"]:
            f, d = _parse_protocol(str(fn))
            foci.add(f)
            defs.add(d)
        focus = next(iter(foci)) if len(foci) == 1 else "mixed"
        prot[sid] = focus
    hb = new_preds["high_blast"].copy()
    hb["focus"] = hb["SAMPLE_ID"].map(prot)
    hb["y"] = (hb["HIGH_BLAST"].astype(str) == "yes").astype(int)
    hb["p"] = hb["HIGH_BLAST_yes"]
    hb = hb[hb.focus.isin(["5", "10"])]
    hb["is10"] = (hb.focus == "10").astype(int)

    # (1) protocol vs label association: proportion 10-pt among high vs low blast
    prop = hb.groupby("y")["is10"].mean()
    # (2) can the model score predict protocol? AUROC(is10 ~ p)
    score_auc = roc_auc_score(hb.is10, hb.p) if hb.is10.nunique() == 2 else np.nan

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    axL.bar(["low blast", "high blast"], [prop.get(0, 0), prop.get(1, 0)],
            color=[STEEL, AMBER], edgecolor="white", width=0.55)
    for i, v in enumerate([prop.get(0, 0), prop.get(1, 0)]):
        axL.text(i, v + 0.01, f"{v:.0%}", ha="center", fontsize=11)
    axL.set_ylabel("fraction scanned at 10 Fokuspunkte")
    axL.set_ylim(0, 1.05)
    axL.set_title("Is scan protocol confounded with the label?", color=NAVY, fontweight="bold", fontsize=11)
    axL.spines[["top", "right"]].set_visible(False)
    # score by protocol
    data = [hb[hb.focus == "5"].p, hb[hb.focus == "10"].p]
    bp = axR.boxplot(data, tick_labels=["5 Fokuspunkte", "10 Fokuspunkte"], patch_artist=True, widths=0.5)
    for patch, c in zip(bp["boxes"], [STEEL, NAVY]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    axR.set_ylabel("model HIGH_BLAST score P(yes)")
    axR.set_title(f"Does the score track scan protocol?\nAUROC(protocol ~ score) = {score_auc:.2f}",
                  color=NAVY, fontweight="bold", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ASSETS / "fig_confounder.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_confounder.png")
    return {"prop10_low": float(prop.get(0, 0)), "prop10_high": float(prop.get(1, 0)),
            "protocol_from_score_auroc": float(score_auc), "n": int(len(hb))}


def main() -> None:
    clini = pd.read_csv(ROOT / "tables" / "stamp_clini.csv")
    slide = pd.read_csv(ROOT / "tables" / "stamp_slide.csv")

    metrics: dict = {"sample": {}, "patient": {}}
    old_preds, new_preds = {}, {}
    for t in TASKS:
        old = pooled(DATA / f"stamp_aml_{t}_uni2")
        new = pooled(DATA / f"stamp_aml_{t}_uni2_patientcv")
        if old is None or new is None:
            raise SystemExit(f"missing pooled preds for {t} (old={old is not None}, new={new is not None}) "
                             "— has training finished?")
        old_preds[t], new_preds[t] = old, new
        metrics["sample"][t] = compute_metrics(old, t)
        metrics["patient"][t] = compute_metrics(new, t)

    fig_sample_vs_patient(metrics)
    fig_auprc_calibration(new_preds)
    metrics["response_beyond_blast"] = fig_response_beyond_blast(new_preds, clini)
    metrics["confounder"] = fig_confounder(new_preds, slide)

    (ASSETS / "metrics.json").write_text(json.dumps(metrics, indent=2, default=lambda o: None))
    print("wrote metrics.json")
    # quick console summary
    for t in TASKS:
        s, p = metrics["sample"][t], metrics["patient"][t]
        if KIND[t] == "reg":
            print(f"  {t:14s} Pearson r  sample {s['pearson']:.2f} -> patient {p['pearson']:.2f} "
                  f"| R2 {s['r2']:.2f}->{p['r2']:.2f}")
        else:
            print(f"  {t:14s} AUROC      sample {s['auroc']:.3f} -> patient {p['auroc']:.3f}")


if __name__ == "__main__":
    main()
