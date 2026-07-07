#!/usr/bin/env python3
"""Identify slides that are good candidates for additional human annotation.

Strategy: pick slides where the model is most confident (extreme prediction)
AND most uncertain (predictions near a decision boundary). Focus on RESPONSE_CR
and BLAST_PERCENT since those are clinically meaningful.
"""

import json
from pathlib import Path

import pandas as pd

EXP_BASE = {
    "response": (
        "/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_response_uni2/crossval",
        "RESPONSE_CR",
        "RESPONSE_CR_yes",
        "classification",
    ),
    "blast_percent": (
        "/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_blast_percent_uni2/crossval",
        "BLAST_PERCENT",
        "pred",
        "regression",
    ),
    "blast_severity": (
        "/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_blast_severity_uni2/crossval",
        "BLAST_SEVERITY",
        None,
        "classification",
    ),
    "high_blast": (
        "/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_high_blast_uni2/crossval",
        "HIGH_BLAST",
        "HIGH_BLAST_yes",
        "classification",
    ),
}

SLIDE_TABLE = pd.read_csv("/home/jeff/Projects/STAMP/tables/stamp_slide.csv")
CLINI_TABLE = pd.read_csv("/home/jeff/Projects/STAMP/tables/stamp_clini.csv")
OUT_DIR = Path("/home/jeff/Projects/STAMP/validation_report/annotation_candidates")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_preds(crossval_dir: Path) -> pd.DataFrame:
    dfs = []
    for sd in sorted(crossval_dir.glob("split-*")):
        p = sd / "patient-preds.csv"
        if p.exists():
            df = pd.read_csv(p)
            df["split"] = sd.name
            dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    summary = {}

    # ============= 1. RESPONSE_CR — highest model confidence in either class =============
    cv = Path(EXP_BASE["response"][0])
    df = load_all_preds(cv)
    df = df.merge(SLIDE_TABLE[["SAMPLE_ID", "FILENAME"]], on="SAMPLE_ID", how="left")
    df["stem"] = df["FILENAME"].str.replace(".h5", "", regex=False)

    # Sort by yes-probability for extremes
    high_yes = df.nlargest(10, "RESPONSE_CR_yes")[
        ["SAMPLE_ID", "stem", "RESPONSE_CR", "RESPONSE_CR_yes", "split"]
    ]
    high_no = df.nsmallest(10, "RESPONSE_CR_yes")[
        ["SAMPLE_ID", "stem", "RESPONSE_CR", "RESPONSE_CR_yes", "split"]
    ]
    boundary = df.iloc[(df["RESPONSE_CR_yes"] - 0.5).abs().argsort()[:10]][
        ["SAMPLE_ID", "stem", "RESPONSE_CR", "RESPONSE_CR_yes", "split"]
    ]

    summary["response"] = {
        "high_confidence_yes": high_yes.to_dict(orient="records"),
        "high_confidence_no": high_no.to_dict(orient="records"),
        "near_boundary": boundary.to_dict(orient="records"),
    }

    # ============= 2. BLAST_PERCENT — extreme predicted, extreme errors =============
    cv = Path(EXP_BASE["blast_percent"][0])
    df = load_all_preds(cv)
    df = df.merge(SLIDE_TABLE[["SAMPLE_ID", "FILENAME"]], on="SAMPLE_ID", how="left")
    df["stem"] = df["FILENAME"].str.replace(".h5", "", regex=False)
    df["abs_err"] = (df["pred"] - df["BLAST_PERCENT"]).abs()

    high_pred = df.nlargest(10, "pred")[
        ["SAMPLE_ID", "stem", "BLAST_PERCENT", "pred", "abs_err", "split"]
    ]
    low_pred = df.nsmallest(10, "pred")[
        ["SAMPLE_ID", "stem", "BLAST_PERCENT", "pred", "abs_err", "split"]
    ]
    big_errors = df.nlargest(10, "abs_err")[
        ["SAMPLE_ID", "stem", "BLAST_PERCENT", "pred", "abs_err", "split"]
    ]

    summary["blast_percent"] = {
        "highest_predicted": high_pred.to_dict(orient="records"),
        "lowest_predicted": low_pred.to_dict(orient="records"),
        "largest_errors": big_errors.to_dict(orient="records"),
    }

    # ============= 3. HIGH_BLAST — high confidence and boundary =============
    cv = Path(EXP_BASE["high_blast"][0])
    df = load_all_preds(cv)
    df = df.merge(SLIDE_TABLE[["SAMPLE_ID", "FILENAME"]], on="SAMPLE_ID", how="left")
    df["stem"] = df["FILENAME"].str.replace(".h5", "", regex=False)

    high_yes = df.nlargest(10, "HIGH_BLAST_yes")[
        ["SAMPLE_ID", "stem", "HIGH_BLAST", "HIGH_BLAST_yes", "split"]
    ]
    high_no = df.nsmallest(10, "HIGH_BLAST_yes")[
        ["SAMPLE_ID", "stem", "HIGH_BLAST", "HIGH_BLAST_yes", "split"]
    ]
    boundary = df.iloc[(df["HIGH_BLAST_yes"] - 0.5).abs().argsort()[:10]][
        ["SAMPLE_ID", "stem", "HIGH_BLAST", "HIGH_BLAST_yes", "split"]
    ]

    summary["high_blast"] = {
        "high_confidence_yes": high_yes.to_dict(orient="records"),
        "high_confidence_no": high_no.to_dict(orient="records"),
        "near_boundary": boundary.to_dict(orient="records"),
    }

    # ============= 4. Overall priority recommendations =============
    # Combine: high-error blast_percent slides + boundary response slides + high-conf response slides with WRONG pred
    cv_resp = load_all_preds(Path(EXP_BASE["response"][0]))
    cv_resp = cv_resp.merge(
        SLIDE_TABLE[["SAMPLE_ID", "FILENAME"]], on="SAMPLE_ID", how="left"
    )
    cv_resp["stem"] = cv_resp["FILENAME"].str.replace(".h5", "", regex=False)
    # find disagreements with high model confidence (likely either label noise OR genuinely difficult slides)
    cv_resp["confidence"] = (cv_resp["RESPONSE_CR_yes"] - 0.5).abs() * 2
    cv_resp["correct"] = cv_resp["pred"] == cv_resp["RESPONSE_CR"]
    confident_mistakes = (
        cv_resp[(~cv_resp["correct"]) & (cv_resp["confidence"] > 0.7)]
        .sort_values("confidence", ascending=False)
        .head(15)[
            [
                "SAMPLE_ID",
                "stem",
                "RESPONSE_CR",
                "pred",
                "RESPONSE_CR_yes",
                "confidence",
            ]
        ]
    )
    summary["confident_mistakes_response"] = confident_mistakes.to_dict(
        orient="records"
    )

    (OUT_DIR / "candidates.json").write_text(json.dumps(summary, indent=2, default=str))

    # Print short overview
    print("=== RESPONSE high-confidence YES (top 5) ===")
    for r in summary["response"]["high_confidence_yes"][:5]:
        print(
            f"  {r['SAMPLE_ID']}  GT={r['RESPONSE_CR']}  pred_yes={r['RESPONSE_CR_yes']:.3f}  | {r['stem'][:55]}"
        )
    print("\n=== RESPONSE high-confidence NO (top 5) ===")
    for r in summary["response"]["high_confidence_no"][:5]:
        print(
            f"  {r['SAMPLE_ID']}  GT={r['RESPONSE_CR']}  pred_yes={r['RESPONSE_CR_yes']:.3f}  | {r['stem'][:55]}"
        )
    print("\n=== RESPONSE near boundary (top 5) ===")
    for r in summary["response"]["near_boundary"][:5]:
        print(
            f"  {r['SAMPLE_ID']}  GT={r['RESPONSE_CR']}  pred_yes={r['RESPONSE_CR_yes']:.3f}  | {r['stem'][:55]}"
        )
    print("\n=== RESPONSE confident mistakes (top 5) ===")
    for r in summary["confident_mistakes_response"][:5]:
        print(
            f"  {r['SAMPLE_ID']}  GT={r['RESPONSE_CR']}  pred={r['pred']}  conf={r['confidence']:.3f}  | {r['stem'][:55]}"
        )
    print("\n=== BLAST_PERCENT largest errors (top 5) ===")
    for r in summary["blast_percent"]["largest_errors"][:5]:
        print(
            f"  {r['SAMPLE_ID']}  GT={r['BLAST_PERCENT']}%  pred={r['pred']:.1f}%  err={r['abs_err']:.1f}%  | {r['stem'][:55]}"
        )

    print(f"\nWrote: {OUT_DIR / 'candidates.json'}")


if __name__ == "__main__":
    main()
