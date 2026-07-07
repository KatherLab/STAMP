#!/usr/bin/env python3
"""Aggregate per-slide-per-experiment heatmaps + predictions into compact panels."""
import json
import shutil
from pathlib import Path

import pandas as pd

EXP_BASE = {
    "response":       Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_response_uni2"),
    "blast_percent":  Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_blast_percent_uni2"),
    "blast_severity": Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_blast_severity_uni2"),
    "high_blast":     Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_high_blast_uni2"),
}

EXP_LABELS = {
    "response":       ("RESPONSE_CR", "classification"),
    "blast_percent":  ("BLAST_PERCENT", "regression"),
    "blast_severity": ("BLAST_SEVERITY", "classification"),
    "high_blast":     ("HIGH_BLAST", "classification"),
}

OUT_DIR = Path("/home/jeff/Projects/STAMP/validation_report/heatmaps")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLIDE_TABLE = pd.read_csv("/home/jeff/Projects/STAMP/tables/stamp_slide.csv")
CLINI_TABLE = pd.read_csv("/home/jeff/Projects/STAMP/tables/stamp_clini.csv")

TARGET_STEMS = [
    "333 19 A Score 29 - 5 Fokuspunkte - entfettet - 2026-03-03 18.42.23",
    "931 19 A Score 30 - 5 Fokuspunkte - entfettet - 2026-03-04 21.23.41",
    "189-19 B Score 31 - 10 Fokuspunkte - nicht entfettet",
    "189-19 C Score 82 - 10 Fokuspunkte - nicht entfettet",
    "258 19 B Score 100- 5 Fokuspunkte - entfettet  - 2026-03-03 17.14.15",
    "S_895 19 B Score 97 - 10 Fokuspunkte - entfettet - 2026-03-04 20.46.51",
]


def find_split_for_stem(exp: str, stem: str) -> int | None:
    sample_id = SLIDE_TABLE.loc[
        SLIDE_TABLE["FILENAME"] == stem + ".h5", "SAMPLE_ID"
    ].iloc[0]
    splits = json.loads((EXP_BASE[exp] / "crossval" / "splits.json").read_text())
    for i, sp in enumerate(splits["splits"]):
        if sample_id in sp["test_patients"]:
            return i
    return None


def get_pred(exp: str, split: int, sample_id: str) -> dict:
    csvp = EXP_BASE[exp] / "crossval" / f"split-{split}" / "patient-preds.csv"
    df = pd.read_csv(csvp)
    row = df[df["SAMPLE_ID"] == sample_id]
    if len(row) == 0:
        return {}
    return row.iloc[0].to_dict()


def find_overview(exp: str, split: int, stem: str) -> Path | None:
    plotdir = EXP_BASE[exp] / "heatmaps" / f"split-{split}" / stem / "plots"
    if not plotdir.exists():
        return None
    for p in plotdir.glob(f"overview-{stem}*.png"):
        return p
    return None


def find_classmap(exp: str, split: int, stem: str) -> Path | None:
    rawdir = EXP_BASE[exp] / "heatmaps" / f"split-{split}" / stem / "raw"
    if not rawdir.exists():
        return None
    for p in rawdir.glob(f"{stem}-classmap.png"):
        return p
    # regression: no classmap; pick raw-overlay or the relevance heatmap instead
    for p in sorted(rawdir.glob("*.png")):
        if "classmap" not in p.name and "thumbnail" not in p.name and "raw-overlay" not in p.name:
            return p
    return None


def main():
    summary = []
    for stem in TARGET_STEMS:
        sample_id = SLIDE_TABLE.loc[
            SLIDE_TABLE["FILENAME"] == stem + ".h5", "SAMPLE_ID"
        ].iloc[0]
        clini = CLINI_TABLE[CLINI_TABLE["SAMPLE_ID"] == sample_id].iloc[0]

        slide_out = OUT_DIR / stem
        slide_out.mkdir(exist_ok=True)

        per_exp = {}
        for exp in EXP_BASE:
            split = find_split_for_stem(exp, stem)
            if split is None:
                continue
            label, task = EXP_LABELS[exp]
            pred = get_pred(exp, split, sample_id)
            overview = find_overview(exp, split, stem)
            classmap = find_classmap(exp, split, stem)

            if overview:
                shutil.copy(overview, slide_out / f"{exp}__overview.png")
            if classmap:
                shutil.copy(classmap, slide_out / f"{exp}__classmap.png")

            per_exp[exp] = {
                "split": split,
                "label": label,
                "task": task,
                "gt": clini[label],
                "pred": pred,
                "has_overview": overview is not None,
                "has_classmap": classmap is not None,
            }

        per_slide = {
            "stem": stem,
            "sample_id": sample_id,
            "clini": {k: clini[k] for k in ["BLAST_PERCENT", "RESPONSE_CR", "BLAST_SEVERITY", "HIGH_BLAST"]},
            "experiments": per_exp,
        }
        summary.append(per_slide)
        print(f"{stem[:50]:<50}  exps_with_heatmap={sum(1 for e in per_exp.values() if e['has_overview'])}/{len(per_exp)}")

    (OUT_DIR / "_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote summary: {OUT_DIR/'_summary.json'}")


if __name__ == "__main__":
    main()
