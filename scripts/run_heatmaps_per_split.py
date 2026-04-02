#!/usr/bin/env python3
"""Run STAMP heatmaps per crossval split with correct checkpoint matching.

For each split, only generates heatmaps for WSI slides whose patients were
in that split's test set (avoiding data leakage). This ensures each heatmap
uses a model that never saw that slide during training.

Usage:
    python scripts/run_heatmaps_per_split.py
"""

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
import yaml

# === Configuration ===
BASE_DIR = Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_response_uni2")
CROSSVAL_DIR = BASE_DIR / "crossval"
HEATMAP_BASE_DIR = BASE_DIR / "heatmaps"
FEATURE_DIR = Path("/mnt/nvme0n1p1/Jeff_projects/B01/features/uni2-0242c340")
WSI_DIR = Path("/mnt/nvme0n1p1/Jeff_projects/B01/data/AG Janssen")
SLIDE_TABLE = Path("/home/jeff/Projects/STAMP/tables/stamp_slide.csv")
STAMP_DIR = Path("/home/jeff/Projects/STAMP")

# Heatmap parameters
DEVICE = "cuda"
TOPK = 8
BOTTOMK = 8
OPACITY = 0.6


def main():
    # Load splits
    with open(CROSSVAL_DIR / "splits.json") as f:
        splits_data = json.load(f)

    # Load slide table to map SAMPLE_ID -> FILENAME
    slide_df = pd.read_csv(SLIDE_TABLE)
    sample_to_filenames = {}
    for _, row in slide_df.iterrows():
        sample_id = row["SAMPLE_ID"]
        filename = row["FILENAME"]
        sample_to_filenames.setdefault(sample_id, []).append(filename)

    # Get available WSI stems (from .ndpi files)
    available_wsis = {p.stem: p.name for p in WSI_DIR.glob("*.ndpi")}
    print(f"Found {len(available_wsis)} available WSI files")

    for split_i, split in enumerate(splits_data["splits"]):
        checkpoint = CROSSVAL_DIR / f"split-{split_i}" / "model.ckpt"
        if not checkpoint.exists():
            print(f"[SKIP] Split {split_i}: no model.ckpt found")
            continue

        test_patients = set(split["test_patients"])

        # Find WSIs for test patients that have actual .ndpi files
        slide_paths = []
        for patient in test_patients:
            filenames = sample_to_filenames.get(patient, [])
            for fname in filenames:
                stem = fname.replace(".h5", "")
                if stem in available_wsis:
                    slide_paths.append(available_wsis[stem])

        if not slide_paths:
            print(f"[SKIP] Split {split_i}: no WSI files for test patients")
            continue

        print(f"\n[SPLIT {split_i}] Generating heatmaps for {len(slide_paths)} slides:")
        for sp in slide_paths:
            print(f"  - {sp}")

        # Create per-split output directory
        split_heatmap_dir = HEATMAP_BASE_DIR / f"split-{split_i}"

        # Build a temporary YAML config for this split
        config = {
            "heatmaps": {
                "output_dir": str(split_heatmap_dir),
                "feature_dir": str(FEATURE_DIR),
                "wsi_dir": str(WSI_DIR),
                "checkpoint_path": str(checkpoint),
                "device": DEVICE,
                "topk": TOPK,
                "bottomk": BOTTOMK,
                "opacity": OPACITY,
                "slide_paths": slide_paths,
            }
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False, prefix=f"heatmap_split{split_i}_"
        ) as tmp:
            yaml.dump(config, tmp)
            tmp_path = tmp.name

        try:
            env = os.environ.copy()
            env["DISABLE_ADDMM_CUDA_LT"] = "1"

            result = subprocess.run(
                ["stamp", "--config", tmp_path, "heatmaps"],
                cwd=str(STAMP_DIR),
                env=env,
                capture_output=False,
                text=True,
            )

            if result.returncode != 0:
                print(f"[ERROR] Split {split_i} failed with return code {result.returncode}")
            else:
                print(f"[DONE] Split {split_i} heatmaps complete")
        finally:
            os.unlink(tmp_path)

    print("\n=== All heatmap generation complete ===")
    # Summarize what was generated
    total = 0
    for split_dir in sorted(HEATMAP_BASE_DIR.glob("split-*")):
        slide_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
        total += len(slide_dirs)
        print(f"  {split_dir.name}: {len(slide_dirs)} slides")
    print(f"  Total: {total} slides")


if __name__ == "__main__":
    main()
