"""
Script to train mock STAMP models for backward-compatibility testing.

Uses a handful of real .h5 feature files from a local experiment directory,
assigns fake labels, and trains each model type for a minimal number of epochs.
Resulting checkpoints are saved to weights/test_models/.

Run from the repo root:
    .venv/bin/python tests/make_test_models.py
"""

import os
import random
import shutil
import tempfile
from pathlib import Path

import pandas as pd
import torch

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

FEATURE_DIR = Path(
    "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/arianna"
    "/STAMP_raw_xiyuewang-ctranspath-7c998680"
)
OUTPUT_DIR = Path(__file__).parent.parent / "weights" / "test_models"

# Grab the first N h5 files for fast training
N_FILES = 10
H5_FILES = sorted(FEATURE_DIR.glob("*.h5"))[:N_FILES]
PATIENT_IDS = [f.stem for f in H5_FILES]

assert len(H5_FILES) >= N_FILES, (
    f"Expected at least {N_FILES} .h5 files in {FEATURE_DIR}, "
    f"found {len(H5_FILES)}"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _clini_slide_tables(
    tmp_dir: Path,
    *,
    patients: list[str],
    ground_truth_col: str,
    ground_truths: list,
    extra_cols: dict[str, list] | None = None,
) -> tuple[Path, Path]:
    """Write clini.csv and slide.csv, then return their paths."""
    clini_rows: dict[str, list] = {
        "PATIENT": patients,
        ground_truth_col: ground_truths,
    }
    if extra_cols:
        clini_rows.update(extra_cols)

    clini_path = tmp_dir / "clini.csv"
    slide_path = tmp_dir / "slide.csv"

    pd.DataFrame(clini_rows).to_csv(clini_path, index=False)
    pd.DataFrame(
        {"PATIENT": patients, "FILENAME": [f"{p}.h5" for p in patients]}
    ).to_csv(slide_path, index=False)

    return clini_path, slide_path


def _train(
    *,
    task: str,
    model_name: str,
    output_name: str,
    clini_path: Path,
    slide_path: Path | None,
    feature_dir: Path,
    ground_truth_label,
    categories=None,
    time_label=None,
    status_label=None,
) -> None:
    from stamp.modeling.config import (
        AdvancedConfig,
        MlpModelParams,
        ModelParams,
        TrainConfig,
        VitModelParams,
    )
    from stamp.modeling.registry import ModelName
    from stamp.modeling.train import train_categorical_model_

    output_dir = OUTPUT_DIR / f"_tmp_{output_name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        task=task,
        clini_table=clini_path,
        slide_table=slide_path,
        feature_dir=feature_dir,
        output_dir=output_dir,
        patient_label="PATIENT",
        filename_label="FILENAME",
        ground_truth_label=ground_truth_label,
        categories=categories,
        time_label=time_label,
        status_label=status_label,
    )

    advanced = AdvancedConfig(
        bag_size=64,
        num_workers=min(os.cpu_count() or 1, 4),
        batch_size=4,
        max_epochs=2,
        patience=2,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        model_name=ModelName(model_name),
        model_params=ModelParams(vit=VitModelParams(), mlp=MlpModelParams()),
    )

    train_categorical_model_(config=config, advanced=advanced)

    ckpt_src = output_dir / "model.ckpt"
    ckpt_dst = OUTPUT_DIR / f"{output_name}.ckpt"
    shutil.copy2(ckpt_src, ckpt_dst)
    shutil.rmtree(output_dir)
    print(f"  saved → {ckpt_dst.relative_to(Path.cwd())}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    random.seed(42)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    patients = PATIENT_IDS

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        # --- Classification labels (2 classes) ---
        categories = ["A", "B"]
        cls_labels = [categories[i % 2] for i in range(len(patients))]

        clini_cls, slide_cls = _clini_slide_tables(
            tmp_dir,
            patients=patients,
            ground_truth_col="label",
            ground_truths=cls_labels,
        )

        # --- Regression labels ---
        reg_labels = [round(random.uniform(0.0, 100.0), 4) for _ in patients]

        clini_reg, slide_reg = _clini_slide_tables(
            tmp_dir,
            patients=patients,
            ground_truth_col="target",
            ground_truths=reg_labels,
        )

        # --- Survival labels ---
        surv_times = [round(random.uniform(100.0, 3000.0), 1) for _ in patients]
        surv_status = [random.randint(0, 1) for _ in patients]

        clini_surv, slide_surv = _clini_slide_tables(
            tmp_dir,
            patients=patients,
            ground_truth_col="time",
            ground_truths=surv_times,
            extra_cols={"status": surv_status},
        )

        # --- Multi-target classification labels (barspoon) ---
        sub_cats = ["X", "Y"]
        grade_cats = ["1", "2", "3"]
        sub_labels = [sub_cats[i % 2] for i in range(len(patients))]
        grade_labels = [grade_cats[i % 3] for i in range(len(patients))]

        clini_bar = tmp_dir / "clini_bar.csv"
        pd.DataFrame(
            {
                "PATIENT": patients,
                "subtype": sub_labels,
                "grade": grade_labels,
            }
        ).to_csv(clini_bar, index=False)

        # ------------------------------------------------------------------ #
        print("\n=== Training mock models ===\n")

        specs: list[dict] = [
            dict(
                task="classification",
                model_name="vit",
                output_name="vit_tile_classification",
                clini_path=clini_cls,
                slide_path=slide_cls,
                feature_dir=FEATURE_DIR,
                ground_truth_label="label",
                categories=categories,
            ),
            dict(
                task="classification",
                model_name="mlp",
                output_name="mlp_tile_classification",
                clini_path=clini_cls,
                slide_path=slide_cls,
                feature_dir=FEATURE_DIR,
                ground_truth_label="label",
                categories=categories,
            ),
            dict(
                task="classification",
                model_name="trans_mil",
                output_name="transmil_tile_classification",
                clini_path=clini_cls,
                slide_path=slide_cls,
                feature_dir=FEATURE_DIR,
                ground_truth_label="label",
                categories=categories,
            ),
            dict(
                task="regression",
                model_name="vit",
                output_name="vit_tile_regression",
                clini_path=clini_reg,
                slide_path=slide_reg,
                feature_dir=FEATURE_DIR,
                ground_truth_label="target",
            ),
            dict(
                task="survival",
                model_name="vit",
                output_name="vit_tile_survival",
                clini_path=clini_surv,
                slide_path=slide_surv,
                feature_dir=FEATURE_DIR,
                ground_truth_label=None,
                time_label="time",
                status_label="status",
            ),
        ]

        for spec in specs:
            print(f"  [{spec['model_name']} / {spec['task']}] {spec['output_name']}")
            _train(**spec)

        # Barspoon (multi-target) needs separate handling
        print("  [barspoon / classification] barspoon_tile_classification")
        _train(
            task="classification",
            model_name="barspoon",
            output_name="barspoon_tile_classification",
            clini_path=clini_bar,
            slide_path=slide_cls,
            feature_dir=FEATURE_DIR,
            ground_truth_label=["subtype", "grade"],
            categories=sub_cats + grade_cats,
        )

    print("\nDone. Checkpoints written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
