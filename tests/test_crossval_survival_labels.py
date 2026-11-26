from pathlib import Path

import h5py
import numpy as np
import pandas as pd

from stamp.modeling.config import (
    AdvancedConfig,
    CrossvalConfig,
    MlpModelParams,
    ModelParams,
)
from stamp.modeling.crossval import categorical_crossval_
from stamp.statistics import compute_stats_
from stamp.utils.seed import Seed


def test_survival_crossval_uses_configured_label_names(tmp_path: Path) -> None:
    Seed.set(0)
    feat_dir = tmp_path / "feats"
    feat_dir.mkdir()
    rng = np.random.default_rng(0)
    rows = []
    for i in range(20):
        pid = f"P{i:02d}"
        with h5py.File(feat_dir / f"{pid}.h5", "w") as h5:
            h5.create_dataset("feats", data=rng.standard_normal(8).astype(np.float32))
            h5.attrs["feat_type"] = "patient"
        rows.append(
            {"PATIENT": pid, "OS_MONTHS": float(i + 1) * 10.0, "OS_EVENT": i % 2}
        )

    clini_path = tmp_path / "clini.csv"
    pd.DataFrame(rows).to_csv(clini_path, index=False)
    out_dir = tmp_path / "crossval"

    config = CrossvalConfig(
        task="survival",
        output_dir=out_dir,
        clini_table=clini_path,
        slide_table=None,
        feature_dir=feat_dir,
        patient_label="PATIENT",
        time_label="OS_MONTHS",
        status_label="OS_EVENT",
        n_splits=2,
    )
    advanced = AdvancedConfig(
        seed=0,
        batch_size=4,
        max_epochs=1,
        patience=1,
        num_workers=0,
        accelerator="cpu",
        model_params=ModelParams(mlp=MlpModelParams(dim_hidden=16, num_layers=1)),
    )

    categorical_crossval_(config=config, advanced=advanced)

    pred_csvs = [out_dir / f"split-{i}" / "patient-preds.csv" for i in range(2)]
    for csv_path in pred_csvs:
        df = pd.read_csv(csv_path)
        assert "OS_MONTHS" in df.columns
        assert "OS_EVENT" in df.columns

    stats_dir = tmp_path / "stats"
    compute_stats_(
        task="survival",
        output_dir=stats_dir,
        pred_csvs=pred_csvs,
        time_label="OS_MONTHS",
        status_label="OS_EVENT",
    )
    assert (stats_dir / "survival-stats_individual.csv").is_file()
