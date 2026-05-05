"""
Backward-compatibility tests for STAMP model checkpoints.

Each test case loads a checkpoint (from a local weights directory or a
GitHub-release URL) and verifies that the current codebase can:
  1. deserialise the checkpoint without errors
  2. produce predictions with the expected output shape

Checkpoints are created by tests/make_test_models.py.
URLs are left blank here; fill them in after uploading to GitHub releases.

Skips gracefully when neither the local file nor a download URL is available.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest
import torch

from stamp.modeling.data import PatientData, patient_feature_dataloader, tile_bag_dataloader
from stamp.modeling.deploy import _predict, load_model_from_ckpt
from stamp.types import FeaturePath, PatientId, Task
from stamp.utils.seed import Seed

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "test_models"

# ---------------------------------------------------------------------------
# Model registry
# (model_name, task, local_ckpt, url, sha256)
# Fill url / sha256 once the file is uploaded to GitHub releases.
# ---------------------------------------------------------------------------

_MODELS = [
    (
        "vit",
        "classification",
        "vit_tile_classification.ckpt",
        "",  # url
        "",  # sha256
    ),
    (
        "mlp",
        "classification",
        "mlp_tile_classification.ckpt",
        "",  # url
        "",  # sha256
    ),
    (
        "trans_mil",
        "classification",
        "transmil_tile_classification.ckpt",
        "",  # url
        "",  # sha256
    ),
    (
        "vit",
        "regression",
        "vit_tile_regression.ckpt",
        "",  # url
        "",  # sha256
    ),
    (
        "vit",
        "survival",
        "vit_tile_survival.ckpt",
        "",  # url
        "",  # sha256
    ),
    (
        "barspoon",
        "classification",
        "barspoon_tile_classification.ckpt",
        "",  # url
        "",  # sha256
    ),
]


# ---------------------------------------------------------------------------
# Feature-file helpers
# ---------------------------------------------------------------------------


def _make_tile_feature_file(*, n_tiles: int, dim_input: int) -> io.BytesIO:
    """In-memory .h5 tile-level feature file."""
    buf = io.BytesIO()
    feats = torch.randn(n_tiles, dim_input).numpy()
    coords = np.random.rand(n_tiles, 2).astype(np.float32) * 2508
    with h5py.File(buf, "w") as h5:
        h5["feats"] = feats
        h5["coords"] = coords
        h5.attrs["extractor"] = "random-test-generator"
        h5.attrs["unit"] = "um"
        h5.attrs["tile_size"] = 2508
    return buf


def _make_patient_feature_file(
    tmp_path: Path, *, patient_id: str, dim_input: int
) -> FeaturePath:
    """On-disk .h5 patient-level feature file."""
    path = tmp_path / f"{patient_id}.h5"
    feats = torch.randn(1, dim_input).numpy()
    with h5py.File(path, "w") as h5:
        h5["feats"] = feats
        h5.attrs["feat_type"] = "patient"
        h5.attrs["encoder"] = "random-test-generator"
    return FeaturePath(path)


# ---------------------------------------------------------------------------
# Checkpoint resolver
# ---------------------------------------------------------------------------


def _resolve_checkpoint(ckpt_filename: str, url: str, sha256: str) -> Path | None:
    """Return a local path to the checkpoint, or None if unavailable."""
    local = _WEIGHTS_DIR / ckpt_filename
    if local.exists():
        return local

    if url:
        from stamp.cache import download_file

        return download_file(url=url, file_name=ckpt_filename, sha256sum=sha256)

    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:GPU available but not used")
@pytest.mark.filterwarnings(
    "ignore:The 'predict_dataloader' does not have many workers"
)
@pytest.mark.parametrize(
    "model_name,task,ckpt_filename,url,sha256",
    [
        pytest.param(*row, id=row[2].replace(".ckpt", ""))
        for row in _MODELS
    ],
)
def test_backward_compatibility(
    model_name: str,
    task: Task,
    ckpt_filename: str,
    url: str,
    sha256: str,
    tmp_path: Path,
) -> None:
    Seed.set(42)

    ckpt_path = _resolve_checkpoint(ckpt_filename, url, sha256)
    if ckpt_path is None:
        pytest.skip(
            f"Checkpoint '{ckpt_filename}' not found locally and no URL provided. "
            "Run tests/make_test_models.py to generate it."
        )

    model = load_model_from_ckpt(ckpt_path)

    dim_input: int = model.hparams["dim_input"]
    feature_type: str = model.hparams.get("supported_features", "tile")
    n_patients = 3
    n_tiles = 32

    patient_ids = [PatientId(f"pat_{i:02d}") for i in range(n_patients)]

    # ---- build PatientData and dataloader -------------------------------- #
    if feature_type == "patient":
        files = [
            _make_patient_feature_file(tmp_path, patient_id=str(pid), dim_input=dim_input)
            for pid in patient_ids
        ]
        patient_to_data = {
            pid: PatientData(ground_truth=None, feature_files={f})
            for pid, f in zip(patient_ids, files)
        }
        test_dl, _ = patient_feature_dataloader(
            patient_data=list(patient_to_data.values()),
            categories=(
                list(model.categories) if hasattr(model, "categories") and isinstance(model.categories, list) else None
            ),
            batch_size=1,
            shuffle=False,
            num_workers=1,
            transform=None,
        )
    else:  # tile-level
        feature_files = [
            _make_tile_feature_file(n_tiles=n_tiles, dim_input=dim_input)
            for _ in patient_ids
        ]
        patient_to_data = {
            pid: PatientData(ground_truth=None, feature_files={cast(FeaturePath, ff)})
            for pid, ff in zip(patient_ids, feature_files)
        }
        categories: list[str] | None = None
        if task == "classification" and hasattr(model, "categories"):
            cats = model.categories
            if isinstance(cats, list):
                categories = cats
            elif isinstance(cats, dict):
                # barspoon: dict[target_label, list[str]]
                categories = [c for v in cats.values() for c in v]

        test_dl, _ = tile_bag_dataloader(
            task=task,
            patient_data=list(patient_to_data.values()),
            bag_size=None,
            categories=categories,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            transform=None,
        )

    # ---- run predictions ------------------------------------------------- #
    predictions = _predict(
        model=model,
        test_dl=test_dl,
        patient_ids=patient_ids,
        accelerator="cpu",
    )

    assert len(predictions) == n_patients, (
        f"expected {n_patients} predictions, got {len(predictions)}"
    )

    for pid in patient_ids:
        pred = predictions[pid]

        if isinstance(pred, dict):
            # barspoon multi-target output
            for target_label, tensor in pred.items():
                assert isinstance(tensor, torch.Tensor), (
                    f"barspoon head '{target_label}' prediction should be a Tensor"
                )
                assert not torch.isnan(tensor).any(), (
                    f"NaN in barspoon prediction for '{target_label}'"
                )
        else:
            pred_tensor = cast(torch.Tensor, pred)
            assert not torch.isnan(pred_tensor).any(), f"NaN in prediction for {pid}"

            if task == "classification":
                assert pred_tensor.ndim == 1, (
                    f"classification output should be 1-D, got shape {pred_tensor.shape}"
                )
            elif task == "regression":
                assert pred_tensor.numel() == 1, (
                    f"regression output should be scalar, got shape {pred_tensor.shape}"
                )
            else:  # survival
                assert pred_tensor.numel() >= 1, (
                    f"survival output should be non-empty, got shape {pred_tensor.shape}"
                )
