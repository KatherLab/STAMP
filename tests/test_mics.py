from pathlib import Path

import h5py
import numpy as np
import torch

from typing import cast, Sequence

from stamp.modeling.config import TrainConfig
from stamp.modeling.data import (
    PatientData,
    _log_patient_slide_feature_inconsistencies,
    create_dataloader,
)
from stamp.modeling.models.__init__ import LitSlideClassifier
from stamp.modeling.models.mlp import MLP
from stamp.modeling.registry import ModelName, load_model_class
from stamp.types import FeaturePath
from stamp.utils.config import StampConfig


def _make_h5(path: Path, vec: np.ndarray) -> None:
    with h5py.File(path, "w") as h5:
        h5.create_dataset("feats", data=vec)


def test_create_dataloader_infers_categories_patient(tmp_path: Path) -> None:
    # create two patient-level feature files
    p1 = tmp_path / "p1.h5"
    p2 = tmp_path / "p2.h5"
    _make_h5(p1, np.zeros((1, 4), dtype=np.float32))
    _make_h5(p2, np.zeros((1, 4), dtype=np.float32))

    pd1 = PatientData(ground_truth="A", feature_files=[FeaturePath(p1)])
    pd2 = PatientData(ground_truth="B", feature_files=[FeaturePath(p2)])

    dl, cats = create_dataloader(
        feature_type="patient",
        task="classification",
        patient_data=[pd1, pd2],
        bag_size=None,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        transform=None,
        categories=None,
    )

    assert set(cats) == {"A", "B"}


def test_create_dataloader_survival_tuple_and_legacy_string(tmp_path: Path) -> None:
    p1 = tmp_path / "s1.h5"
    p2 = tmp_path / "s2.h5"
    _make_h5(p1, np.zeros((1, 3), dtype=np.float32))
    _make_h5(p2, np.zeros((1, 3), dtype=np.float32))

    # tuple (time, event) and legacy string (will be parsed defensively)
    pd1 = PatientData(ground_truth=(5.0, 1), feature_files=[FeaturePath(p1)])
    pd2 = PatientData(ground_truth="7 0", feature_files=[FeaturePath(p2)])

    dl, cats = create_dataloader(
        feature_type="patient",
        task="survival",
        patient_data=cast(Sequence[PatientData], [pd1, pd2]),
        bag_size=None,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        transform=None,
        categories=None,
    )

    batch_feats, labels = next(iter(dl))
    # labels: shape (2,2) -> (time, event)
    assert labels.shape[0] == 2
    # first sample is (5.0, 1)
    assert torch.isclose(labels[0, 0], torch.tensor(5.0))
    assert labels[0, 1] == 1.0
    # legacy string parsing yields an event token parsed to 0 (censored)
    assert labels[1, 1] == 0.0


def test_predict_dtype_casting_no_error() -> None:
    # Build a minimal LitSlideClassifier with MLP backbone
    categories = ["A", "B"]
    category_weights = torch.tensor([1.0, 1.0], dtype=torch.float32)

    model = LitSlideClassifier(
        model_class=MLP,
        ground_truth_label="y",
        categories=categories,
        category_weights=category_weights,
        dim_input=4,
        # provide backbone hyperparams
        dim_hidden=8,
        num_layers=2,
        dropout=0.1,
        # Base required args
        total_steps=10,
        max_lr=1e-3,
        div_factor=25.0,
        train_patients=[],
        valid_patients=[],
    )

    # force model params to half precision to simulate mixed-precision checkpoints
    model.model.half()

    feats = torch.rand((1, 4), dtype=torch.float32)
    labels = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    # predict_step should cast feats to model dtype (half) and not raise
    out = model.predict_step((feats, labels), 0)
    assert isinstance(out, torch.Tensor)
    assert out.dtype == next(model.model.parameters()).dtype


def test_log_missing_feature_filenames(caplog, tmp_path: Path) -> None:
    missing = tmp_path / "missing.h5"
    # slide_to_patient maps FeaturePath -> patient id
    slide_to_patient = {FeaturePath(missing): "P1"}
    patient_to_ground_truth = {"P1": "A"}

    with caplog.at_level("WARNING"):
        _log_patient_slide_feature_inconsistencies(
            patient_to_ground_truth=patient_to_ground_truth,
            slide_to_patient=slide_to_patient,
        )

    assert "some feature files could not be found" in caplog.text
    # ensure only filename is logged (not full path)
    assert "missing.h5" in caplog.text


def test_model_registry_returns_classes() -> None:
    LitClass, ModelClass = load_model_class("classification", "patient", ModelName.MLP)
    assert callable(LitClass)
    assert callable(ModelClass)


def test_stampconfig_training_task_default() -> None:
    cfg = StampConfig.model_validate(
        {
            "training": {
                "output_dir": "out",
                "clini_table": "cl.csv",
                "feature_dir": "feats",
                "ground_truth_label": "gt",
            }
        }
    )
    assert cfg.training is not None
    assert (
        cfg.training.task
        == TrainConfig(
            output_dir=Path("out"),
            clini_table=Path("cl.csv"),
            feature_dir=Path("feats"),
        ).task
    )
