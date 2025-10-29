from pathlib import Path

import numpy as np
import pytest
import torch
from random_data import create_random_patient_level_feature_file, make_old_feature_file

from stamp.modeling.data import (
    PatientData,
    patient_feature_dataloader,
    tile_bag_dataloader,
)
from stamp.modeling.deploy import (
    _predict,
    _to_prediction_df,
    _to_regression_prediction_df,
    _to_survival_prediction_df,
)
from stamp.modeling.models import (
    LitPatientClassifier,
    LitTileClassifier,
    LitTileRegressor,
    LitTileSurvival,
)
from stamp.modeling.models.mlp import MLP
from stamp.modeling.models.vision_tranformer import VisionTransformer
from stamp.seed import Seed
from stamp.types import GroundTruth, PatientId, Task


def test_predict_patient_level(
    tmp_path: Path, categories: list[str] = ["foo", "bar", "baz"], dim_feats: int = 12
):
    model = LitPatientClassifier(
        model_class=MLP,
        categories=categories,
        category_weights=torch.rand(len(categories)),
        dim_input=dim_feats,
        dim_hidden=32,
        num_layers=2,
        dropout=0.2,
        ground_truth_label="test",
        train_patients=["pat1", "pat2"],
        valid_patients=["pat3", "pat4"],
        # these values do not affect at inference time
        total_steps=320,
        max_lr=1e-4,
        div_factor=25.0,
    )

    # Create 3 random patient-level feature files on disk
    patient_ids = [PatientId(f"pat{i}") for i in range(5, 8)]
    labels = ["foo", "bar", "baz"]
    files = [
        create_random_patient_level_feature_file(
            tmp_path=tmp_path, feat_dim=dim_feats, feat_filename=str(pid)
        )
        for pid in patient_ids
    ]
    patient_to_data = {
        pid: PatientData(
            ground_truth=label,
            feature_files={file},
        )
        for pid, label, file in zip(patient_ids, labels, files)
    }

    test_dl, _ = patient_feature_dataloader(
        patient_data=list(patient_to_data.values()),
        categories=categories,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        transform=None,
    )

    predictions = _predict(
        model=model,
        test_dl=test_dl,
        patient_ids=patient_ids,
        accelerator="cpu",
    )

    assert len(predictions) == len(patient_to_data)
    for pid in patient_ids:
        assert predictions[pid].shape == torch.Size([3]), "expected one score per class"

    # Check if scores are consistent between runs and different for different patients
    more_patient_ids = [PatientId(f"pat{i}") for i in range(8, 11)]
    more_labels = ["foo", "bar", "baz"]
    more_files = [
        create_random_patient_level_feature_file(
            tmp_path=tmp_path, feat_dim=dim_feats, feat_filename=str(pid)
        )
        for pid in more_patient_ids
    ]
    more_patient_to_data = {
        pid: PatientData(
            ground_truth=label,
            feature_files={file},
        )
        for pid, label, file in zip(more_patient_ids, more_labels, more_files)
    }
    # Add the original patient for repeatability check
    all_patient_ids = more_patient_ids + [patient_ids[0]]

    more_test_dl, _ = patient_feature_dataloader(
        patient_data=[more_patient_to_data[pid] for pid in more_patient_ids]
        + [patient_to_data[patient_ids[0]]],
        categories=categories,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        transform=None,
    )

    more_predictions = _predict(
        model=model,
        test_dl=more_test_dl,
        patient_ids=all_patient_ids,
        accelerator="cpu",
    )

    assert len(more_predictions) == len(all_patient_ids)
    # Different patients should give different results
    assert not torch.allclose(
        more_predictions[more_patient_ids[0]], more_predictions[more_patient_ids[1]]
    ), "different inputs should give different results"
    # The same patient should yield the same result
    assert torch.allclose(
        predictions[patient_ids[0]], more_predictions[patient_ids[0]]
    ), "the same inputs should repeatedly yield the same results"


@pytest.mark.parametrize("task", ["classification", "regression", "survival"])
def test_to_prediction_df(task: str) -> None:
    if task == "classification":
        ModelClass = LitTileClassifier
    elif task == "regression":
        ModelClass = LitTileRegressor
    else:
        ModelClass = LitTileSurvival
    n_heads = 7
    model = ModelClass(
        model_class=VisionTransformer,
        categories=["foo", "bar", "baz"],
        category_weights=torch.tensor([0.1, 0.2, 0.7]),
        dim_input=12,
        dim_model=n_heads * 3,
        dim_feedforward=56,
        n_heads=n_heads,
        n_layers=2,
        dropout=0.5,
        ground_truth_label="test",
        time_label="time",
        status_label="status",
        train_patients=np.array(["pat1", "pat2"]),
        valid_patients=np.array(["pat3", "pat4"]),
        use_alibi=False,
        total_steps=1000,
        max_lr=1e-4,
        div_factor=25,
    )
    if task == "classification":
        preds_df = _to_prediction_df(
            categories=list(model.categories),  # type: ignore
            patient_to_ground_truth={
                PatientId("pat5"): GroundTruth("foo"),
                PatientId("pat6"): None,
                PatientId("pat7"): GroundTruth("baz"),
            },
            patient_label="patient",
            ground_truth_label="target",
            predictions={
                PatientId("pat5"): torch.rand((3)),
                PatientId("pat6"): torch.rand((3)),
                PatientId("pat7"): torch.rand((3)),
            },
        )

        # Check if all expected columns are included
        assert {
            "patient",
            "target",
            "pred",
            "target_foo",
            "target_bar",
            "target_baz",
            "loss",
        } <= set(preds_df.columns)
        assert len(preds_df) == 3

        # Check if no loss / target is given for targets with missing ground truths
        no_ground_truth = preds_df[preds_df["patient"].isin(["pat6"])]
        assert no_ground_truth["target"].isna().all()  # pyright: ignore[reportGeneralTypeIssues,reportAttributeAccessIssue]
        assert no_ground_truth["loss"].isna().all()  # pyright: ignore[reportGeneralTypeIssues,reportAttributeAccessIssue]

        # Check if loss / target is given for targets with ground truths
        with_ground_truth = preds_df[preds_df["patient"].isin(["pat5", "pat7"])]
        assert (~with_ground_truth["target"].isna()).all()  # pyright: ignore[reportGeneralTypeIssues,reportAttributeAccessIssue]
        assert (~with_ground_truth["loss"].isna()).all()  # pyright: ignore[reportGeneralTypeIssues,reportAttributeAccessIssue]

    elif task == "regression":
        patient_to_ground_truth = {}
        predictions = {PatientId(f"pat{i}"): torch.randn(1) for i in range(5)}
        preds_df = _to_regression_prediction_df(
            patient_to_ground_truth=patient_to_ground_truth,
            patient_label="patient",
            ground_truth_label="target",
            predictions=predictions,
        )
        assert "patient" in preds_df.columns
        assert "pred" in preds_df.columns
        assert len(preds_df) > 0

        assert "loss" in preds_df.columns
        assert preds_df["loss"].isna().all()
    else:
        patient_to_ground_truth = {
            PatientId("p1"): "10.0 1",
            PatientId("p2"): "12.3 0",
        }
        predictions = {
            PatientId("p1"): torch.tensor([0.8]),
            PatientId("p2"): torch.tensor([0.2]),
        }

        preds_df = _to_survival_prediction_df(
            patient_to_ground_truth=patient_to_ground_truth,
            patient_label="patient",
            ground_truth_label="target",
            predictions=predictions,
        )
        assert "patient" in preds_df.columns
        assert "pred_score" in preds_df.columns
        assert len(preds_df) > 0


@pytest.mark.filterwarnings("ignore:GPU available but not used")
@pytest.mark.filterwarnings(
    "ignore:The 'predict_dataloader' does not have many workers"
)
@pytest.mark.parametrize("task", ["classification", "regression", "survival"])
def test_mil_predict_generic(tmp_path: Path, task: Task) -> None:
    Seed.set(42)
    dim_feats = 12
    categories = ["foo", "bar", "baz"]

    if task == "classification":
        model = LitTileClassifier(
            model_class=VisionTransformer,
            categories=categories,
            category_weights=torch.rand(len(categories)),
            dim_input=dim_feats,
            dim_model=32,
            dim_feedforward=64,
            n_heads=4,
            n_layers=2,
            dropout=0.2,
            ground_truth_label="target",
            train_patients=np.array(["pat1", "pat2"]),
            valid_patients=np.array(["pat3"]),
            use_alibi=False,
            total_steps=100,
            max_lr=1e-4,
            div_factor=25.0,
        )
    elif task == "regression":
        model = LitTileRegressor(
            model_class=MLP,
            dim_input=dim_feats,
            dim_hidden=32,
            num_layers=2,
            dropout=0.1,
            ground_truth_label="target",
            train_patients=["pat1", "pat2"],
            valid_patients=["pat3"],
            total_steps=100,
            max_lr=1e-4,
            div_factor=25.0,
        )
    else:  # survival
        model = LitTileSurvival(
            model_class=MLP,
            dim_input=dim_feats,
            dim_hidden=32,
            num_layers=2,
            dropout=0.1,
            time_label="time",
            status_label="status",
            train_patients=["pat1", "pat2"],
            valid_patients=["pat3"],
            total_steps=100,
            max_lr=1e-4,
            div_factor=25.0,
        )

    # ---- Build tile-level feature file so batch = (bags, coords, bag_sizes, gt)
    if task == "classification":
        feature_file = make_old_feature_file(
            feats=torch.rand(23, dim_feats), coords=torch.rand(23, 2)
        )
        gt = GroundTruth("foo")
    elif task == "regression":
        feature_file = make_old_feature_file(
            feats=torch.rand(30, dim_feats), coords=torch.rand(30, 2)
        )
        gt = GroundTruth(42.5)  # numeric target wrapped for typing
    else:  # survival
        feature_file = make_old_feature_file(
            feats=torch.rand(40, dim_feats), coords=torch.rand(40, 2)
        )
        gt = GroundTruth("12  0")  # (time, status)

    patient_to_data = {
        PatientId("pat_test"): PatientData(
            ground_truth=gt,
            feature_files={feature_file},
        )
    }

    # ---- Use tile_bag_dataloader for ALL tasks (so batch has 4 elements)
    test_dl, _ = tile_bag_dataloader(
        task=task,  # "classification" | "regression" | "survival"
        patient_data=list(patient_to_data.values()),
        bag_size=None,
        categories=(categories if task == "classification" else None),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        transform=None,
    )

    predictions = _predict(
        model=model,
        test_dl=test_dl,
        patient_ids=list(patient_to_data.keys()),
        accelerator="cpu",
    )

    assert len(predictions) == 1
    pred = list(predictions.values())[0]
    if task == "classification":
        assert pred.shape == torch.Size([len(categories)])
    elif task == "regression":
        assert pred.shape == torch.Size([1])
    else:  # survival
        # Cox model → scalar log-risk, KM → vector or matrix
        assert pred.ndim in (0, 1, 2), f"unexpected survival output shape: {pred.shape}"

    # Repeatability
    predictions2 = _predict(
        model=model,
        test_dl=test_dl,
        patient_ids=list(patient_to_data.keys()),
        accelerator="cpu",
    )
    for pid in predictions:
        assert torch.allclose(predictions[pid], predictions2[pid])
