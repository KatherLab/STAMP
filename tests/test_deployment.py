from pathlib import Path

import numpy as np
import pytest
import torch
from random_data import create_random_patient_level_feature_file, make_old_feature_file

from stamp.modeling.classifier import LitPatientlassifier, LitTileClassifier
from stamp.modeling.classifier.mlp import MLPClassifier
from stamp.modeling.classifier.vision_tranformer import VisionTransformer
from stamp.modeling.data import (
    PatientData,
    patient_feature_dataloader,
    tile_bag_dataloader,
)
from stamp.modeling.deploy import _predict, _to_prediction_df
from stamp.types import GroundTruth, PatientId


@pytest.mark.filterwarnings("ignore:GPU available but not used")
@pytest.mark.filterwarnings(
    "ignore:The 'predict_dataloader' does not have many workers which may be a bottleneck"
)
def test_predict(
    categories: list[str] = ["foo", "bar", "baz"],
    n_heads: int = 7,
    dim_input: int = 12,
) -> None:
    model = LitTileClassifier(
        categories=list(categories),
        category_weights=torch.rand(len(categories)),
        dim_input=dim_input,
        model=VisionTransformer(
            dim_input=dim_input,
            dim_output=len(categories),
            dim_model=n_heads * 3,
            dim_feedforward=56,
            n_heads=n_heads,
            n_layers=2,
            dropout=0.5,
            use_alibi=False,
        ),
        ground_truth_label="test",
        train_patients=np.array(["pat1", "pat2"]),
        valid_patients=np.array(["pat3", "pat4"]),
        use_alibi=False,
        # these values do not affect at inference time
        total_steps=320,
        max_lr=1e-4,
        div_factor=25.0,
    )

    patient_to_data = {
        PatientId("pat5"): PatientData(
            ground_truth=GroundTruth("foo"),
            feature_files={
                make_old_feature_file(
                    feats=torch.rand(23, dim_input), coords=torch.rand(23, 2)
                )
            },
        )
    }

    test_dl, _ = tile_bag_dataloader(
        patient_data=list(patient_to_data.values()),
        bag_size=None,
        categories=list(model.categories),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        transform=None,
    )

    predictions = _predict(
        model=model,
        test_dl=test_dl,
        patient_ids=list(patient_to_data.keys()),
        accelerator="cpu",
    )

    assert len(predictions) == len(patient_to_data)
    assert predictions[PatientId("pat5")].shape == torch.Size([3]), (
        "expected one score per class"
    )

    # Check if scores are consistent between runs
    more_patients_to_data = {
        PatientId("pat6"): PatientData(
            ground_truth=GroundTruth("bar"),
            feature_files={
                make_old_feature_file(
                    feats=torch.rand(12, dim_input), coords=torch.rand(12, 2)
                )
            },
        ),
        **patient_to_data,
        PatientId("pat7"): PatientData(
            ground_truth=GroundTruth("baz"),
            feature_files={
                make_old_feature_file(
                    feats=torch.rand(56, dim_input), coords=torch.rand(56, 2)
                )
            },
        ),
    }

    more_test_dl, _ = tile_bag_dataloader(
        patient_data=list(more_patients_to_data.values()),
        bag_size=None,
        categories=list(model.categories),
        batch_size=1,
        shuffle=False,
        num_workers=2,
        transform=None,
    )

    more_predictions = _predict(
        model=model,
        test_dl=more_test_dl,
        patient_ids=list(more_patients_to_data.keys()),
        accelerator="cpu",
    )

    assert len(more_predictions) == len(more_patients_to_data)
    assert not torch.allclose(
        more_predictions[PatientId("pat5")], more_predictions[PatientId("pat6")]
    ), "different inputs should give different results"
    assert torch.allclose(
        predictions[PatientId("pat5")], more_predictions[PatientId("pat5")]
    ), "the same inputs should repeatedly yield the same results"


def test_predict_patient_level(
    tmp_path: Path, categories: list[str] = ["foo", "bar", "baz"], dim_feats: int = 12
):
    model = LitPatientlassifier(
        categories=categories,
        category_weights=torch.rand(len(categories)),
        model=MLPClassifier(
            dim_output=len(categories),
            dim_input=dim_feats,
            dim_hidden=32,
            num_layers=2,
            dropout=0.2,
        ),
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

def test_to_prediction_df(
    categories: list[str] = ["foo", "bar", "baz"],
    n_heads: int = 7,
) -> None:
    model = LitTileClassifier(
        categories=list(categories),
        category_weights=torch.tensor([0.1, 0.2, 0.7]),
        model=VisionTransformer(
            dim_output=len(categories),
            dim_input=12,
            dim_model=n_heads * 3,
            dim_feedforward=56,
            n_heads=n_heads,
            n_layers=2,
            dropout=0.5,
            use_alibi=False,
        ),
        ground_truth_label="test",
        train_patients=np.array(["pat1", "pat2"]),
        valid_patients=np.array(["pat3", "pat4"]),
        use_alibi=False,
        total_steps=1000,
        max_lr=1e-4,
        div_factor=25,
    )

    preds_df = _to_prediction_df(
        categories=list(model.categories),
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
