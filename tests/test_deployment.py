import numpy as np
import numpy.typing as npt
import pytest
import torch
from random_data import make_feature_file

from stamp.modeling.data import GroundTruth, PatientData, PatientId
from stamp.modeling.deploy import _predict, _to_prediction_df
from stamp.modeling.lightning_model import LitVisionTransformer


@pytest.mark.filterwarnings("ignore:GPU available but not used")
def test_predict(
    categories: npt.NDArray = np.array(["foo", "bar", "baz"]),
    n_heads: int = 7,
    dim_input: int = 12,
) -> None:
    model = LitVisionTransformer(
        categories=list(categories),
        category_weights=torch.rand(len(categories)),
        dim_input=dim_input,
        dim_model=n_heads * 3,
        dim_feedforward=56,
        n_heads=n_heads,
        n_layers=2,
        dropout=0.5,
        ground_truth_label="test",
        train_patients=np.array(["pat1", "pat2"]),
        valid_patients=np.array(["pat3", "pat4"]),
        use_alibi=False,
    )

    patient_to_data = {
        PatientId("pat5"): PatientData(
            ground_truth=GroundTruth("foo"),
            feature_files={
                make_feature_file(
                    feats=torch.rand(23, dim_input), coords=torch.rand(23, 2)
                )
            },
        )
    }

    predictions = _predict(
        model=model,
        patient_to_data=patient_to_data,
        num_workers=2,
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
                make_feature_file(
                    feats=torch.rand(12, dim_input), coords=torch.rand(12, 2)
                )
            },
        ),
        **patient_to_data,
        PatientId("pat7"): PatientData(
            ground_truth=GroundTruth("baz"),
            feature_files={
                make_feature_file(
                    feats=torch.rand(56, dim_input), coords=torch.rand(56, 2)
                )
            },
        ),
    }

    more_predictions = _predict(
        model=model,
        patient_to_data=more_patients_to_data,
        num_workers=2,
        accelerator="cpu",
    )

    assert len(more_predictions) == len(more_patients_to_data)
    assert not torch.allclose(
        more_predictions[PatientId("pat5")], more_predictions[PatientId("pat6")]
    ), "different inputs should give different results"
    assert torch.allclose(
        predictions[PatientId("pat5")], more_predictions[PatientId("pat5")]
    ), "the same inputs should repeatedly yield the same results"


def test_to_prediction_df() -> None:
    n_heads = 7
    model = LitVisionTransformer(
        categories=["foo", "bar", "baz"],
        category_weights=torch.tensor([0.1, 0.2, 0.7]),
        dim_input=12,
        dim_model=n_heads * 3,
        dim_feedforward=56,
        n_heads=n_heads,
        n_layers=2,
        dropout=0.5,
        ground_truth_label="test",
        train_patients=np.array(["pat1", "pat2"]),
        valid_patients=np.array(["pat3", "pat4"]),
        use_alibi=False,
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
