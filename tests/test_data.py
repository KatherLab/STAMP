import tempfile
from pathlib import Path

import pytest
import torch
from random_data import make_feature_file
from torch.utils.data import DataLoader

from stamp.modeling.data import (
    BagDataset,
    BagSize,
    FeaturePath,
    GroundTruth,
    PatientData,
    PatientId,
    filter_complete_patient_data_,
)


@pytest.mark.filterwarnings("ignore:some patients have no associated slides")
@pytest.mark.filterwarnings("ignore:some feature files could not be found")
def test_get_cohort_df(tmp_path: Path) -> None:
    with (
        tempfile.NamedTemporaryFile(dir=tmp_path) as slide_a1,
        tempfile.NamedTemporaryFile(dir=tmp_path) as slide_b1,
        tempfile.NamedTemporaryFile(dir=tmp_path) as slide_b2,
        tempfile.NamedTemporaryFile(dir=tmp_path) as slide_c1,
    ):
        patients_with_complete_data = filter_complete_patient_data_(
            patient_to_ground_truth={
                # patient with one slide
                PatientId("Patient A"): GroundTruth("mutated"),
                # patient with two slides
                PatientId("Patient B"): GroundTruth("mutated"),
                # patient with two slides, one of which has no feature file
                PatientId("Patient C"): GroundTruth("wild type"),
                # patient without slides
                PatientId("Patient D"): GroundTruth("wild type"),
                # patient one slide but without corresponding features
                PatientId("Patient E"): GroundTruth("wild type"),
            },
            slide_to_patient={
                FeaturePath(Path(slide_a1.name)): PatientId("Patient A"),
                FeaturePath(Path(slide_b1.name)): PatientId("Patient B"),
                FeaturePath(Path(slide_b2.name)): PatientId("Patient B"),
                FeaturePath(Path(slide_c1.name)): PatientId("Patient C"),
            },
            drop_patients_with_missing_ground_truth=True,
        )

        assert patients_with_complete_data == {
            "Patient A": PatientData(
                ground_truth=GroundTruth("mutated"),
                feature_files={FeaturePath(Path(slide_a1.name))},
            ),
            "Patient B": PatientData(
                ground_truth=GroundTruth("mutated"),
                feature_files={
                    FeaturePath(Path(slide_b1.name)),
                    FeaturePath(Path(slide_b2.name)),
                },
            ),
            "Patient C": PatientData(
                ground_truth=GroundTruth("wild type"),
                feature_files={FeaturePath(Path(slide_c1.name))},
            ),
        }


def test_dataset(
    bag_size: BagSize = BagSize(5),
    dim_feats: int = 34,
    batch_size: int = 2,
) -> None:
    ds = BagDataset(
        bags=[
            [
                make_feature_file(
                    feats=torch.rand((12, dim_feats)), coords=torch.rand(12, 2)
                )
            ],
            [
                make_feature_file(
                    feats=torch.rand((8, dim_feats)), coords=torch.rand(8, 2)
                )
            ],
            [
                make_feature_file(
                    feats=torch.rand((34, dim_feats)), coords=torch.rand(34, 2)
                )
            ],
        ],
        bag_size=bag_size,
        ground_truths=torch.rand(3, 4) > 0.5,
        transform=None,
    )

    assert len(ds) == 3

    # Test single dataset item
    item_bag, coords, item_bag_size, _ = ds[0]
    assert item_bag.shape == (bag_size, dim_feats)
    assert coords.shape == (bag_size, 2)
    assert item_bag_size <= bag_size

    # Test batching
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    bag, coords, bag_sizes, _ = next(iter(dl))
    assert bag.shape == (batch_size, bag_size, dim_feats)
    assert coords.shape == (batch_size, bag_size, 2)
    assert (bag_sizes <= bag_size).all()
