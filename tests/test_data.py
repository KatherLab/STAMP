import tempfile
from io import BytesIO
from pathlib import Path

import h5py
import pytest
import torch
from random_data import make_feature_file, make_old_feature_file
from torch.utils.data import DataLoader

from stamp.modeling.data import (
    BagDataset,
    BagSize,
    CoordsInfo,
    FeaturePath,
    GroundTruth,
    PatientData,
    PatientId,
    filter_complete_patient_data_,
    get_coords,
)
from stamp.preprocessing.tiling import Microns, SlideMPP, SlidePixels


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
                make_old_feature_file(
                    feats=torch.rand((12, dim_feats)), coords=torch.rand(12, 2)
                )
            ],
            [
                make_old_feature_file(
                    feats=torch.rand((8, dim_feats)), coords=torch.rand(8, 2)
                )
            ],
            [
                make_old_feature_file(
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


def test_get_coords_with_mpp() -> None:
    # Test new feature file with valid mpp calculation
    file_bytes = make_feature_file(
        feats=torch.rand((34, 34)),
        coords=torch.rand(34, 2),
        tile_size_um=Microns(2508.0),
        tile_size_px=SlidePixels(512),
    )
    with h5py.File(file_bytes, "r") as h5:
        coords_info = get_coords(h5)
        assert type(coords_info) is CoordsInfo
        from math import isclose

        assert isclose(coords_info.mpp, (2508.0 / 512), rel_tol=1e-9)

    # Test old feature file without mpp calculation
    file_bytes = make_old_feature_file(
        feats=torch.rand((34, 34)),
        coords=torch.rand(34, 2),
        tile_size_um=Microns(2508.0),
    )
    with h5py.File(file_bytes, "r") as h5:
        coords_info = get_coords(h5)
        assert type(coords_info) is CoordsInfo
        assert coords_info.tile_size_um == Microns(2508.0)
        assert coords_info.tile_size_px is None
        with pytest.raises(RuntimeError, match="tile size in pixels is not available"):
            _ = coords_info.mpp


def test_get_coords_invalid_file() -> None:
    # Test invalid feature file with missing attributes
    file_bytes = BytesIO()
    with h5py.File(file_bytes, "w") as h5:
        h5.create_dataset("coords", data=torch.rand(34, 2).numpy())
    with h5py.File(file_bytes, "r") as h5:
        with pytest.raises(RuntimeError, match="unable to infer coordinates"):
            get_coords(h5)


def test_get_coords_historic_format() -> None:
    # Test historic STAMP format with inferred mpp
    file_bytes = make_old_feature_file(
        feats=torch.rand((34, 34)),
        coords=torch.rand(34, 2),
    )
    with h5py.File(file_bytes, "w") as h5:
        h5.attrs["tile_size"] = 224
        h5.attrs["unit"] = "px"
        h5.create_dataset("coords_historic", data=torch.rand(34, 2).numpy())
    with h5py.File(file_bytes, "r") as h5:
        coords_info = get_coords(h5)
        assert type(coords_info) is CoordsInfo
        assert coords_info.tile_size_um == Microns(256.0)
        assert coords_info.tile_size_px == SlidePixels(224)
        assert coords_info.mpp == SlideMPP(256.0 / 224)
