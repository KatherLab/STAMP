import tempfile
from io import BytesIO
from pathlib import Path

import h5py
import pytest
import torch
from random_data import (
    create_random_patient_level_feature_file,
    make_feature_file,
    make_old_feature_file,
    create_good_and_bad_slide__tables,
)
from torch.utils.data import DataLoader

from stamp.modeling.data import (
    BagDataset,
    CoordsInfo,
    PatientData,
    PatientFeatureDataset,
    filter_complete_patient_data_,
    get_coords,
    slide_to_patient_from_slide_table_,
)
from stamp.types import (
    BagSize,
    FeaturePath,
    GroundTruth,
    Microns,
    PatientId,
    SlideMPP,
    TilePixels,
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


@pytest.mark.parametrize(
    "feature_file_creator",
    [make_feature_file, make_old_feature_file],
)
def test_bag_dataset(
    feature_file_creator,
    bag_size: BagSize = BagSize(5),
    dim_feats: int = 34,
    batch_size: int = 2,
) -> None:
    ds = BagDataset(
        bags=[
            [
                feature_file_creator(
                    feats=torch.rand((12, dim_feats)), coords=torch.rand(12, 2)
                )
            ],
            [
                feature_file_creator(
                    feats=torch.rand((8, dim_feats)), coords=torch.rand(8, 2)
                )
            ],
            [
                feature_file_creator(
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


def test_patient_feature_dataset(
    tmp_path: Path, dim_feats: int = 16, batch_size: int = 2
) -> None:
    # Create 3 random patient-level feature files on disk
    files = [
        create_random_patient_level_feature_file(tmp_path=tmp_path, feat_dim=dim_feats)
        for _ in range(3)
    ]
    # One-hot encoded labels for 3 samples, 4 categories
    labels = torch.eye(4)[:3]

    ds = PatientFeatureDataset(files, labels, transform=None)
    assert len(ds) == 3

    # Test single dataset item
    feats, label = ds[0]
    assert feats.shape == (dim_feats,)
    assert torch.allclose(label, labels[0])

    # Test batching
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
    feats_batch, labels_batch = next(iter(dl))
    assert feats_batch.shape == (batch_size, dim_feats)
    assert labels_batch.shape == (batch_size, 4)


def test_get_coords_with_mpp() -> None:
    # Test new feature file with valid mpp calculation
    file_bytes = make_feature_file(
        feats=torch.rand((34, 34)),
        coords=torch.rand(34, 2),
        tile_size_um=Microns(2508.0),
        tile_size_px=TilePixels(512),
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
        assert coords_info.tile_size_px == TilePixels(224)
        assert coords_info.mpp == SlideMPP(256.0 / 224)


def test_slide_table_h5_validation (tmp_path: Path):
    """
    Test that an error is raised in 
    slide_to_patient_from_slide_table_() when no .h5 files are in the slide
    table.
    """
    feature_dir = tmp_path

    good_slide_path, bad_slide_path = create_good_and_bad_slide__tables(tmp_path=tmp_path)
    # remember that PandasLabel is just a string

    # Test Good Slide Path (should be no error or error that doesn't contain the error I made)
    # assert slide_to_patient_from_slide_table_(slide_table_path=good_slide_path, feature_dir=feature_dir, patient_label="PATIENT", filename_label="FILENAME")
    with pytest.raises(ValueError, match="No .h5 extensions found in the slide table's feature path"):
        slide_to_patient_from_slide_table_(
            slide_table_path=good_slide_path,
            feature_dir=feature_dir,
            patient_label="PATIENT",
            filename_label="FILENAME")
    # Test Bad Slide Path
    with pytest.raises(ValueError, match="No .h5 extensions found in the slide table's feature path"):
        slide_to_patient_from_slide_table_(
            slide_table_path=bad_slide_path,
            feature_dir=feature_dir,
            patient_label="PATIENT",
            filename_label="FILENAME")

# def test_slide_table_h5_validation_random(tmp_path: Path, ):
#     """
#     Test that an error is raised in 
#     slide_to_patient_from_slide_table_() when no .h5 files are in the slide
#     table.
#     """

#     slide_path = dir / "slide.csv"



#     # Create temp paths
#     (tmp_path / "test_data").mkdir()

#     # Create bad slide table
#     bad_slide_table = "bad_slide"
#     bad_feature_dir = ""
#     bad_patient_label = 1
#     bad_filename_label = 1

    
#     # Create a good slide table

#     clini_pathj = slide_path, feat_dir, categories = create_random_dataset(
        
#     )
    
#     good_slide_table= 2
#     good_feature_dir = 2
#     good_patient_label = 2
#     good_filename_label = 2

#     # This should raise an error   
#     slide_df = pd.DataFrame(
#         slide_path_to_patient.items()
#     )
#     assert
#     slide_to_patient_from_slide_table_(
#     slide_table_path=bad_slide_table,
#             feature_dir=bad_feature_dir,
#             patient_label=bad_patient_label,
#             filename_label=bad_filename_label,
# ) 
#     # This should not raise an error
#     slide_to_patient_from_slide_table_(
#     slide_table_path=good_slide_table,
#             feature_dir=good_feature_dir,
#             patient_label=good_patient_label,
#             filename_label=good_filename_label,
# ) 