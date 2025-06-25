import random
import shutil
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
from huggingface_hub.errors import GatedRepoError
from random_data import create_random_dataset, create_random_feature_file, random_string

from stamp.cache import download_file
from stamp.encoding import (
    EncoderName,
    init_patient_encoder_,
    init_slide_encoder_,
)
from stamp.preprocessing.config import ExtractorName

# Contains an accepted input patch-level feature encoder
# TODO: Make a class for each extractor instead of a function. This class
# will contain as properties the extractor name and output dimension.
input_dims = {
    ExtractorName.CTRANSPATH: 768,
    ExtractorName.CHIEF_CTRANSPATH: 768,
    ExtractorName.CONCH: 512,
    ExtractorName.CONCH1_5: 768,
    ExtractorName.GIGAPATH: 1536,
    ExtractorName.VIRCHOW_FULL: 2560,
    ExtractorName.VIRCHOW2: 1280,
}

# They are not all, just one case that is accepted for each encoder
used_extractor = {
    EncoderName.CHIEF: ExtractorName.CHIEF_CTRANSPATH,
    EncoderName.COBRA: ExtractorName.CTRANSPATH,
    EncoderName.EAGLE: ExtractorName.CTRANSPATH,
    EncoderName.GIGAPATH: ExtractorName.GIGAPATH,
    EncoderName.MADELEINE: ExtractorName.CONCH,
    EncoderName.TITAN: ExtractorName.CONCH1_5,
    # EncoderName.PRISM: ExtractorName.VIRCHOW_FULL,
}


@pytest.mark.slow
@pytest.mark.parametrize("encoder", EncoderName)
@pytest.mark.filterwarnings("ignore:Importing from timm.models.layers is deprecated")
@pytest.mark.filterwarnings(
    "ignore:You are using `torch.load` with `weights_only=False`"
)
@pytest.mark.filterwarnings(
    "ignore:Importing from timm.models.registry is deprecated, please import via timm.models"
)
@pytest.mark.filterwarnings(
    "ignore:`torch.backends.cuda.sdp_kernel()` is deprecated:FutureWarning"
)
def test_if_encoding_crashes(*, tmp_path: Path, encoder: EncoderName):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    _, slide_path, feature_dir, _ = create_random_dataset(
        dir=tmp_path,
        n_patients=1,
        min_slides_per_patient=2,
        max_slides_per_patient=2,
        min_tiles_per_slide=32,
        max_tiles_per_slide=32,
        feat_dim=input_dims[used_extractor[encoder]],
        extractor_name=used_extractor[encoder],
        n_categories=2,
    )

    cuda_required = [
        EncoderName.CHIEF,
        EncoderName.COBRA,
        EncoderName.GIGAPATH,
        EncoderName.MADELEINE,
        EncoderName.EAGLE,
    ]

    if encoder in cuda_required and not torch.cuda.is_available():
        pytest.skip(f"Skipping {encoder} as CUDA is not available")

    agg_feat_dir = None
    if encoder == EncoderName.EAGLE:
        # Eagle requires the aggregated features, so we generate new ones
        # with same name and coordinates as the other ctranspath feats.
        agg_feat_dir = tmp_path / "agg_output"
        agg_feat_dir.mkdir()
        slide_df = pd.read_csv(slide_path)
        feature_filenames = [Path(path).stem for path in slide_df["slide_path"]]

        for feat_filename in feature_filenames:
            # Read the coordinates from the ctranspath feature file
            ctranspath_file = feature_dir / f"{feat_filename}.h5"
            with h5py.File(ctranspath_file, "r") as h5_file:
                coords: np.ndarray = h5_file["coords"][:]  # type: ignore
            create_random_feature_file(
                tmp_path=agg_feat_dir,
                min_tiles=32,
                max_tiles=32,
                feat_dim=input_dims[ExtractorName.VIRCHOW2],
                extractor_name=ExtractorName.VIRCHOW2,
                feat_filename=feat_filename,
                coords=coords,
            )
    elif encoder == EncoderName.TITAN:
        # A random conch1_5 feature does not work with titan so we just download
        # a real one
        downloaded_dir = tmp_path / "downloaded_output"
        downloaded_dir.mkdir()

        feat_path = download_file(
            url="https://github.com/KatherLab/STAMP/releases/download/2.1.0/conch15feats.h5",
            file_name="conch15feats.h5",
            sha256sum="f55866f5d38528a008c1381f689db36b9db57a48756b1a56b17b53357e29cb3a",
        )
        shutil.move(feat_path, downloaded_dir / feat_path.name)

        feat_path = downloaded_dir / feat_path.name
        feature_dir = downloaded_dir

        slide_df = pd.read_csv(slide_path)
        slide_df = slide_df.assign(slide_path=str(feat_path))
        slide_df.to_csv(slide_path)

    # Create random subdirectories for slide and patient features
    slide_output_dir = tmp_path / f"slide_output_{random_string(16)}"
    patient_output_dir = tmp_path / f"patient_output_{random_string(16)}"
    slide_output_dir.mkdir(exist_ok=True)
    patient_output_dir.mkdir(exist_ok=True)

    try:
        init_slide_encoder_(
            encoder=encoder,
            output_dir=slide_output_dir,
            feat_dir=Path(feature_dir),
            device="cuda" if torch.cuda.is_available() else "cpu",
            agg_feat_dir=agg_feat_dir,
        )

        init_patient_encoder_(
            encoder=encoder,
            output_dir=patient_output_dir,
            feat_dir=Path(feature_dir),
            slide_table_path=slide_path,
            patient_label="patient",
            filename_label="slide_path",
            device="cuda" if torch.cuda.is_available() else "cpu",
            agg_feat_dir=agg_feat_dir,
        )
    except ModuleNotFoundError:
        pytest.skip(f"dependencies for {encoder} not installed")
    except GatedRepoError:
        pytest.skip(f"cannot access gated repo for {encoder}")

    # Check if the slide file has any contents
    slide_files = list(slide_output_dir.glob("*.h5"))
    assert len(slide_files) > 0, "No slide feature files were generated."
    for slide_file in slide_files:
        with h5py.File(slide_file, "r") as h5_file:
            slide_datasets = list(h5_file.keys())
            # Check that there are slides
            assert len(slide_datasets) > 0
            for slide_dataset in slide_datasets:
                # Check feature contents
                assert len(h5_file[slide_dataset][:]) > 0  # type: ignore

    # Check if the patient file has any contents
    patient_files = list(patient_output_dir.glob("*.h5"))
    assert len(patient_files) > 0, "No patient feature files were generated."
    for patient_file in patient_files:
        with h5py.File(patient_file, "r") as h5_file:
            patient_datasets = list(h5_file.keys())
            # Check that the amount of patient feats is 1
            assert len(patient_datasets) == 1
            for patient_dataset in patient_datasets:
                # Check feature contents
                assert len(h5_file[patient_dataset][:]) > 0  # type: ignore
