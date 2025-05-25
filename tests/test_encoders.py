import random
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from huggingface_hub.errors import GatedRepoError
from random_data import create_random_dataset, create_random_feature_file

from stamp.encoding import EncoderName, get_pat_embs, get_slide_embs
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
    EncoderName.PRISM: ExtractorName.VIRCHOW_FULL,
    EncoderName.TITAN: ExtractorName.CONCH1_5,
}



@pytest.mark.slow
@pytest.mark.parametrize("encoder", [EncoderName.TITAN])
@pytest.mark.filterwarnings("ignore:Importing from timm.models.layers is deprecated")
@pytest.mark.filterwarnings(
    "ignore:You are using `torch.load` with `weights_only=False`"
)
@pytest.mark.filterwarnings(
    "ignore:Importing from timm.models.registry is deprecated, please import via timm.models"
)
@pytest.mark.filterwarnings(
    "ignore:` torch.backends.cuda.sdp_kernel()` is deprecated. In the future, this context manager will be removed. Please see `torch.nn.attention.sdpa_kernel()` for the new context manager, with updated signature."
)
def test_if_encoding_crashes(*, tmp_path: Path, encoder: EncoderName):
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)

    _, slide_path, feature_dir, _ = create_random_dataset(
        dir=tmp_path,
        n_patients=2,
        min_slides_per_patient=2,
        max_slides_per_patient=3,
        min_tiles_per_slide=32,
        max_tiles_per_slide=32,
        feat_dim=input_dims[used_extractor[encoder]],
        extractor_name=used_extractor[encoder],
        n_categories=2,
    )

    agg_feat_dir = None
    if encoder == EncoderName.EAGLE:
        agg_feat_dir = tmp_path / "agg_output"
        agg_feat_dir.mkdir()
        slide_df = pd.read_csv(slide_path)
        feature_filenames = [Path(path).name for path in slide_df["slide_path"]]

        for feat_filename in feature_filenames:
            create_random_feature_file(
                tmp_path=tmp_path,
                min_tiles=32,
                max_tiles=32,
                feat_dim=input_dims[ExtractorName.VIRCHOW2],
                extractor_name=ExtractorName.VIRCHOW2,
            )
    elif encoder == EncoderName.TITAN:
        # A random conch1_5 feature does not work with titan so we just download
        # a real one
        feature_dir = "/mnt/bulk-sirius/juan/pap_screening/datasets/example/features/mahmood-conch1_5-df7541f1"
        slide_df = pd.read_csv(slide_path)
        slide_df.assign(slide_path=feature_dir)

    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)

    try:
        get_slide_embs(
            encoder=encoder,
            output_dir=output_dir,
            feat_dir=Path(feature_dir),
            device="cuda" if torch.cuda.is_available() else "cpu",
            agg_feat_dir=agg_feat_dir,
        )

        # TODO: test patient encoding
    except ModuleNotFoundError:
        pytest.skip(f"dependencies for {encoder} not installed")
    except GatedRepoError:
        pytest.skip(f"cannot access gated repo for {encoder}")

    # TODO: test that the contents are not empty
