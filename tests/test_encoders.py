import os
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from huggingface_hub.errors import GatedRepoError

from stamp.cache import download_file
from stamp.encoding import EncoderName


@pytest.mark.slow
@pytest.mark.parametrize("encoder", EncoderName)
@pytest.mark.filterwarnings("ignore:Importing from timm.models.layers is deprecated")
@pytest.mark.filterwarnings(
    "ignore:You are using `torch.load` with `weights_only=False`"
)
def test_if_slide_encoder_crashes(*, tmp_path: Path, encoder: EncoderName):
    pass
