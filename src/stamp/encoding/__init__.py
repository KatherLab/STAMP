import math
import os
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder


def get_pat_embs(
    encoder_name: EncoderName,
    output_dir: Path,
    feat_dir: Path,
    slide_table_path: Path,
    device: DeviceLikeType,
    agg_feat_dir: Path | None = None,
) -> None:
    """"""
    match encoder_name:
        case EncoderName.COBRA:
            from stamp.encoding.encoder.cobra import Cobra

            encoder: Encoder = Cobra()

        case EncoderName.EAGLE:
            from stamp.encoding.encoder.eagle import Eagle

            encoder: Encoder = Eagle()
    # TODO: Add other encoders and empty case

    encoder.encode_patients(
        output_dir, feat_dir, slide_table_path, device, agg_feat_dir=agg_feat_dir
    )


def get_slide_embs(
    encoder_name: EncoderName,
    output_dir: Path,
    feat_dir: Path,
    device: DeviceLikeType,
) -> None:
    match encoder_name:
        case EncoderName.TITAN:
            from stamp.encoding.encoder.titan import Titan

            encoder: Encoder = Titan()
        case EncoderName.COBRA:
            from stamp.encoding.encoder.cobra import Cobra

            encoder: Encoder = Cobra()

    encoder.encode_slides(output_dir, feat_dir, device)
