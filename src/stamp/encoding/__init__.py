from pathlib import Path
from typing import assert_never

from torch._prims_common import DeviceLikeType  # type: ignore

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder


def get_pat_embs(
    encoder: EncoderName | Encoder,
    output_dir: Path,
    feat_dir: Path,
    slide_table_path: Path,
    device: DeviceLikeType,
    agg_feat_dir: Path | None = None,
) -> None:
    """"""
    match encoder:
        case EncoderName.COBRA:
            from stamp.encoding.encoder.cobra import Cobra

            selected_encoder: Encoder = Cobra()

        case EncoderName.EAGLE:
            from stamp.encoding.encoder.eagle import Eagle

            selected_encoder: Encoder = Eagle()

        case EncoderName.TITAN:
            from stamp.encoding.encoder.titan import Titan

            selected_encoder: Encoder = Titan()

        case Encoder():
            selected_encoder = encoder

        case _ as unreachable:
            assert_never(unreachable)

    selected_encoder.encode_patients(
        output_dir, feat_dir, slide_table_path, device, agg_feat_dir=agg_feat_dir
    )


def get_slide_embs(
    encoder: EncoderName | Encoder,
    output_dir: Path,
    feat_dir: Path,
    device: DeviceLikeType,
    agg_feat_dir: Path | None = None,
) -> None:
    match encoder:
        case EncoderName.TITAN:
            from stamp.encoding.encoder.titan import Titan

            selected_encoder: Encoder = Titan()
        case EncoderName.COBRA:
            from stamp.encoding.encoder.cobra import Cobra

            selected_encoder: Encoder = Cobra()

        case EncoderName.EAGLE:
            from stamp.encoding.encoder.eagle import Eagle

            selected_encoder: Encoder = Eagle()

        case EncoderName.GIGAPATH:
            from stamp.encoding.encoder.gigapath import Gigapath

            selected_encoder: Encoder = Gigapath()

        case EncoderName.CHIEF:
            from stamp.encoding.encoder.chief import CHIEF

            selected_encoder: Encoder = CHIEF()

        case Encoder():
            selected_encoder = encoder

        case _ as unreachable:
            assert_never(unreachable)

    selected_encoder.encode_slides(
        output_dir, feat_dir, device, agg_feat_dir=agg_feat_dir
    )
