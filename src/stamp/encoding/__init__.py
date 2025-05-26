from pathlib import Path
from typing import assert_never

from torch._prims_common import DeviceLikeType  # type: ignore

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import PandasLabel

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"

def get_pat_embs(
    encoder: EncoderName | Encoder,
    output_dir: Path,
    feat_dir: Path,
    slide_table_path: Path,
    patient_label: PandasLabel,
    filename_label: PandasLabel,
    device: DeviceLikeType,
    agg_feat_dir: Path | None = None,
    generate_hash: bool = True,
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

        case EncoderName.GIGAPATH:
            from stamp.encoding.encoder.gigapath import Gigapath

            selected_encoder: Encoder = Gigapath()

        case EncoderName.CHIEF:
            from stamp.encoding.encoder.chief import CHIEF

            selected_encoder: Encoder = CHIEF()

        case EncoderName.PRISM:
            from stamp.encoding.encoder.prism import Prism

            selected_encoder: Encoder = Prism()

        case EncoderName.MADELEINE:
            from stamp.encoding.encoder.madeleine import Madeleine

            selected_encoder: Encoder = Madeleine()

        case Encoder():
            selected_encoder = encoder

        case _ as unreachable:
            assert_never(unreachable)

    selected_encoder.encode_patients(
        output_dir=output_dir,
        feat_dir=feat_dir,
        slide_table_path=slide_table_path,
        patient_label=patient_label,
        filename_label=filename_label,
        device=device,
        agg_feat_dir=agg_feat_dir,
        generate_hash=generate_hash,
    )


def get_slide_embs(
    encoder: EncoderName | Encoder,
    output_dir: Path,
    feat_dir: Path,
    device: DeviceLikeType,
    agg_feat_dir: Path | None = None,
    generate_hash: bool = True,
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

        case EncoderName.PRISM:
            from stamp.encoding.encoder.prism import Prism

            selected_encoder: Encoder = Prism()

        case EncoderName.MADELEINE:
            from stamp.encoding.encoder.madeleine import Madeleine

            selected_encoder: Encoder = Madeleine()

        case Encoder():
            selected_encoder = encoder

        case _ as unreachable:
            assert_never(unreachable)

    selected_encoder.encode_slides(
        output_dir=output_dir,
        feat_dir=feat_dir,
        device=device,
        agg_feat_dir=agg_feat_dir,
        generate_hash=generate_hash,
    )
