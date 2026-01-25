from pathlib import Path
from typing import assert_never

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.types import DeviceLikeType, PandasLabel

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"


def init_slide_encoder_(
    encoder: EncoderName | Encoder,
    output_dir: Path,
    feat_dir: Path,
    device: DeviceLikeType,
    agg_feat_dir: Path | None = None,
    generate_hash: bool = True,
) -> None:
    """
    Encode patch-level features to a single feature per slide using a given encoder.

    Args:
        encoder: The encoder to use for feature extraction.
            Can be an instance of `Encoder` or an `EncoderName` enum value.
        output_dir: Directory where the encoded patient features will be saved.
        feat_dir: Directory containing the input features for encoding.
        device: Device to use for computation (e.g., "cpu", "mps" or "cuda").
        agg_feat_dir: Directory for aggregated features. Defaults to None.
        generate_hash: Whether to generate a hash for the output file name. Defaults to True.

    Side Effects:
        - Saves encoded patient features to `output_dir` in a single .h5 file
        containing each patient as a dataset.
    """
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

        case EncoderName.CHIEF_CTRANSPATH:
            from stamp.encoding.encoder.chief import CHIEF

            selected_encoder: Encoder = CHIEF()

        case EncoderName.MADELEINE:
            from stamp.encoding.encoder.madeleine import Madeleine

            selected_encoder: Encoder = Madeleine()

        case EncoderName.PRISM:
            from stamp.encoding.encoder.prism import Prism

            selected_encoder: Encoder = Prism()

        case EncoderName.TICON:
            from stamp.encoding.encoder.ticon_encoder import TiconEncoder

            selected_encoder: Encoder = TiconEncoder()

        case Encoder():
            selected_encoder = encoder

        case _ as unreachable:
            assert_never(unreachable)  # type: ignore

    selected_encoder.encode_slides_(
        output_dir=output_dir,
        feat_dir=feat_dir,
        device=device,
        agg_feat_dir=agg_feat_dir,
        generate_hash=generate_hash,
    )


def init_patient_encoder_(
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
    """
    Encode patch-level features to a single feature per patient using a given encoder.

    This function selects an encoder based on the provided `encoder` parameter and
    uses it to encode patient-level features. Each patient's slide is concatenated
    along the x-axis to create a single 'virtual' slide for encoding.

    Args:
        encoder: The encoder to use for feature extraction.
            Can be an instance of `Encoder` or an `EncoderName` enum value.
        output_dir: Directory where the encoded patient features will be saved.
        feat_dir: Directory containing the input features for encoding.
        slide_table_path: Path to the slide table file (CSV) containing metadata.
        patient_label: Column name in the slide table representing patient IDs.
        filename_label: Column name in the slide table representing slide filenames.
        device: Device to use for computation (e.g., "cpu", "mps" or "cuda").
        agg_feat_dir: Directory for aggregated features. Defaults to None.
        generate_hash: Whether to generate a hash for the output file name. Defaults to True.

    Side Effects:
        - Saves encoded patient features to `output_dir` in a single .h5 file
        containing each patient as a dataset.
    """

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

        case EncoderName.CHIEF_CTRANSPATH:
            from stamp.encoding.encoder.chief import CHIEF

            selected_encoder: Encoder = CHIEF()

        case EncoderName.MADELEINE:
            from stamp.encoding.encoder.madeleine import Madeleine

            selected_encoder: Encoder = Madeleine()

        case EncoderName.PRISM:
            from stamp.encoding.encoder.prism import Prism

            selected_encoder: Encoder = Prism()

        case EncoderName.TICON:
            from stamp.encoding.encoder.ticon_encoder import TiconEncoder

            selected_encoder: Encoder = TiconEncoder()

        case Encoder():
            selected_encoder = encoder

        case _ as unreachable:
            assert_never(unreachable)

    selected_encoder.encode_patients_(
        output_dir=output_dir,
        feat_dir=feat_dir,
        slide_table_path=slide_table_path,
        patient_label=patient_label,
        filename_label=filename_label,
        device=device,
        agg_feat_dir=agg_feat_dir,
        generate_hash=generate_hash,
    )
