"""TICON Encoder - Slide-level contextualization of tile embeddings."""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

try:
    from torch.amp.autocast_mode import autocast
except (ImportError, AttributeError):
    try:
        from torch.cuda.amp import autocast
    except ImportError:
        from torch.amp import autocast  # type: ignore

from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder, EncoderName

# , _resolve_extractor_name
from stamp.modeling.data import CoordsInfo

# , get_coords
from stamp.modeling.models.ticon_architecture import (
    TILE_EXTRACTOR_TO_TICON,
    get_ticon_key,
    load_ticon_backbone,
)

#    TiconBackbone,
from stamp.preprocessing.config import ExtractorName
from stamp.types import DeviceLikeType

_logger = logging.getLogger("stamp")


class TiconEncoder(Encoder):
    """
    TICON Encoder for slide-level contextualization.

    Inherits from Encoder ABC to reuse existing infrastructure.
    """

    def __init__(
        self,
        device: DeviceLikeType = "cuda",
        precision: torch.dtype = torch.float32,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        ticon_model = load_ticon_backbone(device=device)

        super().__init__(
            model=ticon_model,
            identifier=EncoderName.TICON,
            precision=precision,
            required_extractors=list(TILE_EXTRACTOR_TO_TICON.keys()),
        )

        self._device = torch.device(device)
        self._current_extractor = ExtractorName.H_OPTIMUS_1

    def _validate_and_read_features(
        self,
        h5_path: str,
    ) -> tuple[Tensor, CoordsInfo]:
        """Extended validation returning extractor info."""
        feats, coords, extractor = self._read_h5(h5_path)

        if extractor not in self.required_extractors:
            raise ValueError(
                f"Features must be extracted with one of {self.required_extractors}.  "
                f"Got: {extractor}"
            )
        self._current_extractor = ExtractorName(extractor)
        return feats, coords

    def _prepare_coords(self, coords: CoordsInfo, num_tiles: int) -> Tensor:
        """Prepare coordinates tensor for TICON."""
        if coords is None:
            return torch.zeros(
                1, num_tiles, 2, device=self._device, dtype=torch.float32
            )
        # if CoordsInfo
        if isinstance(coords, CoordsInfo):
            # coords_data = coords.coords_um

            if coords.tile_size_um and coords.tile_size_um > 0:
                # Umrechnung in Grid-Indizes (Gleitkomma, um relative Position zu erhalten)
                coords_data = coords.coords_um / coords.tile_size_um
            else:
                coords_data = coords.coords_um

        # Dictionary
        elif isinstance(coords, dict):
            if "coords" not in coords:
                _logger.warning("coords dict missing 'coords' key, using zeros")
                return torch.zeros(
                    1, num_tiles, 2, device=self._device, dtype=torch.float32
                )
            coords_data = coords["coords"]

        # already tensor or array
        else:
            coords_data = coords

        # convert to tensor
        if not isinstance(coords_data, torch.Tensor):
            coords_data = np.array(coords_data)
            coords_tensor = torch.from_numpy(coords_data)
        else:
            coords_tensor = coords_data

        # adapt dimensions (add batch dim)
        if coords_tensor.dim() == 2:
            coords_tensor = coords_tensor.unsqueeze(0)

        return coords_tensor.to(self._device, dtype=torch.float32)

    def _generate_slide_embedding(
        self,
        feats: torch.Tensor,
        device: DeviceLikeType,
        coords: CoordsInfo,
        **kwargs,
    ) -> np.ndarray:
        """Generate contextualized slide embedding using TICON."""

        extractor = self._current_extractor
        if extractor is None:
            raise ValueError("extractor must be provided for TICON encoding")

        # Convert string to ExtractorName to be sure
        if isinstance(extractor, str):
            extractor = ExtractorName(extractor)

        tile_encoder_key, _ = get_ticon_key(extractor)
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)

        feats = feats.to(self._device, dtype=torch.float32)

        coords_tensor = self._prepare_coords(coords, feats.shape[1])

        # check pytorch version for autocast compatibility
        is_legacy_autocast = "torch.cuda.amp" in autocast.__module__

        ac_kwargs = {
            "enabled": (self._device.type == "cuda"),
            "dtype": torch.bfloat16,
        }
        # if its the new version: add device_type
        if not is_legacy_autocast:
            ac_kwargs["device_type"] = "cuda"
        with torch.no_grad():
            with autocast(**ac_kwargs):
                contextualized = self.model(
                    x=feats,
                    relative_coords=coords_tensor,
                    tile_encoder_key=tile_encoder_key,
                )

        return contextualized.detach().squeeze(0).cpu().numpy()

    # only pseudo-code so TiconEncoder can be instantiated
    def _generate_patient_embedding(
        self,
        feats_list: list[torch.Tensor],
        device: DeviceLikeType,
        coords_list: list[CoordsInfo],
        **kwargs,
    ) -> np.ndarray:
        """Generate patient embedding by contextualizing each slide."""
        contextualized = [
            self._generate_slide_embedding(feats, device, **kwargs)
            for feats in feats_list
        ]
        return np.concatenate(contextualized, axis=0)

    def encode_slides_(
        self,
        output_dir: Path,
        feat_dir: Path,
        device: DeviceLikeType,
        generate_hash: bool = True,
        **kwargs,
    ) -> None:
        """Override to pass extractor info to _generate_slide_embedding."""
        if generate_hash:
            encode_dir = f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}"
        else:
            encode_dir = f"{self.identifier}-slide"

        encode_dir = output_dir / encode_dir
        os.makedirs(encode_dir, exist_ok=True)

        self.model.to(device).eval()

        h5_files = [f for f in os.listdir(feat_dir) if f.endswith(".h5")]

        for filename in (progress := tqdm(h5_files)):
            h5_path = os.path.join(feat_dir, filename)
            slide_name = Path(filename).name
            progress.set_description(slide_name)

            output_path = (encode_dir / slide_name).with_suffix(".h5")
            if output_path.exists():
                _logger.info(f"Skipping {slide_name}: output exists")
                continue

            try:
                feats, coords, extractor = self._read_h5(h5_path)
            except ValueError as e:
                tqdm.write(str(e))
                continue

            target_extractor = ExtractorName(extractor)

            slide_embedding = self._generate_slide_embedding(
                feats, device, coords=coords, extractor=target_extractor
            )

            self._save_features_(
                output_path=output_path, feats=slide_embedding, feat_type="slide"
            )


def ticon_encoder(
    device: DeviceLikeType = "cuda",
    precision: torch.dtype = torch.float32,
) -> TiconEncoder:
    """Create a TICON encoder for slide-level contextualization."""
    return TiconEncoder(device=device, precision=precision)


__all__ = ["TiconEncoder", "ticon_encoder"]
