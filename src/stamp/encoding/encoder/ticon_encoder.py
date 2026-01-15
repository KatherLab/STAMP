"""TICON Encoder - Slide-level contextualization of tile embeddings."""

import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

# try:
#     from torch.amp.autocast_mode import autocast
# except (ImportError, AttributeError):
#     try:
#         from torch.cuda.amp import autocast
#     except ImportError:
#         from torch.amp import autocast  # type: ignore
from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder, EncoderName
from stamp.modeling.data import CoordsInfo
from stamp.modeling.models.ticon_architecture import (
    TILE_EXTRACTOR_TO_TICON,
    get_ticon_key,
    load_ticon_backbone,
)
from stamp.preprocessing.config import ExtractorName
from stamp.types import DeviceLikeType

_logger = logging.getLogger("stamp")


class TiconEncoder(Encoder):
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
        self._current_extractor = None

    def _prepare_coords(self, coords: CoordsInfo, num_tiles: int) -> Tensor:
        """Prepare coordinates tensor for TICON."""
        if coords is None:
            print("No coords provided, using zeros.")
            return torch.zeros(
                1, num_tiles, 2, device=self._device, dtype=torch.float32
            )
        # CoordsInfo: get relative positions
        if isinstance(coords, CoordsInfo):
            coords_data = coords.coords_um
            if coords.tile_size_um and coords.tile_size_um > 0:
                # converting to grid-indices to get relative positions (is optional only, can be left out)
                coords_data = coords.coords_um / coords.tile_size_um
            else:
                coords_data = coords.coords_um
        else:
            coords_data = coords

        # convert CoordsInfo to tensor
        if not isinstance(coords_data, torch.Tensor):
            coords_data = np.array(coords_data)
            coords_tensor = torch.from_numpy(coords_data)
        else:
            coords_tensor = coords_data

        # adapt dimensions (add batch dim)
        if coords_tensor.dim() == 2:
            coords_tensor = coords_tensor.unsqueeze(0)  # [1, N, 2]
        assert (
            coords_tensor.shape[1] == num_tiles
        )  # number of coords-pairs must match number of tiles
        return coords_tensor.to(self._device, dtype=torch.float32)

    def _generate_slide_embedding(
        self,
        feats: torch.Tensor,
        device: DeviceLikeType,
        **kwargs,
    ) -> np.ndarray:
        """Generate contextualized slide embedding using TICON."""

        # get extractor from kwargs
        extractor = kwargs.get("extractor")
        if extractor is None:
            raise ValueError("extractor must be provided for TICON encoding")

        # Convert extractor-string to ExtractorName to be sure
        if isinstance(extractor, str):
            extractor = ExtractorName(extractor)

        tile_encoder_key, _ = get_ticon_key(extractor)
        print(f"Using tile extractor: {tile_encoder_key} for ticon")
        if feats.dim() == 2:
            feats = feats.unsqueeze(0)  # add batch dim
        feats = feats.to(self._device, dtype=torch.float32)

        # get coords from kwargs
        coords_tensor = kwargs.get("coords", None)
        print(
            f"Coords tensor shape: {coords_tensor.shape}"
            if coords_tensor is not None
            else "No coords tensor provided"
        )
        # # check pytorch version for autocast compatibility
        # is_legacy_autocast = "torch.cuda.amp" in autocast.__module__

        # ac_kwargs = {
        #     "enabled": (self._device.type == "cuda"),
        #     "dtype": torch.bfloat16,
        # }
        # # if its the new version: add device_type
        # if not is_legacy_autocast:
        #     ac_kwargs["device_type"] = "cuda"

        # Inference mode only/ without autocast
        with torch.no_grad():
            try:
                contextualized = self.model(
                    x=feats,
                    relative_coords=coords_tensor,
                    tile_encoder_key=tile_encoder_key,
                )
            except RuntimeError as e:
                _logger.error(
                    f"RuntimeError during TICON encoding without autocast: {e}. Retrying with autocast."
                )
                raise e

            # try:
            #     with autocast(**ac_kwargs):
            #         contextualized = self.model(
            #             x=feats,
            #             relative_coords=coords_tensor,
            #             tile_encoder_key=tile_encoder_key,
            #         )
            # except RuntimeError as e:
            #     _logger.error(
            #         f"RuntimeError during TICON encoding with autocast {ac_kwargs}: {e}. Retrying without autocast."
            #     )
            #     contextualized = self.model(
            #         x=feats,
            #         relative_coords=coords_tensor,
            #         tile_encoder_key=tile_encoder_key,
            #     )

        return contextualized.detach().squeeze(0).cpu().numpy()

    # only pseudo-code so TiconEncoder can be instantiated
    def _generate_patient_embedding(
        self,
        feats_list: list[torch.Tensor],
        device: DeviceLikeType,
        **kwargs,
    ) -> np.ndarray:
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
            #
            try:
                feats, coords = self._validate_and_read_features(h5_path)
            except ValueError as e:
                tqdm.write(s=str(e))
                continue
            try:
                feats, coords, extractor = self._read_h5(h5_path)
            except ValueError as e:
                tqdm.write(str(e))
                continue
            try:
                target_extractor = ExtractorName(extractor)  # str → Enum
            except ValueError:
                target_extractor = extractor  # Schon Enum

            # option to save coords because it is not a classical slide, also set feat_type to tile
            coords_um_np = coords.coords_um
            print(
                f"Coords um shape: {coords_um_np.shape}"
                if coords is not None
                else "No coords found"
            )

            # CoordsInfo -> absolute coords in µm
            if isinstance(coords_um_np, torch.Tensor):
                coords_um_np = coords_um_np.detach().cpu().numpy()
                print(f"Converted coords to numpy array, shape: {coords_um_np.shape}")
            else:
                coords_um_np = np.asarray(coords_um_np)
                print(f"Coords as numpy array, shape: {coords_um_np.shape}")

            slide_embedding = self._generate_slide_embedding(
                feats,
                device,
                coords=self._prepare_coords(coords, feats.shape[0]),
                extractor=target_extractor,
            )

            self._save_features_(
                output_path=output_path,
                feats=slide_embedding,
                feat_type="tile",
                coords=coords_um_np,
                tile_size_um=float(coords.tile_size_um)
                if coords.tile_size_um is not None
                else None,
                tile_size_px=int(coords.tile_size_px)
                if coords.tile_size_px is not None
                else None,
                unit="um",
            )

