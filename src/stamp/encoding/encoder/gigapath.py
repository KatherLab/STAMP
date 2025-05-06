import os
from pathlib import Path

import h5py
import numpy as np
import torch
from gigapath import slide_encoder
from torch import Tensor
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo, get_coords


class Gigapath(Encoder):
    def __init__(self) -> None:
        try:
            model = slide_encoder.create_model(
                "hf_hub:prov-gigapath/prov-gigapath", "gigapath_slide_enc12l768d", 1536
            )
        except AssertionError:
            raise ModuleNotFoundError(
                "Gigapath requires flash-attn. "
                "Install it with: pip install flash-attn --no-build-isolation"
            )
        # I cant add flash-attn to the pyproject.toml because it requires torch
        # beforehand, I add torch to build-system requires but throws the
        # same bloody error
        super().__init__(model=model, identifier="gigapath")
    # TODO: Make this a shared function for all encoders.
    def _read_h5(self, h5_path: str) -> tuple[Tensor, CoordsInfo, str]:
        if not os.path.exists(h5_path) or not h5_path.endswith(".h5"):
            raise FileNotFoundError("File does not exist or is not an h5 file")
        with h5py.File(h5_path, "r") as f:
            feats: Tensor = torch.tensor(f["feats"][:], dtype=torch.float16)  # type: ignore
            coords: CoordsInfo = get_coords(f)
            extractor: str = f.attrs.get("extractor", "no extractor name")
            return feats, coords, extractor

    # TODO: This can be reused too, give extractor name as parameter
    def _validate_and_read_features(self, h5_path) -> tuple[Tensor, CoordsInfo]:
        feats, coords, extractor = self._read_h5(h5_path)
        if "gigapath" not in extractor:
            raise ValueError(
                f"Features must be extracted with gigapath. "
                f"Features located in {h5_path} are extracted with {extractor}"
            )
        return feats, coords

    def _convert_coords(
        self,
        coords,
        total_wsi_width,
        max_wsi_height,
        n_grid,
        current_x_offset,
    ):
        """
        Normalize the x and y coordinates relative to the total WSI width and max height, using the same grid [0, 1000].
        Thanks Peter!
        """
        # Normalize x-coordinates based on total WSI width (taking into account the current x offset)
        normalized_x = (coords[:, 0] + current_x_offset) / total_wsi_width * n_grid

        # Normalize y-coordinates based on the maximum WSI height
        normalized_y = coords[:, 1] / max_wsi_height * n_grid

        # Stack normalized x and y coordinates
        converted_coords = np.stack([normalized_x, normalized_y], axis=-1)

        return np.array(converted_coords, dtype=np.float32)

    def encode_slides(
        self,
        output_dir: Path,
        feat_dir: Path,
        device: DeviceLikeType,
        **kwargs,
    ) -> None:
        output_name = (
            f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        )
        output_file = os.path.join(output_dir, output_name)

        slide_dict = {}
        self.model.to(device).half().eval()

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        for tile_feats_filename in tqdm(os.listdir(feat_dir), desc="Processing slides"):
            h5_path = os.path.join(feat_dir, tile_feats_filename)
            slide_name: str = Path(tile_feats_filename).stem

            try:
                feats, coords = self._validate_and_read_features(h5_path)
            except FileNotFoundError as e:
                tqdm.write(s=str(e))
                continue

            # Calculated obtaining the tile with rightmost x coord
            # and the tile width is added as the coord is from top left
            slide_width = max(coords.coords_um[:, 0]) + coords.tile_size_um
            slide_height = max(coords.coords_um[:, 1]) + coords.tile_size_um

            # For some reason gigapaths requires normalized coords in a
            # [0,1000] grid
            n_grid = 1000

            # TODO: Check np.stack behaviour
            norm_coords = self._convert_coords(
                coords.coords_um, slide_width, slide_height, n_grid, current_x_offset=0
            )

            norm_coords = (
                torch.tensor(norm_coords, dtype=torch.float32)
                .unsqueeze(1)
                .to(device)
                .half()
            )
            feats = feats.unsqueeze(1).half().to(device)

            with torch.inference_mode():
                slide_embedding = self.model(feats, norm_coords)

            if isinstance(slide_embedding, list):  # Ensure slide_feats is not a list
                slide_embedding = torch.cat(slide_embedding, dim=0)

            slide_embedding = slide_embedding.detach().squeeze().cpu().numpy()

            slide_dict[slide_name] = {
                "feats": slide_embedding,
            }

        # TODO: Reutilice this function
        # TODO: Add codebase hash to h5 file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for slide_name, data in slide_dict.items():
                f.create_dataset(f"{slide_name}", data=data["feats"])
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = self.identifier
                f.attrs["precision"] = str(torch.float16)
            # Check if the file is empty
            if len(f) == 0:
                tqdm.write("Extraction failed: file empty")
                os.remove(output_file)
            tqdm.write(f"Finished encoding, saved to {output_file}")

    def encode_patients(
        self, output_dir, feat_dir, slide_table_path, device, **kwargs
    ) -> None:
        pass
