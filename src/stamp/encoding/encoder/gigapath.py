import math
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from gigapath import slide_encoder
from torch import Tensor, dtype
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo, get_coords
from stamp.preprocessing.tiling import SlideMPP


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
    def _read_h5(self, h5_path: str, dtype: dtype) -> tuple[Tensor, CoordsInfo, str]:
        if not os.path.exists(h5_path) or not h5_path.endswith(".h5"):
            raise FileNotFoundError("File does not exist or is not an h5 file")
        with h5py.File(h5_path, "r") as f:
            feats: Tensor = torch.tensor(f["feats"][:], dtype=dtype)  # type: ignore
            coords: CoordsInfo = get_coords(f)
            extractor: str = f.attrs.get("extractor", "no extractor name")
            return feats, coords, extractor

    # TODO: This can be reused too, give extractor name as parameter
    def _validate_and_read_features(
        self, h5_path, dtype: dtype
    ) -> tuple[Tensor, CoordsInfo]:
        feats, coords, extractor = self._read_h5(h5_path, dtype=dtype)
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
                feats, coords = self._validate_and_read_features(h5_path, torch.float16)
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
        """Generate one virtual slide concatenating all the slides of a
        patient over the x axis."""
        output_name = (
            f"{self.identifier}-pat-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        )
        slide_table = pd.read_csv(slide_table_path)
        patient_groups = slide_table.groupby("PATIENT")

        output_file = os.path.join(output_dir, output_name)

        patient_dict = {}
        self.model.to(device).half().eval()

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        for patient_id, group in tqdm(patient_groups, leave=False):
            all_feats_list = []
            all_coords_list = []
            total_wsi_width = 0
            max_wsi_height = 0
            slides_mpp = SlideMPP(-1)

            slide_info = []
            # Concatenate all slides over x axis adding the offset to each feature x coordinate.
            for _, row in group.iterrows():
                slide_filename = row["FILENAME"]
                h5_path = os.path.join(feat_dir, slide_filename)

                try:
                    feats, coords = self._validate_and_read_features(
                        h5_path, torch.float16
                    )
                except FileNotFoundError as e:
                    tqdm.write(s=str(e))
                    continue

                # Get the mpp of one slide and check that the rest have the same
                if slides_mpp < 0:
                    slides_mpp = coords.mpp
                elif not math.isclose(slides_mpp, coords.mpp, rel_tol=1e-5):
                    raise ValueError(
                        "All patient slides must have the same mpp value. "
                        "Try reprocessing the slides using the same tile_size_um and "
                        "tile_size_px values for all of them."
                    )

                wsi_width = max(coords.coords_um[:, 0]) + coords.tile_size_um
                wsi_height = max(coords.coords_um[:, 1]) + coords.tile_size_um

                total_wsi_width += wsi_width  # Sum the widths of all slides
                max_wsi_height = max(max_wsi_height, wsi_height)  # Track the max height

                slide_info.append((wsi_width, wsi_height, feats, coords))

            current_x_offset = 0

            for wsi_width, wsi_height, feats, coords in slide_info:
                all_feats = (
                    torch.tensor(feats, dtype=torch.float16)
                    .unsqueeze(1)
                    .to(device)
                    .half()
                    .detach()
                )

                norm_coords = self._convert_coords(
                    coords=coords.coords_um,
                    total_wsi_width=total_wsi_width,
                    max_wsi_height=max_wsi_height,
                    n_grid=1000,
                    current_x_offset=current_x_offset,
                )

                # Update x-coordinates by shifting them based on the current_x_offset
                current_x_offset += (
                    wsi_width  # Move the x_offset forward for the next slide
                )

                norm_coords = (
                    torch.tensor(norm_coords, dtype=torch.float32)
                    .unsqueeze(1)
                    .to(device)
                    .half()
                )

                all_feats_list.append(all_feats)
                all_coords_list.append(norm_coords)

            if not all_feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            all_feats_cat = torch.cat(all_feats_list, dim=0).unsqueeze(0)
            all_coords_cat = torch.cat(all_coords_list, dim=0).unsqueeze(0)

            breakpoint()

            with torch.inference_mode():
                patient_embedding = self.model(all_feats_cat, all_coords_cat)

            if isinstance(patient_embedding, list):  # Ensure slide_feats is not a list
                patient_embedding = torch.cat(patient_embedding, dim=0)

            patient_embedding = patient_embedding.detach().squeeze().cpu().numpy()

            patient_dict[patient_id] = {
                "feats": patient_embedding,
            }
