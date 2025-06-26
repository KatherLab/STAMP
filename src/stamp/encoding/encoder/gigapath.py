import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gigapath import slide_encoder
from tqdm import tqdm

from stamp.cache import get_processing_code_hash
from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo
from stamp.preprocessing.config import ExtractorName
from stamp.types import PandasLabel, SlideMPP

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"
__credits__ = [
    "Xu, et al. (https://github.com/prov-gigapath/prov-gigapath)",
    "Peter Neidlinger",
]

_logger = logging.getLogger("stamp")


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
        super().__init__(
            model=model,
            identifier=EncoderName.GIGAPATH,
            precision=torch.float16,
            required_extractors=[ExtractorName.GIGAPATH],
        )

    def _generate_slide_embedding(
        self, feats, device, coords: CoordsInfo | None = None, **kwargs
    ) -> np.ndarray:
        if not coords:
            raise ValueError("Tile coords are required for encoding")

        # Calculate slide dimensions
        slide_width = max(coords.coords_um[:, 0]) + coords.tile_size_um
        slide_height = max(coords.coords_um[:, 1]) + coords.tile_size_um

        # Normalize coordinates to a [0, 1000] grid
        n_grid = 1000
        norm_coords = self._convert_coords(
            coords.coords_um, slide_width, slide_height, n_grid, current_x_offset=0
        )
        norm_coords = (
            torch.tensor(norm_coords, dtype=torch.float32)
            .unsqueeze(0)
            .to(device)
            .half()
        )
        feats = feats.unsqueeze(0).half().to(device)

        self.model.to(self.precision)

        with torch.inference_mode():
            slide_embedding = self.model(feats, norm_coords)

        if isinstance(slide_embedding, list):  # Ensure slide_embedding is not a list
            slide_embedding = torch.cat(slide_embedding, dim=0)

        return slide_embedding.detach().squeeze().cpu().numpy()

    def encode_patients_(
        self,
        output_dir: Path,
        feat_dir: Path,
        slide_table_path: Path,
        patient_label: PandasLabel,
        filename_label: PandasLabel,
        device,
        generate_hash: bool,
        **kwargs,
    ) -> None:
        """Generate one virtual slide concatenating all the slides of a
        patient over the x axis."""
        slide_table = pd.read_csv(slide_table_path)
        patient_groups = slide_table.groupby(patient_label)

        # generate the name for the folder containing the feats
        if generate_hash:
            encode_dir = (
                f"{self.identifier}-pat-{get_processing_code_hash(Path(__file__))[:8]}"
            )
        else:
            encode_dir = f"{self.identifier}-pat"
        encode_dir = output_dir / encode_dir
        os.makedirs(encode_dir, exist_ok=True)

        self.model.to(device).half().eval()

        for patient_id, group in (progress := tqdm(patient_groups)):
            progress.set_description(str(patient_id))

            # skip patient in case feature file already exists
            output_path = (encode_dir / str(patient_id)).with_suffix(".h5")
            if output_path.exists():
                _logger.debug(
                    f"skipping {str(patient_id)} because {output_path} already exists"
                )
                continue

            all_feats_list = []
            all_coords_list = []
            total_wsi_width = 0
            max_wsi_height = 0
            slides_mpp = SlideMPP(-1)

            slide_info = []
            # Concatenate all slides over x axis adding the offset to each feature x coordinate.
            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                h5_path = os.path.join(feat_dir, slide_filename)

                feats, coords = self._validate_and_read_features(h5_path=h5_path)

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
                    .unsqueeze(0)
                    .to(device)
                    .half()
                )

                all_feats_list.append(feats.unsqueeze(dim=0).to(device))
                all_coords_list.append(norm_coords)

            if not all_feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            patient_embedding = self._generate_patient_embedding(
                all_feats_list, device, coords_list=all_coords_list
            )

            self._save_features_(output_path=output_path, feats=patient_embedding)

    def _generate_patient_embedding(
        self, feats_list, device, coords_list: list | None = None, **kwargs
    ) -> np.ndarray:
        if not coords_list:
            raise ValueError("Tile coords are required for encoding")
        all_feats_cat = torch.cat(feats_list, dim=1)
        all_coords_cat = torch.cat(coords_list, dim=1)

        with torch.inference_mode():
            patient_embedding = self.model(all_feats_cat, all_coords_cat)

        if isinstance(patient_embedding, list):  # Ensure slide_feats is not a list
            patient_embedding = torch.cat(patient_embedding, dim=0)

        return patient_embedding.detach().squeeze().cpu().numpy()

    def _convert_coords(
        self,
        coords,
        total_wsi_width,
        max_wsi_height,
        n_grid,
        current_x_offset,
    ) -> np.ndarray:
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
