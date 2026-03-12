import logging
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from gigapath import slide_encoder
from tqdm import tqdm

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo
from stamp.preprocessing.config import ExtractorName
from stamp.types import PandasLabel, SlideMPP
from stamp.utils.cache import get_processing_code_hash

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
                "hf_hub:prov-gigapath/prov-gigapath",
                "gigapath_slide_enc12l768d",
                1536,
                global_pool=True,
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

        coords_px = coords.coords_um / coords.mpp
        norm_coords = (
            torch.tensor(coords_px, dtype=torch.float32).unsqueeze(0).to(device).half()
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
            slides_mpp = SlideMPP(-1)

            slide_info = []
            # Concatenate all slides over x axis adding the offset to each feature x coordinate.
            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                h5_path = os.path.join(feat_dir, slide_filename)

                # Skip if not an .h5 file
                if not h5_path.endswith(".h5"):
                    tqdm.write(f"Skipping {slide_filename} (not an .h5 file)")
                    continue

                try:
                    feats, coords = self._validate_and_read_features(h5_path=h5_path)
                except (FileNotFoundError, ValueError, OSError) as e:
                    tqdm.write(f"Skipping {slide_filename}: {e}")
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
                slide_info.append((wsi_width, feats, coords))

            current_x_offset = 0

            for wsi_width, feats, coords in slide_info:
                offset_coords_um = coords.coords_um.copy()
                offset_coords_um[:, 0] += current_x_offset

                current_x_offset += wsi_width

                coords_px = offset_coords_um / coords.mpp

                norm_coords = (
                    torch.tensor(coords_px, dtype=torch.float32)
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

            self._save_features_(
                output_path=output_path, feats=patient_embedding, feat_type="patient"
            )

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
