import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm
from transformers import AutoModel

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo, PandasLabel
from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.tiling import Microns, SlideMPP


class Titan(Encoder):
    def __init__(self) -> None:
        model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        super().__init__(
            model=model,
            identifier=EncoderName.TITAN,
            precision=torch.float32,
            required_extractor=[ExtractorName.CONCH1_5],
        )

    def _generate_slide_embedding(
        self,
        feats: Tensor,
        device: DeviceLikeType,
        coords: CoordsInfo | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Helper method to encode a single slide."""
        if coords is None:
            raise ValueError("Coords must be provided.")

        coords_tensor = torch.tensor(coords.coords_um, dtype=self.precision)

        # Convert coordinates from microns to pixels
        patch_size_lvl0 = math.floor(256 / coords.mpp)  # Inferred from TITAN docs
        coords_px = coords_tensor / coords.mpp  # Convert to pixels
        coords_px = coords_px.to(torch.int64).to(device)  # Convert to integer

        feats = feats.to(device=device)

        with torch.inference_mode():
            slide_embedding = self.model.encode_slide_from_patch_features(
                feats, coords_px, patch_size_lvl0
            )
            return slide_embedding.detach().squeeze().cpu().numpy()

    def _generate_patient_embedding(
        self,
        feats_list: list,
        device: DeviceLikeType,
        coords_list: list[CoordsInfo] | None = None,
        **kwargs,
    ) -> np.ndarray:
        if coords_list is None:
            raise ValueError("coords_list must be provided.")

        # Concatenate all feature to a single slide tensor
        all_feats_cat = torch.cat(feats_list, dim=0).unsqueeze(0)

        # Create a single CoordsInfo item for the virtual slide
        # Already validated that mpp values are all equal within patient slides
        tile_size_um: Microns = coords_list[0].tile_size_um
        tile_size_px = coords_list[0].tile_size_px
        # Combine all slide coords to a single virtual slide set of coordinates
        coords_um = np.concatenate([coord.coords_um for coord in coords_list], axis=0)
        # Create virtual slide's Coords Info object
        coords = CoordsInfo(coords_um, tile_size_um, tile_size_px)

        return self._generate_slide_embedding(all_feats_cat, device, coords)

    def encode_patients(
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

        output_file = self._generate_output_path(
            output_dir=output_dir, generate_hash=generate_hash
        )

        patient_dict = {}
        self.model.to(device).eval()

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        for patient_id, group in tqdm(patient_groups, leave=False):
            all_feats_list = []
            all_coords_list = []
            current_x_offset = 0
            slides_mpp = SlideMPP(-1)

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

                # Add the offset to tile coordinates in x axis
                for coord in coords.coords_um:
                    coord[0] += current_x_offset

                # get the coordinates of the rightmost tile and add the
                # tile width as these coordinates are from the top-left
                # point. With that you get the total width of the slide.
                current_x_offset = max(coords.coords_um[:, 0]) + coords.tile_size_um

                # Add tile feats and coords to the patient virtual slide
                all_feats_list.append(feats)
                all_coords_list.append(coords)

            if not all_feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            patient_embedding = self._generate_patient_embedding(
                all_feats_list, device, all_coords_list
            )
            patient_dict[patient_id] = {
                "feats": patient_embedding,
            }

        self._save_features(output_file, entry_dict=patient_dict)
