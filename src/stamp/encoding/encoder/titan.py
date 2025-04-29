import math
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm
from transformers import AutoModel

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder
from stamp.modeling.data import CoordsInfo, get_coords
from stamp.preprocessing.tiling import SlideMPP


class Titan(Encoder):
    def __init__(self) -> None:
        model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
        super().__init__(model=model, identifier="mahmood-titan")

    def _read_h5(self, h5_path: str) -> tuple[Tensor, CoordsInfo, str]:
        if not os.path.exists(h5_path) or not h5_path.endswith(".h5"):
            raise FileNotFoundError("File does not exist or is not an h5 file")
        with h5py.File(h5_path, "r") as f:
            feats: Tensor = torch.tensor(f["feats"][:], dtype=torch.float32)  # type: ignore
            coords: CoordsInfo = get_coords(f)
            extractor: str = f.attrs.get("extractor", "no extractor name")
            return feats, coords, extractor

    def _validate_and_read_features(self, h5_path) -> tuple[Tensor, CoordsInfo]:
        feats, coords, extractor = self._read_h5(h5_path)
        if "conch1_5" not in extractor:
            raise ValueError(
                f"Features must be extracted with conch1_5. "
                f"Features located in {h5_path} are extracted with {extractor}"
            )
        return feats, coords

    def _encode_slide(
        self, feats: Tensor, coords_um: Tensor, mpp: SlideMPP, device: DeviceLikeType
    ) -> np.ndarray:
        """Helper method to encode a single slide."""
        # Convert coordinates from microns to pixels
        patch_size_lvl0 = math.floor(256 / mpp)  # Inferred from TITAN docs
        coords_px = coords_um / mpp  # Convert to pixels
        coords_px = coords_px.to(torch.int64).to(device)  # Convert to integer

        feats = feats.to(device)

        with torch.inference_mode():
            slide_embedding = self.model.encode_slide_from_patch_features(
                feats, coords_px, patch_size_lvl0
            )
            return slide_embedding.to(torch.float32).detach().squeeze().cpu().numpy()

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
        self.model.to(device).eval()

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

            coords_tensor = torch.tensor(coords.coords_um, dtype=torch.float32)

            slide_embedding = self._encode_slide(
                feats, coords_tensor, coords.mpp, device
            )
            slide_dict[slide_name] = {
                "feats": slide_embedding,
            }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for slide_name, data in slide_dict.items():
                f.create_dataset(f"{slide_name}", data=data["feats"])
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = self.identifier
                f.attrs["precision"] = str(torch.float32)
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
                slide_filename = row["FILENAME"]
                h5_path = os.path.join(feat_dir, slide_filename)

                try:
                    feats, coords = self._validate_and_read_features(h5_path)
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

                # Add the offset to tile coordinates in x axis
                for coord in coords.coords_um:
                    coord[0] += current_x_offset

                # get the coordinates of the rightmost tile and add the
                # tile width as these coordinates are from the top-left
                # point. With that you get the total width of the slide.
                current_x_offset = max(coords.coords_um[:, 0]) + coords.tile_size_um

                coords_tensor = torch.tensor(coords.coords_um, dtype=torch.float32)

                # Add tile feats and coords to the patient virtual slide
                all_feats_list.append(feats)
                all_coords_list.append(coords_tensor)

            if not all_feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            all_feats_cat = torch.cat(all_feats_list, dim=0).unsqueeze(0)
            all_coords_cat = torch.cat(all_coords_list, dim=0).unsqueeze(0)

            patient_embedding = self._encode_slide(
                all_feats_cat, all_coords_cat, slides_mpp, device
            )
            patient_dict[patient_id] = {
                "feats": patient_embedding,
            }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for slide_name, data in patient_dict.items():
                f.create_dataset(f"{slide_name}", data=data["feats"])
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = self.identifier
                f.attrs["precision"] = str(torch.float32)
            # Check if the file is empty
            if len(f) == 0:
                tqdm.write("Extraction failed: file empty")
                os.remove(output_file)
            tqdm.write(f"Finished encoding, saved to {output_file}")