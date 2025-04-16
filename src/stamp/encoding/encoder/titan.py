import math
import os
from pathlib import Path

import h5py
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
        if "titan" not in extractor:
            raise ValueError(
                f"Features must be extracted with titan. "
                f"Features located in {h5_path} are extracted with {extractor}"
            )
        return feats, coords

    def _encode_slide(
        self, feats: Tensor, coords: CoordsInfo, device: DeviceLikeType
    ) -> torch.Tensor:
        """Helper method to encode a single slide."""
        # Convert coordinates from microns to pixels
        patch_size_lvl0 = math.floor(256 / coords.mpp)  # Inferred from TITAN docs
        coords_px = coords.coords_um / coords.mpp  # Convert to pixels
        coords_px = torch.tensor(coords_px, dtype=torch.float32).to(device)
        coords_px = coords_px.to(torch.int64).to(device)  # Convert to integer

        feats = torch.tensor(feats, dtype=torch.float32).to(device)

        with torch.inference_mode():
            slide_embedding = self.model.encode_slide_from_patch_features(
                feats, coords_px, patch_size_lvl0
            )
            return slide_embedding.to(torch.float32).detach().squeeze()

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

            slide_embedding = self._encode_slide(feats, coords, device)
            slide_dict[slide_name] = {
                "feats": slide_embedding,
            }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for slide_name, data in slide_dict.items():
                f.create_dataset(f"{slide_name}", data=data["feats"])
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = self.identifier
                f.attrs["precision"] = torch.float32
            # Check if the file is empty
            if len(f) == 0:
                tqdm.write("Extraction failed: file empty")
                os.remove(output_file)
                return
            tqdm.write(f"Finished encoding, saved to {output_file}")

    def encode_patients(self, output_dir, feat_dir, slide_table_path, device, **kwargs):
        output_name = (
            f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        )
        slide_table = pd.read_csv(slide_table_path)
        patient_groups = slide_table.groupby("PATIENT")

        output_file = os.path.join(output_dir, output_name)

        patient_dict = {}
        self.model.to(device).eval()

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        # Not implemented yet :P
        pass

