import math
import os
from pathlib import Path

import h5py
import torch
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

    # TODO: Check precision

    def encode_slides(self, output_dir, feat_dir, device, **kwargs) -> None:
        # Ensure model weights and biases are on the same device as the input
        self.model.to(device).eval()

        output_name = (
            f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        )
        output_file = os.path.join(output_dir, output_name)

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        with h5py.File(output_file, "w") as h5_file:
            h5_file.attrs["encoder"] = self.identifier
            h5_file.attrs["stamp_version"] = stamp.__version__

            for tile_feats_filename in tqdm(
                os.listdir(feat_dir), desc="Processing slides"
            ):
                h5_path = os.path.join(feat_dir, tile_feats_filename)
                if not os.path.exists(h5_path) or not h5_path.endswith(".h5"):
                    tqdm.write(
                        f"File {h5_path} does not exist or is not an h5 file, skipping"
                    )
                    continue

                with h5py.File(h5_path, "r") as f:
                    feats = f["feats"][:]  # type: ignore
                    coords: CoordsInfo = get_coords(f)  # type: ignore

                # Convert coordinates from microns to pixels
                patch_size_lvl0 = math.floor(
                    256 / coords.mpp
                )  # Inferred from TITAN docs
                coords_px = coords.coords_um / coords.mpp  # Convert to pixels
                coords_px = torch.tensor(coords_px, dtype=torch.float32).to(device)
                coords_px = coords_px.to(torch.int64).to(device)  # Convert to integer

                feats = torch.tensor(feats, dtype=torch.float32).to(device)

                with torch.inference_mode():
                    slide_embedding = self.model.encode_slide_from_patch_features(
                        feats, coords_px, patch_size_lvl0
                    )
                    slide_embedding = (
                        slide_embedding.to(torch.float32)
                        .detach()
                        .squeeze()
                        .cpu()
                        .numpy()
                    )

                slide_name = Path(tile_feats_filename).stem
                h5_file.create_dataset(slide_name, data=slide_embedding)

        tqdm.write(f"Finished encoding, saved all slide embeddings to {output_file}")

    def encode_patients(self, output_dir, feat_dir, slide_table_path, device):
        return NotImplementedError("Not implemented yet :P but soon!")
