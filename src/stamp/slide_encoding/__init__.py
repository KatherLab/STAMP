import os
import pdb
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.slide_encoding.config import EncoderName
from stamp.slide_encoding.encoder import Encoder


def get_pat_embs(
    encoder_name: EncoderName,
    output_dir: Path,
    feat_dir: Path,
    slide_table_path: Path,
    device: DeviceLikeType,
) -> None:
    """"""
    match encoder_name:
        case EncoderName.COBRA:
            from stamp.slide_encoding.encoder.cobra import cobra

            encoder: Encoder = cobra()
        case EncoderName.TITAN:
            from stamp.slide_encoding.encoder.titan import titan

            encoder: Encoder = titan()
        case EncoderName.CHIEF:
            from stamp.slide_encoding.encoder.chief import chief

            encoder: Encoder = chief()

        # TODO: Add other encoders

    dtype = torch.float32
    model = encoder.model.to(device).eval()
    # TODO: dtype depends con CUDA capabilities. Check end of
    # extract_feat_patients.py on how to handle this

    slide_table = pd.read_csv(slide_table_path)
    patient_groups = slide_table.groupby("PATIENT")
    slide_dict = {}

    output_name = f"{encoder.identifier}-{get_processing_code_hash(Path(__file__))[:8]}"
    output_file = os.path.join(output_dir, output_name)

    if os.path.exists(output_file):
        tqdm.write(f"Output file {output_file} already exists, skipping")
        return

    for patient_id, group in tqdm(patient_groups, leave=False):
        all_feats_list = []

        for _, row in group.iterrows():
            slide_filename = row["FILENAME"]
            h5_path = os.path.join(feat_dir, slide_filename)
            if not os.path.exists(h5_path):
                tqdm.write(f"File {h5_path} does not exist, skipping")
                continue
            with h5py.File(h5_path, "r") as f:
                feats = f["feats"][:]  # type: ignore

            feats = torch.tensor(feats).to(device)
            all_feats_list.append(feats)

        if all_feats_list:
            # Concatenate all features for this patient along the second dimension
            all_feats_cat = torch.cat(all_feats_list, dim=0).unsqueeze(0)

            with torch.inference_mode():
                assert all_feats_cat.ndim == 3, (
                    f"Expected 3D tensor, got {all_feats_cat.ndim}"
                )
                slide_feats = model(all_feats_cat.to(dtype))
                slide_dict[patient_id] = {
                    "feats": slide_feats.to(torch.float32)
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy(),
                    "encoder": encoder.identifier,
                    "precision": dtype,
                }
        else:
            tqdm.write(f"No features found for patient {patient_id}, skipping")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for patient_id, data in slide_dict.items():
            f.create_dataset(f"{patient_id}", data=data["feats"])
            f.attrs["encoder"] = data["encoder"]
        tqdm.write(f"Finished encoding, saved to {output_file}")


def get_slide_embs(
    encoder_name: EncoderName,
    output_dir: Path,
    feat_dir: Path,
    slide_table_path: Path,
    device: DeviceLikeType,
) -> None:
    match encoder_name:
        case EncoderName.TITAN:
            from stamp.slide_encoding.encoder.titan import titan

            encoder: Encoder = titan()

    slide_dict: dict[str, dict[str, torch.Tensor]] = {}

    # Ensure model weights and biases are on the same device as the input
    model = encoder.model.to(device).eval()

    output_name = f"{encoder.identifier}-{get_processing_code_hash(Path(__file__))[:8]}"
    output_file = os.path.join(output_dir, f"{output_name}.h5")

    if os.path.exists(output_file):
        tqdm.write(f"Output file {output_file} already exists, skipping")
        return

    with h5py.File(output_file, "w") as h5_file:
        h5_file.attrs["encoder"] = encoder.identifier
        h5_file.attrs["stamp_version"] = stamp.__version__

        for tile_feats_filename in tqdm(os.listdir(feat_dir), desc="Processing slides"):
            h5_path = os.path.join(feat_dir, tile_feats_filename)
            if not os.path.exists(h5_path) or not h5_path.endswith(".h5"):
                tqdm.write(
                    f"File {h5_path} does not exist or is not an h5 file, skipping"
                )
                continue

            with h5py.File(h5_path, "r") as f:
                feats = f["feats"][:]  # type: ignore
                coords = f["coords"][:]  # type: ignore

            # Convert coordinates from microns to pixels
            mpp = 1.14  # microns per pixel
            patch_size_lvl0 = 256 / mpp  # Inferred from TITAN docs
            coords = coords / mpp  # Convert to pixels
            coords = torch.tensor(coords, dtype=torch.float32).to(device)
            coords = coords.to(torch.int64).to(device)  # Convert to integer

            feats = torch.tensor(feats, dtype=torch.float32).to(device)

            with torch.inference_mode():
                slide_embedding = model.encode_slide_from_patch_features(
                    feats, coords, patch_size_lvl0
                )
                slide_embedding = (
                    slide_embedding.to(torch.float32).detach().squeeze().cpu().numpy()
                )

            slide_name = Path(tile_feats_filename).stem
            h5_file.create_dataset(slide_name, data=slide_embedding)

    tqdm.write(f"Finished encoding, saved all slide embeddings to {output_file}")
