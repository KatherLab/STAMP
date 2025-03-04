import os
from pathlib import Path

import h5py
import pandas as pd
import torch
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

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
    # TODO: add logger

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
