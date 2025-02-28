from pathlib import Path
import torch
import h5py
import os
import pandas as pd
from tqdm import tqdm
from torch._prims_common import DeviceLikeType
from stamp.cache import get_processing_code_hash
from stamp.slide_encoding.config import EncoderName


def get_pat_embs(
    encoder: EncoderName,
    output_dir: Path,
    feat_dir: Path,
    slide_table: Path,
    device: DeviceLikeType,
    dtype: torch.dtype,
):
    match encoder:
        case EncoderName.COBRA:
            from stamp.slide_encoding.encoder.cobra import cobra

            encoder = cobra()
        
        # TODO: Add other encoders

    slide_table = pd.read_csv(slide_table)
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
                feats = f["feats"][:]
  

            feats = torch.tensor(feats).to(device)
            all_feats_list.append(feats)

        if all_feats_list:
            # Concatenate all features for this patient along the second dimension
            all_feats_cat = torch.cat(all_feats_list, dim=0).unsqueeze(0)

            with torch.inference_mode():
                assert all_feats_cat.ndim == 3, (
                    f"Expected 3D tensor, got {all_feats_cat.ndim}"
                )
                slide_feats = encoder.model(all_feats_cat.to(dtype))
                slide_dict[patient_id] = {
                    "feats": slide_feats.to(torch.float32)
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy(),
                    "encoder": encoder.identifier,
                }
        else:
            tqdm.write(f"No features found for patient {patient_id}, skipping")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with h5py.File(output_file, "w") as f:
        for patient_id, data in slide_dict.items():
            f.create_dataset(f"{patient_id}", data=data["feats"])
            f.attrs["encoder"] = data["encoder"]
        tqdm.write(f"Finished encoding, saved to {output_file}")
