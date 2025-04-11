import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder
from stamp.encoding.encoder.chief import CHIEF

"""From https://github.com/KatherLab/EAGLE/blob/main/eagle/main_feature_extraction.py"""


class Eagle(Encoder):
    def __init__(self) -> None:
        model = CHIEF().model
        super().__init__(model=model, identifier="katherlab-eagle")

    def encode_patients(
        self, output_dir, feat_dir, slide_table_path, device, **kwargs
    ) -> None:
        """Encode patients from slide features."""
        agg_feat_dir: Path | None = kwargs.get("agg_feat_dir")
        if not agg_feat_dir:
            raise ValueError("agg_feat_dir is required for Eagle's encode_patients")

        slide_table = pd.read_csv(slide_table_path)
        patient_groups = slide_table.groupby("PATIENT")
        slide_dict = {}
        self.model.to(device).eval()

        output_name = (
            f"{self.identifier}-pat-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        )
        output_file = os.path.join(output_dir, output_name)

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        for patient_id, group in tqdm(patient_groups, leave=False):
            feats_list = []
            agg_feats_list = []
            slide_ids_list = []

            for _, row in group.iterrows():
                slide_filename = row["FILENAME"]
                slide_id = os.path.splitext(slide_filename)[0]
                h5_ctp = os.path.join(feat_dir, slide_filename)
                h5_vir2 = os.path.join(agg_feat_dir, slide_filename)

                # Read ctranspath features
                if not os.path.exists(h5_ctp):
                    print(f"File {h5_ctp} does not exist, skipping")
                    continue
                try:
                    with h5py.File(h5_ctp, "r") as f:
                        feats = torch.tensor(f["feats"][:], dtype=torch.float32)
                except Exception as e:
                    print(f"Error reading {h5_ctp}: {e}")
                    continue

                # Read virchow2 features used for aggregation
                if not os.path.exists(h5_vir2):
                    print(f"File {h5_vir2} does not exist, skipping")
                    continue
                try:
                    with h5py.File(h5_vir2, "r") as f:
                        agg_feats = torch.tensor(f["feats"][:], dtype=torch.float32)
                except Exception as e:
                    print(f"Error reading {h5_vir2}: {e}")
                    continue
                # TODO: Check that the features have ctranspath id
                # TODO: Check that agg features have virchow2 id
                # TODO: Check that the features have the same number of rows

                feats_list.append(feats)
                agg_feats_list.append(agg_feats)
                slide_ids_list.extend([slide_id] * feats.shape[0])
            if not feats_list:
                print(f"No ctranspath features for patient {patient_id}")
                return
            all_feats = torch.cat(feats_list, dim=0).to(device)
            all_agg_feats = torch.cat(agg_feats_list, dim=0).to(device)
            with torch.no_grad():
                result = self.model(all_feats)
                attention_raw = result["attention_raw"].squeeze(0).cpu()
            k = min(25, attention_raw.shape[0])
            _, topk_indices = torch.topk(attention_raw, k)
            top_indices = topk_indices.numpy()

            # Get top virchow2 features using the top indices from
            # ctranspath features
            top_agg_feats = []
            for i in top_indices:
                top_agg_feats.append(all_agg_feats[i])
            top_agg_feats = torch.stack(top_agg_feats).to(device)
            eagle_embedding = torch.mean(top_agg_feats, dim=0)
            slide_dict[patient_id] = {
                "feats": eagle_embedding.to(torch.float32)
                .detach()
                .squeeze()
                .cpu()
                .numpy(),
                "encoder": self.identifier,
                "precision": torch.float32,
            }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for patient_id, data in slide_dict.items():
                f.create_dataset(f"{patient_id}", data=data["feats"])
                f.attrs["encoder"] = data["encoder"]
            tqdm.write(f"Finished encoding, saved to {output_file}")

    def encode_slides(
        self, output_dir, feat_dir, patch_size_lvl0, *args, **kwargs
    ) -> None:
        """Encode slide from patch features."""
        pass
