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
        # TODO: load chief model
        pass

    def encode_patients(
        self, output_dir, feat_dir, agg_feat_dir, slide_table_path, device
    ) -> None:
        slide_table = pd.read_csv(slide_table_path)
        patient_groups = slide_table.groupby("PATIENT")
        slide_dict = {}

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
            coords_list = []
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
                        coords = torch.tensor(f["coords"][:], dtype=torch.int)
                except Exception as e:
                    print(f"Error reading {h5_ctp}: {e}")
                    continue

                # Read virchow2 features used for aggregation
                # Coords are not used as they are the same as ctranspath
                # TODO: Ask if this is correct
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
                coords_list.append(coords)
                slide_ids_list.extend([slide_id] * feats.shape[0])
            if not feats_list:
                print(f"No ctranspath features for patient {patient_id}")
                return
            all_feats = torch.cat(feats_list, dim=0).to(device)
            all_coords = torch.cat(coords_list, dim=0)
            slide_ids_arr = np.array(slide_ids_list)
            with torch.no_grad():
                result = CHIEF.model(all_feats)
                attention_raw = result["attention_raw"].squeeze(0).cpu()
            k = min(25, attention_raw.shape[0])
            topk_values, topk_indices = torch.topk(attention_raw, k)
            top_indices = topk_indices.numpy()
            top_slide_ids = slide_ids_arr[top_indices]
            # Do the indices from ctranspath work for virchow2? I suppose so
