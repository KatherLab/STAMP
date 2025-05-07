import os
from pathlib import Path

import pandas as pd
import torch
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm
from transformers import AutoModel

from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder


class Prism(Encoder):
    def __init__(self) -> None:
        model = AutoModel.from_pretrained("paige-ai/Prism", trust_remote_code=True)
        super().__init__(model=model, identifier="paigeai-prism")

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
                feats, _ = self._validate_and_read_features(
                    h5_path, "virchow2", torch.float32
                )
            except FileNotFoundError as e:
                tqdm.write(s=str(e))
                continue

            with torch.autocast(str(device), torch.float16), torch.inference_mode():
                slide_embedding = (
                    self.model.slide_representations(feats.to(device))[
                        "image_embedding"
                    ]
                    .detach()
                    .squeeze()
                    .cpu()
                    .numpy()
                )
                slide_dict[slide_name] = {
                    "feats": slide_embedding,
                }

        self._save_features(
            output_file=output_file, entry_dict=slide_dict, precision=torch.float32
        )

    def encode_patients(
        self, output_dir, feat_dir, slide_table_path, device, **kwargs
    ) -> None:
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
            feats_list = []

            # Concatenate all slides over x axis adding the offset to each feature x coordinate.
            for _, row in group.iterrows():
                slide_filename = row["FILENAME"]
                h5_path = os.path.join(feat_dir, slide_filename)

                try:
                    feats, _ = self._validate_and_read_features(
                        h5_path=h5_path,
                        extractor_name="virchow2",
                        precision=torch.float32,
                    )
                except FileNotFoundError as e:
                    tqdm.write(s=str(e))
                    continue

                feats_list.append(feats)

            if not feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            all_feats = torch.cat(feats_list, dim=0).to(device)

            patient_embedding = (
                self.model.slide_representations(all_feats.to(device))[
                    "image_embedding"
                ]
                .detach()
                .squeeze()
                .cpu()
                .numpy()
            )
            patient_dict[patient_id] = {
                "feats": patient_embedding,
            }

        self._save_features(
            output_file=output_file, entry_dict=patient_dict, precision=torch.float32
        )
