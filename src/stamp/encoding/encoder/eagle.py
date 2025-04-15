import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType
from tqdm import tqdm

from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder
from stamp.encoding.encoder.chief import CHIEF
from stamp.modeling.data import CoordsInfo, get_coords

"""From https://github.com/KatherLab/EAGLE/blob/main/eagle/main_feature_extraction.py"""


class Eagle(Encoder):
    def __init__(self) -> None:
        model = CHIEF().model
        super().__init__(model=model, identifier="katherlab-eagle")

    def encode_patients(
        self,
        output_dir: Path,
        feat_dir: Path,
        slide_table_path: Path,
        device: DeviceLikeType,
        **kwargs,
    ) -> None:
        """Encode patients from slide features."""
        agg_feat_dir: Path | None = kwargs.get("agg_feat_dir")
        if not agg_feat_dir:
            raise ValueError(
                "agg_feat_dir that contains virchow2 features"
                " is required for Eagle's encode_patients"
            )

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
                        feats: Tensor = torch.tensor(f["feats"][:], dtype=torch.float32)  # type: ignore
                        coords: CoordsInfo = get_coords(f)
                        extractor: str = f.attrs.get("extractor", "no extractor name")
                        # Check that extractor name contains ctranspath in it
                        if "ctranspath" not in extractor:
                            raise ValueError(
                                f"Features must be extracted with ctranspath or chief-ctranspath. "
                                f"Features located in {h5_ctp} are extracted with {extractor}"
                            )
                except Exception as e:
                    print(f"Error reading {h5_ctp}: {e}")
                    continue

                # Read virchow2 features used for aggregation
                if not os.path.exists(h5_vir2):
                    print(f"File {h5_vir2} does not exist, skipping")
                    continue
                try:
                    with h5py.File(h5_vir2, "r") as f:
                        agg_feats: Tensor = torch.tensor(
                            f["feats"][:], dtype=torch.float32
                        )  # type: ignore
                        agg_coords: CoordsInfo = get_coords(f)
                        extractor: str = f.attrs.get("extractor", "no extractor name")
                        # Check that extractor name contains virchow2 in it
                        if "virchow2" not in extractor:
                            raise ValueError(
                                f"Aggregated features must be extracted with virchow2 "
                                f"Features located in {h5_vir2} are extracted with {extractor}"
                            )
                except Exception as e:
                    print(f"Error reading {h5_vir2}: {e}")
                    continue

                # Check that both feature lists are paired by coords
                if not torch.allclose(
                    coords.coords_um, agg_coords.coords_um, atol=1e-5, rtol=0
                ):
                    raise ValueError(
                        f"Coordinates mismatch between ctranspath and virchow2"
                        f" features for slide {slide_id}. Ensure that both are aligned."
                    )

                feats_list.append(feats)
                agg_feats_list.append(agg_feats)

            if not feats_list:
                print(f"No ctranspath features for patient {patient_id}")
                return
            all_feats = torch.cat(feats_list, dim=0).to(device)
            all_agg_feats = torch.cat(agg_feats_list, dim=0).to(device)

            if all_feats.shape[0] != all_agg_feats.shape[0]:
                raise ValueError(
                    f"Number of ctranspath features and virchow2 features do not match:"
                    f" {all_feats.shape[0]} != {all_agg_feats.shape[0]}"
                )

            with torch.no_grad():
                result = self.model(all_feats)
                attention_raw = result["attention_raw"].squeeze(0).cpu()

            # Get the 25 most relevant features
            k = min(25, attention_raw.shape[0])
            _, topk_indices = torch.topk(attention_raw, k)
            top_indices = topk_indices.numpy()

            # Get top virchow2 features using the top indices from
            # ctranspath features
            top_agg_feats = []
            for i in top_indices:
                top_agg_feats.append(all_agg_feats[i])
            top_agg_feats = torch.stack(top_agg_feats).to(device)
            # Create eagle embedding by averaging the top virchow2 features
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
