import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.encoder import Encoder
from stamp.encoding.encoder.chief import CHIEF
from stamp.modeling.data import CoordsInfo, get_coords

"""From https://github.com/KatherLab/EAGLE/blob/main/eagle/main_feature_extraction.py"""


class Eagle(Encoder):
    def __init__(self) -> None:
        model = CHIEF().model
        super().__init__(model=model, identifier="katherlab-eagle")

    def _read_h5(self, h5_path: str) -> tuple[Tensor, CoordsInfo, str]:
        if not os.path.exists(h5_path) or not h5_path.endswith(".h5"):
            raise FileNotFoundError("File does not exist or is not an h5 file")
        with h5py.File(h5_path, "r") as f:
            feats: Tensor = torch.tensor(f["feats"][:], dtype=torch.float32)  # type: ignore
            coords: CoordsInfo = get_coords(f)
            extractor: str = f.attrs.get("extractor", "no extractor name")
            return feats, coords, extractor

    def _validate_and_read_features(
        self, h5_ctp: str, h5_vir2: str, slide_name: str
    ) -> tuple[Tensor, Tensor]:
        feats, coords, extractor = self._read_h5(h5_ctp)

        if "ctranspath" not in extractor:
            raise ValueError(
                f"Features must be extracted with ctranspath or chief-ctranspath. "
                f"Features located in {h5_ctp} are extracted with {extractor}"
            )

        agg_feats, agg_coords, extractor = self._read_h5(h5_vir2)

        if "virchow2" not in extractor:
            raise ValueError(
                f"Aggregated features must be extracted with virchow2 "
                f"Features located in {h5_vir2} are extracted with {extractor}"
            )

        if feats.shape[0] != agg_feats.shape[0]:
            raise ValueError(
                f"Number of ctranspath features and virchow2 features do not match:"
                f" {feats.shape[0]} != {agg_feats.shape[0]}"
            )

        if not torch.allclose(
            coords.coords_um, agg_coords.coords_um, atol=1e-5, rtol=0
        ):
            raise ValueError(
                f"Coordinates mismatch between ctranspath and virchow2"
                f" features for slide {slide_name}. Ensure that both are aligned."
            )

        return feats, agg_feats

    def _create_eagle_embedding(
        self,
        feats: Tensor,
        agg_feats: Tensor,
        device: DeviceLikeType,
    ) -> np.ndarray:
        """Process features, compute attention, and create an embedding."""
        with torch.no_grad():
            result = self.model(feats.to(device))
            attention_raw = result["attention_raw"].squeeze(0).cpu()

        # Get the 25 most relevant features
        k = min(25, attention_raw.shape[0])
        _, topk_indices = torch.topk(attention_raw, k)
        top_indices = topk_indices.numpy()

        # Get top virchow2 features using the top indices from ctranspath features
        top_agg_feats = [agg_feats[i] for i in top_indices]
        top_agg_feats = torch.stack(top_agg_feats).to(device)

        # Create eagle embedding by averaging the top virchow2 features
        eagle_embedding = torch.mean(top_agg_feats, dim=0)

        return eagle_embedding.to(torch.float32).detach().squeeze().cpu().numpy()

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
        patient_dict = {}
        self.model.to(device).eval()

        output_name = (
            f"{self.identifier}-pat-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        )
        output_file: str = os.path.join(output_dir, output_name)

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        for patient_id, group in tqdm(patient_groups, leave=False):
            feats_list = []
            agg_feats_list = []

            for _, row in group.iterrows():
                slide_filename = row["FILENAME"]
                slide_name = Path(slide_filename).stem
                h5_ctp = os.path.join(feat_dir, slide_filename)
                h5_vir2 = os.path.join(agg_feat_dir, slide_filename)

                # Validate and read features
                try:
                    feats, agg_feats = self._validate_and_read_features(
                        h5_ctp, h5_vir2, slide_name
                    )
                except FileNotFoundError as e:
                    tqdm.write(str(e))
                    continue

                feats_list.append(feats)
                agg_feats_list.append(agg_feats)

            if not feats_list:
                tqdm.write(f"No ctranspath features for patient {patient_id}")
                continue
            all_feats = torch.cat(feats_list, dim=0).to(device)
            all_agg_feats = torch.cat(agg_feats_list, dim=0).to(device)

            eagle_embedding = self._create_eagle_embedding(
                all_feats, all_agg_feats, device
            )
            patient_dict[patient_id] = {
                "feats": eagle_embedding,
            }

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for patient_id, data in patient_dict.items():
                f.create_dataset(f"{patient_id}", data=data["feats"])
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = self.identifier
                f.attrs["precision"] = str(torch.float32)
            if len(f) == 0:
                tqdm.write("Encoding failed: file empty")
                os.remove(output_file)
                return
            tqdm.write(f"Finished encoding, saved to {output_file}")

    def encode_slides(
        self,
        output_dir: Path,
        feat_dir: Path,
        device: DeviceLikeType,
        **kwargs,
    ) -> None:
        """Encode slide from patch features."""
        agg_feat_dir: Path | None = kwargs.get("agg_feat_dir")
        if not agg_feat_dir:
            raise ValueError(
                "agg_feat_dir that contains virchow2 features"
                " is required for Eagle's encode_patients"
            )

        output_name = (
            f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        )
        output_file = os.path.join(output_dir, output_name)
        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        slide_dict = {}
        self.model.to(device).eval()

        for tile_feats_filename in tqdm(os.listdir(feat_dir), desc="Processing slides"):
            h5_ctp = os.path.join(feat_dir, tile_feats_filename)
            h5_vir2 = os.path.join(agg_feat_dir, tile_feats_filename)
            slide_name: str = Path(tile_feats_filename).stem

            try:
                feats, agg_feats = self._validate_and_read_features(
                    h5_ctp, h5_vir2, slide_name
                )
            except FileNotFoundError as e:
                tqdm.write(s=str(e))
                continue

            eagle_embedding = self._create_eagle_embedding(feats, agg_feats, device)
            slide_dict[slide_name] = {
                "feats": eagle_embedding,
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
                tqdm.write("Encoding failed: file empty")
                os.remove(output_file)
                return
            tqdm.write(f"Finished Encoding, saved to {output_file}")
