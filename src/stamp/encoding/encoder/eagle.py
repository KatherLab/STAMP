import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm

from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.encoding.encoder.chief import CHIEF
from stamp.modeling.data import CoordsInfo, PandasLabel
from stamp.preprocessing.config import ExtractorName

"""From https://github.com/KatherLab/EAGLE/blob/main/eagle/main_feature_extraction.py"""


class Eagle(Encoder):
    def __init__(self) -> None:
        self.required_agg_extractor = ExtractorName.VIRCHOW2
        super().__init__(
            model=CHIEF().model,
            identifier=EncoderName.EAGLE,
            precision=torch.float32,
            required_extractor=[
                ExtractorName.CTRANSPATH,
                ExtractorName.CHIEF_CTRANSPATH,
            ],
        )

    def _validate_and_read_features_with_agg(
        self, h5_ctp: str, h5_vir2: str, slide_name: str
    ) -> tuple[Tensor, Tensor]:
        feats: Tensor
        coords: CoordsInfo
        extractor: str
        feats, coords, extractor = self._read_h5(h5_ctp)

        # TODO: Eagle requires ctranspath features extracted with 2mpp
        # magnification. Validate this.

        if extractor not in self.required_extractor:
            raise ValueError(
                f"Features must be extracted with one of {self.required_extractor}. "
                f"Features located in {h5_ctp} are extracted with {extractor}"
            )

        agg_feats, agg_coords, extractor = self._read_h5(h5_vir2)

        if extractor != self.required_agg_extractor:
            raise ValueError(
                f"Aggregated features must be extracted with {self.required_agg_extractor} "
                f"Features located in {h5_vir2} are extracted with {extractor}"
            )

        if feats.shape[0] != agg_feats.shape[0]:
            raise ValueError(
                f"Number of ctranspath features and virchow2 features do not match:"
                f" {feats.shape[0]} != {agg_feats.shape[0]}"
            )

        if not np.allclose(coords.coords_um, agg_coords.coords_um, atol=1e-5, rtol=0):
            raise ValueError(
                f"Coordinates mismatch between ctranspath and virchow2"
                f" features for slide {slide_name}. Ensure that both are aligned."
            )

        return feats, agg_feats

    def _generate_slide_embedding(
        self,
        feats: Tensor,
        device: DeviceLikeType,
        agg_feats: Tensor | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Process features, compute attention, and create an embedding."""

        if agg_feats is None:
            raise ValueError("agg_feats is required for slide embedding")

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

    def _generate_patient_embedding(
        self,
        feats_list: list[Tensor],
        device: DeviceLikeType,
        agg_feats_list: list[Tensor] | None = None,
        **kwargs,
    ) -> np.ndarray:
        if agg_feats_list is None:
            raise ValueError("agg_feats_list is required for patient embedding")
        all_feats = torch.cat(feats_list, dim=0).to(device)
        all_agg_feats = torch.cat(agg_feats_list, dim=0).to(device)
        return self._generate_slide_embedding(all_feats, device, all_agg_feats)

    def encode_slides(
        self,
        output_dir: Path,
        feat_dir: Path,
        device: DeviceLikeType,
        generate_hash: bool,
        **kwargs,
    ) -> None:
        """Encode slide from patch features."""
        agg_feat_dir: Path | None = kwargs.get("agg_feat_dir")
        if not agg_feat_dir:
            raise ValueError(
                "agg_feat_dir that contains virchow2 features"
                " is required for Eagle's encode_patients"
            )

        output_file = self._generate_output_path(
            output_dir=output_dir, generate_hash=generate_hash
        )

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
                feats, agg_feats = self._validate_and_read_features_with_agg(
                    h5_ctp, h5_vir2, slide_name
                )
            except FileNotFoundError as e:
                tqdm.write(s=str(e))
                continue

            eagle_embedding = self._generate_slide_embedding(feats, device, agg_feats)

            slide_dict[slide_name] = {
                "feats": eagle_embedding,
            }

        self._save_features(output_file, entry_dict=slide_dict)

    def encode_patients(
        self,
        output_dir: Path,
        feat_dir: Path,
        slide_table_path: Path,
        patient_label: PandasLabel,
        filename_label: PandasLabel,
        device: DeviceLikeType,
        generate_hash: bool,
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
        patient_groups = slide_table.groupby(patient_label)
        patient_dict = {}
        self.model.to(device).eval()

        output_file = self._generate_output_path(
            output_dir=output_dir, generate_hash=generate_hash
        )

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        for patient_id, group in tqdm(patient_groups, leave=False):
            feats_list = []
            agg_feats_list = []

            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                slide_name = Path(slide_filename).stem
                h5_ctp = os.path.join(feat_dir, slide_filename)
                h5_vir2 = os.path.join(agg_feat_dir, slide_filename)

                # Validate and read features
                try:
                    feats, agg_feats = self._validate_and_read_features_with_agg(
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

            eagle_embedding = self._generate_patient_embedding(
                feats_list, device, agg_feats_list
            )
            patient_dict[patient_id] = {
                "feats": eagle_embedding,
            }

        self._save_features(output_file, entry_dict=patient_dict)
