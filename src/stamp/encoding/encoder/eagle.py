import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm

from stamp.cache import get_processing_code_hash
from stamp.encoding.config import EncoderName
from stamp.encoding.encoder import Encoder
from stamp.encoding.encoder.chief import CHIEF
from stamp.modeling.data import CoordsInfo
from stamp.preprocessing.config import ExtractorName
from stamp.types import DeviceLikeType, PandasLabel

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"
__credits__ = ["Neidlinger, et al. (https://github.com/KatherLab/EAGLE)"]

_logger = logging.getLogger("stamp")


class Eagle(Encoder):
    def __init__(self) -> None:
        self.required_agg_extractor = ExtractorName.VIRCHOW2
        super().__init__(
            model=CHIEF().model,
            identifier=EncoderName.EAGLE,
            precision=torch.float32,
            required_extractors=[
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

        if extractor not in self.required_extractors:
            raise ValueError(
                f"Features must be extracted with one of {self.required_extractors}. "
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

    def encode_slides_(
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

        if generate_hash:
            encode_dir = f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}"
        else:
            encode_dir = f"{self.identifier}-slide"
        encode_dir = output_dir / encode_dir
        os.makedirs(encode_dir, exist_ok=True)

        self.model.to(device).eval()

        for tile_feats_filename in (progress := tqdm(os.listdir(feat_dir))):
            h5_ctp = os.path.join(feat_dir, tile_feats_filename)
            h5_vir2 = os.path.join(agg_feat_dir, tile_feats_filename)
            slide_name: str = Path(tile_feats_filename).stem
            progress.set_description(slide_name)

            # skip patient in case feature file already exists
            output_path = (encode_dir / slide_name).with_suffix(".h5")
            if output_path.exists():
                _logger.debug(
                    f"skipping {str(slide_name)} because {output_path} already exists"
                )
                continue

            try:
                feats, agg_feats = self._validate_and_read_features_with_agg(
                    h5_ctp, h5_vir2, slide_name
                )
            except ValueError as e:
                tqdm.write(s=str(e))
                continue

            slide_embedding = self._generate_slide_embedding(feats, device, agg_feats)
            self._save_features_(
                output_path=output_path, feats=slide_embedding, feat_type="slide"
            )

    # TODO: Add @override decorator on each encoder once it is added to python
    def encode_patients_(
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
        self.model.to(device).eval()

        # generate the name for the folder containing the feats
        if generate_hash:
            encode_dir = (
                f"{self.identifier}-pat-{get_processing_code_hash(Path(__file__))[:8]}"
            )
        else:
            encode_dir = f"{self.identifier}-pat"
        encode_dir = output_dir / encode_dir
        os.makedirs(encode_dir, exist_ok=True)

        for patient_id, group in (progress := tqdm(patient_groups)):
            progress.set_description(str(patient_id))

            # skip patient in case feature file already exists
            output_path = (encode_dir / str(patient_id)).with_suffix(".h5")
            if output_path.exists():
                _logger.debug(
                    f"skipping {str(patient_id)} because {output_path} already exists"
                )
                continue

            feats_list = []
            agg_feats_list = []

            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                slide_name = Path(slide_filename).stem
                h5_ctp = os.path.join(feat_dir, slide_filename)
                h5_vir2 = os.path.join(agg_feat_dir, slide_filename)

                feats, agg_feats = self._validate_and_read_features_with_agg(
                    h5_ctp, h5_vir2, slide_name
                )
                feats_list.append(feats)
                agg_feats_list.append(agg_feats)

            if not feats_list:
                tqdm.write(f"No ctranspath features for patient {patient_id}")
                continue

            patient_embedding = self._generate_patient_embedding(
                feats_list, device, agg_feats_list
            )
            self._save_features_(
                output_path=output_path, feats=patient_embedding, feat_type="patient"
            )
