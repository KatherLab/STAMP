import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypeVar

import h5py
import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch._prims_common import DeviceLikeType  # type: ignore
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.config import EncoderName
from stamp.modeling.data import CoordsInfo, PandasLabel, get_coords
from stamp.preprocessing.config import ExtractorName

EncoderModel = TypeVar("EncoderModel", bound=nn.Module)


class Encoder(ABC):
    def __init__(
        self,
        model,
        identifier: EncoderName,
        precision: torch.dtype,
        required_extractor: list[ExtractorName],
    ):
        self.model = model
        self.identifier = identifier
        self.precision = precision
        self.required_extractor = required_extractor

    def encode_slides(
        self,
        output_dir: Path,
        feat_dir: Path,
        device: DeviceLikeType,
        generate_hash: bool,
        **kwargs,
    ) -> None:
        """General method to encode slides."""
        output_file = self._generate_output_path(
            output_dir=output_dir, generate_hash=generate_hash
        )

        slide_dict = {}
        self.model.to(device).eval()

        if self.precision == torch.float16:
            self.model.half()

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        for tile_feats_filename in tqdm(os.listdir(feat_dir), desc="Processing slides"):
            h5_path = os.path.join(feat_dir, tile_feats_filename)
            slide_name: str = Path(tile_feats_filename).stem

            try:
                feats, coords = self._validate_and_read_features(h5_path)
            except ValueError as e:
                tqdm.write(s=str(e))
                continue

            slide_embedding = self._generate_slide_embedding(
                feats, device, coords=coords
            )
            slide_dict[slide_name] = {"feats": slide_embedding}

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
        """General method to encode patients with features only.
        Override this method if coords are needed for encoding"""
        output_file = self._generate_output_path(
            output_dir=output_dir, generate_hash=generate_hash
        )

        patient_dict = {}
        self.model.to(device).eval()

        if self.precision == torch.float16:
            self.model.half()

        if os.path.exists(output_file):
            tqdm.write(f"Output file {output_file} already exists, skipping")
            return

        slide_table = self._read_slide_table(slide_table_path)
        patient_groups = slide_table.groupby(patient_label)

        for patient_id, group in tqdm(patient_groups, leave=False):
            feats_list = []

            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                h5_path = os.path.join(feat_dir, slide_filename)
                feats, _ = self._validate_and_read_features(h5_path)
                feats_list.append(feats)

            if not feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            patient_embedding = self._generate_patient_embedding(
                feats_list, device, **kwargs
            )
            patient_dict[patient_id] = {"feats": patient_embedding}

        self._save_features(output_file, entry_dict=patient_dict)

    @abstractmethod
    def _generate_slide_embedding(
        self, feats: torch.Tensor, device, **kwargs
    ) -> np.ndarray:
        """Generate slide embedding. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _generate_patient_embedding(
        self,
        feats_list: list,
        device,
        **kwargs,
    ) -> np.ndarray:
        """Generate patient embedding. Must be implemented by subclasses."""
        pass

    @staticmethod
    def _read_slide_table(slide_table_path: Path) -> pd.DataFrame:
        return pd.read_csv(slide_table_path)

    def _validate_and_read_features(self, h5_path: str) -> tuple[Tensor, CoordsInfo]:
        feats, coords, extractor = self._read_h5(h5_path)
        if extractor not in self.required_extractor:
            raise ValueError(
                f"Features must be extracted with one of {self.required_extractor}. "
                f"Features located in {h5_path} are extracted with {extractor}"
            )
        return feats, coords

    def _read_h5(
        self,
        h5_path: str,
    ) -> tuple[Tensor, CoordsInfo, str]:
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"File does not exist: {h5_path}")
        elif not h5_path.endswith(".h5"):
            raise ValueError(f"File is not of type .h5: {os.path.basename(h5_path)}")
        with h5py.File(h5_path, "r") as f:
            feats: Tensor = torch.tensor(f["feats"][:], dtype=self.precision)  # type: ignore
            coords: CoordsInfo = get_coords(f)
            extractor: str = f.attrs.get("extractor", "no extractor name")
            return feats, coords, extractor

    def _save_features(self, output_file: str, entry_dict: dict) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for entry_name, data in entry_dict.items():
                f.create_dataset(f"{entry_name}", data=data["feats"])
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = str(self.identifier)
                f.attrs["precision"] = str(self.precision)
            if len(f) == 0:
                tqdm.write("Encoding failed: file empty")
                os.remove(output_file)
            else:
                tqdm.write(f"Finished encoding, saved to {output_file}")

    def _generate_output_path(self, output_dir: Path, generate_hash: bool) -> str:
        if generate_hash:
            output_name = f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}.h5"
        else:
            output_name = f"{self.identifier}-slide.h5"
        return os.path.join(output_dir, output_name)
