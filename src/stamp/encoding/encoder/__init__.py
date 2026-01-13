import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.encoding.config import EncoderName
from stamp.modeling.data import CoordsInfo, get_coords, read_table
from stamp.preprocessing.config import ExtractorName
from stamp.types import DeviceLikeType, PandasLabel

__author__ = "Juan Pablo Ricapito"
__copyright__ = "Copyright (C) 2025 Juan Pablo Ricapito"
__license__ = "MIT"

_logger = logging.getLogger("stamp")


class Encoder(ABC):
    def __init__(
        self,
        model,
        identifier: EncoderName,
        precision: torch.dtype,
        required_extractors: list[ExtractorName],
    ):
        self.model = model
        self.identifier = identifier
        self.precision = precision
        self.required_extractors = required_extractors

    def encode_slides_(
        self,
        output_dir: Path,
        feat_dir: Path,
        device: DeviceLikeType,
        generate_hash: bool,
        **kwargs,
    ) -> None:
        """General method for encoding slide-level features. Called by init_slide_encoder_.
        Override this function if coords are required. See init_slide_encoder_ for full description"""
        # generate the name for the folder containing the feats
        if generate_hash:
            encode_dir = f"{self.identifier}-slide-{get_processing_code_hash(Path(__file__))[:8]}"
        else:
            encode_dir = f"{self.identifier}-slide"
        encode_dir = output_dir / encode_dir
        os.makedirs(encode_dir, exist_ok=True)

        self.model.to(device).eval()

        if self.precision == torch.float16:
            self.model.half()

        for tile_feats_filename in (progress := tqdm(os.listdir(feat_dir))):
            h5_path = os.path.join(feat_dir, tile_feats_filename)
            slide_name: str = Path(tile_feats_filename).name
            progress.set_description(slide_name)

            # skip patient in case feature file already exists
            output_path = (encode_dir / slide_name).with_suffix(".h5")
            if output_path.exists():
                _logger.info(
                    f"skipping {str(slide_name)} because {output_path} already exists"
                )
                continue

            try:
                feats, coords = self._validate_and_read_features(h5_path)
            except ValueError as e:
                tqdm.write(s=str(e))
                continue

            slide_embedding = self._generate_slide_embedding(
                feats, device, coords=coords
            )
            self._save_features_(
                output_path=output_path, feats=slide_embedding, feat_type="slide"
            )

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
        """General method for encoding patient-level features. Called by init_patient_encoder_.
        Override this function if coords are required. See init_patient_encoder_ for full description"""
        # generate the name for the folder containing the feats
        if generate_hash:
            encode_dir = (
                f"{self.identifier}-pat-{get_processing_code_hash(Path(__file__))[:8]}"
            )
        else:
            encode_dir = f"{self.identifier}-pat"
        encode_dir = output_dir / encode_dir
        os.makedirs(encode_dir, exist_ok=True)

        self.model.to(device).eval()

        if self.precision == torch.float16:
            self.model.half()

        slide_table = read_table(slide_table_path)
        patient_groups = slide_table.groupby(patient_label)

        for patient_id, group in (progress := tqdm(patient_groups)):
            progress.set_description(str(patient_id))

            # skip patient in case feature file already exists
            output_path = (encode_dir / str(patient_id)).with_suffix(".h5")
            if output_path.exists():
                _logger.info(
                    f"skipping {str(patient_id)} because {output_path} already exists"
                )
                continue

            feats_list = []

            for _, row in group.iterrows():
                slide_filename = row[filename_label]
                h5_path = os.path.join(feat_dir, slide_filename)
                feats, coords = self._validate_and_read_features(h5_path)
                feats_list.append(feats)

            if not feats_list:
                tqdm.write(f"No features found for patient {patient_id}, skipping.")
                continue

            patient_embedding = self._generate_patient_embedding(
                feats_list, device, **kwargs
            )
            self._save_features_(
                output_path=output_path, feats=patient_embedding, feat_type="patient"
            )

    @abstractmethod
    def _generate_slide_embedding(
        self, feats: torch.Tensor, device, coords, **kwargs
    ) -> np.ndarray:
        """Generate slide embedding. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _generate_patient_embedding(
        self,
        feats_list: list,
        device,
        coords_list: list,
        **kwargs,
    ) -> np.ndarray:
        """Generate patient embedding. Must be implemented by subclasses."""
        pass

    def _validate_and_read_features(self, h5_path: str) -> tuple[Tensor, CoordsInfo]:
        feats, coords, extractor = self._read_h5(h5_path)
        if extractor not in self.required_extractors:
            raise ValueError(
                f"Features must be extracted with one of {self.required_extractors}. "
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
            extractor: str = f.attrs.get("extractor", "")
            if extractor == "":
                raise ValueError(
                    f"Feature file does not have extractor's name in the metadata: {os.path.basename(h5_path)}"
                )

            return feats, coords, _resolve_extractor_name(extractor)

    def _save_features_(
        self, output_path: Path, feats: np.ndarray, feat_type: str
    ) -> None:
        with (
            NamedTemporaryFile(dir=output_path.parent, delete=False) as tmp_h5_file,
            h5py.File(tmp_h5_file, "w") as f,
        ):
            try:
                f["feats"] = feats
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = str(self.identifier)
                f.attrs["precision"] = str(self.precision)
                f.attrs["stamp_version"] = stamp.__version__
                f.attrs["code_hash"] = get_processing_code_hash(Path(__file__))[:8]
                f.attrs["feat_type"] = feat_type
                # TODO: Add more metadata like tile-level extractor name
                # and maybe tile size in pixels and microns
            except Exception:
                _logger.exception(f"error while writing {output_path}")
                if tmp_h5_file is not None:
                    Path(tmp_h5_file.name).unlink(missing_ok=True)

            Path(tmp_h5_file.name).rename(output_path)
            _logger.debug(f"saved features to {output_path}")


def _resolve_extractor_name(raw: str) -> ExtractorName:
    """
    Resolve an extractor string to a valid ExtractorName.

    Handles:
      - exact matches ('gigapath', 'virchow-full')
      - versioned strings like 'gigapath-ae23d', 'virchow-full-2025abc'
    Raises ValueError if the base name is not recognized.
    """
    if not raw:
        raise ValueError("Empty extractor string")

    name = str(raw).strip().lower()

    # Exact match
    for e in ExtractorName:
        if name == e.value.lower():
            return e

    # Versioned form: '<enum-value>-something'
    for e in ExtractorName:
        if name.startswith(e.value.lower() + "-"):
            return e

    # Otherwise fail
    raise ValueError(
        f"Unknown extractor '{raw}'. "
        f"Expected one of {[e.value for e in ExtractorName]} "
        f"or a versioned variant like '<name>-<hash>'."
    )
