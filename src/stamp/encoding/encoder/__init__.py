import os
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

import h5py
import torch
from torch import Tensor, dtype, nn
from tqdm import tqdm

import stamp
from stamp.modeling.data import CoordsInfo, get_coords

EncoderModel = TypeVar("EncoderModel", bound=nn.Module)


@dataclass()
class Encoder(ABC, Generic[EncoderModel]):
    _: KW_ONLY
    model: EncoderModel
    identifier: str
    """An ID _uniquely_ identifying the model and extractor.
    
    If possible, it should include the digest of the model weights etc.
    so that any change in the model also changes its ID.
    """

    @abstractmethod
    def encode_slides(self, output_dir, feat_dir, device, **kwargs) -> None:
        """Abstract method to encode slide from patch features."""
        pass

    @abstractmethod
    def encode_patients(
        self, output_dir, feat_dir, slide_table_path, device, **kwargs
    ) -> None:
        """Abstract method to encode patient from slide features."""
        pass

    def _read_h5(
        self, h5_path: str, precision: dtype
    ) -> tuple[Tensor, CoordsInfo, str]:
        if not os.path.exists(h5_path) or not h5_path.endswith(".h5"):
            raise FileNotFoundError("File does not exist or is not an h5 file")
        with h5py.File(h5_path, "r") as f:
            feats: Tensor = torch.tensor(f["feats"][:], dtype=precision)  # type: ignore
            coords: CoordsInfo = get_coords(f)
            extractor: str = f.attrs.get("extractor", "no extractor name")
            return feats, coords, extractor

    def _validate_and_read_features(
        self, h5_path, extractor_name, precision
    ) -> tuple[Tensor, CoordsInfo]:
        feats, coords, extractor = self._read_h5(h5_path, precision=precision)
        if extractor_name not in extractor:
            raise ValueError(
                f"Features must be extracted with {extractor_name}. "
                f"Features located in {h5_path} are extracted with {extractor}"
            )
        return feats, coords

    def _save_features(self, output_file, entry_dict, precision) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for entry_name, data in entry_dict.items():
                f.create_dataset(f"{entry_name}", data=data["feats"])
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = self.identifier
                f.attrs["precision"] = str(precision)
            # Check if the file is empty
            if len(f) == 0:
                tqdm.write("Extraction failed: file empty")
                os.remove(output_file)
            else:
                tqdm.write(f"Finished encoding, saved to {output_file}")
            # TODO: Add codebase hash to h5 file
