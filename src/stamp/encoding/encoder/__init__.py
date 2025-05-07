import os
from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

import h5py
from torch import nn
from tqdm import tqdm

import stamp

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

    def save_features(self, output_file, slide_dict, precision) -> None:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with h5py.File(output_file, "w") as f:
            for slide_name, data in slide_dict.items():
                f.create_dataset(f"{slide_name}", data=data["feats"])
                f.attrs["version"] = stamp.__version__
                f.attrs["encoder"] = self.identifier
                f.attrs["precision"] = precision
            # Check if the file is empty
            if len(f) == 0:
                tqdm.write("Extraction failed: file empty")
                os.remove(output_file)
            tqdm.write(f"Finished encoding, saved to {output_file}")
            # TODO: Add codebase hash to h5 file