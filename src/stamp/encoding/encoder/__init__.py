from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

from torch import nn

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
