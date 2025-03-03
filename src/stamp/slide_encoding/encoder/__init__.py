from dataclasses import KW_ONLY, dataclass
from typing import Generic, TypeVar

from torch import nn

EncoderModel = TypeVar("EncoderModel", bound=nn.Module)

@dataclass(frozen=True)
class Encoder(Generic[EncoderModel]):
    _: KW_ONLY
    model: EncoderModel
    identifier: str
    """An ID _uniquely_ identifying the model and extractor.
    
    If possible, it should include the digest of the model weights etc.
    so that any change in the model also changes its ID.
    """