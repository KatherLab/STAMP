"""KRONOS2 multiplex spatial-proteomics feature extractor."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from transformers import AutoModel

from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import MultiplexExtractor, MultiplexFeatures


_MODEL_ID = "MahmoodLab/KRONOS2"


def register_additional_markers(
    model: torch.nn.Module,
    marker_metadata_csv: Path,
) -> None:
    """Register novel marker metadata with KRONOS2 before inference."""

    register = getattr(model, "register_additional_markers", None)
    if not callable(register):
        raise TypeError(
            "The selected model does not support registering novel markers. "
            "Use the official KRONOS2 model with a valid marker_metadata_csv."
        )
    register(str(marker_metadata_csv))


def _preferred_nuclear_marker(marker_names: Sequence[str]) -> str | None:
    """Select the upstream-preferred nuclear stain when it is present."""

    normalized = {name.strip().upper(): name for name in marker_names}
    return normalized.get("DAPI") or normalized.get("DRAQ5")


def _preprocess_kronos2(
    model: torch.nn.Module,
    batch: Tensor,
    marker_names: Sequence[str],
) -> Tensor:
    """Apply KRONOS2's marker-aware, upstream-provided normalization on CPU."""

    # ``SPImage.to_patches`` has already applied its one dtype-dependent scale.
    patches = np.ascontiguousarray(
        batch.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    preprocess = getattr(model, "preprocess", None)
    if not callable(preprocess):
        raise TypeError("KRONOS2 model does not expose its required preprocess method.")
    normalized = preprocess(
        patches,
        list(marker_names),
        preferred_dapi=_preferred_nuclear_marker(marker_names),
    )
    return torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32))


def _forward_kronos2(
    model: torch.nn.Module,
    batch: Tensor,
    marker_names: Sequence[str],
) -> MultiplexFeatures:
    """Return KRONOS2's published 768-dimensional CLS embedding."""

    feats = model(batch.float(), list(marker_names))
    if not isinstance(feats, Tensor):
        raise TypeError(
            "KRONOS2 returned a non-tensor output; expected its CLS feature tensor."
        )
    return MultiplexFeatures(feats=feats)


def _missing_marker_aware_forward(
    _model: torch.nn.Module,
    _batch: Tensor,
) -> MultiplexFeatures:
    raise RuntimeError("KRONOS2 requires channel-ordered marker names.")


def kronos2() -> MultiplexExtractor[torch.nn.Module]:
    """Load the gated KRONOS2 model from Hugging Face.

    Access to ``MahmoodLab/KRONOS2`` must be approved for the active Hugging Face
    account.  The model includes its own marker vocabulary and normalization data.
    """

    model = AutoModel.from_pretrained(_MODEL_ID, trust_remote_code=True)
    return MultiplexExtractor(
        model=model,
        identifier=ExtractorName.KRONOS2,
        forward=_missing_marker_aware_forward,
        preprocess=_preprocess_kronos2,
        forward_with_markers=_forward_kronos2,
    )
