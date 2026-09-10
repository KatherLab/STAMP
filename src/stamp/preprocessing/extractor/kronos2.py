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


def _validate_marker_batch(batch: Tensor, marker_names: Sequence[str]) -> None:
    if batch.ndim != 4 or batch.shape[1] != len(marker_names) or not marker_names:
        raise ValueError(
            "KRONOS2 per-marker extraction requires (batch, markers, height, width) "
            "input and one name per channel."
        )
    if len(set(marker_names)) != len(marker_names):
        raise ValueError("KRONOS2 per-marker extraction requires unique marker names.")


def _preprocess_kronos2_per_marker(
    model: torch.nn.Module,
    batch: Tensor,
    marker_names: Sequence[str],
) -> Tensor:
    """Normalize each channel independently, matching single-marker inference.

    Pass only the current marker and its nuclear-stain preference to the
    upstream normalizer, just as for a separate single-channel extraction.
    """
    _validate_marker_batch(batch, marker_names)
    return torch.cat(
        [
            _preprocess_kronos2(model, batch[:, index : index + 1], [name])
            for index, name in enumerate(marker_names)
        ],
        dim=1,
    )


def _forward_kronos2_per_marker(
    model: torch.nn.Module,
    batch: Tensor,
    marker_names: Sequence[str],
) -> MultiplexFeatures:
    """Encode aligned tiles separately per channel, preserving marker order.

    ``feats`` is the mean of independent CLS vectors, not a joint-panel CLS.
    MarkerFusion must consume ``marker_embeddings`` instead.
    """
    _validate_marker_batch(batch, marker_names)
    embeddings = []
    for index, name in enumerate(marker_names):
        feats = _forward_kronos2(model, batch[:, index : index + 1], [name]).feats
        if feats.ndim != 2 or feats.shape[0] != batch.shape[0]:
            raise ValueError("KRONOS2 must return one CLS feature vector per tile.")
        embeddings.append(feats)
    stacked = torch.stack(embeddings, dim=1)
    return MultiplexFeatures(feats=stacked.mean(dim=1), marker_embeddings=stacked)


def kronos2_per_marker() -> MultiplexExtractor[torch.nn.Module]:
    """Load KRONOS2 for independent single-channel CLS feature extraction."""
    model = AutoModel.from_pretrained(_MODEL_ID, trust_remote_code=True)
    return MultiplexExtractor(
        model=model,
        identifier=ExtractorName.KRONOS2_PER_MARKER,
        forward=_missing_marker_aware_forward,
        preprocess=_preprocess_kronos2_per_marker,
        forward_with_markers=_forward_kronos2_per_marker,
    )
