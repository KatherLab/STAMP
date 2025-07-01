"""Lightning wrapper around the model"""

from collections.abc import Iterable, Sequence
from typing import TypeAlias

import lightning
import numpy as np
import torch
from jaxtyping import Bool, Float
from packaging.version import Version
from torch import Tensor, nn, optim
from torchmetrics.classification import MulticlassAUROC

import stamp
from stamp.modeling.vision_transformer import VisionTransformer
from stamp.types import (
    Bags,
    BagSizes,
    Category,
    CoordinatesBatch,
    EncodedTargets,
    PandasLabel,
    PatientId,
)

Loss: TypeAlias = Float[Tensor, ""]


class LitVisionTransformer(lightning.LightningModule):
    """
    PyTorch Lightning wrapper for the Vision Transformer (ViT) model used in weakly supervised
    learning settings, such as Multiple Instance Learning (MIL) for whole-slide images or patch-based data.

    This class encapsulates training, validation, testing, and prediction logic, along with:
    - Masking logic that ensures only valid tiles (patches) participate in attention during training.
    - AUROC metric tracking during validation for multiclass classification.
    - Compatibility checks based on the `stamp` framework version.
    - Integration of class imbalance handling through weighted cross-entropy loss.

    The attention mask is applied *only* during training to hide paddings
    and is skipped during evaluation and inference for reducing memory usage.

    Args:
        categories: List of class labels.
        category_weights: Class weights for cross-entropy loss to handle imbalance.
        dim_input: Input feature dimensionality per tile.
        dim_model: Latent dimensionality used inside the transformer.
        dim_feedforward: Dimensionality of the transformer MLP block.
        n_heads: Number of self-attention heads.
        n_layers: Number of transformer layers.
        dropout: Dropout rate used throughout the model.
        use_alibi: Whether to use ALiBi-style positional bias in attention (optional).
        ground_truth_label: Column name for accessing ground-truth labels from metadata.
        train_patients: List of patient IDs used for training.
        valid_patients: List of patient IDs used for validation.
        stamp_version: Version of the `stamp` framework used during training.
        **metadata: Additional metadata to store with the model.
    """

    def __init__(
        self,
        *,
        categories: Sequence[Category],
        category_weights: Float[Tensor, "category_weight"],  # noqa: F821
        dim_input: int,
        dim_model: int,
        dim_feedforward: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        # Experimental features
        # TODO remove default values for stamp 3; they're only here for backwards compatibility
        use_alibi: bool = False,
        # Metadata used by other parts of stamp, but not by the model itself
        ground_truth_label: PandasLabel,
        train_patients: Iterable[PatientId],
        valid_patients: Iterable[PatientId],
        stamp_version: Version = Version(stamp.__version__),
        # Other metadata
        **metadata,
    ) -> None:
        super().__init__()

        if len(categories) != len(category_weights):
            raise ValueError(
                "the number of category weights has to match the number of categories!"
            )

        self.vision_transformer = VisionTransformer(
            dim_output=len(categories),
            dim_input=dim_input,
            dim_model=dim_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_alibi=use_alibi,
        )
        self.class_weights = category_weights
        self.valid_auroc = MulticlassAUROC(len(categories))

        # Used during deployment
        self.ground_truth_label = ground_truth_label
        self.categories = np.array(categories)
        self.train_patients = train_patients
        self.valid_patients = valid_patients

        _ = metadata  # unused, but saved in model

        # Check if version is compatible.
        # This should only happen when the model is loaded,
        # otherwise the default value will make these checks pass.
        if stamp_version < Version("2.0.0.dev8"):
            # Update this as we change our model in incompatible ways!
            raise ValueError(
                f"model has been built with stamp version {stamp_version} "
                f"which is incompatible with the current version."
            )
        elif stamp_version > Version(stamp.__version__):
            # Let's be strict with models "from the future",
            # better fail deadly than have broken results.
            raise ValueError(
                "model has been built with a stamp version newer than the installed one "
                f"({stamp_version} > {stamp.__version__}). "
                "Please upgrade stamp to a compatible version."
            )

        self.save_hyperparameters()

    def forward(
        self,
        bags: Bags,
    ) -> Float[Tensor, "batch logit"]:
        return self.vision_transformer(bags)

    def _step(
        self,
        *,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        step_name: str,
        use_mask: bool,
    ) -> Loss:
        bags, coords, bag_sizes, targets = batch

        mask = _mask_from_bags(bags=bags, bag_sizes=bag_sizes) if use_mask else None

        logits = self.vision_transformer(bags, coords=coords, mask=mask)

        loss = nn.functional.cross_entropy(
            logits,
            targets.type_as(logits),
            weight=self.class_weights.type_as(logits),
        )

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if step_name == "validation":
            # TODO this is a bit ugly, we'd like to have `_step` without special cases
            self.valid_auroc.update(logits, targets.long().argmax(dim=-1))
            self.log(
                f"{step_name}_auroc",
                self.valid_auroc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return loss

    def training_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(batch=batch, step_name="training", use_mask=True)

    def validation_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(batch=batch, step_name="validation", use_mask=False)

    def test_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(batch=batch, step_name="test", use_mask=False)

    def predict_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Float[Tensor, "batch logit"]:
        bags, coords, bag_sizes, _ = batch
        # adding a mask here will *drastically* and *unbearably* increase memory usage
        return self.vision_transformer(bags, coords=coords, mask=None)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


def _mask_from_bags(
    *,
    bags: Bags,
    bag_sizes: BagSizes,
) -> Bool[Tensor, "batch tile"]:
    max_possible_bag_size = bags.size(1)
    mask = torch.arange(max_possible_bag_size).type_as(bag_sizes).unsqueeze(0).repeat(
        len(bags), 1
    ) >= bag_sizes.unsqueeze(1)

    return mask
