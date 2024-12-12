"""Lightning wrapper around the model"""

from typing import NewType

import lightning
import numpy.typing as npt
from packaging.version import Version
from torch import Tensor, nn, optim
from torchmetrics.classification import MulticlassAUROC

import stamp
from stamp.modeling.data import (
    Bags,
    BagSizes,
    Category,
    EncodedTargets,
    PandasLabel,
    PatientId,
)
from stamp.modeling.vision_transformer import VisionTransformer

Losses = NewType("Losses", Tensor)


class LitVisionTransformer(lightning.LightningModule):
    def __init__(
        self,
        *,
        categories: npt.NDArray[Category],
        category_weights: Tensor,
        dim_input: int,
        dim_model: int,
        dim_feedforward: int,
        n_heads: int,
        n_layers: int,
        dropout: float,
        # Metadata used by other parts of stamp, but not by the model itself
        ground_truth_label: PandasLabel,
        train_patients: npt.NDArray[PatientId],
        valid_patients: npt.NDArray[PatientId],
        stamp_version: Version = Version(stamp.__version__),
        # Other metadata
        **metadata,
    ) -> None:
        """
        Args:
            metadata:
                Any additional information to be saved in the models,
                but not directly influencing the model.
        """
        super().__init__()

        if len(categories) != len(category_weights):
            raise ValueError(
                "the number of category weights has to mathc the number of categories!"
            )

        self.model = VisionTransformer(
            dim_output=len(categories),
            dim_input=dim_input,
            dim_model=dim_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.class_weights = category_weights

        # Check if version is compatible.
        # This should only happen when the model is loaded,
        # otherwise the default value will make these checks pass.
        if stamp_version < Version("2.0.0.dev0"):
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

        self.valid_auroc = MulticlassAUROC(len(categories))

        # Used during deployment
        self.ground_truth_label = ground_truth_label
        self.categories = categories
        self.train_patients = train_patients
        self.valid_patients = valid_patients

        _ = metadata  # unused

        self.save_hyperparameters()

    def _step(
        self,
        *,
        step_name: str,
        batch: tuple[Bags, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Losses:
        _ = batch_idx  # unused

        bags, _, targets = batch
        assert (
            bags.ndim == 3
        ), "expected bags to have dimensions [batch_size, n_tiles, n_features]"

        logits = self.model(bags)

        loss = nn.functional.cross_entropy(
            logits, targets, weight=self.class_weights.type_as(logits)
        )

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if step_name == "valid":
            # TODO this is a bit ugly, we'd like to have `_step` without special cases
            self.valid_auroc.update(logits, targets.argmax(-1))
            self.log(
                f"{step_name}_auroc",
                self.valid_auroc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )

        return Losses(loss)

    def training_step(
        self, batch: tuple[Bags, BagSizes, EncodedTargets], batch_idx: int
    ) -> Losses:
        return self._step(
            step_name="training",
            batch=batch,
            batch_idx=batch_idx,
        )

    def validation_step(
        self, batch: tuple[Bags, BagSizes, EncodedTargets], batch_idx: int
    ) -> Losses:
        return self._step(
            step_name="validation",
            batch=batch,
            batch_idx=batch_idx,
        )

    def test_step(
        self, batch: tuple[Bags, BagSizes, EncodedTargets], batch_idx: int
    ) -> Losses:
        return self._step(
            step_name="test",
            batch=batch,
            batch_idx=batch_idx,
        )

    def predict_step(
        self, batch: tuple[Bags, BagSizes, EncodedTargets], batch_idx: int
    ) -> Tensor:
        bags, _, _ = batch
        return self.model(bags)

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
