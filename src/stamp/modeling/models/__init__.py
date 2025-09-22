"""Lightning wrapper around the model"""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import TypeAlias

import lightning
import numpy as np
import torch
from jaxtyping import Bool, Float
from packaging.version import Version
from torch import Tensor, nn, optim
from torchmetrics.classification import MulticlassAUROC
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError, PearsonCorrCoef

import stamp
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


class LitBaseClassifier(lightning.LightningModule, ABC):
    """
    PyTorch Lightning wrapper for tile level and patient level clasification.

    This class encapsulates training, validation, testing, and prediction logic, along with:
    - Masking logic that ensures only valid tiles (patches) participate in attention during training (deactivated)
    - AUROC metric tracking during validation for multiclass classification.
    - Compatibility checks based on the `stamp` framework version.
    - Integration of class imbalance handling through weighted cross-entropy loss.

    The attention mask is currently deactivated to reduce memory usage.

    Args:
        categories: List of class labels.
        category_weights: Class weights for cross-entropy loss to handle imbalance.
        dim_input: Input feature dimensionality per tile.
        total_steps: Number of steps done in the LR Scheduler cycle.
        max_lr: max learning rate.
        div_factor: Determines the initial learning rate via initial_lr = max_lr/div_factor
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
        # Learning Rate Scheduler params, not used in inference
        total_steps: int,
        max_lr: float,
        div_factor: float,
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

        # self.model: nn.Module = self.build_backbone(
        #     dim_input, len(categories), metadata
        # )

        self.class_weights = category_weights
        self.valid_auroc = MulticlassAUROC(len(categories))
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.div_factor = div_factor

        # Used during deployment
        self.ground_truth_label = ground_truth_label
        self.categories = np.array(categories)
        self.train_patients = train_patients
        self.valid_patients = valid_patients
        self.stamp_version = str(stamp_version)

        _ = metadata  # unused, but saved in model

        # Check if version is compatible.
        # This should only happen when the model is loaded,
        # otherwise the default value will make these checks pass.
        # TODO: Change this on version change
        if stamp_version < Version("2.3.0"):
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

    @abstractmethod
    def build_backbone(
        self, dim_input: int, dim_output: int, metadata: dict
    ) -> nn.Module:
        pass

    @staticmethod
    def get_model_params(model_class: type[nn.Module], metadata: dict) -> dict:
        keys = [
            k for k in inspect.signature(model_class.__init__).parameters if k != "self"
        ]
        return {k: v for k, v in metadata.items() if k in keys}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.total_steps,
            max_lr=self.max_lr,
            div_factor=self.div_factor,
        )
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


class LitTileClassifier(LitBaseClassifier):
    """
    PyTorch Lightning wrapper for the model used in weakly supervised
    learning settings, such as Multiple Instance Learning (MIL) for whole-slide images or patch-based data.
    """

    supported_features = ["tile"]

    def __init__(self, *, dim_input: int, **kwargs):
        super().__init__(dim_input=dim_input, **kwargs)

        self.vision_transformer: nn.Module = self.build_backbone(
            dim_input, len(self.categories), kwargs
        )

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

        mask = (
            self._mask_from_bags(bags=bags, bag_sizes=bag_sizes) if use_mask else None
        )

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
        return self._step(batch=batch, step_name="training", use_mask=False)

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

    def _mask_from_bags(
        *,
        bags: Bags,
        bag_sizes: BagSizes,
    ) -> Bool[Tensor, "batch tile"]:
        max_possible_bag_size = bags.size(1)
        mask = torch.arange(max_possible_bag_size).type_as(bag_sizes).unsqueeze(
            0
        ).repeat(len(bags), 1) >= bag_sizes.unsqueeze(1)

        return mask


class LitPatientlassifier(LitBaseClassifier):
    """
    PyTorch Lightning wrapper for MLPClassifier.
    """

    supported_features = ["patient"]

    def __init__(self, *, dim_input: int, **kwargs):
        super().__init__(dim_input=dim_input, **kwargs)

        self.model: nn.Module = self.build_backbone(
            dim_input, len(self.categories), kwargs
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def _step(self, batch, step_name: str):
        feats, targets = batch
        logits = self.model(feats)
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
            self.valid_auroc.update(logits, targets.long().argmax(dim=-1))
            self.log(
                f"{step_name}_auroc",
                self.valid_auroc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "training")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "validation")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def predict_step(self, batch, batch_idx):
        feats, _ = batch
        return self.model(feats)


class LitBaseRegressor(lightning.LightningModule, ABC):
    """
    PyTorch Lightning wrapper for tile-level / patient-level regression.

    Adds a selectable loss:
      - 'l1' : mean absolute error
      - 'cc' : correlation-coefficient loss = 1 - Pearson r

    Args:
        dim_input: Input feature dimensionality per tile.
        loss_type: 'l1'.
        total_steps: Number of steps for OneCycleLR.
        max_lr: Maximum LR for OneCycleLR.
        div_factor: initial_lr = max_lr / div_factor.
        ground_truth_label: Column name for ground-truth values in metadata.
        train_patients: IDs used for training.
        valid_patients: IDs used for validation.
        stamp_version: Version of `stamp` used during training.
        **metadata: Stored alongside the model checkpoint.
    """

    def __init__(
        self,
        *,
        dim_input: int,
        # Learning Rate Scheduler params, not used in inference
        total_steps: int,
        max_lr: float,
        div_factor: float,
        # Metadata used by other parts of stamp, but not by the model itself
        ground_truth_label: PandasLabel,
        train_patients: Iterable[PatientId],
        valid_patients: Iterable[PatientId],
        stamp_version: Version = Version(stamp.__version__),
        # Other metadata
        **metadata,
    ) -> None:
        super().__init__()

        self.model: nn.Module = self.build_backbone(dim_input, metadata)

        self.valid_mae = MeanAbsoluteError()
        self.valid_mse = MeanSquaredError()
        self.valid_pearson = PearsonCorrCoef()

        # LR scheduler config
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.div_factor = div_factor

        # Deployment
        self.ground_truth_label = ground_truth_label
        self.train_patients = train_patients
        self.valid_patients = valid_patients
        self.stamp_version = str(stamp_version)

        _ = metadata  # unused here, but saved in model

        # Check if version is compatible.
        # This should only happen when the model is loaded,
        # otherwise the default value will make these checks pass.
        # TODO: Change this on version change
        if stamp_version < Version("2.3.0"):
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

    @abstractmethod
    def build_backbone(self, dim_input: int, metadata: dict) -> nn.Module:
        pass

    @staticmethod
    def get_model_params(model_class: type[nn.Module], metadata: dict) -> dict:
        keys = [
            k for k in inspect.signature(model_class.__init__).parameters if k != "self"
        ]
        return {k: v for k, v in metadata.items() if k in keys}

    @staticmethod
    def _l1_loss(pred: Tensor, target: Tensor) -> Loss:
        # expects shapes [..., 1] or [...]
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)
        return torch.mean(torch.abs(pred - target))

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.total_steps,
            max_lr=self.max_lr,
            div_factor=self.div_factor,
        )
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )


class LitTileRegressor(LitBaseRegressor):
    """
    PyTorch Lightning wrapper for weakly supervised / MIL regression at tile/patient level.
    Produces a single continuous output per bag (dim_output = 1).
    """

    supported_features = ["tile"]

    def forward(
        self,
        bags: Bags,
        coords: CoordinatesBatch | None = None,
        mask: Bool[Tensor, "batch tile"] | None = None,
    ) -> Float[Tensor, "batch 1"]:
        # Mirror the classifier’s call signature to the backbone
        # (most ViT backbones accept coords/mask even if unused)
        return self.model(bags, coords=coords, mask=mask)

    def _step(
        self,
        *,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        step_name: str,
        use_mask: bool,
    ) -> Loss:
        bags, coords, bag_sizes, targets = batch

        mask = (
            self._mask_from_bags(bags=bags, bag_sizes=bag_sizes) if use_mask else None
        )

        preds = self.model(bags, coords=coords, mask=mask)  # (B, 1) preferred
        # Ensure numeric/dtype/shape compatibility
        y = targets.to(preds).float()
        if y.ndim == preds.ndim - 1:
            y = y.unsqueeze(-1)

        loss = self._l1_loss(preds, y)

        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        if step_name == "validation":
            # Optional regression metrics from base (MAE/MSE/Pearson)
            p = preds.squeeze(-1)
            t = y.squeeze(-1)
            self.valid_mae.update(p, t)
            self.valid_mse.update(p, t)
            self.valid_pearson.update(p, t)

        return loss

    def training_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(batch=batch, step_name="training", use_mask=False)

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
    ) -> Float[Tensor, "batch 1"]:
        bags, coords, bag_sizes, _ = batch
        # keep memory usage low as in classifier
        return self.model(bags, coords=coords, mask=None)

    def _mask_from_bags(
        *,
        bags: Bags,
        bag_sizes: BagSizes,
    ) -> Bool[Tensor, "batch tile"]:
        max_possible_bag_size = bags.size(1)
        mask = torch.arange(max_possible_bag_size).type_as(bag_sizes).unsqueeze(
            0
        ).repeat(len(bags), 1) >= bag_sizes.unsqueeze(1)

        return mask
