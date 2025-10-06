"""Lightning wrapper around the model"""

import inspect
from abc import ABC
from collections.abc import Iterable, Sequence
from typing import Any, TypeAlias

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

__author__ = "Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2025 Minh Duc Nguyen"
__license__ = "MIT"

Loss: TypeAlias = Float[Tensor, ""]


class Base(lightning.LightningModule, ABC):
    """
    PyTorch Lightning wrapper for tile level and patient level clasification/regression.

    - Compatibility checks based on the `stamp` framework version.

    Args:
        total_steps: Number of steps done in the LR Scheduler cycle.
        max_lr: max learning rate.
        div_factor: Determines the initial learning rate via initial_lr = max_lr/div_factor
        train_patients: List of patient IDs used for training.
        valid_patients: List of patient IDs used for validation.
        stamp_version: Version of the `stamp` framework used during training.
        **metadata: Additional metadata to store with the model.
    """

    def __init__(
        self,
        *,
        # Learning Rate Scheduler params, not used in inference
        total_steps: int,
        max_lr: float,
        div_factor: float,
        # Metadata used by other parts of stamp, but not by the model itself
        train_patients: Iterable[PatientId],
        valid_patients: Iterable[PatientId],
        stamp_version: Version = Version(stamp.__version__),
        # Other metadata
        **metadata,
    ) -> None:
        super().__init__()

        # LR scheduler config
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.div_factor = div_factor

        # Deployment
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

        supported_features = getattr(self, "supported_features", None)
        if supported_features is not None:
            self.hparams["supported_features"] = supported_features[0]
        self.save_hyperparameters()

    @staticmethod
    def _get_model_params(model_class: type[nn.Module], metadata: dict) -> dict:
        keys = [
            k for k in inspect.signature(model_class.__init__).parameters if k != "self"
        ]
        return {k: v for k, v in metadata.items() if k in keys}

    def _build_backbone(
        self,
        model_class: type[nn.Module],
        dim_input: int,
        dim_output: int,
        metadata: dict,
    ) -> nn.Module:
        params = self._get_model_params(model_class, metadata)
        return model_class(
            dim_input=dim_input,
            dim_output=dim_output,
            **params,
        )

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


class LitBaseClassifier(Base):
    """
    PyTorch Lightning wrapper for tile level and patient level clasification.

    This class encapsulates training, validation, testing, and prediction logic, along with:
    - Masking logic that ensures only valid tiles (patches) participate in attention during training (deactivated)
    - AUROC metric tracking during validation for multiclass classification.
    - Integration of class imbalance handling through weighted cross-entropy loss.

    The attention mask is currently deactivated to reduce memory usage.

    Args:
        model_class: model backbone
        categories: List of class labels.
        ground_truth_label: Column name for accessing ground-truth labels from metadata.
        category_weights: Class weights for cross-entropy loss to handle imbalance.
        dim_input: Input feature dimensionality per tile.
    """

    def __init__(
        self,
        *,
        model_class: type[nn.Module],
        ground_truth_label: PandasLabel,
        categories: Sequence[Category],
        category_weights: Float[Tensor, "category_weight"],  # noqa: F821
        dim_input: int,
        **kwargs,
    ) -> None:
        super().__init__(
            model_class=model_class,
            ground_truth_label=ground_truth_label,
            categories=categories,
            category_weights=category_weights,
            dim_input=dim_input,
            **kwargs,
        )
        self.ground_truth_label = ground_truth_label

        if len(categories) != len(category_weights):
            raise ValueError(
                "the number of category weights has to match the number of categories!"
            )

        self.model: nn.Module = self._build_backbone(
            model_class, dim_input, len(categories), kwargs
        )

        self.class_weights = category_weights
        self.valid_auroc = MulticlassAUROC(len(categories))
        # Number classes
        self.categories = np.array(categories)

        self.hparams["task"] = "classification"


class LitTileClassifier(LitBaseClassifier):
    """
    PyTorch Lightning wrapper for the model used in weakly supervised
    learning settings, such as Multiple Instance Learning (MIL) for whole-slide images or patch-based data.
    """

    supported_features = ["tile"]

    def forward(
        self,
        bags: Bags,
    ) -> Float[Tensor, "batch logit"]:
        return self.model(bags)

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

        logits = self.model(bags, coords=coords, mask=mask)

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


class LitPatientClassifier(LitBaseClassifier):
    """
    PyTorch Lightning wrapper for MLPClassifier.
    """

    supported_features = ["patient"]

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


class LitBaseRegressor(Base):
    """
    PyTorch Lightning wrapper for tile-level / patient-level regression.

    Adds a selectable loss:
      - 'l1' : mean absolute error
      - 'cc' : correlation-coefficient loss = 1 - Pearson r

    Args:
        dim_input: Input feature dimensionality per tile.
        model_clas: Model backbone
        loss_type: 'l1'.
    """

    def __init__(
        self,
        *,
        dim_input: int,
        model_class: type[nn.Module],
        ground_truth_label: PandasLabel | None,
        **kwargs,
    ) -> None:
        super().__init__(
            dim_input=dim_input,
            model_class=model_class,
            ground_truth_label=ground_truth_label,
            **kwargs,
        )

        self.model: nn.Module = self._build_backbone(model_class, dim_input, 1, kwargs)
        self.ground_truth_label = ground_truth_label
        self.hparams["task"] = "regression"

    @staticmethod
    def _compute_loss(y_true: Tensor, y_pred: Tensor) -> Loss:
        return nn.functional.l1_loss(y_true, y_pred)


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

        loss = self._compute_loss(preds, y)
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
            self.log(
                "validation_loss",
                torch.nn.functional.l1_loss(p, t),
                prog_bar=True,
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


class LitTileSurvival(LitTileRegressor):
    """
    PyTorch Lightning module for survival analysis with Cox proportional hazards loss.
    Expects dataloader batches like:
      (bags, coords, bag_sizes, targets)
    where targets is shape (B,2): [:,0]=time, [:,1]=event (1=event, 0=censored).
    """

    def __init__(
        self,
        time_label: PandasLabel,
        status_label: PandasLabel,
        method: str = "cox",
        **kwargs,
    ):
        super().__init__(time_label=time_label, status_label=status_label, **kwargs)
        self.hparams["task"] = "survival"
        self.method = method
        self.time_label = time_label
        self.status_label = status_label
        # storage for validation accumulation
        self._val_scores, self._val_times, self._val_events = [], [], []

    @staticmethod
    def cox_loss(
        scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        """
        Breslow negative partial log-likelihood.
        scores: (N,) risk scores (higher = riskier)
        times:  (N,) survival/censoring times
        events: (N,) 1=event, 0=censored
        """
        scores = scores.flatten()
        events = events.bool().flatten()
        times = times.flatten()

        # event times and indices
        if not events.any():
            return scores.sum() * 0.0  # keep graph

        t_event = times[events]  # (R,)
        # risk set mask: j is at risk for event i if T_j >= T_i
        # (use >= per standard Cox; vectorized broadcast)
        risk_mask = t_event[:, None] <= times[None, :]  # (R, N)

        # log-sum-exp over risk sets for numerical stability
        # log sum_j exp(score_j) for each event i
        max_scores = scores.max()  # stability
        lse = (
            torch.log((risk_mask * torch.exp(scores - max_scores)).sum(dim=1))
            + max_scores
        )  # (R,)

        # sum over events: s_i - log sum_{j in R_i} exp(s_j)
        loglik = scores[events] - lse
        npll = -loglik.mean()  # mean reduction
        return npll

    @staticmethod
    def logistic_hazard_loss(
        logits: torch.Tensor, times: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        """
        logits: (B, L) raw predictions for each interval
        times: (B,) discrete event/censoring time (int)
        events: (B,) 1=event, 0=censored
        """
        B, L = logits.shape
        hazard = torch.sigmoid(logits)
        log_survival = torch.cumsum(
            torch.log(1 - nn.functional.pad(hazard, (1, 0))), dim=-1
        )

        likelihood = -(
            events * torch.log(hazard[torch.arange(B), times])
            + (1 - events) * torch.log(1 - hazard[torch.arange(B), times])
            + log_survival[torch.arange(B), times]
        )
        return likelihood.mean()

    @staticmethod
    def c_index(
        scores: torch.Tensor, times: torch.Tensor, events: torch.Tensor
    ) -> torch.Tensor:
        """
        Concordance index: proportion of correctly ordered comparable pairs.
        """
        N = len(times)
        if N <= 1:
            return torch.tensor(float("nan"), device=scores.device)

        t_i = times.view(-1, 1).expand(N, N)
        t_j = times.view(1, -1).expand(N, N)
        e_i = events.view(-1, 1).expand(N, N)

        mask = (t_i < t_j) & e_i.bool()
        if mask.sum() == 0:
            return torch.tensor(float("nan"), device=scores.device)

        s_i = scores.view(-1, 1).expand(N, N)[mask]
        s_j = scores.view(1, -1).expand(N, N)[mask]

        conc = (s_i > s_j).float()
        ties = (s_i == s_j).float() * 0.5
        return (conc + ties).sum() / mask.sum()

    def training_step(self, batch, batch_idx):
        bags, coords, bag_sizes, targets = batch
        preds = self.model(bags, coords=coords, mask=None)
        y = targets.to(preds.device, dtype=torch.float32)
        times, events = y[:, 0], y[:, 1]

        if self.method == "cox":
            preds = preds.squeeze(-1)  # (B,)
            loss = self.cox_loss(preds, times, events)
        elif self.method == "logistic-hazard":
            # preds expected shape (B, L)
            loss = self.logistic_hazard_loss(preds, times, events)
        else:
            raise ValueError(f"Unknown method: {self.method}")

        self.log(
            "train_cox_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def validation_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Any:
        bags, coords, bag_sizes, targets = batch
        preds = self.model(bags, coords=coords, mask=None).squeeze(-1)

        y = targets.to(preds.device, dtype=torch.float32)
        times, events = y[:, 0], y[:, 1]

        # accumulate on CPU to save GPU memory
        self._val_scores.append(preds.detach().cpu())
        self._val_times.append(times.detach().cpu())
        self._val_events.append(events.detach().cpu())

    def on_validation_epoch_end(self):
        if len(self._val_scores) == 0:
            return

        scores = torch.cat(self._val_scores).to(self.device)
        times = torch.cat(self._val_times).to(self.device)
        events = torch.cat(self._val_events).to(self.device)

        val_loss = self.cox_loss(scores, times, events)
        val_ci = self.c_index(scores, times, events)

        self.log("cox_loss", val_loss, prog_bar=True, sync_dist=True)
        self.log("val_cindex", val_ci, prog_bar=True, sync_dist=True)

        self._val_scores.clear()
        self._val_times.clear()
        self._val_events.clear()
