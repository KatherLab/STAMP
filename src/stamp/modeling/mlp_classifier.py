from collections.abc import Iterable, Sequence

import lightning
import numpy as np
import torch
from packaging.version import Version
from torch import Tensor, nn, optim
from torchmetrics.classification import MulticlassAUROC

import stamp
from stamp.types import Category, PandasLabel, PatientId


class MLPClassifier(nn.Module):
    """
    Simple MLP for classification from a single feature vector.
    """

    def __init__(
        self,
        dim_input: int,
        dim_hidden: int,
        dim_output: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        layers = []
        in_dim = dim_input
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, dim_hidden))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = dim_hidden
        layers.append(nn.Linear(in_dim, dim_output))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class LitMLPClassifier(lightning.LightningModule):
    """
    PyTorch Lightning wrapper for MLPClassifier.
    """

    supported_features = ["patient"]

    def __init__(
        self,
        *,
        categories: Sequence[Category],
        category_weights: torch.Tensor,
        dim_input: int,
        dim_hidden: int,
        num_layers: int,
        dropout: float,
        ground_truth_label: PandasLabel,
        train_patients: Iterable[PatientId],
        valid_patients: Iterable[PatientId],
        stamp_version: Version = Version(stamp.__version__),
        # Learning Rate Scheduler params, used only in training
        total_steps: int,
        max_lr: float,
        div_factor: float,
        **metadata,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = MLPClassifier(
            dim_input=dim_input,
            dim_hidden=dim_hidden,
            dim_output=len(categories),
            num_layers=num_layers,
            dropout=dropout,
        )
        self.class_weights = category_weights
        self.valid_auroc = MulticlassAUROC(len(categories))
        self.ground_truth_label = ground_truth_label
        self.categories = np.array(categories)
        self.train_patients = train_patients
        self.valid_patients = valid_patients
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.div_factor = div_factor
        self.stamp_version = str(stamp_version)

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

    def configure_optimizers(
        self,
    ) -> tuple[list[optim.Optimizer], list[optim.lr_scheduler.LRScheduler]]:
        optimizer = optim.AdamW(
            self.parameters(), lr=1e-3
        )  # this lr value should be ignored with the scheduler

        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            total_steps=self.total_steps,
            max_lr=self.max_lr,
            div_factor=25.0,
        )
        return [optimizer], [scheduler]

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Log learning rate at the end of each training batch
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log(
            "learning_rate",
            current_lr,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
