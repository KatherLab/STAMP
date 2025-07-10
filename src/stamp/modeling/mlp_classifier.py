from typing import Iterable, Sequence

import lightning
import numpy as np
import torch
from packaging.version import Version
from torch import Tensor, nn
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

    # TODO: Add version check with version 2.2.1, for both MLP and Transformer

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

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)
