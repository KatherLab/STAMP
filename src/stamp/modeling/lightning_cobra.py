"""Lightning wrapper around the model"""

from collections.abc import Iterable, Sequence
from typing import TypeAlias
import os
import lightning
import numpy as np
import torch
from jaxtyping import Bool, Float
from packaging.version import Version
from torch import Tensor, nn, optim
from torchmetrics.classification import MulticlassAUROC

import stamp
from stamp.cache import STAMP_CACHE_DIR
from stamp.modeling.data import (
    Bags,
    BagSizes,
    Category,
    CoordinatesBatch,
    EncodedTargets,
    PandasLabel,
    PatientId,
)

try:
    from cobra.utils.load_cobra import get_cobraII
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "cobra dependencies not installed."
        " Please update your venv using `uv sync --extra cobra`"
    ) from e
Loss: TypeAlias = Float[Tensor, ""]


class LitCobra(lightning.LightningModule):
    def __init__(
        self,
        *,
        categories: Sequence[Category],
        category_weights: Float[Tensor, "category_weight"],  # noqa: F821
        # dropout: float,
        # Experimental features
        # TODO remove default values for stamp 3; they're only here for backwards compatibility
        # use_alibi: bool = False,
        # Metadata used by other parts of stamp, but not by the model itself
        feat_dim: int,
        lr: float = 1e-5,
        ground_truth_label: PandasLabel,
        train_patients: Iterable[PatientId],
        valid_patients: Iterable[PatientId],
        stamp_version: Version = Version(stamp.__version__),
        freeze: str = "None",
        dropout: float = 0.5,
        hidden_dim: int = 512,
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
        # if not os.path.exists("../../../weights/cobraII.pth.tar"): #FIXME
        # project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
        # weights_path = os.path.join(project_dir, "weights", "cobraII.pth.tar")
        model_path = STAMP_CACHE_DIR / "cobraII.pth.tar"
        self.cobra = get_cobraII(
            download_weights=(not os.path.exists(model_path)),
            checkpoint_path=model_path,
        )
        self.class_weights = category_weights
        self.head = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(categories)),
        )
        self.lr = lr

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

        self.valid_auroc = MulticlassAUROC(len(categories))

        # Used during deployment
        self.ground_truth_label = ground_truth_label
        self.categories = np.array(categories)
        self.train_patients = train_patients
        self.valid_patients = valid_patients

        _ = metadata  # unused, but saved in model

        self.save_hyperparameters()

        if freeze == "base":
            print("Freezing base of COBRA")
            for param in self.cobra.embed.parameters():
                param.requires_grad = False
            for param in self.cobra.mamba_enc.parameters():
                param.requires_grad = False
            for param in self.cobra.norm.parameters():
                param.requires_grad = False
        elif freeze == "emb":
            print("Freezing embedding of COBRA")
            for param in self.cobra.embed.parameters():
                param.requires_grad = False
        elif freeze == "full":
            print("Freezing COBRA")
            for param in self.cobra.parameters():
                param.requires_grad = False
        elif freeze != "None":
            raise ValueError(
                "Invalid freeze option, implemented options: 'None', 'base', 'emb', 'full'"
            )

    def forward(
        self,
        bags: Bags,
    ) -> Float[Tensor, "batch logit"]:
        return self.head(self.cobra(bags))

    def _step(
        self,
        *,
        step_name: str,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        _ = batch_idx  # unused

        bags, coords, bag_sizes, targets = batch

        # logits = self.vision_transformer(
        #     bags, coords=coords, mask=_mask_from_bags(bags=bags, bag_sizes=bag_sizes)
        # )
        logits = self(bags)

        loss = nn.functional.cross_entropy(
            logits, targets.type_as(logits), weight=self.class_weights.type_as(logits)
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
            self.valid_auroc.update(logits, targets.long().argmax(-1))
            self.log(
                f"{step_name}_auroc",
                self.valid_auroc,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,
            )

        return loss

    def training_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(
            step_name="training",
            batch=batch,
            batch_idx=batch_idx,
        )

    def validation_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(
            step_name="validation",
            batch=batch,
            batch_idx=batch_idx,
        )

    def test_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int,
    ) -> Loss:
        return self._step(
            step_name="test",
            batch=batch,
            batch_idx=batch_idx,
        )

    def predict_step(
        self,
        batch: tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets],
        batch_idx: int = -1,
    ) -> Float[Tensor, "batch logit"]:
        bags, coords, bag_sizes, _ = batch
        return self.head(self.cobra(bags))

    def configure_optimizers(self) -> optim.Optimizer:
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


# def _mask_from_bags(
#     *,
#     bags: Bags,
#     bag_sizes: BagSizes,
# ) -> Bool[Tensor, "batch tile"]:
#     max_possible_bag_size = bags.size(1)
#     mask = torch.arange(max_possible_bag_size).type_as(bag_sizes).unsqueeze(0).repeat(
#         len(bags), 1
#     ) >= bag_sizes.unsqueeze(1)

#     return mask
