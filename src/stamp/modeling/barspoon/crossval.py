from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pytorch_lightning as pl
import tomli
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import KFold

from stamp.modeling.barspoon.data import make_crossval_dataloaders
from stamp.modeling.barspoon.model import LitEncDecTransformer
from stamp.modeling.barspoon.target_file import encode_targets
from stamp.modeling.barspoon.utils import (
    flatten_batched_dicts,
    make_dataset_df,
    make_preds_df,
)
from stamp.seed import Seed


def crossval_mixin(
    output_dir: Path,
    clini_tables: list[Path],
    slide_tables: list[Path],
    feature_dirs: list[Path],
    target_file: Path,
    patient_col: str,
    filename_col: str,
    group_by: str | None,
    instances_per_bag: int = 4096,
    batch_size: int = 4,
    accumulate_grad_samples: int = 32,
    num_workers: int = 4,
    patience: int = 16,
    max_epochs: int = 256,
    seed: int = 42,
    num_splits: int = 5,
    accelerator: str = "auto",
    **kwargs,
) -> None:
    """Cross-validation mixin for BarSpoon modeling"""

    Seed.set(seed)
    torch.set_float32_matmul_precision("medium")

    with open(target_file, "rb") as target_toml_file:
        target_info = tomli.load(target_toml_file)
    target_labels = list(target_info["targets"].keys())

    dataset_df = make_dataset_df(
        clini_tables=clini_tables,
        slide_tables=slide_tables,
        feature_dirs=feature_dirs,
        patient_col=patient_col,
        filename_col=filename_col,
        group_by=group_by,
        target_labels=target_labels,
    )

    for fold_no, (train_idx, valid_idx, test_idx) in enumerate(
        get_splits(dataset_df.index.values, n_splits=num_splits)
    ):
        fold_dir = output_dir / f"{fold_no=}"
        fold_dir.mkdir(exist_ok=True, parents=True)
        train_df, valid_df, test_df = (
            dataset_df.loc[train_idx],
            dataset_df.loc[valid_idx],
            dataset_df.loc[test_idx],
        )

        assert not (overlap := set(train_df.index) & set(valid_df.index)), (
            f"overlap between training and testing set: {overlap}"
        )

        train_encoded_targets = encode_targets(
            train_df, target_labels=target_labels, **target_info
        )

        valid_encoded_targets = encode_targets(
            valid_df, target_labels=target_labels, **target_info
        )

        test_encoded_targets = encode_targets(
            test_df, target_labels=target_labels, **target_info
        )

        train_dl, valid_dl, test_dl = make_crossval_dataloaders(
            train_bags=train_df.path.to_list(),
            train_targets={k: v.encoded for k, v in train_encoded_targets.items()},
            valid_bags=valid_df.path.to_list(),
            valid_targets={k: v.encoded for k, v in valid_encoded_targets.items()},
            test_bags=test_df.path.to_list(),
            test_targets={k: v.encoded for k, v in test_encoded_targets.items()},
            instances_per_bag=instances_per_bag,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        example_feats, _, _ = next(iter(train_dl))
        d_features = example_feats.size(-1)

        model = LitEncDecTransformer(
            d_features=d_features,
            target_labels=target_labels,
            weights={k: v.weight for k, v in train_encoded_targets.items()},
            # Other hparams
            version="barspoon-transformer 3.1",
            categories={k: v.categories for k, v in train_encoded_targets.items()},
            target_file=target_info,
            **{
                f"train_{train_df.index.name}": list(train_df.index),
                f"valid_{valid_df.index.name}": list(valid_df.index),
                f"test_{test_df.index.name}": list(test_df.index),
            },
            **{k: v for k, v in kwargs.items() if k not in {"target_file"}},
        )

        # FIXME The number of accelerators is currently fixed to one for the
        # following reasons:
        #  1. `trainer.predict()` does not return any predictions if used with
        #     the default strategy no multiple GPUs
        #  2. `stamp.modeling.barspoon.model.SafeMulticlassAUROC` breaks on multiple GPUs.
        trainer = pl.Trainer(
            default_root_dir=fold_dir,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss",
                    mode="min",
                    patience=patience,
                ),
                ModelCheckpoint(
                    monitor="val_loss",
                    mode="min",
                    filename="checkpoint-{epoch:02d}-{val_loss:0.3f}",
                ),
            ],
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=1,
            accumulate_grad_batches=accumulate_grad_samples // batch_size,
            gradient_clip_val=0.5,
            logger=CSVLogger(save_dir=fold_dir),
        )

        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

        trainer.test(model=model, dataloaders=test_dl)
        # Save best validation set predictions
        valid_preds = flatten_batched_dicts(
            trainer.predict(model=model, dataloaders=valid_dl, return_predictions=True)
        )
        valid_preds_df = make_preds_df(
            predictions=valid_preds,
            base_df=valid_df,
            categories={k: v.categories for k, v in train_encoded_targets.items()},
        )
        valid_preds_df.to_csv(fold_dir / "valid-patient-preds.csv")

        # Save test set predictions
        test_preds = flatten_batched_dicts(
            trainer.predict(model=model, dataloaders=test_dl, return_predictions=True)
        )
        test_preds_df = make_preds_df(
            predictions=test_preds,
            base_df=test_df,
            categories={k: v.categories for k, v in train_encoded_targets.items()},
        )
        test_preds_df.to_csv(fold_dir / "patient-preds.csv")


def get_splits(
    items: npt.NDArray[Any], n_splits: int = 6
) -> Iterator[Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]]:
    """Splits a dataset into six training, validation and test sets

    This generator will yield `n_split` sets of training, validation and test
    sets.  To do so, it first splits `items` into `n_splits` different parts,
    and selects a different part as validation and testing set.  The training
    set is made up of the remaining `n_splits`-2 parts.
    """
    splitter = KFold(n_splits=n_splits, shuffle=True)
    # We have to explicitly force `dtype=np.object_` so this doesn't fail for
    # folds of different sizes
    folds = np.array([fold for _, fold in splitter.split(items)], dtype=np.object_)
    for test_fold, test_fold_idxs in enumerate(folds):
        # We have to agressively do `astype()`s here, as, if all folds have the
        # same size, the folds get coerced into one 2D tensor with dtype
        # `object` instead of one with dtype int
        test_fold_idxs = test_fold_idxs.astype(int)
        val_fold = (test_fold + 1) % n_splits
        val_fold_idxs = folds[val_fold].astype(int)

        train_folds = set(range(n_splits)) - {test_fold, val_fold}
        train_fold_idxs = np.concatenate(folds[list(train_folds)]).astype(int)

        yield (
            items[train_fold_idxs],
            items[val_fold_idxs],
            items[test_fold_idxs],
        )


if __name__ == "__main__":
    crossval_mixin(
        output_dir=Path(
            "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/multitarget/crossval"
        ),
        clini_tables=[
            Path(
                "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/survival_prediction/TCGA-CRC-DX_CLINI.xlsx"
            )
        ],
        slide_tables=[
            Path(
                "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/survival_prediction/TCGA-CRC-DX_SLIDE.csv"
            )
        ],
        feature_dirs=[
            Path(
                "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/Narmin/Features/xiyuewang-ctranspath-7c998680-5e630f4e"
            )
        ],
        target_file=Path(
            "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/multitarget/test.toml"
        ),
        patient_col="PATIENT",
        filename_col="FILENAME",
        group_by=None,
        num_encoder_heads=8,
        num_decoder_heads=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        d_model=128,
        dim_feedforward=512,
        positional_encoding=True,
        instances_per_bag=256,
        learning_rate=1e-4,
        batch_size=16,
        accumulate_grad_samples=64,
        num_workers=4,
        patience=10,
        max_epochs=5,
        seed=42,
        num_splits=5,
        accelerator="auto",
    )
