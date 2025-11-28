#!/usr/bin/env python3
from pathlib import Path

import pytorch_lightning as pl
import tomli
import torch
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import train_test_split

from stamp.modeling.barspoon.data import make_train_dataloaders
from stamp.modeling.barspoon.model import LitEncDecTransformer
from stamp.modeling.barspoon.target_file import encode_targets
from stamp.modeling.barspoon.utils import (
    flatten_batched_dicts,
    make_dataset_df,
    make_preds_df,
)
from stamp.seed import Seed


def train_mixin(
    output_dir: Path,
    clini_tables: list[Path],
    slide_tables: list[Path],
    feature_dirs: list[Path],
    patient_col: str,
    filename_col: str,
    group_by: str | None,
    target_file: Path,
    valid_clini_tables: list[Path] | None = None,
    valid_slide_tables: list[Path] | None = None,
    valid_feature_dirs: list[Path] | None = None,
    instances_per_bag: int = 4096,
    batch_size: int = 4,
    num_workers: int = 4,
    # num_encoder_heads: int = 8,
    # num_decoder_heads: int = 8,
    # num_encoder_layers: int = 2,
    # num_decoder_layers: int = 2,
    # d_model: int = 512,
    # dim_feedforward: int = 2048,
    # positional_encoding: bool = True,
    # learning_rate: float = 1e-4,
    accumulate_grad_samples: int = 32,
    patience: int = 16,
    max_epochs: int = 256,
    seed: int = 42,
    accelerator: str = "auto",
    **kargs,
):
    # parser = make_argument_parser()
    # args = parser.parse_args()

    Seed.set(seed)
    torch.set_float32_matmul_precision("medium")

    with open(target_file, "rb") as target_toml_file:
        target_info = tomli.load(target_toml_file)
    target_labels = list(target_info["targets"].keys())

    if valid_clini_tables or valid_slide_tables or valid_feature_dirs:
        # read validation set from separate clini / slide table / feature dir
        train_df = make_dataset_df(
            clini_tables=clini_tables,
            slide_tables=slide_tables,
            feature_dirs=feature_dirs,
            patient_col=patient_col,
            filename_col=filename_col,
            group_by=group_by,
            target_labels=target_labels,
        )
        valid_df = make_dataset_df(
            clini_tables=valid_clini_tables or clini_tables,
            slide_tables=valid_slide_tables or slide_tables,
            feature_dirs=valid_feature_dirs or feature_dirs,
            patient_col=patient_col,
            filename_col=filename_col,
            group_by=group_by,
            target_labels=target_labels,
        )
    else:
        # split validation set off main dataset
        dataset_df = make_dataset_df(
            clini_tables=clini_tables,
            slide_tables=slide_tables,
            feature_dirs=feature_dirs,
            patient_col=patient_col,
            filename_col=filename_col,
            group_by=group_by,
            target_labels=target_labels,
        )
        train_items, valid_items = train_test_split(dataset_df.index, test_size=0.2)
        train_df, valid_df = dataset_df.loc[train_items], dataset_df.loc[valid_items]

    train_encoded_targets = encode_targets(
        train_df, target_labels=target_labels, **target_info
    )

    valid_encoded_targets = encode_targets(
        valid_df, target_labels=target_labels, **target_info
    )

    assert not (overlap := set(train_df.index) & set(valid_df.index)), (
        f"unexpected overlap between training and testing set: {overlap}"
    )

    train_dl, valid_dl = make_train_dataloaders(
        train_bags=train_df.path.to_list(),
        train_targets={k: v.encoded for k, v in train_encoded_targets.items()},
        valid_bags=valid_df.path.to_list(),
        valid_targets={k: v.encoded for k, v in valid_encoded_targets.items()},
        instances_per_bag=instances_per_bag,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    example_bags, _, _ = next(iter(train_dl))
    d_features = example_bags.size(-1)

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
        },
        **{k: v for k, v in kargs.items() if k not in {"target_file"}},
    )

    trainer = pl.Trainer(
        default_root_dir=output_dir,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min", patience=patience),
            ModelCheckpoint(
                monitor="val_loss",
                mode="min",
                filename="checkpoint-{epoch:02d}-{val_loss:0.3f}",
            ),
        ],
        max_epochs=max_epochs,
        # FIXME The number of accelerators is currently fixed to one for the
        # following reasons:
        #  1. `trainer.predict()` does not return any predictions if used with
        #     the default strategy no multiple GPUs
        #  2. `stamp.modeling.barspoon.model.SafeMulticlassAUROC` breaks on multiple GPUs.
        accelerator=accelerator,
        devices=1,
        accumulate_grad_batches=accumulate_grad_samples // batch_size,
        gradient_clip_val=0.5,
        logger=CSVLogger(save_dir=output_dir),
    )

    trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=valid_dl)

    predictions = flatten_batched_dicts(
        trainer.predict(model=model, dataloaders=valid_dl, return_predictions=True)
    )

    preds_df = make_preds_df(
        predictions=predictions,
        base_df=valid_df,
        categories={k: v.categories for k, v in train_encoded_targets.items()},
    )
    preds_df.to_csv(output_dir / "valid-patient-preds.csv")


if __name__ == "__main__":
    train_mixin(
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
        accelerator="auto",
    )
