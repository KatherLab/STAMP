#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from packaging.specifiers import SpecifierSet
from torch.utils.data import DataLoader

from stamp.modeling.barspoon.data import BagDataset
from stamp.modeling.barspoon.model import LitEncDecTransformer
from stamp.modeling.barspoon.utils import (
    flatten_batched_dicts,
    make_dataset_df,
    make_preds_df,
)
from stamp.seed import Seed


def deploy_mixin(
    output_dir: Path,
    clini_tables: list[Path],
    slide_tables: list[Path],
    feature_dirs: list[Path],
    patient_col: str,
    filename_col: str,
    group_by: str | None,
    checkpoint_path: Path,
    num_workers: int = min(os.cpu_count() or 0, 8),
    accelerator: str = "auto",
    seed: int = 42,
):
    Seed.set(seed)
    torch.set_float32_matmul_precision("medium")

    model = LitEncDecTransformer.load_from_checkpoint(checkpoint_path=checkpoint_path)
    name, version = model.hparams.get("version", "undefined 0").split(" ")

    spec = SpecifierSet(">=3.0,<4")

    if not (
        name == "barspoon-transformer"
        and (spec := SpecifierSet(">=3.0,<4")).contains(version)
    ):
        raise ValueError(
            f"model not compatible. Found {name} {version}, expected barspoon-transformer {spec}"
        )

    target_labels = model.hparams["target_labels"]

    dataset_df = make_dataset_df(
        clini_tables=clini_tables,
        slide_tables=slide_tables,
        feature_dirs=feature_dirs,
        patient_col=patient_col,
        filename_col=filename_col,
        group_by=group_by,
        target_labels=target_labels,
    )

    # Make a dataset with faux labels (the labels will be ignored)
    ds = BagDataset(
        bags=list(dataset_df.path),
        targets={},
        instances_per_bag=None,
    )
    dl = DataLoader(ds, shuffle=False, num_workers=num_workers)

    # FIXME The number of accelerators is currently fixed because
    # `trainer.predict()` does not return any predictions if used with the
    # default strategy no multiple GPUs
    trainer = pl.Trainer(
        default_root_dir=output_dir,
        accelerator=accelerator,
        devices=1,
        logger=False,
    )
    predictions = flatten_batched_dicts(trainer.predict(model=model, dataloaders=dl))
    preds_df = make_preds_df(
        predictions={k: v.float() for k, v in predictions.items()},
        base_df=dataset_df.drop(columns="path"),
        categories=model.hparams["categories"],
    )

    # save results
    output_dir.mkdir(exist_ok=True, parents=True)
    preds_df.to_csv(output_dir / "patient-preds.csv")


if __name__ == "__main__":
    deploy_mixin(
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
        checkpoint_path=Path(
            "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/multitarget/crossval/fold_no=1/lightning_logs/version_0/checkpoints/checkpoint-epoch=04-val_loss=0.739.ckpt"
        ),
        patient_col="PATIENT",
        filename_col="FILENAME",
        group_by=None,
        num_workers=4,
        seed=42,
        accelerator="auto",
    )
