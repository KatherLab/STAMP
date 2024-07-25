from typing import Any, Iterable, Optional, Sequence, Tuple, TypeVar
from pathlib import Path
from functools import partial

import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.all import (
    Learner, DataLoader, DataLoaders, RocAuc,
    SaveModelCallback, CSVLogger, EarlyStoppingCallback,
    MixedPrecision, AMPMode, OptimWrapper
)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from .data import make_dataset, SKLearnEncoder
from .TransMIL import TransMIL


__all__ = ['train', 'deploy']


T = TypeVar('T')


def train(
    *,
    bags: Sequence[Iterable[Path]],
    targets: Tuple[SKLearnEncoder, np.ndarray],
    add_features: Iterable[Tuple[SKLearnEncoder, Sequence[Any]]] = [],
    valid_idxs: np.ndarray,
    n_epoch: int = 32,
    patience: int = 8,
    path: Optional[Path] = None,
    batch_size: int = 64,
    cores: int = 8,
    plot: bool = False
) -> Learner:
    """Train a MLP on image features.

    Args:
        bags:  H5s containing bags of tiles.
        targets:  An (encoder, targets) pair.
        add_features:  An (encoder, targets) pair for each additional input.
        valid_idxs:  Indices of the datasets to use for validation.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == "cuda":
        # allow for usage of TensorFloat32 as internal dtype for matmul on modern NVIDIA GPUs
        torch.set_float32_matmul_precision("medium")

    target_enc, targs = targets
    train_ds = make_dataset(
        bags=bags[~valid_idxs],
        targets=(target_enc, targs[~valid_idxs]),
        add_features=[
            (enc, vals[~valid_idxs])
            for enc, vals in add_features],
        bag_size=512)

    valid_ds = make_dataset(
        bags=bags[valid_idxs],
        targets=(target_enc, targs[valid_idxs]),
        add_features=[
            (enc, vals[valid_idxs])
            for enc, vals in add_features],
        bag_size=None)
    
    # build dataloaders
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=cores,
        drop_last=len(train_ds) > batch_size,
        device=device, pin_memory=device.type == "cuda"
    )
    valid_dl = DataLoader(
        valid_ds, batch_size=1, shuffle=False, num_workers=cores,
        device=device, pin_memory=device.type == "cuda"
    )
    batch = train_dl.one_batch()
    feature_dim = batch[0].shape[-1]

    # for binary classification num_classes=2
    model = TransMIL(
        num_classes=len(target_enc.categories_[0]), input_dim=feature_dim,
        dim=512, depth=2, heads=8, mlp_dim=512, dropout=.0
    )
    # TODO:
    # maybe increase mlp_dim? Not necessary 4*dim, but maybe a bit?
    # maybe add at least some dropout?
    
    # model = torch.compile(model)
    model.to(device)
    print(f"Model: {model}", end=" ")
    print(f"[Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}]")

    # weigh inversely to class occurrences
    counts = pd.Series(targs[~valid_idxs]).value_counts()
    weight = counts.sum() / counts
    weight /= weight.sum()
    # reorder according to vocab
    weight = torch.tensor(
        list(map(weight.get, target_enc.categories_[0])), dtype=torch.float32, device=device)
    loss_func = nn.CrossEntropyLoss(weight=weight)

    dls = DataLoaders(train_dl, valid_dl, device=device)

    learn = Learner(
        dls,
        model,
        loss_func=loss_func,
        opt_func = partial(OptimWrapper, opt=torch.optim.AdamW),
        metrics=[RocAuc()],
        path=path,
    )#.to_bf16()

    cbs = [
        SaveModelCallback(monitor='valid_loss', fname=f'best_valid'),
        EarlyStoppingCallback(monitor='valid_loss', patience=patience),
        CSVLogger(),
        # MixedPrecision(amp_mode=AMPMode.BF16)
    ]
    learn.fit_one_cycle(n_epoch=n_epoch, reset_opt=True, lr_max=1e-4, wd=1e-2, cbs=cbs)
    
    # Plot training and validation losses as well as learning rate schedule
    if plot:
        path_plots = path / "plots"
        path_plots.mkdir(parents=True, exist_ok=True)

        learn.recorder.plot_loss()
        plt.savefig(path_plots / 'losses_plot.png')
        plt.close()

        learn.recorder.plot_sched()
        plt.savefig(path_plots / 'lr_scheduler.png')
        plt.close()

    return learn


def deploy(
    test_df: pd.DataFrame, learn: Learner, *,
    target_label: Optional[str] = None,
    cat_labels: Optional[Sequence[str]] = None, cont_labels: Optional[Sequence[str]] = None,
    device: torch.device = torch.device('cpu')
) -> pd.DataFrame:
    assert test_df.PATIENT.nunique() == len(test_df), 'duplicate patients!'
    #assert (len(add_label)
    #        == (n := len(learn.dls.train.dataset._datasets[-2]._datasets))), \
    #    f'not enough additional feature labels: expected {n}, got {len(add_label)}'
    if target_label is None: target_label = learn.target_label
    if cat_labels is None: cat_labels = learn.cat_labels
    if cont_labels is None: cont_labels = learn.cont_labels

    target_enc = learn.dls.dataset._datasets[-1].encode
    categories = target_enc.categories_[0]
    add_features = []
    if cat_labels:
        cat_enc = learn.dls.dataset._datasets[-2]._datasets[0].encode
        add_features.append((cat_enc, test_df[cat_labels].values))
    if cont_labels:
        cont_enc = learn.dls.dataset._datasets[-2]._datasets[-1].encode
        add_features.append((cont_enc, test_df[cont_labels].values))

    test_ds = make_dataset(
        bags=test_df.slide_path.values,
        targets=(target_enc, test_df[target_label].values),
        add_features=add_features,
        bag_size=None)

    test_dl = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=1,
        device=device, pin_memory=device.type == "cuda")

    #removed softmax in forward, but add here to get 0-1 probabilities
    patient_preds, patient_targs = learn.get_preds(dl=test_dl, act=nn.Softmax(dim=1))

    # make into DF w/ ground truth
    patient_preds_df = pd.DataFrame.from_dict({
        'PATIENT': test_df.PATIENT.values,
        target_label: test_df[target_label].values,
        **{f'{target_label}_{cat}': patient_preds[:, i]
            for i, cat in enumerate(categories)}})

    # calculate loss
    patient_preds = patient_preds_df[[
        f'{target_label}_{cat}' for cat in categories]].values
    patient_targs = target_enc.transform(
        patient_preds_df[target_label].values.reshape(-1, 1))
    patient_preds_df['loss'] = F.cross_entropy(
        torch.tensor(patient_preds), torch.tensor(patient_targs),
        reduction='none')

    patient_preds_df['pred'] = categories[patient_preds.argmax(1)]

    # reorder dataframe and sort by loss (best predictions first)
    patient_preds_df = patient_preds_df[[
        'PATIENT',
        target_label,
        'pred',
        *(f'{target_label}_{cat}' for cat in categories),
        'loss']]
    patient_preds_df = patient_preds_df.sort_values(by='loss')

    return patient_preds_df
