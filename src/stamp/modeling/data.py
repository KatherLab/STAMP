"""Helper classes to manage pytorch data."""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from itertools import groupby
from pathlib import Path
from typing import (
    IO,
    Any,
    BinaryIO,
    Dict,
    Final,
    Generic,
    List,
    TextIO,
    TypeAlias,
    Union,
    cast,
)

import h5py
import numpy as np
import pandas as pd
import torch
from packaging.version import Version
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import stamp
from stamp.types import (
    Bags,
    BagSize,
    BagSizes,
    Category,
    CoordinatesBatch,
    EncodedTargets,
    FeaturePath,
    GroundTruth,
    GroundTruthType,
    Microns,
    PandasLabel,
    PatientId,
    SlideMPP,
    Task,
    TilePixels,
)
from stamp.utils.seed import Seed

_logger = logging.getLogger("stamp")


__author__ = "Marko van Treeck, Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck, Minh Duc Nguyen"
__license__ = "MIT"

_Bag: TypeAlias = Tensor
_EncodedTarget: TypeAlias = (
    Tensor | dict[str, Tensor]
)  # Union of encoded targets or multi-target dict
_BinaryIOLike: TypeAlias = Union[BinaryIO, IO[bytes]]
"""The ground truth, encoded numerically
- classification: one-hot float [C]
- regression: float [1]
- multi-target: dict[target_name -> one-hot/regression value]
"""
_Coordinates: TypeAlias = Tensor


@dataclass
class PatientData(Generic[GroundTruthType]):
    """All raw (i.e. non-generated) information we have on the patient."""

    _ = KW_ONLY
    ground_truth: GroundTruthType
    feature_files: Iterable[FeaturePath | BinaryIO]


def tile_bag_dataloader(
    *,
    patient_data: Sequence[PatientData[GroundTruth | None | dict]],
    bag_size: int | None,
    task: Task,
    categories: Sequence[Category] | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform: Callable[[Tensor], Tensor] | None,
) -> tuple[
    DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
    Sequence[Category] | Mapping[str, Sequence[Category]],
]:
    """Creates a dataloader from patient data for tile-level (bagged) features.

    Args:
        task='classification':
            categories:
                Order of classes for one-hot encoding.
                If `None`, classes are inferred from patient data.
        task='regression':
            returns float targets
    """

    targets, cats_out = _parse_targets(
        patient_data=patient_data,
        task=task,
        categories=categories,
    )

    is_multitarget = isinstance(targets[0], dict)

    collate_fn = _collate_multitarget if is_multitarget else _collate_to_tuple

    ds = BagDataset(
        bags=[patient.feature_files for patient in patient_data],
        bag_size=bag_size,
        ground_truths=targets,
        transform=transform,
    )
    dl = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=Seed.get_loader_worker_init() if Seed._is_set() else None,
    )

    return (
        cast(
            DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
            dl,
        ),
        cats_out,
    )


def _parse_targets(
    *,
    patient_data: Sequence,
    task: Task,
    categories: Sequence[Category] | None = None,
    target_spec: dict[str, Any] | None = None,
    target_label: str | None = None,
) -> tuple[
    Union[torch.Tensor, list[dict[str, torch.Tensor]]],
    Sequence[Category] | Mapping[str, Sequence[Category]],
]:
    """
    Parse raw GroundTruth (str) into model-ready tensors.
    This is the ONLY place task semantics live.
    """

    gts = [p.ground_truth for p in patient_data]

    if task == "classification":
        if any(isinstance(gt, dict) for gt in gts if gt is not None):
            # infer target names from the first non-None dict
            first_dict = next(gt for gt in gts if isinstance(gt, dict))
            target_names = list(first_dict.keys())

            # infer categories per target (ignore None patients, ignore None values)
            categories_out: dict[str, list[str]] = {t: [] for t in target_names}
            for gt in gts:
                if not isinstance(gt, dict):
                    continue
                for t in target_names:
                    v = gt.get(t)
                    if v is not None:
                        categories_out[t].append(v)

            # make unique + sorted
            categories_out = {
                t: sorted(set(vals)) for t, vals in categories_out.items()
            }

            # encode per patient; if gt missing -> all zeros
            encoded: list[dict[str, Tensor]] = []
            for gt in gts:
                patient_encoded: dict[str, Tensor] = {}
                for t in target_names:
                    cats = categories_out[t]
                    if not isinstance(gt, dict) or gt.get(t) is None:
                        one_hot = torch.zeros(len(cats), dtype=torch.float32)
                    else:
                        one_hot = torch.tensor(
                            [gt[t] == c for c in cats],
                            dtype=torch.float32,
                        )
                    patient_encoded[t] = one_hot
                encoded.append(patient_encoded)

            # IMPORTANT: return categories as mapping, not list-of-target-names
            return encoded, categories_out

        # single target
        unique = {gt for gt in gts if gt is not None}
        if len(unique) >= 2 or categories is not None:
            raw = np.array([p.ground_truth for p in patient_data])
            categories = categories or list(sorted(unique))
            labels = torch.tensor(
                raw.reshape(-1, 1) == categories,
                dtype=torch.float32,
            )
            return labels, categories

        raise ValueError(
            "Only one unique class found in classification task. "
            "This is usually a data or configuration error."
        )

    elif task == "regression":
        y = torch.tensor(
            [np.nan if gt is None else float(gt) for gt in gts],
            dtype=torch.float32,
        ).reshape(-1, 1)
        return y, []

    elif task == "survival":
        times, events = [], []
        for gt in gts:
            if gt is None:
                times.append(np.nan)
                events.append(np.nan)
                continue

            time_str, status_str = gt.split(" ", 1)
            times.append(np.nan if time_str.lower() == "nan" else float(time_str))
            events.append(_parse_survival_status(status_str))

        y = torch.tensor(np.column_stack([times, events]), dtype=torch.float32)
        return y, []

    else:
        raise ValueError(f"Unsupported task: {task}")


def _collate_to_tuple(
    items: list[tuple[_Bag, _Coordinates, BagSize, _EncodedTarget]],
) -> tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]:
    bags = torch.stack([bag for bag, _, _, _ in items])
    coords = torch.stack([coord for _, coord, _, _ in items])
    bag_sizes = torch.tensor([bagsize for _, _, bagsize, _ in items])

    targets = [et for _, _, _, et in items]

    # Normalize target shapes
    fixed_targets = []
    for et in targets:
        et = torch.as_tensor(et)
        if et.ndim == 0:  # scalar → (1,)
            et = et.unsqueeze(0)
        elif et.ndim > 1:  # e.g. (1,2) → (2,)
            et = et.view(-1)
        fixed_targets.append(et)

    # Stack into (B, D)
    encoded_targets = torch.stack(fixed_targets)

    return (bags, coords, bag_sizes, encoded_targets)


def _collate_multitarget(
    items: list[tuple[_Bag, _Coordinates, BagSize, Dict[str, Tensor]]],
) -> tuple[Bags, CoordinatesBatch, BagSizes, Dict[str, Tensor]]:
    bags = torch.stack([b for b, _, _, _ in items])
    coords = torch.stack([c for _, c, _, _ in items])
    bag_sizes = torch.tensor([s for _, _, s, _ in items])

    acc: Dict[str, List[Tensor]] = {}

    for _, _, _, tdict in items:
        for k, v in tdict.items():
            acc.setdefault(k, []).append(v)

    targets: Dict[str, Tensor] = {k: torch.stack(v, dim=0) for k, v in acc.items()}

    return bags, coords, bag_sizes, targets


def patient_feature_dataloader(
    *,
    patient_data: Sequence[PatientData[GroundTruth | None]],
    categories: Sequence[Category] | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform: Callable[[Tensor], Tensor] | None,
) -> tuple[DataLoader, Sequence[Category]]:
    """
    Creates a dataloader for patient-level features (one feature vector per patient).
    """
    feature_files = [next(iter(p.feature_files)) for p in patient_data]
    raw_ground_truths = np.array([patient.ground_truth for patient in patient_data])
    categories = (
        categories if categories is not None else list(np.unique(raw_ground_truths))
    )
    one_hot = torch.tensor(raw_ground_truths.reshape(-1, 1) == categories)
    ds = PatientFeatureDataset(feature_files, one_hot, transform=transform)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dl, categories


def create_dataloader(
    *,
    feature_type: str,
    task: Task,
    patient_data: Sequence[PatientData[GroundTruth | None | dict]],
    bag_size: int | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform: Callable[[Tensor], Tensor] | None,
    categories: Sequence[Category] | Mapping[str, Sequence[Category]] | None = None,
) -> tuple[DataLoader, Sequence[Category] | Mapping[str, Sequence[Category]]]:
    """Unified dataloader for all feature types and tasks."""
    if feature_type == "tile":
        # For multi-target classification, categories may be a mapping from
        # target name to per-target categories. _parse_targets (used inside
        # tile_bag_dataloader) only consumes explicit categories for the
        # single-target case, so we pass a sequence or None here.
        cats_arg: Sequence[Category] | None
        if isinstance(categories, Mapping):
            cats_arg = None
        else:
            cats_arg = categories

        return tile_bag_dataloader(
            patient_data=patient_data,
            bag_size=bag_size,
            task=task,
            categories=cats_arg,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            transform=transform,
        )
    elif feature_type in {"slide", "patient"}:
        # For slide/patient-level: single feature vector per entry
        feature_files = [next(iter(p.feature_files)) for p in patient_data]

        if task == "classification":
            raw = np.array([p.ground_truth for p in patient_data])
            categories_out = categories or list(np.unique(raw))
            labels = torch.tensor(
                raw.reshape(-1, 1) == categories_out, dtype=torch.float32
            )
        elif task == "regression":
            values: list[float] = []
            for gt in (p.ground_truth for p in patient_data):
                if gt is None:
                    continue
                if isinstance(gt, dict):
                    # Use first value for multi-target regression
                    first_val = next(iter(gt.values()))
                    values.append(float(first_val))
                else:
                    values.append(float(gt))

            labels = torch.tensor(values, dtype=torch.float32).reshape(-1, 1)
        elif task == "survival":
            times, events = [], []
            for p in patient_data:
                if isinstance(p.ground_truth, dict):
                    # Multi-target survival: use first target
                    val = list(p.ground_truth.values())[0]
                    t, e = (val or "nan nan").split(" ", 1)
                else:
                    t, e = (p.ground_truth or "nan nan").split(" ", 1)
                times.append(float(t) if t.lower() != "nan" else np.nan)
                events.append(_parse_survival_status(e))

            labels = torch.tensor(np.column_stack([times, events]), dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported task: {task}")

        ds = PatientFeatureDataset(feature_files, labels, transform)
        dl = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            worker_init_fn=Seed.get_loader_worker_init() if Seed._is_set() else None,
        )
        return dl, categories or []
    else:
        raise ValueError(f"Unknown feature type: {feature_type}")


def detect_feature_type(feature_dir: Path) -> str:
    """
    Detects feature type by inspecting all .h5 files in feature_dir.

    Returns:
        "tile" if all files are tile-level, "patient" if all are patient-level.
        If files have mixed types, raises an error.
        If no .h5 files are found, raises an error.
    """
    feature_types = set()
    files_checked = 0

    for file in feature_dir.glob("*.h5"):
        files_checked += 1
        with h5py.File(file, "r") as h5:
            feat_type = h5.attrs.get("feat_type")
            encoder = h5.attrs.get("encoder")

            if feat_type is not None or encoder is not None:
                feature_types.add(str(feat_type))
            else:
                # If feat_type is missing, always treat as tile-level feature
                feature_types.add("tile")

    if files_checked == 0:
        raise RuntimeError("No .h5 feature files found in feature_dir.")

    if len(feature_types) > 1:
        raise RuntimeError(
            f"Multiple feature types detected in {feature_dir}: {feature_types}. "
            "All feature files must have the same type."
        )

    return feature_types.pop()


def load_patient_level_data(
    *,
    task: Task | None,
    clini_table: Path,
    feature_dir: Path,
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel | Sequence[PandasLabel] | None = None,
    time_label: PandasLabel | None = None,
    status_label: PandasLabel | None = None,
    feature_ext: str = ".h5",
) -> dict[PatientId, PatientData]:
    """
    Loads PatientData for patient-level features, matching patients in the clinical table
    to feature files in feature_dir named {patient_id}.h5.

    Supports:
        - classification / regression via `ground_truth_label`
        - survival via `time_label` + `status_label` (stored as "time status")
    """

    # Load ground truth mapping
    if task == "survival" and time_label is not None and status_label is not None:
        # Survival: use the existing helper
        patient_to_ground_truth = patient_to_survival_from_clini_table_(
            clini_table_path=clini_table,
            patient_label=patient_label,
            time_label=time_label,
            status_label=status_label,
        )
    elif task in ["classification", "regression"] and ground_truth_label is not None:
        # Classification or regression
        patient_to_ground_truth = patient_to_ground_truth_from_clini_table_(
            clini_table_path=clini_table,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
        )
    else:
        raise ValueError(
            "You must provide either `ground_truth_label` "
            "for classification/regression or (`time_label`, `status_label`) for survival when using tile-level or slide-level features."
        )

    # Build PatientData entries
    patient_to_data: dict[PatientId, PatientData] = {}
    missing_features = []
    for pid, gt in patient_to_ground_truth.items():
        feature_file = feature_dir / f"{pid}{feature_ext}"
        if feature_file.exists():
            patient_to_data[pid] = PatientData(
                ground_truth=gt,
                feature_files=[FeaturePath(feature_file)],
            )
        else:
            missing_features.append(pid)

    if missing_features:
        _logger.warning(
            f"Some patients have no feature file in {feature_dir}: {missing_features}"
        )

    return patient_to_data


@dataclass
class BagDataset(Dataset[tuple[_Bag, _Coordinates, BagSize, _EncodedTarget]]):
    """A dataset of bags of instances."""

    _: KW_ONLY
    bags: Sequence[Iterable[FeaturePath | _BinaryIOLike]]
    """The `.h5` files containing the bags.

    Each bag consists of the features taken from one or multiple h5 files.
    Each of the h5 files needs to have a dataset called `feats` of shape N x F,
    where N is the number of instances and F the number of features per instance.
    """

    bag_size: BagSize | None = None
    """The number of instances in each bag.

    For bags containing more instances,
    a random sample of `bag_size` instances will be drawn.
    Smaller bags are padded with zeros.
    If `bag_size` is None, all the samples will be used.
    """

    ground_truths: Tensor | list[dict[str, Tensor]]

    # ground_truths: Bool[Tensor, "index category_is_hot"]
    # """The ground truth for each bag, one-hot encoded."""

    transform: Callable[[Tensor], Tensor] | None

    def __post_init__(self) -> None:
        if len(self.bags) != len(self.ground_truths):
            raise ValueError(
                "the number of ground truths has to match the number of bags"
            )

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(
        self, index: int
    ) -> tuple[_Bag, _Coordinates, BagSize, _EncodedTarget]:
        # Collect all the features
        feats = []
        coords_um = []
        for bag_file in self.bags[index]:
            with h5py.File(bag_file, "r") as h5:
                if "feats" in h5:
                    arr = h5["feats"][:]  # pyright: ignore[reportIndexIssue] # original STAMP files
                else:
                    arr = h5["patch_embeddings"][:]  # type: ignore # your Kronos files

                feats.append(torch.from_numpy(arr))
                coords_um.append(torch.from_numpy(get_coords(h5).coords_um))

        feats = torch.concat(feats).float()
        coords_um = torch.concat(coords_um).float()

        if self.transform is not None:
            feats = self.transform(feats)

        # Sample a subset, if required
        if self.bag_size is not None:
            return (
                *_to_fixed_size_bag(feats, coords=coords_um, bag_size=self.bag_size),
                self.ground_truths[index],
            )
        else:
            return (
                feats,
                coords_um,
                len(feats),
                self.ground_truths[index],
            )


class PatientFeatureDataset(Dataset):
    """
    Dataset for single feature vector per sample (e.g. slide-level or patient-level).
    Each item is a (feature_vector, label_onehot) tuple.
    """

    def __init__(
        self,
        feature_files: Sequence[FeaturePath | BinaryIO],
        ground_truths: Tensor,  # shape: [num_samples, num_classes]
        transform: Callable[[Tensor], Tensor] | None,
    ):
        if len(feature_files) != len(ground_truths):
            raise ValueError("Number of feature files and ground truths must match.")
        self.feature_files = feature_files
        self.ground_truths = ground_truths
        self.transform = transform

    def __len__(self):
        return len(self.feature_files)

    def __getitem__(self, idx: int):
        feature_file = self.feature_files[idx]
        with h5py.File(feature_file, "r") as h5:
            feats = torch.from_numpy(h5["feats"][:])  # pyright: ignore[reportIndexIssue]
            # Accept [V] or [1, V]
            if feats.ndim == 2 and feats.shape[0] == 1:
                feats = feats[0]
            elif feats.ndim == 1:
                pass
            else:
                raise RuntimeError(
                    f"Expected single feature vector (shape [F] or [1, F]), got {feats.shape} in {feature_file}."
                    "Check that the features are patient-level."
                )
            if self.transform is not None:
                feats = self.transform(feats)
        label = self.ground_truths[idx]
        return feats, label


@dataclass
class CoordsInfo:
    coords_um: np.ndarray
    tile_size_um: Microns
    tile_size_px: TilePixels | None = None

    @property
    def mpp(self) -> SlideMPP:
        if not self.tile_size_px:
            raise RuntimeError(
                "tile size in pixels is not available. Please reextract them using `stamp preprocess`."
            )
        return SlideMPP(self.tile_size_um / self.tile_size_px)


def get_coords(feature_h5: h5py.File) -> CoordsInfo:
    # NEW: handle missing coords - multiplex data bypass: no coords found; generated fake coords
    if "coords" not in feature_h5:
        feats_obj = feature_h5["patch_embeddings"]

        if not isinstance(feats_obj, h5py.Dataset):
            raise RuntimeError(
                f"{feature_h5.filename}: expected 'feats' to be an HDF5 dataset but got {type(feats_obj)}"
            )

        n = feats_obj.shape[0]

        coords_um = np.stack([np.arange(n), np.zeros(n)], axis=1).astype(np.float32)
        tile_size_um = Microns(0.0)
        tile_size_px = TilePixels(0)

        return CoordsInfo(coords_um, tile_size_um, tile_size_px)
    coords: np.ndarray = feature_h5["coords"][:]  # type: ignore
    coords_um: np.ndarray | None = None
    tile_size_um: Microns | None = None
    tile_size_px: TilePixels | None = None
    if (tile_size := feature_h5.attrs.get("tile_size", None)) and feature_h5.attrs.get(
        "unit", None
    ) == "um":
        # STAMP v2 format
        tile_size_um = Microns(float(tile_size))
        coords_um = coords
    elif tile_size := feature_h5.attrs.get("tile_size_um", None):
        # Newer STAMP format
        tile_size_um = Microns(float(tile_size))
        coords_um = coords
    elif (
        round(
            feature_h5.attrs.get(
                "tile_size", get_stride(torch.from_numpy(coords).float())
            )
        )
        == 224
    ):
        # Historic STAMP format
        _logger.debug(
            f"{feature_h5.filename}: tile stride is roughly 224, assuming coordinates have unit 256um/224px (historic STAMP format)"
        )
        tile_size_um = Microns(256.0)
        tile_size_px = TilePixels(224)
        coords_um = coords / 224 * 256

    if (version_str := feature_h5.attrs.get("stamp_version")) and (
        extraction_version := Version(version_str)
    ) > Version(stamp.__version__):
        raise RuntimeError(
            f"features were extracted with a newer version of stamp, please update your stamp to at least version {extraction_version}."
        )

    if not tile_size_px and "tile_size_px" in feature_h5.attrs:
        tile_size_px = TilePixels(int(feature_h5.attrs["tile_size_px"]))  # pyright: ignore[reportArgumentType]

    if not tile_size_um or coords_um is None:
        raise RuntimeError(
            "unable to infer coordinates from feature file. Please reextract them using `stamp preprocess`."
        )

    return CoordsInfo(coords_um, tile_size_um, tile_size_px)


def _to_fixed_size_bag(
    bag: _Bag, coords: _Coordinates, bag_size: BagSize
) -> tuple[_Bag, _Coordinates, BagSize]:
    """Samples a fixed-size bag of tiles from an arbitrary one.

    If the original bag did not have enough tiles,
    the bag is zero-padded to the right.
    """
    # get up to bag_size elements
    n_tiles, _dim_feats = bag.shape
    bag_idxs = torch.randperm(n_tiles)[:bag_size]
    bag_samples = bag[bag_idxs]
    coord_samples = coords[bag_idxs]

    # zero-pad if we don't have enough samples
    zero_padded_bag = torch.cat(
        (
            bag_samples,
            torch.zeros(bag_size - bag_samples.shape[0], bag_samples.shape[1]),
        )
    )
    zero_padded_coord = torch.cat(
        (
            coord_samples,
            torch.zeros(bag_size - coord_samples.shape[0], coord_samples.shape[1]),
        )
    )
    return zero_padded_bag, zero_padded_coord, min(bag_size, len(bag))


def patient_to_ground_truth_from_clini_table_(
    *,
    clini_table_path: Path | TextIO,
    patient_label: PandasLabel,
    ground_truth_label: PandasLabel | Sequence[PandasLabel],
) -> (
    dict[PatientId, GroundTruth | None] | dict[PatientId, dict[str, GroundTruth | None]]
):
    """Loads the patients and their ground truths from a clini table.

    `ground_truth_label` may be either a single column name (str) or a sequence
    of column names. In the latter case the returned mapping will contain a
    dict mapping column -> value for each patient (supporting multi-target
    setups).
    """
    # Normalize to list for uniform handling
    if isinstance(ground_truth_label, str):
        cols = [patient_label, ground_truth_label]
        multi = False
        target_cols_inner: list[PandasLabel] = []
    else:
        cols = [patient_label, *list(ground_truth_label)]
        multi = True
        target_cols_inner = [c for c in cols if c != patient_label]

    clini_df = read_table(
        clini_table_path,
        usecols=cols,
        dtype=str,
    )

    # If multi-target, keep rows where at least one target is present; for
    # single target behave like before and drop rows missing the value.
    if multi:
        clini_df = clini_df.dropna(subset=target_cols_inner, how="all")
    else:
        clini_df = clini_df.dropna(subset=[ground_truth_label])

    try:
        if multi:
            # Build mapping patient -> {col: value}
            result: dict[PatientId, dict[str, GroundTruth | None]] = {}
            for _, row in clini_df.iterrows():
                pid = row[patient_label]
                # Convert pandas nan to None and keep strings otherwise
                result[pid] = {
                    col: (None if pd.isna(row[col]) else str(row[col]))
                    for col in target_cols_inner
                }
            return result
        else:
            patient_to_ground_truth: Mapping[PatientId, str] = cast(
                Mapping[PatientId, str],
                clini_df.set_index(patient_label, verify_integrity=True)[
                    cast(PandasLabel, ground_truth_label)
                ].to_dict(),
            )
            return cast(dict[PatientId, GroundTruth | None], patient_to_ground_truth)
    except KeyError as e:
        if patient_label not in clini_df:
            raise ValueError(
                f"{patient_label} was not found in clini table "
                f"(columns in clini table: {clini_df.columns})"
            ) from e
        else:
            raise ValueError(
                f"One or more ground truth columns were not found in clini table "
                f"(columns in clini table: {clini_df.columns})"
            ) from e


def patient_to_survival_from_clini_table_(
    *,
    clini_table_path: Path | TextIO,
    patient_label: PandasLabel,
    time_label: PandasLabel,
    status_label: PandasLabel,
) -> dict[PatientId, GroundTruth]:
    """
    Loads patients and their survival ground truths (time + event) from a clini table.

    Returns
    dict[PatientId, GroundTruth]
        Mapping patient_id -> "time status" (e.g. "302 dead", "476 alive").
    """
    clini_df = read_table(
        clini_table_path,
        usecols=[patient_label, time_label, status_label],
        dtype=str,
    )

    # normalize values
    clini_df[time_label] = clini_df[time_label].replace(
        [
            "NA",
            "NaN",
            "nan",
            "None",
            "none",
            "N/A",
            "n/a",
            "NULL",
            "null",
            "",
            " ",
            "?",
            "-",
            "--",
            "#N/A",
            "#NA",
            "=#VALUE!",
        ],
        np.nan,
    )
    clini_df[status_label] = clini_df[status_label].str.strip().str.lower()

    # Only drop rows where BOTH time and status are missing
    clini_df = clini_df.dropna(subset=[time_label, status_label], how="all")

    patient_to_ground_truth: dict[PatientId, GroundTruth] = {}
    for _, row in clini_df.iterrows():
        pid = row[patient_label]
        time_str = row[time_label]
        status_str = row[status_label]

        # Skip patients missing survival time
        if pd.isna(time_str):
            continue

        # Encode status: keep both dead (event=1) and alive (event=0)
        status = _parse_survival_status(status_str)

        # Encode back to "alive"/"dead" like before
        # status = "dead" if status_val == 1 else "alive"

        patient_to_ground_truth[pid] = f"{time_str} {status}"

    return patient_to_ground_truth


def slide_to_patient_from_slide_table_(
    *,
    slide_table_path: Path,
    feature_dir: Path,
    patient_label: PandasLabel,
    filename_label: PandasLabel,
) -> dict[FeaturePath, PatientId]:
    """
    Creates a slide-to-patient mapping from a slide table.
    Side effects:
        Verifies that all files in the slide tables filename_label
        column has an .h5 extension.
    """
    slide_df = read_table(
        slide_table_path,
        usecols=[patient_label, filename_label],
        dtype=str,
    )

    # Verify the slide table contains a feature path with .h5 extension by
    # checking the filename_label.
    for x in slide_df[filename_label]:
        if not str(x).endswith(".h5"):
            raise ValueError(
                "One or more files are missing the .h5 extension in the "
                "filename_label column. The first file missing the .h5 "
                "extension is: " + str(x) + "."
            )

    slide_to_patient: Mapping[FeaturePath, PatientId] = {
        FeaturePath(feature_dir / cast(str, k)): PatientId(cast(str, patient))
        for k, patient in slide_df.set_index(filename_label, verify_integrity=True)[
            patient_label
        ].items()
    }

    return slide_to_patient


def read_table(path: Path | TextIO, **kwargs) -> pd.DataFrame:
    if not isinstance(path, Path):
        return pd.read_csv(path, **kwargs)
    elif path.suffix == ".xlsx":
        return pd.read_excel(path, **kwargs)
    elif path.suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    else:
        raise ValueError(
            "table to load has to either be an excel (`*.xlsx`) or csv (`*.csv`) file."
        )


def filter_complete_patient_data_(
    *,
    patient_to_ground_truth: Mapping[
        PatientId, GroundTruth | dict[str, GroundTruth] | None
    ],
    slide_to_patient: Mapping[FeaturePath, PatientId],
    drop_patients_with_missing_ground_truth: bool,
) -> Mapping[PatientId, PatientData]:
    """Aggregate information for all patients for which we have complete data.

    This will sort out slides with missing ground truth, missing features, etc.
    Patients with their ground truth set explicitly set to `None` will be _included_.

    Side effects:
        Checks feature paths' existance.
    """

    _log_patient_slide_feature_inconsistencies(
        patient_to_ground_truth=patient_to_ground_truth,
        slide_to_patient=slide_to_patient,
    )

    patient_to_slides: dict[PatientId, set[FeaturePath]] = {
        patient: set(slides)
        for patient, slides in groupby(
            slide_to_patient, lambda slide: slide_to_patient[slide]
        )
    }

    if not drop_patients_with_missing_ground_truth:
        patient_to_ground_truth = {
            **{patient_id: None for patient_id in patient_to_slides},
            **patient_to_ground_truth,
        }

    patients = {
        patient_id: PatientData(
            ground_truth=ground_truth, feature_files=existing_features_for_patient
        )
        for patient_id, ground_truth in patient_to_ground_truth.items()
        # Restrict to only patients which have slides and features
        if (slides := patient_to_slides.get(patient_id)) is not None
        and (
            existing_features_for_patient := {
                feature_path for feature_path in slides if feature_path.exists()
            }
        )
    }

    _logger.info(
        f"Total patients in clinical table: {len(patient_to_ground_truth)}\n"
        f"Patients appearing in slide table: {len(patient_to_slides)}\n"
        f"Final usable patients (complete data): {len(patients)}\n"
    )
    return patients


def _log_patient_slide_feature_inconsistencies(
    *,
    patient_to_ground_truth: Mapping[PatientId, GroundTruthType],
    slide_to_patient: Mapping[FeaturePath, PatientId],
) -> None:
    """Checks whether the arguments are consistent and logs all irregularities.

    Has no side effects outside of logging.
    """
    if (
        patients_without_slides := patient_to_ground_truth.keys()
        - slide_to_patient.values()
    ):
        _logger.warning(
            f"some patients have no associated slides: {patients_without_slides}"
        )

    if patients_without_ground_truth := (
        slide_to_patient.values() - patient_to_ground_truth.keys()
    ):
        _logger.warning(
            f"some patients have no clinical information: {patients_without_ground_truth}"
        )

    if slides_without_features := {
        slide for slide in slide_to_patient.keys() if not slide.exists()
    }:
        _logger.warning(
            f"some feature files could not be found: {slides_without_features}"
        )


def get_stride(coords: Tensor) -> float:
    """Gets the minimum step width between any two coordintes."""
    xs: Tensor = coords[:, 0].unique(sorted=True)
    ys: Tensor = coords[:, 1].unique(sorted=True)
    stride = cast(
        float,
        min(
            (xs[1:] - xs[:-1]).min().item(),
            (ys[1:] - ys[:-1]).min().item(),
        ),
    )
    return stride


def _parse_survival_status(value) -> int | None:
    """
    Parse a survival status value (string, numeric, or None) into a binary indicator.
    Currently assume no None inputs.
    Returns:
        1 -> event/dead
        0 -> censored/alive
        None -> missing (None, NaN, '')

    Raises:
        ValueError if the input is non-missing but unrecognized.

    Examples:
        'dead', '1', 'event', 'yes'  -> 1
        'alive', '0', 'censored', 'no' -> 0
        None, NaN, '' -> None
    """

    # Handle missing inputs gracefully
    # if value is None:
    #     return 0  # treat empty/missing as censored
    # if isinstance(value, float) and math.isnan(value):
    #     return 0  # treat empty/missing as censored

    s = str(value).strip().lower()
    # if s in {"", "nan", "none"}:
    #     return 0  # treat empty/missing as censored

    # Known mappings
    positives = {"1", "event", "dead", "deceased", "yes", "y", "True", "true"}
    negatives = {"0", "alive", "censored", "no", "false"}

    if s in positives:
        return 1
    elif s in negatives:
        return 0

    # Try numeric fallback
    try:
        f = float(s)
        return 1 if f > 0 else 0
    except ValueError:
        raise ValueError(
            f"Unrecognized survival status: '{value}'. "
            f"Expected one of {sorted(positives | negatives)} or a numeric value."
        )


def load_patient_data_(
    *,
    feature_dir: Path,
    clini_table: Path,
    slide_table: Path | None,
    task: Task,
    ground_truth_label: PandasLabel | Sequence[PandasLabel] | None,
    time_label: PandasLabel | None,
    status_label: PandasLabel | None,
    patient_label: PandasLabel,
    filename_label: PandasLabel,
    drop_patients_with_missing_ground_truth: bool = True,
) -> tuple[Mapping[PatientId, PatientData], str]:
    """Load patient data based on feature type (tile, slide, or patient).

    This consolidates the common data loading logic used across train, crossval, and deploy.

    Returns:
        (patient_to_data, feature_type)
    """
    feature_type = detect_feature_type(feature_dir)

    if feature_type in ("tile", "slide"):
        if slide_table is None:
            raise ValueError("A slide table is required for tile/slide-level features")

        # Load ground truth based on task
        if task == "survival":
            if time_label is None or status_label is None:
                raise ValueError(
                    "Both time_label and status_label are required for survival modeling"
                )
            patient_to_ground_truth = patient_to_survival_from_clini_table_(
                clini_table_path=clini_table,
                time_label=time_label,
                status_label=status_label,
                patient_label=patient_label,
            )
        else:
            if ground_truth_label is None:
                raise ValueError(
                    "Ground truth label is required for classification or regression modeling"
                )
            patient_to_ground_truth = patient_to_ground_truth_from_clini_table_(
                clini_table_path=clini_table,
                ground_truth_label=ground_truth_label,
                patient_label=patient_label,
            )

        # Link slides to patients
        slide_to_patient: Final[dict[FeaturePath, PatientId]] = (
            slide_to_patient_from_slide_table_(
                slide_table_path=slide_table,
                feature_dir=feature_dir,
                patient_label=patient_label,
                filename_label=filename_label,
            )
        )

        # Filter to complete patient data
        patient_to_data = filter_complete_patient_data_(
            patient_to_ground_truth=cast(
                Mapping[PatientId, GroundTruth | dict[str, GroundTruth] | None],
                patient_to_ground_truth,
            ),
            slide_to_patient=slide_to_patient,
            drop_patients_with_missing_ground_truth=drop_patients_with_missing_ground_truth,
        )
    elif feature_type == "patient":
        patient_to_data = load_patient_level_data(
            task=task,
            clini_table=clini_table,
            feature_dir=feature_dir,
            patient_label=patient_label,
            ground_truth_label=ground_truth_label,
            time_label=time_label,
            status_label=status_label,
        )
    else:
        raise RuntimeError(f"Unknown feature type: {feature_type}")

    return patient_to_data, feature_type


def log_patient_class_summary(
    *,
    patient_to_data: Mapping[PatientId, PatientData],
    categories: Sequence[Category] | None,
) -> None:
    """
    Logs class distribution.
    Supports both single-target and multi-target classification.
    """

    ground_truths = [
        p.ground_truth for p in patient_to_data.values() if p.ground_truth is not None
    ]

    if not ground_truths:
        _logger.warning("No ground truths available for summary.")
        return

    # Multi-target
    if isinstance(ground_truths[0], dict):
        # Collect per-target values
        per_target: dict[str, list] = {}

        for gt in ground_truths:
            for key, value in gt.items():
                per_target.setdefault(key, []).append(value)

        for target_name, values in per_target.items():
            counts = {}
            for v in values:
                counts[v] = counts.get(v, 0) + 1

            _logger.info(
                f"[Multi-target] Target '{target_name}' distribution: {counts}"
            )

    # Single-target
    else:
        counts = {}
        for gt in ground_truths:
            counts[gt] = counts.get(gt, 0) + 1

        _logger.info(f"Class distribution: {counts}")
