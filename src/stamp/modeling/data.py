"""Helper classes to manage pytorch data."""

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from itertools import groupby
from pathlib import Path
from typing import IO, BinaryIO, Counter, Generic, TextIO, TypeAlias, Union, cast

import h5py
import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from packaging.version import Version
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

import stamp
from stamp.seed import Seed
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

_logger = logging.getLogger("stamp")
_logged_stamp_v1_warning = False


__author__ = "Marko van Treeck, Minh Duc Nguyen"
__copyright__ = "Copyright (C) 2022-2025 Marko van Treeck, Minh Duc Nguyen"
__license__ = "MIT"

_Bag: TypeAlias = Float[Tensor, "tile feature"]
_EncodedTarget: TypeAlias = Float[Tensor, "category_is_hot"] | Float[Tensor, "1"]  # noqa: F821
_BinaryIOLike: TypeAlias = Union[BinaryIO, IO[bytes]]
"""The ground truth, encoded numerically
- classification: one-hot float [C]
- regression: float [1]
"""
_Coordinates: TypeAlias = Float[Tensor, "tile 2"]


@dataclass
class PatientData(Generic[GroundTruthType]):
    """All raw (i.e. non-generated) information we have on the patient."""

    _ = KW_ONLY
    ground_truth: GroundTruthType
    feature_files: Iterable[FeaturePath | BinaryIO]


def tile_bag_dataloader(
    *,
    patient_data: Sequence[PatientData[GroundTruth | None]],
    bag_size: int | None,
    task: Task,
    categories: Sequence[Category] | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform: Callable[[Tensor], Tensor] | None,
) -> tuple[
    DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
    Sequence[Category],
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
    if task == "classification":
        raw_ground_truths = np.array([patient.ground_truth for patient in patient_data])
        categories = (
            categories if categories is not None else list(np.unique(raw_ground_truths))
        )
        # one_hot = torch.tensor(raw_ground_truths.reshape(-1, 1) == categories)
        one_hot = torch.tensor(
            raw_ground_truths.reshape(-1, 1) == categories, dtype=torch.float32
        )
        ds = BagDataset(
            bags=[patient.feature_files for patient in patient_data],
            bag_size=bag_size,
            ground_truths=one_hot,
            transform=transform,
        )
        cats_out: Sequence[Category] = list(categories)

    elif task == "regression":
        raw_targets = np.array(
            [
                np.nan if p.ground_truth is None else float(p.ground_truth)
                for p in patient_data
            ],
            dtype=np.float32,
        )
        y = torch.from_numpy(raw_targets).reshape(-1, 1)

        ds = BagDataset(
            bags=[patient.feature_files for patient in patient_data],
            bag_size=bag_size,
            ground_truths=y,
            transform=transform,
        )
        cats_out = []

    elif task == "survival":  # Not yet support logistic-harzard
        times: list[float] = []
        events: list[float] = []

        for p in patient_data:
            if p.ground_truth is None:
                times.append(np.nan)
                events.append(np.nan)
                continue

            try:
                time_str, status_str = p.ground_truth.split(" ", 1)

                # Handle missing values encoded as "nan"
                if time_str.lower() == "nan":
                    times.append(np.nan)
                else:
                    times.append(float(time_str))

                if status_str.lower() == "nan":
                    events.append(np.nan)
                elif status_str.lower() in {"dead", "event", "1", "Yes", "yes"}:
                    events.append(1.0)
                elif status_str.lower() in {"alive", "censored", "0", "No", "no"}:
                    events.append(0.0)
                else:
                    events.append(np.nan)  # unknown status → mark missing

            except Exception:
                times.append(np.nan)
                events.append(np.nan)

        # Final tensor shape: (N, 2)
        y = torch.tensor(np.column_stack([times, events]), dtype=torch.float32)

        ds = BagDataset(
            bags=[patient.feature_files for patient in patient_data],
            bag_size=bag_size,
            ground_truths=y,
            transform=transform,
        )
        cats_out: Sequence[Category] = []  # survival has no categories

    else:
        raise ValueError(f"Unknown task: {task}")

    return (
        cast(
            DataLoader[tuple[Bags, CoordinatesBatch, BagSizes, EncodedTargets]],
            DataLoader(
                ds,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                collate_fn=_collate_to_tuple,
                worker_init_fn=Seed.get_loader_worker_init()
                if Seed._is_set()
                else None,
            ),
        ),
        cats_out,
    )


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
    patient_data: Sequence[PatientData[GroundTruth | None]],
    bag_size: int | None = None,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    transform: Callable[[Tensor], Tensor] | None,
    categories: Sequence[Category] | None = None,
) -> tuple[DataLoader, Sequence[Category]]:
    """Unified dataloader for all feature types and tasks."""
    if feature_type == "tile":
        return tile_bag_dataloader(
            patient_data=patient_data,
            bag_size=bag_size,
            task=task,
            categories=categories,
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
            categories = categories or list(np.unique(raw))
            labels = torch.tensor(raw.reshape(-1, 1) == categories, dtype=torch.float32)
        elif task == "regression":
            labels = torch.tensor(
                [
                    float(gt)
                    for gt in (p.ground_truth for p in patient_data)
                    if gt is not None
                ],
                dtype=torch.float32,
            ).reshape(-1, 1)
        elif task == "survival":
            times, events = [], []
            for p in patient_data:
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
    ground_truth_label: PandasLabel | None = None,
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

    ground_truths: Float[Tensor, "index category_is_hot"] | Float[Tensor, "index 1"]

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
    # --- NEW: handle missing coords ----multiplex data bypass: no coords found; generated fake coords
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
        # TODO: find a better way to get this warning just once
        global _logged_stamp_v1_warning
        if not _logged_stamp_v1_warning:
            _logger.info(
                f"{feature_h5.filename}: tile stride is roughly 224, assuming coordinates have unit 256um/224px (historic STAMP format)"
            )
            _logged_stamp_v1_warning = True
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
    ground_truth_label: PandasLabel,
) -> dict[PatientId, GroundTruth]:
    """Loads the patients and their ground truths from a clini table."""
    clini_df = read_table(
        clini_table_path,
        usecols=[patient_label, ground_truth_label],
        dtype=str,
    ).dropna()
    try:
        patient_to_ground_truth: Mapping[PatientId, GroundTruth] = clini_df.set_index(
            patient_label, verify_integrity=True
        )[ground_truth_label].to_dict()
    except KeyError as e:
        if patient_label not in clini_df:
            raise ValueError(
                f"{patient_label} was not found in clini table "
                f"(columns in clini table: {clini_df.columns})"
            ) from e
        elif ground_truth_label not in clini_df:
            raise ValueError(
                f"{ground_truth_label} was not found in clini table "
                f"(columns in clini table: {clini_df.columns})"
            ) from e
        else:
            raise e from e

    return patient_to_ground_truth


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
    -------
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
    patient_to_ground_truth: Mapping[PatientId, GroundTruth | None],
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


def get_stride(coords: Float[Tensor, "tile 2"]) -> float:
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

    s = str(value).strip().lower()

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


def log_patient_class_summary(
    *,
    patient_to_data: Mapping[PatientId, PatientData],
    categories: Sequence[Category] | None,
    prefix: str = "",
) -> None:
    ground_truths = [
        pd.ground_truth
        for pd in patient_to_data.values()
        if pd.ground_truth is not None
    ]

    if not ground_truths:
        _logger.warning(f"{prefix}No ground truths available to summarize.")
        return

    cats = categories or sorted(set(ground_truths))
    counter = Counter(ground_truths)

    _logger.info(
        f"{prefix}Total patients: {len(ground_truths)} | "
        + " | ".join([f"Class {c}: {counter.get(c, 0)}" for c in cats])
    )
