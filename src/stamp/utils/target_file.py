"""Automatically generate target information from clini table

# The `barspoon-targets 2.0` File Format

A barspoon target file is a [TOML][1] file with the following entries:

  - A `version` key mapping to a version string `"barspoon-targets <V>"`, where
    `<V>` is a [PEP-440 version string][2] compatible with `2.0`.
  - A `targets` table, the keys of which are target labels (as found in the
    clinical table) and the values specify exactly one of the following:
     1. A categorical target label, marked by the presence of a `categories`
        key-value pair.
     2. A target label to quantize, marked by the presence of a `thresholds`
        key-value pair.
     3. A target format defined in in a later version of barspoon targets.
    A target may only ever have one of the fields `categories` or `thresholds`.
    A definition of these entries can be found below.

[1]: https://toml.io "Tom's Obvious Minimal Language"
[2]: https://peps.python.org/pep-0440/
    "PEP 440 - Version Identification and Dependency Specification"

## Categorical Target Label

A categorical target is a target table with a key-value pair `categories`.
`categories` contains a list of lists of literal strings.  Each list of strings
will be treated as one category, with all literal strings within that list being
treated as one representative for that category.  This allows the user to easily
group related classes into one large class (i.e. `"True", "1", "Yes"` could all
be unified into the same category).

### Category Weights

It is possible to assign a weight to each category, to e.g. weigh rarer classes
more heavily.  The weights are stored in a table `targets.LABEL.class_weights`,
whose keys is the first representative of each category, and the values of which
is the weight of the category as a floating point number.

## Target Label to Quantize

If a target has the `thresholds` option key set, it is interpreted as a
continuous target which has to be quantized.  `thresholds` has to be a list of
floating point numbers [t_0, t_n], n > 1 containing the thresholds of the bins
to quantize the values into.  A categorical target will be quantized into bins

```asciimath
b_0 = [t_0; t_1], b_1 = (t_1; b_2], ... b_(n-1) = (t_(n-1); t_n]
```

The bins will be treated as categories with names
`f"[{t_0:+1.2e};{t_1:+1.2e}]"` for the first bin and
`f"({t_i:+1.2e};{t_(i+1):+1.2e}]"` for all other bins

To avoid confusion, we recommend to also format the `thresholds` list the same
way.

The bins can also be weighted. See _Categorical Target Label: Category Weights_
for details.

  > Experience has shown that many labels contain non-negative values with a
  > disproportionate amount (more than n_samples/n_bins) of zeroes. We thus
  > decided to make the _right_ side of each bin inclusive, as the bin (-A,0]
  > then naturally includes those zero values.
"""

import logging
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    NamedTuple,
    Optional,
    Sequence,
    TextIO,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
import torch.nn.functional as F
from packaging.specifiers import Specifier


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


__all__ = ["build_targets", "decode_targets"]


class TargetSpec(NamedTuple):
    version: str
    targets: Dict[str, Dict[str, Any]]


class EncodedTarget(NamedTuple):
    categories: List[str]
    encoded: torch.Tensor
    weight: torch.Tensor


def encode_category(
    *,
    clini_df: pd.DataFrame,
    target_label: str,
    categories: Sequence[List[str]],
    class_weights: Optional[Dict[str, float]] = None,
    **ignored,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    # Map each category to its index
    category_map = {member: idx for idx, cat in enumerate(categories) for member in cat}

    # Map each item to it's category's index, mapping nans to num_classes+1
    # This way we can easily discard the NaN column later
    indexes = clini_df[target_label].map(lambda c: category_map.get(c, len(categories)))
    indexes = torch.tensor(indexes.values)

    # Discard nan column
    one_hot = F.one_hot(indexes, num_classes=len(categories) + 1)[:, :-1]

    # Class weights
    if class_weights is not None:
        weight = torch.tensor([class_weights[c[0]] for c in categories])
    else:
        # No class weights given; use normalized inverse frequency
        counts = one_hot.sum(dim=0)
        weight = (w := (counts.sum() / counts)) / w.sum()

    # Warn user of unused labels
    if ignored:
        logging.warn(f"ignored labels in target {target_label}: {ignored}")

    return [c[0] for c in categories], one_hot, weight


def encode_quantize(
    *,
    clini_df: pd.DataFrame,
    target_label: str,
    thresholds: npt.NDArray[np.floating[Any]],
    class_weights: Optional[Dict[str, float]] = None,
    **ignored,
) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
    # Warn user of unused labels
    if ignored:
        logging.warn(f"ignored labels in target {target_label}: {ignored}")

    n_bins = len(thresholds) - 1
    numeric_vals = torch.tensor(pd.to_numeric(clini_df[target_label]).values).reshape(
        -1, 1
    )

    # Map each value to a class index as follows:
    #  1. If the value is NaN or less than the left-most threshold, use class
    #     index 0
    #  2. If it is between the left-most and the right-most threshold, set it to
    #     the bin number (starting from 1)
    #  3. If it is larger than the right-most threshold, set it to N_bins + 1
    bin_index = (
        (numeric_vals > torch.tensor(thresholds).reshape(1, -1)).count_nonzero(1)
        # For the first bucket, we have to include the lower threshold
        + (numeric_vals.reshape(-1) == thresholds[0])
    )
    # One hot encode and discard nan columns (first and last col)
    one_hot = F.one_hot(bin_index, num_classes=n_bins + 2)[:, 1:-1]

    # Class weights
    categories = [
        f"[{thresholds[0]:+1.2e};{thresholds[1]:+1.2e}]",
        *(
            f"({lower:+1.2e};{upper:+1.2e}]"
            for lower, upper in zip(thresholds[1:-1], thresholds[2:], strict=True)
        ),
    ]

    if class_weights is not None:
        weight = torch.tensor([class_weights[c] for c in categories])
    else:
        # No class weights given; use normalized inverse frequency
        counts = one_hot.sum(0)
        weight = (w := (np.divide(counts.sum(), counts, where=counts > 0))) / w.sum()

    return categories, one_hot, weight


def decode_targets(
    encoded: torch.Tensor,
    *,
    target_labels: Sequence[str],
    targets: Dict[str, Any],
    version: str = "barspoon-targets 2.0",
    **ignored,
) -> List[np.ndarray]:
    name, version = version.split(" ")
    spec = Specifier("~=2.0")

    if not (name == "barspoon-targets" and spec.contains(version)):
        raise ValueError(
            f"incompatible target file: expected barspoon-targets{spec}, found `{name} {version}`"
        )

    # Warn user of unused labels
    if ignored:
        logging.warn(f"ignored parameters: {ignored}")

    decoded_targets = []
    curr_col = 0
    for target_label in target_labels:
        info = targets[target_label]

        if (categories := info.get("categories")) is not None:
            # Add another column which is one iff all the other values are zero
            encoded_target = encoded[:, curr_col : curr_col + len(categories)]
            is_none = ~encoded_target.any(dim=1).view(-1, 1)
            encoded_target = torch.cat([encoded_target, is_none], dim=1)

            # Decode to class labels
            representatives = np.array([c[0] for c in categories] + [None])
            category_index = encoded_target.argmax(dim=1)
            decoded = representatives[category_index]
            decoded_targets.append(decoded)

            curr_col += len(categories)

        elif (thresholds := info.get("thresholds")) is not None:
            n_bins = len(thresholds) - 1
            encoded_target = encoded[:, curr_col : curr_col + n_bins]
            is_none = ~encoded_target.any(dim=1).view(-1, 1)
            encoded_target = torch.cat([encoded_target, is_none], dim=1)

            bin_edges = [-np.inf, *thresholds, np.inf]
            representatives = np.array(
                [
                    f"[{lower:+1.2e};{upper:+1.2e})"
                    for lower, upper in zip(bin_edges[:-1], bin_edges[1:])
                ]
            )
            decoded = representatives[encoded_target.argmax(dim=1)]

            decoded_targets.append(decoded)

            curr_col += n_bins

        else:
            raise ValueError(f"cannot decode {target_label}: no target info")

    return decoded_targets


def build_targets(
    *,
    clini_tables: Sequence[Path],
    categorical_labels: Sequence[str],
    category_min_count: int = 32,
    quantize: Sequence[tuple[str, int]] = (),
) -> Dict[str, EncodedTarget]:
    clini_df = pd.concat([read_table(c) for c in clini_tables])
    encoded_targets: Dict[str, EncodedTarget] = {}

    # categorical targets
    for target_label in categorical_labels:
        counts = clini_df[target_label].value_counts()
        well_supported = counts[counts >= category_min_count]

        if len(well_supported) <= 1:
            continue

        categories = [[str(cat)] for cat in well_supported.index]

        weights = well_supported.sum() / well_supported
        weights /= weights.sum()

        representatives, encoded, weight = encode_category(
            clini_df=clini_df,
            target_label=target_label,
            categories=categories,
            class_weights=weights.to_dict(),
        )

        encoded_targets[target_label] = EncodedTarget(
            categories=representatives,
            encoded=encoded,
            weight=weight,
        )

    # quantized targets
    for target_label, bincount in quantize:
        vals = pd.to_numeric(clini_df[target_label]).dropna()

        if vals.empty:
            continue

        vals_clamped = vals.replace(
            {
                -np.inf: vals[vals != -np.inf].min(),
                np.inf: vals[vals != np.inf].max(),
            }
        )

        thresholds = np.array(
            [
                -np.inf,
                *np.quantile(vals_clamped, q=np.linspace(0, 1, bincount + 1))[1:-1],
                np.inf,
            ],
            dtype=float,
        )

        representatives, encoded, weight = encode_quantize(
            clini_df=clini_df,
            target_label=target_label,
            thresholds=thresholds,
        )

        if encoded.shape[1] <= 1:
            continue

        encoded_targets[target_label] = EncodedTarget(
            categories=representatives,
            encoded=encoded,
            weight=weight,
        )

    return encoded_targets


if __name__ == "__main__":
    encoded = build_targets(
        clini_tables=[
            Path(
                "/mnt/bulk-neptune/nguyenmin/stamp-dev/experiments/survival_prediction/TCGA-CRC-DX_CLINI.xlsx"
            )
        ],
        categorical_labels=["BRAF", "KRAS", "NRAS"],
        category_min_count=32,
        quantize=[],
    )
    for name, enc in encoded.items():
        print(name, enc.encoded.shape)
