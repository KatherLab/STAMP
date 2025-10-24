import random
from pathlib import Path

import numpy as np
import torch
from random_data import random_patient_preds, random_string

from stamp.statistics import compute_stats_


def test_statistics_integration(
    *,
    tmp_path: Path,
    n_patient_preds: int = 1,
    n_categories: int = 3,
) -> None:
    """Just check if we can compute stats without crashing"""
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    categories = [random_string(8) for _ in range(n_categories)]

    for patient_preds_i in range(n_patient_preds):
        random_patient_preds(
            n_patients=random.randint(100, 1000), categories=categories
        ).to_csv(tmp_path / f"patient-preds-{patient_preds_i}.csv")

    true_class = categories[1]
    compute_stats_(
        task="classification",
        output_dir=tmp_path / "output",
        pred_csvs=[tmp_path / f"patient-preds-{i}.csv" for i in range(n_patient_preds)],
        ground_truth_label="ground-truth",
        true_class=true_class,
    )

    assert (
        tmp_path / "output" / "ground-truth_categorical-stats_aggregated.csv"
    ).is_file()
    assert (
        tmp_path / "output" / "ground-truth_categorical-stats_individual.csv"
    ).is_file()
    assert (tmp_path / "output" / f"roc-curve_ground-truth={true_class}.svg").is_file()
    assert (tmp_path / "output" / f"pr-curve_ground-truth={true_class}.svg").is_file()


def test_statistics_integration_for_multiple_patient_preds(tmp_path: Path) -> None:
    return test_statistics_integration(tmp_path=tmp_path, n_patient_preds=5)
