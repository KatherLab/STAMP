import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from random_data import random_patient_preds, random_string

from stamp.statistics import compute_stats_


def test_statistics_integration(
    *,
    n_patient_preds: int = 1,
    n_categories: int = 3,
) -> None:
    """Just check if we can compute stats without crashing"""
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    categories = [random_string(8) for _ in range(n_categories)]

    with tempfile.TemporaryDirectory(prefix="stamp_test_statistics_") as tmp_dir:
        dir = Path(tmp_dir)
        for patient_preds_i in range(n_patient_preds):
            random_patient_preds(
                n_patients=random.randint(100, 1000), categories=categories
            ).to_csv(dir / f"patient-preds-{patient_preds_i}.csv")

        compute_stats_(
            output_dir=dir / "output",
            pred_csvs=[dir / f"patient-preds-{i}.csv" for i in range(n_patient_preds)],
            ground_truth_label="ground_truth",
            true_class=categories[1],
        )


def test_statistics_integration_for_multiple_patient_preds() -> None:
    return test_statistics_integration(n_patient_preds=5)
