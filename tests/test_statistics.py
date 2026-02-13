import random
from pathlib import Path

import numpy as np
import pandas as pd
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


def test_statistics_survival_integration(
    *,
    tmp_path: Path,
    n_folds: int = 1,
    n_patients: int = 200,
) -> None:
    """Check that survival statistics run without crashing."""
    random.seed(0)
    np.random.seed(0)

    for fold_i in range(n_folds):
        times = np.random.uniform(30, 2000, size=n_patients)
        statuses = np.random.choice([0, 1], size=n_patients, p=[0.3, 0.7])
        risks = np.random.randn(n_patients)
        df = pd.DataFrame(
            {
                "patient": [random_string(8) for _ in range(n_patients)],
                "day": times,
                "status": statuses,
                "pred_score": risks,
            }
        )
        df.to_csv(tmp_path / f"survival-preds-{fold_i}.csv", index=False)

    compute_stats_(
        task="survival",
        output_dir=tmp_path / "output",
        pred_csvs=[tmp_path / f"survival-preds-{i}.csv" for i in range(n_folds)],
        time_label="day",
        status_label="status",
    )

    assert (tmp_path / "output" / "survival-stats_individual.csv").is_file()


def test_statistics_survival_integration_multiple_folds(tmp_path: Path) -> None:
    return test_statistics_survival_integration(tmp_path=tmp_path, n_folds=5)


def test_statistics_regression_integration(
    *,
    tmp_path: Path,
    n_folds: int = 1,
    n_patients: int = 200,
) -> None:
    """Check that regression statistics run without crashing."""
    random.seed(0)
    np.random.seed(0)

    for fold_i in range(n_folds):
        y_true = np.random.uniform(0, 100, size=n_patients)
        y_pred = y_true + np.random.randn(n_patients) * 10  # noisy predictions
        df = pd.DataFrame(
            {
                "patient": [random_string(8) for _ in range(n_patients)],
                "target": y_true,
                "pred": y_pred,
            }
        )
        df.to_csv(tmp_path / f"regression-preds-{fold_i}.csv", index=False)

    compute_stats_(
        task="regression",
        output_dir=tmp_path / "output",
        pred_csvs=[tmp_path / f"regression-preds-{i}.csv" for i in range(n_folds)],
        ground_truth_label="target",
    )

    assert (tmp_path / "output" / "target_regression-stats_individual.csv").is_file()
    assert (tmp_path / "output" / "target_regression-stats_aggregated.csv").is_file()


def test_statistics_regression_integration_multiple_folds(tmp_path: Path) -> None:
    return test_statistics_regression_integration(tmp_path=tmp_path, n_folds=5)


def test_statistics_multi_target_classification_integration(
    *,
    tmp_path: Path,
    n_patient_preds: int = 1,
) -> None:
    """Check that multi-target classification statistics run without crashing.

    Multi-target predictions produce separate ground-truth columns per target.
    We run compute_stats_ once per target, as the statistics pipeline handles
    one target at a time.
    """
    random.seed(0)
    np.random.seed(0)
    torch.random.manual_seed(0)

    categories_per_target = {"subtype": ["A", "B"], "grade": ["1", "2", "3"]}

    for pred_i in range(n_patient_preds):
        n_patients = random.randint(100, 500)
        data: dict[str, list] = {
            "patient": [random_string(8) for _ in range(n_patients)],
        }

        for target_label, cats in categories_per_target.items():
            data[target_label] = [random.choice(cats) for _ in range(n_patients)]
            probs = torch.softmax(torch.rand(len(cats), n_patients), dim=0)
            for j, cat in enumerate(cats):
                data[f"{target_label}_{cat}"] = probs[j].tolist()

        pd.DataFrame(data).to_csv(
            tmp_path / f"multi-target-preds-{pred_i}.csv", index=False
        )

    # Run statistics per target (as the pipeline would do)
    for target_label, cats in categories_per_target.items():
        true_class = cats[0]
        compute_stats_(
            task="classification",
            output_dir=tmp_path / "output" / target_label,
            pred_csvs=[
                tmp_path / f"multi-target-preds-{i}.csv" for i in range(n_patient_preds)
            ],
            ground_truth_label=target_label,
            true_class=true_class,
        )

        assert (
            tmp_path
            / "output"
            / target_label
            / f"{target_label}_categorical-stats_aggregated.csv"
        ).is_file()
        assert (
            tmp_path
            / "output"
            / target_label
            / f"roc-curve_{target_label}={true_class}.svg"
        ).is_file()
        assert (
            tmp_path
            / "output"
            / target_label
            / f"pr-curve_{target_label}={true_class}.svg"
        ).is_file()


def test_statistics_multi_target_classification_multiple_preds(
    tmp_path: Path,
) -> None:
    return test_statistics_multi_target_classification_integration(
        tmp_path=tmp_path, n_patient_preds=3
    )
