from pathlib import Path

import pytest
import torch
from test_train_deploy import (
    # prevent rerunning test_train_deploy_integration
    test_train_deploy_integration as _test_train_deploy_integration,
)

from stamp.modeling.transforms import vary_precision


def test_vary_precision_transform() -> None:
    raw = torch.rand(10000) * 2**10 - 2**9  # Random numbers between -2^9 and 2^9
    for min_fracton_bits in range(1, 10):
        augmented = vary_precision(raw, min_fraction_bits=min_fracton_bits)
        assert (raw.sign() == augmented.sign()).all()
        assert (
            (raw.abs().log2() - augmented.abs().log2()) < (2 ** (-min_fracton_bits))
        ).all()


@pytest.mark.slow
@pytest.mark.filterwarnings("ignore:No positive samples in targets")
def test_varying_precision_integration(tmp_path: Path) -> None:
    _test_train_deploy_integration(
        tmp_path=tmp_path,
        use_vary_precision_transform=True,
    )
