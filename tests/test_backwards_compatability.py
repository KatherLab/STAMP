# %%
import hashlib
import os
import shutil
import urllib.request
from pathlib import Path

import pytest
import torch

from stamp.modeling.data import FeaturePath, PatientData
from stamp.modeling.deploy import _predict
from stamp.modeling.lightning_model import LitVisionTransformer

_STAMP_CACHE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")) / "stamp"
)


def _download_file(*, url: str, file_name: str, sha256sum: str) -> Path:
    outfile_path = _STAMP_CACHE_DIR / file_name
    if (outfile_path).is_file():
        with open(outfile_path, "rb") as weight_file:
            digest = hashlib.file_digest(weight_file, "sha256")
        assert digest.hexdigest() == sha256sum, (
            f"{outfile_path} has the wrong checksum. Try deleting it and rerunning this script."
        )
    else:
        filename, _ = urllib.request.urlretrieve(url)
        with open(filename, "rb") as weight_file:
            digest = hashlib.file_digest(weight_file, "sha256")
        assert digest.hexdigest() == sha256sum, "hash of downloaded file did not match"
        shutil.copy(filename, outfile_path)

    return outfile_path


@pytest.mark.filterwarnings(
    "ignore:The 'predict_dataloader' does not have many workers"
)
def test_backwards_compatability() -> None:
    example_model_path = _download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0-dev8/example-model.ckpt",
        file_name="example-model.ckpt",
        sha256sum="a71dffd4b5fdb82acd5f84064880efd3382e200b07e5a008cb53e03197b6de56",
    )
    example_feature_path = _download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0-dev8/TCGA-AA-3860-01Z-00-DX1.a63df9ca-6141-4bdc-8545-719fd9ae0aa5.h5",
        file_name="TCGA-AA-3860-01Z-00-DX1.a63df9ca-6141-4bdc-8545-719fd9ae0aa5.h5",
        sha256sum="c180f6029ca1defbe0f5a972e9333848a28245f1ca79e343c7ff06c4804a12f7",
    )

    model = LitVisionTransformer.load_from_checkpoint(example_model_path)

    predictions = _predict(
        model=model,
        patient_to_data={
            "TestPatient": PatientData(
                ground_truth=None,
                feature_files=[FeaturePath(example_feature_path)],
            )
        },
        num_workers=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    assert torch.allclose(
        predictions["TestPatient"], torch.tensor([0.0331, 0.9669]), atol=1e-4
    ), f"prediction does not match that of stamp {model.hparams['stamp_version']}"
