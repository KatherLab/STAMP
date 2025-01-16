import pytest
import torch

from stamp.cache import download_file
from stamp.modeling.data import FeaturePath, PatientData
from stamp.modeling.deploy import _predict
from stamp.modeling.lightning_model import LitVisionTransformer


@pytest.mark.filterwarnings(
    "ignore:The 'predict_dataloader' does not have many workers"
)
def test_backwards_compatability() -> None:
    example_checkpoint_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0-dev8/example-model.ckpt",
        file_name="example-model.ckpt",
        sha256sum="a71dffd4b5fdb82acd5f84064880efd3382e200b07e5a008cb53e03197b6de56",
    )
    example_feature_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.0.0-dev8/TCGA-AA-3860-01Z-00-DX1.a63df9ca-6141-4bdc-8545-719fd9ae0aa5.h5",
        file_name="TCGA-AA-3860-01Z-00-DX1.a63df9ca-6141-4bdc-8545-719fd9ae0aa5.h5",
        sha256sum="c180f6029ca1defbe0f5a972e9333848a28245f1ca79e343c7ff06c4804a12f7",
    )

    model = LitVisionTransformer.load_from_checkpoint(example_checkpoint_path)

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
