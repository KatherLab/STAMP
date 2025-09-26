import pytest
import torch

from stamp.cache import download_file
from stamp.modeling.data import PatientData, tile_bag_dataloader
from stamp.modeling.deploy import _predict
from stamp.modeling.models.vision_tranformer import LitVisionTransformer
from stamp.seed import Seed
from stamp.types import FeaturePath, PatientId


@pytest.mark.filterwarnings(
    "ignore:The 'predict_dataloader' does not have many workers"
)
def test_backwards_compatibility() -> None:
    Seed.set(42)
    example_checkpoint_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.2.0/example-model-v2_3_0.ckpt",
        file_name="example-modelv2_3_0.ckpt",
        sha256sum="eb6225fcdea7f33dee80fd5dc4e7a0da6cd0d91a758e3ee9605d6869b30ab657",
    )
    example_feature_path = download_file(
        url="https://github.com/KatherLab/STAMP/releases/download/2.2.0/TCGA-AA-3877-01Z-00-DX1.36902310-bc0b-4437-9f86-6df85703e0ad.h5",
        file_name="TCGA-AA-3877-01Z-00-DX1.36902310-bc0b-4437-9f86-6df85703e0ad.h5",
        sha256sum="9ee5172c205c15d55eb9a8b99e98319c1a75b7fdd6adde7a3ae042d3c991285e",
    )

    model = LitVisionTransformer.load_from_checkpoint(example_checkpoint_path)

    # Prepare PatientData and DataLoader for the test patient
    patient_id = PatientId("TestPatient")
    patient_to_data = {
        patient_id: PatientData(
            ground_truth=None,
            feature_files=[FeaturePath(example_feature_path)],
        )
    }
    test_dl, _ = tile_bag_dataloader(
        task="classification",
        patient_data=list(patient_to_data.values()),
        bag_size=None,
        categories=list(model.categories),
        batch_size=1,
        shuffle=False,
        num_workers=1,
        transform=None,
    )

    predictions = _predict(
        model=model,
        test_dl=test_dl,
        patient_ids=[patient_id],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
    )

    assert torch.allclose(
        predictions["TestPatient"], torch.tensor([0.0083, 0.9917]), atol=1e-4
    ), f"prediction does not match that of stamp {model.hparams['stamp_version']}"
