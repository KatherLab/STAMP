# %%
from pathlib import Path

from stamp.config import StampConfig
from stamp.heatmaps.config import HeatmapConfig
from stamp.modeling.config import (
    AdvancedConfig,
    CrossvalConfig,
    DeploymentConfig,
    MlpModelParams,
    ModelParams,
    TrainConfig,
    VitModelParams,
)
from stamp.preprocessing.config import (
    ExtractorName,
    Microns,
    PreprocessingConfig,
    SlideMPP,
    TilePixels,
)
from stamp.statistics import StatsConfig


def test_config_parsing() -> None:
    config = StampConfig.model_validate(
        {
            "crossval": {
                "categories": None,
                "clini_table": "clini.xlsx",
                "feature_dir": "CRC",
                "filename_label": "FILENAME",
                "ground_truth_label": "isMSIH",
                "output_dir": "test-crossval",
                "patient_label": "PATIENT",
                "slide_table": "slide.csv",
                "use_vary_precision_transform": False,
                "n_splits": 5,
            },
            "deployment": {
                "checkpoint_paths": [
                    "test-crossval/split-0/model.ckpt",
                    "test-crossval/split-1/model.ckpt",
                    "test-crossval/split-2/model.ckpt",
                    "test-crossval/split-3/model.ckpt",
                    "test-crossval/split-4/model.ckpt",
                ],
                "clini_table": "clini.xlsx",
                "feature_dir": "CRC",
                "filename_label": "FILENAME",
                "ground_truth_label": "isMSIH",
                "output_dir": "test-deploy",
                "patient_label": "PATIENT",
                "slide_table": "slide.csv",
            },
            "heatmaps": {
                "bottomk": 5,
                "checkpoint_path": "test-train/model.ckpt",
                "device": "cuda",
                "feature_dir": "feats",
                "output_dir": "test-heatmaps",
                "slide_paths": None,
                "topk": 5,
                "wsi_dir": "wsis",
                "default_slide_mpp": 1.0,
            },
            "preprocessing": {
                "brightness_cutoff": 240,
                "cache_dir": "cache",
                "canny_cutoff": 0.02,
                "device": "cuda",
                "extractor": "ctranspath",
                "max_workers": 8,
                "output_dir": "bla",
                "tile_size_px": 224,
                "tile_size_um": 256.0,
                "wsi_dir": "wsis",
                "default_slide_mpp": 1.0,
            },
            "statistics": {
                "ground_truth_label": "isMSIH",
                "output_dir": "test-stats",
                "pred_csvs": [
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-0/patient-preds.csv",
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-1/patient-preds.csv",
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-2/patient-preds.csv",
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-3/patient-preds.csv",
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-4/patient-preds.csv",
                ],
                "true_class": "MSIH",
            },
            "training": {
                "categories": None,
                "clini_table": "clini.xlsx",
                "feature_dir": "CRC",
                "filename_label": "FILENAME",
                "ground_truth_label": "isMSIH",
                "output_dir": "test-alibi",
                "patient_label": "PATIENT",
                "slide_table": "slide.csv",
                "use_vary_precision_transform": False,
            },
            "advanced_config": {
                "bag_size": 512,
                "num_workers": 16,
                "batch_size": 64,
                "max_epochs": 64,
                "patience": 16,
                "accelerator": "gpu",
                "model_params": {
                    "vit": {
                        "dim_model": 512,
                        "dim_feedforward": 512,
                        "n_heads": 8,
                        "n_layers": 2,
                        "dropout": 0.25,
                        "use_alibi": True,
                    },
                    "mlp": {
                        "dim_hidden": 512,
                        "num_layers": 2,
                        "dropout": 0.25,
                    },
                },
            },
        }
    )

    assert config == StampConfig(
        preprocessing=PreprocessingConfig(
            output_dir=Path("bla"),
            wsi_dir=Path("wsis"),
            cache_dir=Path("cache"),
            tile_size_um=Microns(256.0),
            tile_size_px=TilePixels(224),
            extractor=ExtractorName.CTRANSPATH,
            max_workers=8,
            device="cuda",
            brightness_cutoff=240,
            canny_cutoff=0.02,
            default_slide_mpp=SlideMPP(1.0),
        ),
        training=TrainConfig(
            output_dir=Path("test-alibi"),
            clini_table=Path("clini.xlsx"),
            slide_table=Path("slide.csv"),
            feature_dir=Path("CRC"),
            ground_truth_label="isMSIH",
            categories=None,
            patient_label="PATIENT",
            filename_label="FILENAME",
            params_path=None,
            use_vary_precision_transform=False,
        ),
        crossval=CrossvalConfig(
            output_dir=Path("test-crossval"),
            clini_table=Path("clini.xlsx"),
            slide_table=Path("slide.csv"),
            feature_dir=Path("CRC"),
            ground_truth_label="isMSIH",
            categories=None,
            patient_label="PATIENT",
            filename_label="FILENAME",
            params_path=None,
            use_vary_precision_transform=False,
            n_splits=5,
        ),
        deployment=DeploymentConfig(
            output_dir=Path("test-deploy"),
            checkpoint_paths=[
                Path("test-crossval/split-0/model.ckpt"),
                Path("test-crossval/split-1/model.ckpt"),
                Path("test-crossval/split-2/model.ckpt"),
                Path("test-crossval/split-3/model.ckpt"),
                Path("test-crossval/split-4/model.ckpt"),
            ],
            clini_table=Path("clini.xlsx"),
            slide_table=Path("slide.csv"),
            feature_dir=Path("CRC"),
            ground_truth_label="isMSIH",
            patient_label="PATIENT",
            filename_label="FILENAME",
        ),
        statistics=StatsConfig(
            output_dir=Path("test-stats"),
            pred_csvs=[
                Path(
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-0/patient-preds.csv"
                ),
                Path(
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-1/patient-preds.csv"
                ),
                Path(
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-2/patient-preds.csv"
                ),
                Path(
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-3/patient-preds.csv"
                ),
                Path(
                    "/mnt/bulk-neptune/mvantreeck/stamp2/test-crossval/split-4/patient-preds.csv"
                ),
            ],
            ground_truth_label="isMSIH",
            true_class="MSIH",
        ),
        heatmaps=HeatmapConfig(
            output_dir=Path("test-heatmaps"),
            feature_dir=Path("feats"),
            wsi_dir=Path("wsis"),
            checkpoint_path=Path("test-train/model.ckpt"),
            slide_paths=None,
            device="cuda",
            topk=5,
            bottomk=5,
            default_slide_mpp=SlideMPP(1.0),
        ),
        advanced_config=AdvancedConfig(
            bag_size=512,
            num_workers=16,
            batch_size=64,
            max_epochs=64,
            patience=16,
            accelerator="gpu",
            model_params=ModelParams(
                vit=VitModelParams(
                    dim_model=512,
                    dim_feedforward=512,
                    n_heads=8,
                    n_layers=2,
                    dropout=0.25,
                    use_alibi=True,
                ),
                mlp=MlpModelParams(
                    dim_hidden=512,
                    num_layers=2,
                    dropout=0.25,
                ),
            ),
        ),
    )
