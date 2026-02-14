import argparse
import logging
import shutil
import sys
from pathlib import Path

import yaml

from stamp.config import StampConfig
from stamp.modeling.config import (
    AdvancedConfig,
    MlpModelParams,
    ModelParams,
    VitModelParams,
)
from stamp.seed import Seed

STAMP_FACTORY_SETTINGS = Path(__file__).with_name("config.yaml")

# Set up the logger
_logger = logging.getLogger("stamp")
_logger.setLevel(logging.DEBUG)
_formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

_stream_handler = logging.StreamHandler(sys.stderr)
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(_formatter)
_logger.addHandler(_stream_handler)


def _create_config_file(config_file: Path) -> None:
    """Create a new config file at the specified path (by copying the default config file)."""
    if not config_file.exists():
        # Copy original config file
        shutil.copy(STAMP_FACTORY_SETTINGS, config_file)
        _logger.info(f"Created new config file at {config_file.absolute()}")
    else:
        _logger.info(
            f"Refusing to overwrite existing config file at {config_file.absolute()}"
        )


def _run_cli(args: argparse.Namespace) -> None:
    # Handle init command
    if args.command == "init":
        _create_config_file(args.config_file_path)
        return

    # Load YAML configuration
    with open(args.config_file_path, "r") as config_yaml:
        config = StampConfig.model_validate(yaml.safe_load(config_yaml))

    # use default advanced config in case none is provided
    if config.advanced_config is None:
        config.advanced_config = AdvancedConfig(
            model_params=ModelParams(vit=VitModelParams(), mlp=MlpModelParams()),
        )

    # Set global random seed
    if config.advanced_config.seed is not None:
        Seed.set(config.advanced_config.seed)

    match args.command:
        case "init":
            raise RuntimeError("this case should be handled above")

        case "config":
            print(yaml.dump(config.model_dump(mode="json", exclude_none=True)))

        case "preprocess":
            from stamp.preprocessing import extract_

            if config.preprocessing is None:
                raise ValueError("no preprocessing configuration supplied")

            _add_file_handle_(_logger, output_dir=config.preprocessing.output_dir)
            _logger.info(
                "using the following configuration:\n"
                f"{yaml.dump(config.preprocessing.model_dump(mode='json', exclude_none=True))}"
            )
            extract_(
                output_dir=config.preprocessing.output_dir,
                wsi_dir=config.preprocessing.wsi_dir,
                wsi_list=config.preprocessing.wsi_list,
                mpp_list=config.preprocessing.mpp_list,
                cache_dir=config.preprocessing.cache_dir,
                tile_size_um=config.preprocessing.tile_size_um,
                tile_size_px=config.preprocessing.tile_size_px,
                extractor=config.preprocessing.extractor,
                max_workers=config.preprocessing.max_workers,
                device=config.preprocessing.device,
                default_slide_mpp=config.preprocessing.default_slide_mpp,
                brightness_cutoff=config.preprocessing.brightness_cutoff,
                canny_cutoff=config.preprocessing.canny_cutoff,
                cache_tiles_ext=config.preprocessing.cache_tiles_ext,
                generate_hash=config.preprocessing.generate_hash,
            )

        case "encode_slides":
            from stamp.encoding import init_slide_encoder_

            if config.slide_encoding is None:
                raise ValueError("no slide encoding configuration supplied")

            _add_file_handle_(_logger, output_dir=config.slide_encoding.output_dir)
            _logger.info(
                "using the following configuration:\n"
                f"{yaml.dump(config.slide_encoding.model_dump(mode='json', exclude_none=True))}"
            )
            init_slide_encoder_(
                encoder=config.slide_encoding.encoder,
                output_dir=config.slide_encoding.output_dir,
                feat_dir=config.slide_encoding.feat_dir,
                device=config.slide_encoding.device,
                agg_feat_dir=config.slide_encoding.agg_feat_dir,
                generate_hash=config.slide_encoding.generate_hash,
            )

        case "encode_patients":
            from stamp.encoding import init_patient_encoder_

            if config.patient_encoding is None:
                raise ValueError("no patient encoding configuration supplied")

            _add_file_handle_(_logger, output_dir=config.patient_encoding.output_dir)
            _logger.info(
                "using the following configuration:\n"
                f"{yaml.dump(config.patient_encoding.model_dump(mode='json', exclude_none=True))}"
            )
            init_patient_encoder_(
                encoder=config.patient_encoding.encoder,
                output_dir=config.patient_encoding.output_dir,
                feat_dir=config.patient_encoding.feat_dir,
                slide_table_path=config.patient_encoding.slide_table,
                patient_label=config.patient_encoding.patient_label,
                filename_label=config.patient_encoding.filename_label,
                device=config.patient_encoding.device,
                agg_feat_dir=config.patient_encoding.agg_feat_dir,
                generate_hash=config.patient_encoding.generate_hash,
            )

        case "train":
            from stamp.modeling.train import train_categorical_model_

            if config.training is None:
                raise ValueError("no training configuration supplied")

            _add_file_handle_(_logger, output_dir=config.training.output_dir)
            _logger.info(
                "using the following configuration:\n"
                f"{yaml.dump(config.training.model_dump(mode='json', exclude_none=True))}"
            )

            if config.training.task is None:
                raise ValueError("task must be set in training configuration")

            train_categorical_model_(
                config=config.training, advanced=config.advanced_config
            )

        case "deploy":
            from stamp.modeling.deploy import deploy_categorical_model_

            if config.deployment is None:
                raise ValueError("no deployment configuration supplied")

            _add_file_handle_(_logger, output_dir=config.deployment.output_dir)
            _logger.info(
                "using the following configuration:\n"
                f"{yaml.dump(config.deployment.model_dump(mode='json', exclude_none=True))}"
            )
            deploy_categorical_model_(
                output_dir=config.deployment.output_dir,
                checkpoint_paths=config.deployment.checkpoint_paths,
                clini_table=config.deployment.clini_table,
                slide_table=config.deployment.slide_table,
                feature_dir=config.deployment.feature_dir,
                patient_label=config.deployment.patient_label,
                filename_label=config.deployment.filename_label,
                num_workers=config.deployment.num_workers,
                accelerator=config.deployment.accelerator,
                ground_truth_label=config.deployment.ground_truth_label,
                time_label=config.deployment.time_label,
                status_label=config.deployment.status_label,
            )

        case "crossval":
            from stamp.modeling.crossval import categorical_crossval_

            if config.crossval is None:
                raise ValueError("no crossval configuration supplied")

            if config.crossval.task is None:
                raise ValueError("task must be set in crossval configuration")

            _add_file_handle_(_logger, output_dir=config.crossval.output_dir)
            _logger.info(
                "using the following configuration:\n"
                f"{yaml.dump(config.crossval.model_dump(mode='json', exclude_none=True))}"
            )

            categorical_crossval_(
                config=config.crossval,
                advanced=config.advanced_config,
            )

        case "statistics":
            from stamp.statistics import compute_stats_

            if config.statistics is None:
                raise ValueError("no statistics configuration supplied")

            _add_file_handle_(_logger, output_dir=config.statistics.output_dir)
            _logger.info(
                "using the following configuration:\n"
                f"{yaml.dump(config.statistics.model_dump(mode='json', exclude_none=True))}"
            )

            compute_stats_(
                task=config.statistics.task,
                output_dir=config.statistics.output_dir,
                pred_csvs=config.statistics.pred_csvs,
                ground_truth_label=config.statistics.ground_truth_label,
                true_class=config.statistics.true_class,
                time_label=config.statistics.time_label,
                status_label=config.statistics.status_label,
            )

        case "heatmaps":
            from stamp.heatmaps import heatmaps_

            if config.heatmaps is None:
                raise ValueError("no heatmaps configuration supplied")

            _add_file_handle_(_logger, output_dir=config.heatmaps.output_dir)
            _logger.info(
                "using the following configuration:\n"
                f"{yaml.dump(config.heatmaps.model_dump(mode='json'))}"
            )
            heatmaps_(
                feature_dir=config.heatmaps.feature_dir,
                wsi_dir=config.heatmaps.wsi_dir,
                checkpoint_path=config.heatmaps.checkpoint_path,
                output_dir=config.heatmaps.output_dir,
                slide_paths=config.heatmaps.slide_paths,
                device=config.heatmaps.device,
                topk=config.heatmaps.topk,
                bottomk=config.heatmaps.bottomk,
                default_slide_mpp=config.heatmaps.default_slide_mpp,
                opacity=config.heatmaps.opacity,
            )

        case _:
            raise RuntimeError(
                "unreachable: the argparser should only allow valid commands"
            )


def _add_file_handle_(logger: logging.Logger, *, output_dir: Path) -> None:
    output_dir.mkdir(exist_ok=True, parents=True)

    file_handler = logging.FileHandler(output_dir / "logfile.log")
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="stamp", description="STAMP: Solid Tumor Associative Modeling in Pathology"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        dest="config_file_path",
        default=Path("config.yaml"),
        help="Path to config file. Default: config.yaml",
    )

    commands = parser.add_subparsers(dest="command")
    commands.add_parser(
        "init",
        help="Create a new STAMP configuration file at the path specified by --config",
    )
    commands.add_parser(
        "preprocess", help="Preprocess whole-slide images into feature vectors"
    )
    commands.add_parser(
        "encode_slides",
        help="Encode patch-level features into slide-level embeddings",
    )
    commands.add_parser(
        "encode_patients",
        help="Encode features into patient-level embeddings",
    )
    commands.add_parser("train", help="Train a Vision Transformer model")
    commands.add_parser(
        "crossval",
        help="Train a Vision Transformer model with cross validation for modeling.n_splits folds",
    )
    commands.add_parser("deploy", help="Deploy a trained Vision Transformer model")
    commands.add_parser(
        "statistics",
        help="Generate AUROCs and AUPRCs with 95%%CI for a trained Vision Transformer model",
    )
    commands.add_parser("config", help="Print the loaded configuration")
    commands.add_parser("heatmaps", help="Generate heatmaps for a trained model")

    args = parser.parse_args()

    # If no command is given, print help and exit
    if args.command is None:
        parser.print_help()
        exit(1)

    # Run the CLI
    try:
        _run_cli(args)
    except Exception as e:
        _logger.exception(e)
        exit(1)


if __name__ == "__main__":
    main()
