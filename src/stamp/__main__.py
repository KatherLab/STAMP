import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import assert_never

import numpy as np
import yaml

from stamp.config import StampConfig

STAMP_FACTORY_SETTINGS = Path(__file__).with_name("config.yaml")

# Set up the logger
logger = logging.getLogger("stamp")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


def _create_config_file(config_file: Path) -> None:
    """Create a new config file at the specified path (by copying the default config file)."""
    if not config_file.exists():
        # Copy original config file
        shutil.copy(STAMP_FACTORY_SETTINGS, config_file)
        logger.info(f"Created new config file at {config_file.absolute()}")
    else:
        logger.info(
            f"Refusing to overwrite existing config file at {config_file.absolute()}"
        )


def run_cli(args: argparse.Namespace) -> None:
    # Handle init command
    if args.command == "init":
        _create_config_file(args.config)
        return

    # Load YAML configuration
    with open(args.config_file_path, "r") as config_yaml:
        config = StampConfig.model_validate(yaml.safe_load(config_yaml))

    match args.command:
        case "init":
            assert_never(
                "this case should be handled above"  # pyright: ignore[reportArgumentType]
            )

        case "config":
            print(yaml.dump(config.model_dump(mode="json")))

        case "preprocess":
            from stamp.preprocessing.extract import extract_

            if config.preprocessing is None:
                raise ValueError("no preprocessing configuration supplied")

            _add_file_handle_(logger, output_dir=config.preprocessing.output_dir)
            extract_(**vars(config.preprocessing))

        case "train":
            from stamp.modeling.train import train_categorical_model_

            if config.training is None:
                raise ValueError("no training configuration supplied")

            _add_file_handle_(logger, output_dir=config.training.output_dir)
            # We pass every parameter explicitly so our type checker can do its work.
            train_categorical_model_(
                output_dir=config.training.output_dir,
                clini_table=config.training.clini_table,
                slide_table=config.training.slide_table,
                feature_dir=config.training.feature_dir,
                patient_label=config.training.patient_label,
                ground_truth_label=config.training.ground_truth_label,
                filename_label=config.training.filename_label,
                categories=(
                    np.array(config.training.categories)
                    if config.training.categories is not None
                    else None
                ),
                # Dataset and -loader parameters
                bag_size=config.training.bag_size,
                num_workers=config.training.num_workers,
                # Training paramenters
                batch_size=config.training.batch_size,
                max_epochs=config.training.max_epochs,
                patience=config.training.patience,
                accelerator=config.training.accelerator,
            )

        case "deploy":
            from stamp.modeling.deploy import deploy_categorical_model_

            if config.deployment is None:
                raise ValueError("no deployment configuration supplied")

            _add_file_handle_(logger, output_dir=config.deployment.output_dir)
            deploy_categorical_model_(
                output_dir=config.deployment.output_dir,
                checkpoint_path=config.deployment.checkpoint_path,
                clini_table=config.deployment.clini_table,
                slide_table=config.deployment.slide_table,
                feature_dir=config.deployment.feature_dir,
                ground_truth_label=config.deployment.ground_truth_label,
                patient_label=config.deployment.patient_label,
                filename_label=config.deployment.filename_label,
                num_workers=config.deployment.num_workers,
                accelerator=config.deployment.accelerator,
            )

        case "crossval":
            raise NotImplementedError()
            from stamp.modeling.transformer.helpers import categorical_crossval_

            if config.crossval is None:
                raise ValueError("no crossval configuration supplied")

            _add_file_handle_(logger, output_dir=config.crossval.output_dir)
            categorical_crossval_(**vars(config.crossval))

        case "statistics":
            raise NotImplementedError()
            from stamp.modeling.statistics import compute_stats_

            if config.statistics is None:
                raise ValueError("no statistics configuration supplied")

            _add_file_handle_(logger, output_dir=config.statistics.output_dir)

            compute_stats_(**vars(config.statistics))

        case "heatmaps":
            from stamp.heatmaps import heatmaps_

            if config.heatmaps is None:
                raise ValueError("no heatmaps configuration supplied")

            _add_file_handle_(logger, output_dir=config.heatmaps.output_dir)

            heatmaps_(**vars(config.heatmaps))

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
    commands.add_parser("setup", help="Download required resources")
    commands.add_parser(
        "preprocess", help="Preprocess whole-slide images into feature vectors"
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
        run_cli(args)
    except Exception as e:
        logger.exception(e)
        exit(1)


if __name__ == "__main__":
    main()
