import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional, assert_never

import yaml

from stamp.config import StampConfig
from stamp.modeling.config import CrossvalConfig, DeploymentConfig, TrainConfig

DEFAULT_CONFIG_FILE = Path("config.yaml")
STAMP_FACTORY_SETTINGS = Path(__file__).with_name("config.yaml")

# Set up the logger
logger = logging.getLogger("stamp")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

stream_handler = logging.StreamHandler(sys.stderr)
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)


class ConfigurationError(Exception):
    pass


def create_config_file(config_file: Path | None) -> Path:
    """Create a new config file at the specified path (by copying the default config file)."""
    config_file = config_file or DEFAULT_CONFIG_FILE
    # Locate original config file
    if not STAMP_FACTORY_SETTINGS.exists():
        raise ConfigurationError(
            f"Default STAMP config file not found at {STAMP_FACTORY_SETTINGS}"
        )
    if not config_file.exists():
        # Copy original config file
        shutil.copy(STAMP_FACTORY_SETTINGS, config_file)
        logger.info(f"Created new config file at {config_file.absolute()}")
    else:
        logger.info(
            f"Refusing to overwrite existing config file at {config_file.absolute()}"
        )

    return config_file


def resolve_config_file_path(config_file: Optional[Path]) -> Path:
    """Resolve the path to the config file, falling back to the default config file if not specified."""
    if config_file is None:
        if DEFAULT_CONFIG_FILE.exists():
            config_file = DEFAULT_CONFIG_FILE
        else:
            config_file = STAMP_FACTORY_SETTINGS
            print(
                f"Falling back to default STAMP config file because {DEFAULT_CONFIG_FILE.absolute()} does not exist"
            )
            if not config_file.exists():
                raise ConfigurationError(
                    f"Default STAMP config file not found at {config_file}"
                )
    if not config_file.exists():
        raise ConfigurationError(
            f"Config file {Path(config_file).absolute()} not found (run `stamp init` to create the config file or use the `--config` flag to specify a different config file)"
        )
    return config_file


def run_cli(args: argparse.Namespace) -> None:
    # Handle init command
    if args.command == "init":
        create_config_file(args.config)
        return

    # Load YAML configuration
    config_file_path = resolve_config_file_path(args.config)
    with open(config_file_path, "r") as config_yaml:
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

            add_file_handle(logger, output_dir=config.preprocessing.output_dir)
            extract_(**vars(config.preprocessing))
        case "train":
            from .modeling.marugoto.transformer.helpers import train_categorical_model_

            # if config.training is None:   #TODO uncomment this once we have deprecated old config options
            if not isinstance(config.training, TrainConfig):
                raise ValueError("no training configuration supplied")

            add_file_handle(logger, output_dir=config.training.output_dir)
            train_categorical_model_(**vars(config.training))
        case "crossval":
            from .modeling.marugoto.transformer.helpers import categorical_crossval_

            # if config.crossval is None:   #TODO uncomment this once we have deprecated old config options
            if not isinstance(config.crossval, CrossvalConfig):
                raise ValueError("no crossval configuration supplied")

            add_file_handle(logger, output_dir=config.crossval.output_dir)
            categorical_crossval_(**vars(config.crossval))
        case "deploy":
            from .modeling.marugoto.transformer.helpers import deploy_categorical_model_

            # if config.deployment is None: #TODO uncomment this once we have deprecated old config options
            if not isinstance(config.deployment, DeploymentConfig):
                raise ValueError("no deployment configuration supplied")

            add_file_handle(logger, output_dir=config.deployment.output_dir)
            deploy_categorical_model_(**vars(config.deployment))
        case "statistics":
            from .modeling.statistics import compute_stats_

            if config.statistics is None:
                raise ValueError("no statistics configuration supplied")

            add_file_handle(logger, output_dir=config.statistics.output_dir)

            compute_stats_(**vars(config.statistics))
        case "heatmaps":
            from stamp.heatmaps import heatmaps_

            if config.heatmaps is None:
                raise ValueError("no heatmaps configuration supplied")

            add_file_handle(logger, output_dir=config.heatmaps.output_dir)

            heatmaps_(**vars(config.heatmaps))
        case _:
            assert_never(
                "the argparser should only allow valid commands"  # pyright: ignore[reportArgumentType]
            )


def add_file_handle(logger: logging.Logger, *, output_dir: Path) -> None:
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
        default=None,
        help=f"Path to config file (if unspecified, defaults to {DEFAULT_CONFIG_FILE.absolute()} or the default STAMP config file shipped with the package if {DEFAULT_CONFIG_FILE.absolute()} does not exist)",
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
    except ConfigurationError as e:
        print(e)
        exit(1)


if __name__ == "__main__":
    main()
