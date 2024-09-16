import argparse
import logging
import os
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Optional, assert_never

from omegaconf import DictConfig, OmegaConf
from omegaconf.listconfig import ListConfig

DEFAULT_CONFIG_FILE = Path("config.yaml")
STAMP_FACTORY_SETTINGS = Path(__file__).with_name("config.yaml")


class ConfigurationError(Exception):
    pass


def check_path_exists(path):
    directories = path.split(os.path.sep)
    current_path = os.path.sep
    for directory in directories:
        current_path = os.path.join(current_path, directory)
        if not os.path.exists(current_path):
            return False, directory
    return True, None


def check_and_handle_path(path, path_key, prefix):
    exists, directory = check_path_exists(path)
    if not exists:
        print(f"From input path: '{path}'")
        print(f"Directory '{directory}' does not exist.")
        print(f"Check the input path of '{path_key}' from the '{prefix}' section.")
        raise SystemExit(f"Stopping {prefix} due to faulty user input...")


def _config_has_key(cfg: DictConfig, key: str):
    try:
        for k in key.split("."):
            cfg = cfg[k]
        if cfg is None:
            return False
    except KeyError:
        return False
    return True


def require_configs(
    cfg: DictConfig,
    keys: Iterable[str],
    prefix: Optional[str] = None,
    paths_to_check: Iterable[str] = [],
) -> None:
    keys = [f"{prefix}.{k}" for k in keys]
    missing = [k for k in keys if not _config_has_key(cfg, k)]
    if len(missing) > 0:
        raise ConfigurationError(f"Missing required configuration keys: {missing}")

    # Check if paths exist
    for path_key in paths_to_check:
        try:
            # for all but modeling.statistics
            path = cfg[prefix][path_key]
        except:
            # for modeling.statistics, handling the pred_csvs
            path = OmegaConf.select(cfg, f"{prefix}.{path_key}")
        if isinstance(path, ListConfig):
            for p in path:
                check_and_handle_path(p, path_key, prefix)
        else:
            check_and_handle_path(path, path_key, prefix)


def create_config_file(config_file: Optional[Path]):
    """Create a new config file at the specified path (by copying the default config file)."""
    config_file = config_file or DEFAULT_CONFIG_FILE
    # Locate original config file
    if not STAMP_FACTORY_SETTINGS.exists():
        raise ConfigurationError(
            f"Default STAMP config file not found at {STAMP_FACTORY_SETTINGS}"
        )
    # Copy original config file
    shutil.copy(STAMP_FACTORY_SETTINGS, config_file)
    print(f"Created new config file at {config_file.absolute()}")


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
    # Set up the logger
    logger = logging.getLogger("stamp")
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(levelname)s\t%(message)s")

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # Handle init command
    if args.command == "init":
        create_config_file(args.config)
        return

    # Load YAML configuration
    config_file_path = resolve_config_file_path(args.config)
    cfg = OmegaConf.load(config_file_path)
    assert isinstance(cfg, DictConfig), "expected config to be a dict, not a list"

    match args.command:
        case "init":
            assert_never("this case should be handled above")
        case "config":
            print(OmegaConf.to_yaml(cfg, resolve=True))
        case "preprocess":
            require_configs(
                cfg,
                [
                    "output_dir",
                    "wsi_dir",
                    "cache_dir",
                    "microns",
                    "cores",
                    "device",
                    "feat_extractor",
                ],
                prefix="preprocessing",
                paths_to_check=["wsi_dir"],
            )
            c = cfg.preprocessing

            Path(c.output_dir).mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(Path(c.output_dir) / "logfile.log")
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            from stamp.preprocessing.extract import Microns, TilePixels, extract_

            extract_(
                wsi_dir=Path(c.wsi_dir),
                output_dir=Path(c.output_dir),
                cache_dir=Path(c.cache_dir),
                extractor=c.feat_extractor,
                tile_size_px=TilePixels(224),
                tile_size_um=Microns(c.microns),
                max_workers=c.cores,
                device=c.device,
            )
        case "train":
            require_configs(
                cfg,
                [
                    "clini_table",
                    "slide_table",
                    "output_dir",
                    "feature_dir",
                    "target_label",
                    "cat_labels",
                    "cont_labels",
                ],
                prefix="modeling",
                paths_to_check=["clini_table", "slide_table", "feature_dir"],
            )
            c = cfg.modeling

            Path(c.output_dir).mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(Path(c.output_dir) / "logfile.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            from .modeling.marugoto.transformer.helpers import train_categorical_model_

            train_categorical_model_(
                clini_table=Path(c.clini_table),
                slide_table=Path(c.slide_table),
                feature_dir=Path(c.feature_dir),
                output_path=Path(c.output_dir),
                target_label=c.target_label,
                cat_labels=c.cat_labels,
                cont_labels=c.cont_labels,
                categories=c.categories,
            )
        case "crossval":
            require_configs(
                cfg,
                [
                    "clini_table",
                    "slide_table",
                    "output_dir",
                    "feature_dir",
                    "target_label",
                    "cat_labels",
                    "cont_labels",
                    "n_splits",
                ],  # this one requires the n_splits key!
                prefix="modeling",
                paths_to_check=["clini_table", "slide_table", "feature_dir"],
            )
            c = cfg.modeling

            Path(c.output_dir).mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(Path(c.output_dir) / "logfile.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            from .modeling.marugoto.transformer.helpers import categorical_crossval_

            categorical_crossval_(
                clini_table=Path(c.clini_table),
                slide_table=Path(c.slide_table),
                feature_dir=Path(c.feature_dir),
                output_path=Path(c.output_dir),
                target_label=c.target_label,
                cat_labels=c.cat_labels,
                cont_labels=c.cont_labels,
                categories=c.categories,
                n_splits=c.n_splits,
            )
        case "deploy":
            require_configs(
                cfg,
                [
                    "clini_table",
                    "slide_table",
                    "output_dir",
                    "deploy_feature_dir",
                    "target_label",
                    "cat_labels",
                    "cont_labels",
                    "model_path",
                ],  # this one requires the model_path key!
                prefix="modeling",
                paths_to_check=["clini_table", "slide_table", "deploy_feature_dir"],
            )
            c = cfg.modeling

            Path(c.output_dir).mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(Path(c.output_dir) / "logfile.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            from .modeling.marugoto.transformer.helpers import deploy_categorical_model_

            deploy_categorical_model_(
                clini_table=Path(c.clini_table),
                slide_table=Path(c.slide_table),
                feature_dir=Path(c.deploy_feature_dir),
                output_path=Path(c.output_dir),
                target_label=c.target_label,
                cat_labels=c.cat_labels,
                cont_labels=c.cont_labels,
                model_path=Path(c.model_path),
            )
            print("Successfully deployed models")
        case "statistics":
            require_configs(
                cfg,
                ["pred_csvs", "target_label", "true_class", "output_dir"],
                prefix="modeling.statistics",
                paths_to_check=["pred_csvs"],
            )
            c = cfg.modeling.statistics

            Path(c.output_dir).mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(Path(c.output_dir) / "logfile.log")
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            if isinstance(c.pred_csvs, str):
                c.pred_csvs = [c.pred_csvs]
            from .modeling.statistics import compute_stats

            compute_stats(
                pred_csvs=[Path(x) for x in c.pred_csvs],
                target_label=c.target_label,
                true_class=c.true_class,
                output_dir=Path(c.output_dir),
            )
            print("Successfully calculated statistics")
        case "heatmaps":
            require_configs(
                cfg,
                [
                    "feature_dir",
                    "wsi_dir",
                    "model_path",
                    "output_dir",
                    "n_toptiles",
                    "overview",
                ],
                prefix="heatmaps",
                paths_to_check=["feature_dir", "wsi_dir", "model_path"],
            )
            c = cfg.heatmaps

            Path(c.output_dir).mkdir(exist_ok=True, parents=True)
            file_handler = logging.FileHandler(c.output_dir)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

            from .heatmaps.__main__ import main

            main(
                slide_name=str(c.slide_name),
                feature_dir=Path(c.feature_dir),
                wsi_dir=Path(c.wsi_dir),
                model_path=Path(c.model_path),
                output_dir=Path(c.output_dir),
                n_toptiles=int(c.n_toptiles),
                overview=c.overview,
            )
            print("Successfully produced heatmaps")
        case _:
            raise ConfigurationError(f"Unknown command {args.command}")


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
