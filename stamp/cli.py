from omegaconf import OmegaConf, DictConfig
from omegaconf.listconfig import ListConfig
import argparse
from pathlib import Path
import os
from typing import Iterable, Optional
import shutil

NORMALIZATION_TEMPLATE_URL = "https://github.com/Avic3nna/STAMP/blob/main/resources/normalization_template.jpg?raw=true"
CTRANSPATH_WEIGHTS_URL = "https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download"
DEFAULT_RESOURCES_DIR = Path(__file__).with_name("resources")
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

def require_configs(cfg: DictConfig, keys: Iterable[str], prefix: Optional[str] = None,
                    paths_to_check: Iterable[str] = []):
    keys = [f"{prefix}.{k}" for k in keys]
    missing = [k for k in keys if not _config_has_key(cfg, k)]
    if len(missing) > 0:
        raise ConfigurationError(f"Missing required configuration keys: {missing}")

    # Check if paths exist
    for path_key in paths_to_check:
        try:
            #for all but modeling.statistics
            path = cfg[prefix][path_key]
        except:
            #for modeling.statistics, handling the pred_csvs
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
        raise ConfigurationError(f"Default STAMP config file not found at {STAMP_FACTORY_SETTINGS}")
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
            print(f"Falling back to default STAMP config file because {DEFAULT_CONFIG_FILE.absolute()} does not exist")
            if not config_file.exists():
                raise ConfigurationError(f"Default STAMP config file not found at {config_file}")
    if not config_file.exists():
        raise ConfigurationError(f"Config file {Path(config_file).absolute()} not found (run `stamp init` to create the config file or use the `--config` flag to specify a different config file)")
    return config_file

def run_cli(args: argparse.Namespace):
    # Handle init command
    if args.command == "init":
        create_config_file(args.config)
        return

    # Load YAML configuration
    config_file = resolve_config_file_path(args.config)
    cfg = OmegaConf.load(config_file)

    # Set environment variables
    if "STAMP_RESOURCES_DIR" not in os.environ:
        os.environ["STAMP_RESOURCES_DIR"] = str(DEFAULT_RESOURCES_DIR)
    
    match args.command:
        case "init":
            return # this is handled above
        case "setup":
            # Download normalization template
            normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
            normalization_template_path.parent.mkdir(parents=True, exist_ok=True)
            if normalization_template_path.exists():
                print(f"Skipping download, normalization template already exists at {normalization_template_path}")
            else:
                print(f"Downloading normalization template to {normalization_template_path}")
                import requests
                r = requests.get(NORMALIZATION_TEMPLATE_URL)
                with normalization_template_path.open("wb") as f:
                    f.write(r.content)
            # Download feature extractor model
            feat_extractor = cfg.preprocessing.feat_extractor
            if feat_extractor == 'ctp':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/ctranspath.pth")
            elif feat_extractor == 'uni':
                model_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/uni/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin")
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if model_path.exists():
                print(f"Skipping download, feature extractor model already exists at {model_path}")
            else:
                if feat_extractor == 'ctp':
                    print(f"Downloading CTransPath weights to {model_path}")
                    import gdown
                    gdown.download(CTRANSPATH_WEIGHTS_URL, str(model_path))
                elif feat_extractor == 'uni':
                    print(f"Downloading UNI weights")
                    from uni.get_encoder import get_encoder
                    get_encoder(enc_name='uni', checkpoint='pytorch_model.bin', assets_dir=f"{os.environ['STAMP_RESOURCES_DIR']}/uni")
        case "config":
            print(OmegaConf.to_yaml(cfg, resolve=True))
        case "preprocess":
            require_configs(
                cfg,
                ["output_dir", "wsi_dir", "cache_dir", "microns", "cores", "norm", "del_slide", "only_feature_extraction", "device", "feat_extractor"],
                prefix="preprocessing",
                paths_to_check=["wsi_dir"]
            )
            c = cfg.preprocessing
            # Some checks
            normalization_template_path = Path(f"{os.environ['STAMP_RESOURCES_DIR']}/normalization_template.jpg")
            if c.norm and not Path(normalization_template_path).exists():
                raise ConfigurationError(f"Normalization template {normalization_template_path} does not exist, please run `stamp setup` to download it.")
            if c.feat_extractor == 'ctp':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/ctranspath.pth"
            elif c.feat_extractor == 'uni':
                model_path = f"{os.environ['STAMP_RESOURCES_DIR']}/uni/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin"
            if not Path(model_path).exists():
                raise ConfigurationError(f"Feature extractor model {model_path} does not exist, please run `stamp setup` to download it.")
            from .preprocessing.wsi_norm import preprocess
            preprocess(
                output_dir=Path(c.output_dir),
                wsi_dir=Path(c.wsi_dir),
                model_path=Path(model_path),
                cache_dir=Path(c.cache_dir),
                feat_extractor=c.feat_extractor,
                # patch_size=c.patch_size,
                target_microns=c.microns,
                cores=c.cores,
                norm=c.norm,
                del_slide=c.del_slide,
                cache=c.cache if 'cache' in c else True,
                only_feature_extraction=c.only_feature_extraction,
                keep_dir_structure=c.keep_dir_structure if 'keep_dir_structure' in c else False,
                device=c.device,
                normalization_template=normalization_template_path
            )
        case "train":
            require_configs(
                cfg,
                ["clini_table", "slide_table", "output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels"],
                prefix="modeling",
                paths_to_check=["clini_table", "slide_table", "feature_dir"]
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import train_categorical_model_
            train_categorical_model_(clini_table=Path(c.clini_table), 
                                     slide_table=Path(c.slide_table),
                                     feature_dir=Path(c.feature_dir), 
                                     output_path=Path(c.output_dir),
                                     target_label=c.target_label, 
                                     cat_labels=c.cat_labels,
                                     cont_labels=c.cont_labels, 
                                     categories=c.categories)
        case "crossval":
            require_configs(
                cfg,
                ["clini_table", "slide_table", "output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels", "n_splits"], # this one requires the n_splits key!
                prefix="modeling",
                paths_to_check=["clini_table", "slide_table", "feature_dir"]
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import categorical_crossval_
            categorical_crossval_(clini_table=Path(c.clini_table), 
                                  slide_table=Path(c.slide_table),
                                  feature_dir=Path(c.feature_dir),
                                  output_path=Path(c.output_dir),
                                  target_label=c.target_label,
                                  cat_labels=c.cat_labels,
                                  cont_labels=c.cont_labels,
                                  categories=c.categories,
                                  n_splits=c.n_splits)
        case "deploy":
            require_configs(
                cfg,
                ["clini_table", "slide_table", "output_dir", "deploy_feature_dir", "target_label", "cat_labels", "cont_labels", "model_path"], # this one requires the model_path key!
                prefix="modeling",
                paths_to_check=["clini_table", "slide_table", "deploy_feature_dir"]
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import deploy_categorical_model_
            deploy_categorical_model_(clini_table=Path(c.clini_table),
                                      slide_table=Path(c.slide_table),
                                      feature_dir=Path(c.deploy_feature_dir),
                                      output_path=Path(c.output_dir),
                                      target_label=c.target_label,
                                      cat_labels=c.cat_labels,
                                      cont_labels=c.cont_labels,
                                      model_path=Path(c.model_path))
            print("Successfully deployed models")
        case "statistics":
            require_configs(
                cfg,
                ["pred_csvs", "target_label", "true_class", "output_dir"],
                prefix="modeling.statistics",
                paths_to_check=["pred_csvs"]
            )
            from .modeling.statistics import compute_stats
            c = cfg.modeling.statistics
            if isinstance(c.pred_csvs,str):
                c.pred_csvs = [c.pred_csvs]
            compute_stats(pred_csvs=[Path(x) for x in c.pred_csvs],
                          target_label=c.target_label,
                          true_class=c.true_class,
                          output_dir=Path(c.output_dir))
            print("Successfully calculated statistics")
        case "heatmaps":
            require_configs(
                cfg,
                ["feature_dir","wsi_dir","model_path","output_dir", "n_toptiles", "overview"], 
                prefix="heatmaps",
                paths_to_check=["feature_dir","wsi_dir","model_path"]
            )
            c = cfg.heatmaps
            from .heatmaps.__main__ import main
            main(slide_name=str(c.slide_name),
                 feature_dir=Path(c.feature_dir),
                 wsi_dir=Path(c.wsi_dir),
                 model_path=Path(c.model_path),
                 output_dir=Path(c.output_dir),
                 n_toptiles=int(c.n_toptiles),
                 overview=c.overview)
            print("Successfully produced heatmaps")
        case _:
            raise ConfigurationError(f"Unknown command {args.command}")

def main() -> None:
    parser = argparse.ArgumentParser(prog="stamp", description="STAMP: Solid Tumor Associative Modeling in Pathology")
    parser.add_argument("--config", "-c", type=Path, default=None, help=f"Path to config file (if unspecified, defaults to {DEFAULT_CONFIG_FILE.absolute()} or the default STAMP config file shipped with the package if {DEFAULT_CONFIG_FILE.absolute()} does not exist)")

    commands = parser.add_subparsers(dest="command")
    commands.add_parser("init", help="Create a new STAMP configuration file at the path specified by --config")
    commands.add_parser("setup", help="Download required resources")
    commands.add_parser("preprocess", help="Preprocess whole-slide images into feature vectors")
    commands.add_parser("train", help="Train a Vision Transformer model")
    commands.add_parser("crossval", help="Train a Vision Transformer model with cross validation for modeling.n_splits folds")
    commands.add_parser("deploy", help="Deploy a trained Vision Transformer model")
    commands.add_parser("statistics", help="Generate AUROCs and AUPRCs with 95%%CI for a trained Vision Transformer model")
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
