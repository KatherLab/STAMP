from omegaconf import OmegaConf
import argparse
from pathlib import Path
from omegaconf.dictconfig import DictConfig
from functools import partial
import os
from typing import Iterable, Optional

NORMALIZATION_TEMPLATE_URL = "https://github.com/Avic3nna/STAMP/blob/main/preprocessing/normalization_template.jpg?raw=true"
CTRANSPATH_WEIGHTS_URL = "https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download"

class ConfigurationError(Exception):
    pass

def _config_has_key(cfg: DictConfig, key: str):
    try:
        for k in key.split("."):
            cfg = cfg[k]
        if cfg is None:
            return False
    except KeyError:
        return False
    return True

def require_configs(cfg: DictConfig, keys: Iterable[str], prefix: Optional[str] = None):
    prefix = f"{prefix}." if prefix else ""
    keys = [f"{prefix}{k}" for k in keys]
    missing = [k for k in keys if not _config_has_key(cfg, k)]
    if len(missing) > 0:
        raise ConfigurationError(f"Missing required configuration keys: {missing}")

def run_cli(args: argparse.Namespace):
    # Load YAML configuration
    try:
        cfg = OmegaConf.load(args.config)
    except FileNotFoundError:
        raise ConfigurationError(f"Config file {args.config} not found (use the --config flag to specify a different config file)")
    
    # Set environment variables
    if "STAMP_RESOURCES_DIR" not in os.environ:
        os.environ["STAMP_RESOURCES_DIR"] = str(Path(args.config).with_name("resources"))
    
    match args.command:
        case "setup":
            # Download normalization template
            normalization_template_path = Path(cfg.preprocessing.normalization_template)
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
            model_path = Path(cfg.preprocessing.model_path)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if model_path.exists():
                print(f"Skipping download, feature extractor model already exists at {model_path}")
            else:
                print(f"Downloading CTransPath weights to {model_path}")
                import gdown
                gdown.download(CTRANSPATH_WEIGHTS_URL, str(model_path))
        case "config":
            print(OmegaConf.to_yaml(cfg, resolve=True))
        case "preprocess":
            require_configs(
                cfg,
                ["output_dir", "wsi_dir", "model_path", "cache_dir", "patch_size", "mpp", "cores", "norm", "del_slide", "only_feature_extraction", "device", "normalization_template"],
                prefix="preprocessing"
            )
            c = cfg.preprocessing
            # Some checks
            if not Path(c.normalization_template).exists():
                raise ConfigurationError(f"Normalization template {c.normalization_template} does not exist, please run `stamp setup` to download it.")
            if not Path(c.model_path).exists():
                raise ConfigurationError(f"Feature extractor model {c.model_path} does not exist, please run `stamp setup` to download it.")
            from .preprocessing.wsi_norm import preprocess
            preprocess(
                output_dir=Path(c.output_dir),
                wsi_dir=Path(c.wsi_dir),
                model_path=Path(c.model_path),
                cache_dir=Path(c.cache_dir),
                patch_size=c.patch_size,
                target_mpp=c.mpp,
                cores=c.cores,
                norm=c.norm,
                del_slide=c.del_slide,
                only_feature_extraction=c.only_feature_extraction,
                device=c.device,
                normalization_template=Path(c.normalization_template)
            )
        case "train":
            require_configs(
                cfg,
                ["output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels"],
                prefix="modeling"
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import train_categorical_model_
            train_categorical_model_(clini_table=Path(c.clini_table), 
                                     slide_csv=Path(c.slide_csv),
                                     feature_dir=Path(c.feature_dir), 
                                     output_path=Path(c.output_dir),
                                     target_label=c.target_label, 
                                     cat_labels=c.cat_labels,
                                     cont_labels=c.cont_labels, 
                                     categories=c.categories)
        case "crossval":
            require_configs(
                cfg,
                ["output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels", "n_splits"], # this one requires the n_splits key!
                prefix="modeling"
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import categorical_crossval_
            categorical_crossval_(clini_table=Path(c.clini_table), 
                                  slide_csv=Path(c.slide_csv),
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
                ["output_dir", "feature_dir", "target_label", "cat_labels", "cont_labels", "model_path"], # this one requires the model_path key!
                prefix="modeling"
            )
            c = cfg.modeling
            from .modeling.marugoto.transformer.helpers import deploy_categorical_model_
            deploy_categorical_model_(clini_table=Path(c.clini_table),
                                      slide_csv=Path(c.slide_csv),
                                      feature_dir=Path(c.feature_dir),
                                      output_path=Path(c.output_dir),
                                      target_label=c.target_label,
                                      cat_labels=c.cat_labels,
                                      cont_labels=c.cont_labels,
                                      model_path=Path(c.model_path))
        case "stats":
            require_configs(
                cfg,
                ["pred_csvs", "target_label", "true_class", "output_dir", "n_bootstrap_samples", "figure_width"],
                prefix="modeling.statistics")
            from .modeling.statistics import compute_stats
            c = cfg.modeling.statistics
            compute_stats(pred_csvs=[Path(x) for x in c.pred_csvs],
                          target_label=c.target_label,
                          true_class=c.true_class,
                          output_dir=Path(c.output_dir),
                          n_bootstrap_samples=c.n_bootstrap_samples,
                          figure_width=c.figure_width,
                          threshold_cmap=c.threshold_cmap)
        case "heatmaps":
            raise NotImplementedError("Heatmaps are not yet implemented")
        case _:
            raise ConfigurationError(f"Unknown command {args.command}")

def main() -> None:
    parser = argparse.ArgumentParser(prog="stamp", description="STAMP: Solid Tumor Associative Modeling in Pathology")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config file")

    commands = parser.add_subparsers(dest="command")
    commands.add_parser("setup", help="Download required resources")
    commands.add_parser("preprocess", help="Preprocess data")
    commands.add_parser("train", help="Train a vision transformer model")
    commands.add_parser("crossval", help="Train a vision transformer model with cross validation for modeling.n_splits folds")
    commands.add_parser("deploy", help="Deploy a trained vision transformer model")
    commands.add_parser("stats", help="Generate ROC curves for a trained model")
    commands.add_parser("config", help="Print the loaded configuation")
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