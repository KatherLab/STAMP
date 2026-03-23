from __future__ import annotations

import os
from importlib import import_module
from copy import deepcopy
from shutil import which
import subprocess


def _field(
    name: str,
    label: str,
    kind: str,
    *,
    default=None,
    required: bool = False,
    help_text: str = "",
    options: list[str] | None = None,
    placeholder: str | None = None,
    path_role: str | None = None,
    path_type: str | None = None,
    presentation: str | None = None,
    coerce_single: bool = False,
) -> dict:
    return {
        "name": name,
        "label": label,
        "kind": kind,
        "default": default,
        "required": required,
        "help": help_text,
        "options": options or [],
        "placeholder": placeholder,
        "path_role": path_role,
        "path_type": path_type,
        "presentation": presentation,
        "coerce_single": coerce_single,
    }


DEFAULT_NUM_WORKERS = min(os.cpu_count() or 1, 16)
COMMON_TASK_OPTIONS = ["classification", "regression", "survival"]


def _load_torch():
    try:
        return import_module("torch")
    except Exception:
        return None


def _detect_device_options() -> list[str]:
    torch = _load_torch()
    options: list[str] = []

    if torch is not None:
        try:
            if torch.cuda.is_available():
                options = ["cuda"]
                options.extend([f"cuda:{index}" for index in range(torch.cuda.device_count())])
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                options = ["mps"]
            elif hasattr(torch, "xpu") and torch.xpu.is_available():
                options = ["xpu"]
                count = getattr(torch.xpu, "device_count", lambda: 0)()
                options.extend([f"xpu:{index}" for index in range(count)])
        except Exception:
            options = []

    if not options and which("nvidia-smi"):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            )
            indices = [line.strip() for line in output.splitlines() if line.strip()]
            if indices:
                options = ["cuda", *[f"cuda:{index}" for index in indices]]
        except Exception:
            pass

    options.append("cpu")
    return list(dict.fromkeys(options))


def _detect_accelerator_options() -> list[str]:
    return _detect_device_options()


DEVICE_OPTIONS = _detect_device_options()
ACCELERATOR_OPTIONS = _detect_accelerator_options()
DEVICE_DEFAULT = DEVICE_OPTIONS[0]
ACCELERATOR_DEFAULT = ACCELERATOR_OPTIONS[0]

EXTRACTOR_OPTIONS = [
    "ctranspath",
    "chief-ctranspath",
    "conch",
    "conch1_5",
    "uni",
    "uni2",
    "dino-bloom",
    "gigapath",
    "h-optimus-0",
    "h-optimus-1",
    "virchow",
    "virchow-full",
    "virchow2",
    "musk",
    "mstar",
    "plip",
    "keep",
    "ticon",
    "red-dino",
]
ENCODER_OPTIONS = [
    "cobra",
    "eagle",
    "chief",
    "titan",
    "gigapath",
    "madeleine",
    "prism",
]
MODEL_NAME_OPTIONS = ["vit", "mlp", "trans_mil", "linear", "barspoon"]

TASK_COMMANDS = {
    "preprocessing": "preprocess",
    "slide_encoding": "encode_slides",
    "patient_encoding": "encode_patients",
    "training": "train",
    "crossval": "crossval",
    "deployment": "deploy",
    "statistics": "statistics",
    "heatmaps": "heatmaps",
}

TASK_CATALOG = {
    "preprocessing": {
        "title": "Preprocess",
        "summary": "Tile whole-slide images and extract patch-level features.",
        "fields": [
            _field(
                "output_dir",
                "Output Directory",
                "path",
                required=True,
                path_role="output",
                path_type="dir",
                placeholder="/data/project/features",
            ),
            _field(
                "wsi_dir",
                "WSI Directory",
                "path",
                required=True,
                path_role="input",
                path_type="dir",
                placeholder="/data/project/wsis",
            ),
            _field(
                "extractor",
                "Extractor",
                "select",
                default="ctranspath",
                options=EXTRACTOR_OPTIONS,
                help_text="Tile-level backbone used during feature extraction.",
            ),
            _field("device", "Device", "select", default=DEVICE_DEFAULT, options=DEVICE_OPTIONS),
            _field(
                "cache_dir",
                "Cache Directory",
                "path",
                path_role="output",
                path_type="dir",
                placeholder="/data/project/cache",
            ),
            _field("max_workers", "Max Workers", "integer", default=8),
            _field("tile_size_um", "Tile Size (um)", "number", default=256.0),
            _field("tile_size_px", "Tile Size (px)", "integer", default=224),
            _field("default_slide_mpp", "Fallback Slide MPP", "number"),
        ],
    },
    "slide_encoding": {
        "title": "Encode Slides",
        "summary": "Aggregate patch features into one feature vector per slide.",
        "fields": [
            _field(
                "output_dir",
                "Output Directory",
                "path",
                required=True,
                path_role="output",
                path_type="dir",
                placeholder="/data/project/slide-features",
            ),
            _field(
                "feat_dir",
                "Feature Directory",
                "path",
                required=True,
                path_role="input",
                path_type="dir",
                placeholder="/data/project/tile-features",
            ),
            _field(
                "encoder",
                "Encoder",
                "select",
                default="chief",
                options=ENCODER_OPTIONS,
            ),
            _field("device", "Device", "select", default=DEVICE_DEFAULT, options=DEVICE_OPTIONS),
            _field(
                "agg_feat_dir",
                "Auxiliary Feature Directory",
                "path",
                path_role="input",
                path_type="dir",
                help_text="Optional, used by encoders such as eagle or prism.",
            ),
            _field("generate_hash", "Generate Hash", "boolean", default=True),
        ],
    },
    "patient_encoding": {
        "title": "Encode Patients",
        "summary": "Aggregate slide features into one feature vector per patient.",
        "fields": [
            _field(
                "output_dir",
                "Output Directory",
                "path",
                required=True,
                path_role="output",
                path_type="dir",
                placeholder="/data/project/patient-features",
            ),
            _field(
                "feat_dir",
                "Feature Directory",
                "path",
                required=True,
                path_role="input",
                path_type="dir",
                placeholder="/data/project/tile-features",
            ),
            _field(
                "slide_table",
                "Slide Table",
                "path",
                required=True,
                path_role="input",
                path_type="file",
                placeholder="/data/project/slide.csv",
            ),
            _field(
                "encoder",
                "Encoder",
                "select",
                default="chief",
                options=ENCODER_OPTIONS,
            ),
            _field("patient_label", "Patient Column", "text", default="PATIENT"),
            _field("filename_label", "Filename Column", "text", default="FILENAME"),
            _field("device", "Device", "select", default=DEVICE_DEFAULT, options=DEVICE_OPTIONS),
            _field(
                "agg_feat_dir",
                "Auxiliary Feature Directory",
                "path",
                path_role="input",
                path_type="dir",
            ),
            _field("generate_hash", "Generate Hash", "boolean", default=True),
        ],
    },
    "training": {
        "title": "Train",
        "summary": "Train a model on one training cohort.",
        "fields": [
            _field(
                "output_dir",
                "Output Directory",
                "path",
                required=True,
                path_role="output",
                path_type="dir",
                placeholder="/data/project/train-output",
            ),
            _field(
                "clini_table",
                "Clinical Table",
                "path",
                required=True,
                path_role="input",
                path_type="file",
                placeholder="/data/project/clini.csv",
            ),
            _field(
                "feature_dir",
                "Feature Directory",
                "path",
                required=True,
                path_role="input",
                path_type="dir",
                placeholder="/data/project/features",
            ),
            _field(
                "slide_table",
                "Slide Table",
                "path",
                path_role="input",
                path_type="file",
                placeholder="/data/project/slide.csv",
            ),
            _field(
                "task",
                "Task",
                "select",
                default="classification",
                options=COMMON_TASK_OPTIONS,
            ),
            _field(
                "ground_truth_label",
                "Ground Truth Label",
                "list",
                presentation="csv",
                coerce_single=True,
                help_text="Comma-separated for multi-target classification.",
                placeholder="KRAS or KRAS,BRAF,NRAS",
            ),
            _field(
                "categories",
                "Categories",
                "list",
                presentation="csv",
                placeholder="mutated,wild type",
            ),
            _field("status_label", "Status Label", "text", placeholder="event"),
            _field("time_label", "Time Label", "text", placeholder="time"),
            _field("patient_label", "Patient Column", "text", default="PATIENT"),
            _field("filename_label", "Filename Column", "text", default="FILENAME"),
            _field(
                "use_vary_precision_transform",
                "Use Vary Precision Transform",
                "boolean",
                default=False,
            ),
        ],
    },
    "crossval": {
        "title": "Cross-Validation",
        "summary": "Run k-fold cross-validation on one cohort.",
        "fields": [
            _field(
                "output_dir",
                "Output Directory",
                "path",
                required=True,
                path_role="output",
                path_type="dir",
                placeholder="/data/project/crossval-output",
            ),
            _field(
                "clini_table",
                "Clinical Table",
                "path",
                required=True,
                path_role="input",
                path_type="file",
                placeholder="/data/project/clini.csv",
            ),
            _field(
                "feature_dir",
                "Feature Directory",
                "path",
                required=True,
                path_role="input",
                path_type="dir",
                placeholder="/data/project/features",
            ),
            _field(
                "slide_table",
                "Slide Table",
                "path",
                path_role="input",
                path_type="file",
                placeholder="/data/project/slide.csv",
            ),
            _field(
                "task",
                "Task",
                "select",
                default="classification",
                options=COMMON_TASK_OPTIONS,
            ),
            _field(
                "ground_truth_label",
                "Ground Truth Label",
                "list",
                presentation="csv",
                coerce_single=True,
                placeholder="KRAS or KRAS,BRAF,NRAS",
            ),
            _field(
                "categories",
                "Categories",
                "list",
                presentation="csv",
                placeholder="mutated,wild type",
            ),
            _field("status_label", "Status Label", "text", placeholder="event"),
            _field("time_label", "Time Label", "text", placeholder="time"),
            _field("patient_label", "Patient Column", "text", default="PATIENT"),
            _field("filename_label", "Filename Column", "text", default="FILENAME"),
            _field("n_splits", "Number of Folds", "integer", default=5),
            _field(
                "use_vary_precision_transform",
                "Use Vary Precision Transform",
                "boolean",
                default=False,
            ),
        ],
    },
    "deployment": {
        "title": "Deploy",
        "summary": "Run one or more trained checkpoints on a target cohort.",
        "fields": [
            _field(
                "output_dir",
                "Output Directory",
                "path",
                required=True,
                path_role="output",
                path_type="dir",
                placeholder="/data/project/deploy-output",
            ),
            _field(
                "checkpoint_paths",
                "Checkpoint Paths",
                "list",
                required=True,
                presentation="lines",
                path_role="input",
                path_type="file",
                placeholder="/data/project/model.ckpt",
            ),
            _field(
                "clini_table",
                "Clinical Table",
                "path",
                path_role="input",
                path_type="file",
                placeholder="/data/project/clini.csv",
            ),
            _field(
                "slide_table",
                "Slide Table",
                "path",
                required=True,
                path_role="input",
                path_type="file",
                placeholder="/data/project/slide.csv",
            ),
            _field(
                "feature_dir",
                "Feature Directory",
                "path",
                required=True,
                path_role="input",
                path_type="dir",
                placeholder="/data/project/features",
            ),
            _field(
                "ground_truth_label",
                "Ground Truth Label",
                "list",
                presentation="csv",
                coerce_single=True,
                placeholder="KRAS or KRAS,BRAF,NRAS",
            ),
            _field("status_label", "Status Label", "text", placeholder="event"),
            _field("time_label", "Time Label", "text", placeholder="time"),
            _field("patient_label", "Patient Column", "text", default="PATIENT"),
            _field("filename_label", "Filename Column", "text", default="FILENAME"),
            _field("num_workers", "Num Workers", "integer", default=DEFAULT_NUM_WORKERS),
            _field("accelerator", "Accelerator", "select", default=ACCELERATOR_DEFAULT, options=ACCELERATOR_OPTIONS),
        ],
    },
    "statistics": {
        "title": "Statistics",
        "summary": "Compute evaluation metrics and plots from prediction CSVs.",
        "fields": [
            _field(
                "output_dir",
                "Output Directory",
                "path",
                required=True,
                path_role="output",
                path_type="dir",
                placeholder="/data/project/statistics",
            ),
            _field(
                "task",
                "Task",
                "select",
                default="classification",
                options=COMMON_TASK_OPTIONS,
            ),
            _field(
                "pred_csvs",
                "Prediction CSVs",
                "list",
                required=True,
                presentation="lines",
                path_role="input",
                path_type="file",
                placeholder="/data/project/patient-preds.csv",
            ),
            _field(
                "ground_truth_label",
                "Ground Truth Label",
                "list",
                presentation="csv",
                coerce_single=True,
                placeholder="KRAS or KRAS,BRAF,NRAS",
            ),
            _field("true_class", "Positive Class", "text", placeholder="mutated"),
            _field("status_label", "Status Label", "text", placeholder="event"),
            _field("time_label", "Time Label", "text", placeholder="time"),
        ],
    },
    "heatmaps": {
        "title": "Heatmaps",
        "summary": "Generate attention maps and top or bottom tile exports.",
        "fields": [
            _field(
                "output_dir",
                "Output Directory",
                "path",
                required=True,
                path_role="output",
                path_type="dir",
                placeholder="/data/project/heatmaps",
            ),
            _field(
                "feature_dir",
                "Feature Directory",
                "path",
                required=True,
                path_role="input",
                path_type="dir",
                placeholder="/data/project/features",
            ),
            _field(
                "wsi_dir",
                "WSI Directory",
                "path",
                required=True,
                path_role="input",
                path_type="dir",
                placeholder="/data/project/wsis",
            ),
            _field(
                "checkpoint_path",
                "Checkpoint Path",
                "path",
                required=True,
                path_role="input",
                path_type="file",
                placeholder="/data/project/model.ckpt",
            ),
            _field(
                "slide_paths",
                "Specific Slide Paths",
                "list",
                presentation="lines",
                help_text="Relative to the WSI directory. Leave empty to process all slides.",
                placeholder="slide1.svs",
            ),
            _field("device", "Device", "select", default=DEVICE_DEFAULT, options=DEVICE_OPTIONS),
            _field("opacity", "Overlay Opacity", "number", default=0.6),
            _field("topk", "Top K Tiles", "integer", default=0),
            _field("bottomk", "Bottom K Tiles", "integer", default=0),
            _field("default_slide_mpp", "Fallback Slide MPP", "number"),
        ],
    },
}

ADVANCED_CONFIG_SCHEMA = [
    _field("seed", "Seed", "integer", default=42),
    _field("bag_size", "Bag Size", "integer", default=512),
    _field("num_workers", "Num Workers", "integer", default=DEFAULT_NUM_WORKERS),
    _field("batch_size", "Batch Size", "integer", default=64),
    _field("max_epochs", "Max Epochs", "integer", default=32),
    _field("patience", "Patience", "integer", default=16),
    _field("accelerator", "Accelerator", "select", default=ACCELERATOR_DEFAULT, options=ACCELERATOR_OPTIONS),
    _field("max_lr", "Max Learning Rate", "number", default=1e-4),
    _field("div_factor", "LR Div Factor", "number", default=25.0),
    _field(
        "model_name",
        "Model Backbone",
        "select",
        default="vit",
        options=MODEL_NAME_OPTIONS,
    ),
]

PIPELINE_TEMPLATES = [
    {
        "id": "quickstart-classification",
        "title": "Quickstart Classification",
        "description": "Preprocess slides, run cross-validation, then compute statistics.",
        "sections": ["preprocessing", "crossval", "statistics"],
    },
    {
        "id": "train-and-deploy",
        "title": "Train and Deploy",
        "description": "Train on one cohort, deploy on another cohort, then summarize metrics.",
        "sections": ["training", "deployment", "statistics", "heatmaps"],
    },
    {
        "id": "patient-level-modeling",
        "title": "Patient-Level Encoding",
        "description": "Create patient embeddings, train a patient-level model, then deploy it.",
        "sections": ["patient_encoding", "training", "deployment", "statistics"],
    },
]


def default_advanced_config() -> dict:
    return {
        "seed": 42,
        "bag_size": 512,
        "num_workers": DEFAULT_NUM_WORKERS,
        "batch_size": 64,
        "max_epochs": 32,
        "patience": 16,
        "accelerator": ACCELERATOR_DEFAULT,
        "max_lr": 1e-4,
        "div_factor": 25.0,
        "model_name": "vit",
        "model_params": {
            "vit": {
                "dim_model": 512,
                "dim_feedforward": 512,
                "n_heads": 8,
                "n_layers": 2,
                "dropout": 0.0,
                "use_alibi": False,
            },
            "trans_mil": {
                "dim_hidden": 512,
            },
            "mlp": {
                "dim_hidden": 512,
                "num_layers": 2,
                "dropout": 0.25,
            },
            "linear": {
                "num_encoder_heads": 8,
                "num_decoder_heads": 8,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 2048,
                "positional_encoding": True,
                "learning_rate": 1e-4,
            },
            "barspoon": {
                "d_model": 512,
                "num_encoder_heads": 8,
                "num_decoder_heads": 8,
                "num_encoder_layers": 2,
                "num_decoder_layers": 2,
                "dim_feedforward": 2048,
                "positional_encoding": True,
                "learning_rate": 1e-4,
            },
        },
    }


def catalog_payload() -> dict:
    return {
        "tasks": deepcopy(TASK_CATALOG),
        "advanced_fields": deepcopy(ADVANCED_CONFIG_SCHEMA),
        "advanced_defaults": default_advanced_config(),
        "templates": deepcopy(PIPELINE_TEMPLATES),
    }
