import asyncio
import logging
import os
import platform
import subprocess
import tempfile
from typing import Annotated

import torch
import yaml
from fastmcp import Context, FastMCP
from pydantic import Field

# Initialize the FastMCP server
mcp = FastMCP("STAMP MCP Server")


STAMP_LOGGER = logging.getLogger("stamp")
# TODO: add proper filesystem management
base_dir = "./"


class MCPLogHandler(logging.Handler):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx

    def emit(self, record):
        msg = self.format(record)
        # Fire-and-forget the coroutine
        asyncio.create_task(self.ctx.log(msg))


async def _run_stamp(mode, config, ctx):
    """
    Run the STAMP command as a subprocess and capture its console output.

    Args:
        mode (str): The mode to run the STAMP command in (e.g., "preprocess", "train").
        config (dict): The configuration dictionary to pass to the command.

    Returns:
        str: The combined stdout and stderr output from the command.
    """
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, suffix=".yaml"
    ) as tmp_config:
        yaml.dump(config, tmp_config)
        tmp_config_path = tmp_config.name

    handler = MCPLogHandler(ctx)
    handler.setLevel(logging.DEBUG)
    STAMP_LOGGER.addHandler(handler)

    print("Running command...")

    try:
        cmd = ["stamp", "--config", tmp_config_path, mode]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("Result returned...")
        print(f"Command completed successfully:\n{result.stdout}\n{result.stderr}")
        return f"Command completed successfully:\n{result.stdout}\n{result.stderr}"
    except subprocess.CalledProcessError as e:
        return f"Command failed with error:\n{e.stdout}\n{e.stderr}"
    finally:
        os.remove(tmp_config_path)
        STAMP_LOGGER.removeHandler(handler)


@mcp.tool
async def preprocess_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to savethe results to")
    ],
    wsi_dir: Annotated[
        str, Field(description="Path of the directory containing whole slide images")
    ],
    extractor: Annotated[
        str,
        Field(
            description="Name of the extractor to use "
            'for feature extraction. Possible options are "ctranspath" and "uni"'
        ),
    ] = "ctranspath",
    cache_dir: Annotated[
        str | None,
        Field(
            description="Directory to save preprocessed slide tiles for optimizing future preprocessing."
        ),
    ] = None,
    device: Annotated[
        str,
        Field(
            description="The device to use for computation. "
            "Possible options are 'cuda' for NVIDIA GPUs, 'cpu' for general-purpose "
            "processors, and 'mps' for Apple Silicon GPUs. Default is 'cuda'."
        ),
    ] = "cuda",
    max_workers: Annotated[
        int,
        Field(
            description="The maximum number of parallel "
            "workers to use for computation. Increasing this value can improve performance "
            "on systems with multiple CPU cores or GPUs. Default is 8."
        ),
    ] = 8,
    tile_size_um: Annotated[
        float,
        Field(
            description="Size of each tile in micrometers. "
            "The default works well for most domains."
        ),
    ] = 256.0,
    tile_size_px: Annotated[
        int,
        Field(
            description="Resolution of the slide's tiles. "
            "The default works well for most domains."
        ),
    ] = 224,
    default_slide_mpp: Annotated[
        str | None,
        Field(
            description="MPP of the slide to use if none can be inferred from the WSI"
        ),
    ] = None,
) -> str:
    """
    Preprocess WSIs and extract patch-level features using STAMP.
    Extracts features with the selected model, and optionally caches tiles.
    Outputs include features (e.g., .h5) and tile preview images.

    Returns:
        str: message indicating success or failure.

    Example:
        >>> preprocess_stamp(
                output_dir="output/features",
                wsi_dir="input/slides",
                extractor="ctranspath",
                cache_dir="cache/tiles",
                device="cuda",
                max_workers=8
            )
        "Command completed successfully: ..."
    """
    config = {
        "preprocessing": {
            "output_dir": output_dir,
            "wsi_dir": wsi_dir,
            "extractor": extractor,
            "cache_dir": cache_dir,
            "device": device,
            "max_workers": max_workers,
            "tile_size_um": tile_size_um,
            "tile_size_px": tile_size_px,
            "default_slide_mpp": default_slide_mpp,
        }
    }
    return await _run_stamp(mode="preprocess", config=config, ctx=ctx)


@mcp.tool
async def train_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to savethe results to")
    ],
    clini_table: Annotated[
        str, Field(description="Path to a CSV or Excel to readclinical data from")
    ],
    feature_dir: Annotated[
        str, Field(description="Path to a Directory containingfeature files")
    ],
    slide_table: Annotated[
        str,
        Field(
            description="Path to a CSV or Excel to readpatient-slide associations from"
        ),
    ],
    ground_truth_label: Annotated[
        str,
        Field(description="Name of categorical column in clinical table to train on"),
    ],
    categories: Annotated[
        list[str],
        Field(
            description="The categories "
            "occurring in the target label column of the clini table."
        ),
    ],
    patient_label: Annotated[
        str,
        Field(
            description="Name of the column "
            "in the clini table that contains a unique ID for each patient"
        ),
    ] = "PATIENT",
    filename_label: Annotated[
        str,
        Field(
            description="Name of the column "
            "in the slide table containing the feature file path relative to `feature_dir`"
        ),
    ] = "FILENAME",
    bag_size: Annotated[
        int,
        Field(
            description="Amount of tiles to sample when training. "
            "Reducing this value reduces memory usage, but it is not recommended as the model can miss"
            "relevant regions of the slide. Default value works well on H&E tissue images."
        ),
    ] = 512,
    batch_size: Annotated[
        int, Field(description="Amount of bags processed together.")
    ] = 64,
) -> str:
    """
    Train a model using clinical data and WSI-derived features via STAMP.
    Takes in a clinical table, slide associations, and extracted features
    to train a model on a specified label.

    Returns:
        str: message indicating the success or failure of the training operation,
        along with any relevant console output.

    Example:
        >>> train_stamp(
                output_dir="output/models",
                clini_table="input/clinical_data.csv",
                feature_dir="input/features",
                slide_table="input/slide_table.csv",
                ground_truth_label="OUTCOME",
                categories=["Positive", "Negative"],
                patient_label="PATIENT",
                filename_label="FILENAME"
            )
        "Command completed successfully: ..."
    """
    config = {
        "training": {
            "output_dir": output_dir,
            "clini_table": clini_table,
            "feature_dir": feature_dir,
            "slide_table": slide_table,
            "ground_truth_label": ground_truth_label,
            "categories": categories,
            "patient_label": patient_label,
            "filename_label": filename_label,
            "bag_size": bag_size,
            "batch_size": batch_size,
        }
    }
    return await _run_stamp(mode="train", config=config, ctx=ctx)


@mcp.tool
async def crossval_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to savethe results to")
    ],
    clini_table: Annotated[
        str, Field(description="Path of aCSV or Excel to readclinical data from")
    ],
    feature_dir: Annotated[
        str, Field(description="Path of the directory containingfeature files")
    ],
    slide_table: Annotated[
        str,
        Field(
            description="Path to a CSV or Excel to readpatient-slide associations from"
        ),
    ],
    ground_truth_label: Annotated[
        str,
        Field(description="Name of categorical column in clinical table to train on"),
    ],
    categories: Annotated[
        list[str],
        Field(
            description="The categories "
            "occurring in the target label column of the clini table."
        ),
    ],
    patient_label: Annotated[
        str,
        Field(
            description="Name of the column "
            "in the clini table that contains a unique ID for each patient"
        ),
    ] = "PATIENT",
    filename_label: Annotated[
        str,
        Field(
            description="Name of the column "
            "in the slide table containing the feature file path relative to `feature_dir`"
        ),
    ] = "FILENAME",
    n_folds: Annotated[
        int, Field("Number of folds to split the data into for cross-validation")
    ] = 3,
    bag_size: Annotated[
        int,
        Field(
            description="Amount of tiles to sample when training. "
            "Reducing this value reduces memory usage, but it is not recommended as the model can miss"
            "relevant regions of the slide. Default value works well on H&E tissue images."
        ),
    ] = 512,
    batch_size: Annotated[
        int, Field(description="Amount of bags processed together.")
    ] = 64,
) -> str:
    """
    Perform cross-validation for model training using STAMP.
    Splits the data into folds and trains a model on each to assess
    generalization. Uses clinical data, features, and slide mappings.

    Returns:
        str: A message indicating the success or failure of the cross-validation operation, along with
             any relevant console output from the STAMP pipeline.

    Example:
        >>> crossval_stamp(
                output_dir="output/crossval",
                clini_table="input/clinical_data.csv",
                feature_dir="input/features",
                slide_table="input/slide_table.csv",
                ground_truth_label="OUTCOME",
                categories=["Positive", "Negative"],
                patient_label="PATIENT",
                filename_label="FILENAME",
                n_folds=5
            )
        "Command completed successfully: ..."
    """
    config = {
        "training": {
            "output_dir": output_dir,
            "clini_table": clini_table,
            "feature_dir": feature_dir,
            "slide_table": slide_table,
            "ground_truth_label": ground_truth_label,
            "categories": categories,
            "patient_label": patient_label,
            "filename_label": filename_label,
            "n_folds": n_folds,
            "bag_size": bag_size,
            "batch_size": batch_size,
        }
    }
    return await _run_stamp(mode="crossval", config=config, ctx=ctx)


@mcp.tool
async def deploy_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to save the results to")
    ],
    clini_table: Annotated[
        str, Field(description="Path to a CSV or Excel to read clinical data from")
    ],
    feature_dir: Annotated[
        str, Field(description="Path of the directory containing feature files")
    ],
    slide_table: Annotated[
        str,
        Field(
            description="Path to a CSV or Excel to read patient-slide associations from"
        ),
    ],
    ground_truth_label: Annotated[
        str,
        Field(description="Name of categorical column in clinical table to train on"),
    ],
    categories: Annotated[
        list[str],
        Field(
            description="The categories "
            "occurring in the target label column of the clini table."
        ),
    ],
    checkpoint_paths: Annotated[
        list[str],
        Field(
            description="You can also "
            "combine multiple models to get a majority vote from multiple models. "
            "This is especially handy to e.g. combine the models from a cross-validation."
        ),
    ],
    patient_label: Annotated[
        str,
        Field(
            description="Name of the column "
            "in the clini table that contains a unique ID for each patient"
        ),
    ] = "PATIENT",
    filename_label: Annotated[
        str,
        Field(
            description="Name of the column "
            "in the slide table containing the feature file path relative to `feature_dir`"
        ),
    ] = "FILENAME",
) -> str:
    """
    Run inference using trained STAMP model(s).

    Generates predictions for patients using one or more model checkpoints and associated feature data.

    Returns:
        str: A message indicating the success or failure of the deployment operation, along with
             any relevant console output from the STAMP pipeline.

    Example:
        >>> deploy_stamp(
                output_dir="output/predictions",
                clini_table="input/clinical_data.csv",
                feature_dir="input/features",
                slide_table="input/slide_table.csv",
                ground_truth_label="OUTCOME",
                categories=["Positive", "Negative"],
                checkpoint_paths=["models/checkpoint1.pth", "models/checkpoint2.pth"],
                patient_label="PATIENT",
                filename_label="FILENAME"
            )
        "Command completed successfully: ..."
    """
    config = {
        "training": {
            "output_dir": output_dir,
            "clini_table": clini_table,
            "feature_dir": feature_dir,
            "slide_table": slide_table,
            "ground_truth_label": ground_truth_label,
            "categories": categories,
            "patient_label": patient_label,
            "filename_label": filename_label,
            "checkpoint_paths": checkpoint_paths,
        }
    }
    return await _run_stamp(mode="deploy", config=config, ctx=ctx)


@mcp.tool
async def statistics_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to save the results to")
    ],
    ground_truth_label: Annotated[str, Field(description="Name of the target label.")],
    true_class: Annotated[
        str,
        Field(description="The positive class to calculate ove-vs-all statistics for."),
    ],
    pred_csvs: Annotated[
        list[str],
        Field(
            description="List of CSV filepaths containing "
            "patient predictions to generate statistics from."
        ),
    ],
) -> str:
    """
    Generate evaluation metrics for model predictions.

    Computes classification statistics (e.g., precision, recall, F1)
    from prediction CSVs, using the specified target and positive class.

    Returns:
        str: A message indicating the success or failure of the statistics generation operation,
             along with any relevant console output from the STAMP pipeline.

    Example:
        >>> statistics_stamp(
                output_dir="output/statistics",
                ground_truth_label="OUTCOME",
                true_class="Positive",
                pred_csvs=["predictions/fold1.csv", "predictions/fold2.csv"]
            )
        "Command completed successfully: ..."
    """
    config = {
        "statistics": {
            "output_dir": output_dir,
            "ground_truth_label": ground_truth_label,
            "true_class": true_class,
            "pred_csvs": pred_csvs,
        }
    }
    return await _run_stamp(mode="statistics", config=config, ctx=ctx)


@mcp.tool
async def heatmaps_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to savethe results to")
    ],
    feature_dir: Annotated[
        str, Field(description="Path of the directory containingfeature files")
    ],
    wsi_dir: Annotated[
        str, Field(description="Path of the directory containing whole slide images")
    ],
    checkpoint_path: Annotated[
        str, Field(description="Path of the model to generate the heatmaps with.")
    ],
    slide_paths: Annotated[
        list[str] | None,
        Field(
            description="List of slide paths relative "
            "to `wsi_dir` to generate heatmaps for. If not specified, heatmaps will be generated "
            "for all slides in `wsi_dir`."
        ),
    ] = None,
    topk: Annotated[
        int | None, Field(description="Number of top-scoring tiles to extract")
    ] = None,
    bottomk: Annotated[
        int | None, Field(description="Number of bottom-scoring tiles to extract")
    ] = None,
) -> str:
    """
    Generate heatmaps and tile scorings from WSIs using a trained model.
    Produces visual explanations and optionally extracts top/bottom
    scoring tiles.

    Returns:
        str: A message indicating the success or failure of the heatmap generation operation,
             along with any relevant console output from the STAMP pipeline.

    Example:
        >>> heatmaps_stamp(
                output_dir="output/heatmaps",
                feature_dir="input/features",
                wsi_dir="input/slides",
                checkpoint_path="models/checkpoint.pth",
                slide_paths=["slide1.svs", "slide2.svs"],
                topk=10,
                bottomk=5
            )
        "Command completed successfully: ..."
    """
    config = {
        "heatmaps": {
            "output_dir": output_dir,
            "feature_dir": feature_dir,
            "wsi_dir": wsi_dir,
            "checkpoint_path": checkpoint_path,
            "slide_paths": slide_paths,
            "topk": topk,
            "bottomk": bottomk,
        }
    }
    return await _run_stamp(mode="heatmaps", config=config, ctx=ctx)


@mcp.tool
async def encode_slides_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to savethe results to")
    ],
    feature_dir: Annotated[
        str, Field(description="Path of the directory containingfeature files")
    ],
    wsi_dir: Annotated[
        str, Field(description="Path of the directory containing whole slide images")
    ],
    encoder: Annotated[
        str,
        Field(
            description="Name of the encoder to use "
            'for feature extraction. Possible options are "chief" and "cobra"'
        ),
    ] = "chief",
    device: Annotated[
        str,
        Field(
            description="The device to use for computation. "
            "Possible options are 'cuda' for NVIDIA GPUs, 'cpu' for general-purpose "
            "processors, and 'mps' for Apple Silicon GPUs. Default is 'cuda'."
        ),
    ] = "cuda",
) -> str:
    """Tile-Level features can be enconded into a single feature per slide,
    this is useful when trying to capture global patterns across whole slides.
    This tool takes as input tile-level features stored in .h5 files (one file per slide
    containing the features in the field "feats" concatenated in a tensor).
    The output is one .h5 file per slide.

    Returns:
        str: A message indicating the success or failure of the encoding operation, along with
             any relevant console output from the STAMP pipeline.

    Example:
        >>> encode_slides_stamp(
                output_dir="output/features",
                wsi_dir="input/slides",
                encoder="chief",
                device="cuda",
                feature_dir="input/features"
            )
        "Command completed successfully: ..."
    """
    config = {
        "slide_encoding": {
            "output_dir": output_dir,
            "wsi_dir": wsi_dir,
            "encoder": encoder,
            "device": device,
            "feature_dir": feature_dir,
        }
    }

    return await _run_stamp(mode="encode_slides", config=config, ctx=ctx)


@mcp.tool
async def encode_patients_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to savethe results to")
    ],
    feature_dir: Annotated[
        str, Field(description="Path of the directory containingfeature files")
    ],
    wsi_dir: Annotated[
        str, Field(description="Path of the directory containing whole slide images")
    ],
    slide_table: Annotated[
        str,
        Field(
            description="Path to a CSV or Excel to readpatient-slide associations from"
        ),
    ],
    encoder: Annotated[
        str,
        Field(
            description="Name of the encoder to use "
            'for feature extraction. Possible options are "chief" and "cobra"'
        ),
    ] = "chief",
    device: Annotated[
        str,
        Field(
            description="The device to use for computation. "
            "Possible options are 'cuda' for NVIDIA GPUs, 'cpu' for general-purpose "
            "processors, and 'mps' for Apple Silicon GPUs. Default is 'cuda'."
        ),
    ] = "cuda",
) -> str:
    """Tile-Level features can be enconded into a single feature per patient,
    this is useful when trying to capture global patterns across whole slides.
    This tool takes as input tile-level features stored in .h5 files (one file per slide
    containing the features in the field "feats" concatenated in a tensor).
    The output is one .h5 file per feature.

    Returns:
        str: A message indicating the success or failure of the encoding operation, along with
             any relevant console output from the STAMP pipeline.

    Example:
        >>> encode_patients_stamp(
                output_dir="output/features",
                wsi_dir="input/slides",
                encoder="chief",
                device="cuda",
                feature_dir="input/features"
                slide_table="input/slide_table.csv",
            )
        "Command completed successfully: ..."
    """
    config = {
        "patient_encoding": {
            "output_dir": output_dir,
            "wsi_dir": wsi_dir,
            "encoder": encoder,
            "device": device,
            "feature_dir": feature_dir,
            "slide_table": slide_table,
        }
    }

    return await _run_stamp(mode="encode_patients", config=config, ctx=ctx)


def _resolve_path(path: str) -> str:
    """
    Resolves the absolute path and ensures it's within the allowed base directory.

    Args:
        path (str): Relative path from the allowed base directory.

    Returns:
        str: Safe, absolute path.

    Raises:
        PermissionError: If the resolved path is outside the allowed directory.
    """
    abs_path = os.path.abspath(os.path.join(base_dir, path))
    if not abs_path.startswith(base_dir):
        raise PermissionError(f"Access denied to path: {path}")
    return abs_path


@mcp.tool
def read_file(path: str) -> str:
    """
    Read the contents of a file inside the allowed folder.

    Args:
        path (str): Relative path to the file.

    Returns:
        str: Content of the file.
    """
    safe_path = _resolve_path(path)
    with open(safe_path, "r", encoding="utf-8") as f:
        return f.read()


@mcp.tool
def list_files(subdir: str = "") -> list:
    """
    List all files under the given subdirectory (default is root), recursively,
    returning paths relative to the base directory.

    Args:
        subdir (str): Relative subdirectory path to list files from.

    Returns:
        list: List of relative file paths found.
    """
    safe_path = _resolve_path(subdir)
    file_list = []
    for root, _, files in os.walk(safe_path):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, base_dir)
            file_list.append(rel_path)
    return file_list


@mcp.tool
def check_available_devices() -> str:
    """
    Check which computation devices are available on the system.
    This includes checking for cuda (NVIDIA GPUs) and mps (Apple Silicon GPUs).

    Returns:
        A string describing the available devices.
    """
    devices = []

    # Check for CUDA availability
    if torch.cuda.is_available():
        devices.append("cuda")

    # Check for MPS availability (Apple Silicon GPUs)
    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        devices.append("mps")

    # Check for CPU (always available)
    devices.append("cpu")

    # Format the result
    if devices:
        return f"Available devices: {', '.join(devices)}"
    else:
        return "No computation devices are available."


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
