import asyncio
import logging
import os
from pathlib import Path
import platform
import tempfile
from typing import Annotated
import argparse

import torch
import yaml
from fastmcp import Context, FastMCP
from pydantic import Field
import pandas as pd
from stamp.__main__ import _run_cli


# Initialize the FastMCP server
mcp = FastMCP("STAMP MCP Server")


STAMP_LOGGER = logging.getLogger("stamp")
# TODO: add proper filesystem management
WORKSPACE_FOLDER = "./"  # Folder where the agent can work on.
WORKSPACE_PATH = Path(WORKSPACE_FOLDER).resolve()
# List of additional allowed paths outside workspace
ALLOWED_EXTERNAL_PATHS = [
    "/mnt/bulk-curie/peter/fmbenchmark/images/tcga_crc",
    "/mnt/bulk-curie/peter/fmbenchmark/20mag_experiments/features/tcga_crc/ctranspath/STAMP_raw_xiyuewang-ctranspath-7c998680",
    # Add other specific paths you want to allow
]
MAX_ITEMS = 100  # Max amount of files listed with list_files tool.
# Big values could exceed LLM's context length. When it exceeds, values are summarized.


class MCPLogHandler(logging.Handler):
    def __init__(self, ctx):
        super().__init__()
        self.ctx = ctx
        self.captured_logs = []  # Store captured logs

    def emit(self, record):
        msg = self.format(record)
        # Store the log message
        self.captured_logs.append(msg)
        # Fire-and-forget the coroutine
        asyncio.create_task(self.ctx.log(msg))


async def _run_stamp(mode, config, ctx):
    """
    Run the STAMP command directly by calling _run_cli() instead of subprocess.

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

    # Set up logging handler to capture STAMP logs
    handler = MCPLogHandler(ctx)
    handler.setLevel(logging.INFO)
    STAMP_LOGGER.addHandler(handler)

    try:
        # Create argparse Namespace object to mimic command line arguments
        args = argparse.Namespace(command=mode, config_file_path=Path(tmp_config_path))

        # Call the STAMP CLI function directly
        _run_cli(args)

        # Get captured logs
        captured_logs_text = (
            "\n".join(handler.captured_logs)
            if handler.captured_logs
            else "Command completed successfully (no logs captured)"
        )
        return f"Command completed successfully:\n{captured_logs_text}"

    except Exception as e:
        captured_logs_text = (
            "\n".join(handler.captured_logs) if handler.captured_logs else ""
        )
        error_msg = f"Command failed with error: {str(e)}\n{captured_logs_text}"
        return error_msg

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
) -> str:
    """
    Train a model using clinical data and WSI-derived features via STAMP.
    Takes in a clinical table, slide associations, and extracted features
    to train a model on a specified label. Best option when an external cohort is available.

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
        }
    }
    return await _run_stamp(mode="train", config=config, ctx=ctx)


@mcp.tool
async def crossval_stamp(
    ctx: Context,
    output_dir: Annotated[
        str, Field(description="Path of the directory to save the results to")
    ],
    clini_table: Annotated[
        str, Field(description="Path of a CSV or Excel to read clinical data from")
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
    n_splits: Annotated[
        int,
        Field(
            description="Number of folds to split the data into for cross-validation"
        ),
    ] = 5,
) -> str:
    """
    Perform cross-validation for model training using STAMP.
    Splits the data into folds and trains a model on each to assess
    generalization. Uses clinical data, features, and slide mappings.
    Best option when only one cohort is available.

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
                n_splits=5
            )
        "Command completed successfully: ..."
    """
    config = {
        "crossval": {  # Changed from "training" to "crossval"
            "output_dir": output_dir,
            "clini_table": clini_table,
            "feature_dir": feature_dir,
            "slide_table": slide_table,
            "ground_truth_label": ground_truth_label,
            "categories": categories,
            "patient_label": patient_label,
            "filename_label": filename_label,
            "n_splits": n_splits,
        },
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
                pred_csvs=["/pathto/split-0/patient-preds.csv", "/pathto/split-1/patient-preds.csv"]
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
        list[str],
        Field(
            description="List of slide paths relative to `wsi_dir` to "
            "generate heatmaps for. The slide paths HAVE to be specified relative to `wsi_dir`.",
            min_length=1,
        ),
    ],
    topk: Annotated[
        int | None, Field(description="Number of top-scoring tiles to extract")
    ] = None,
    bottomk: Annotated[
        int | None, Field(description="Number of bottom-scoring tiles to extract")
    ] = None,
    device: Annotated[
        str | None,
        Field(
            description="The device to use for computation. "
            "Possible options are 'cuda' for NVIDIA GPUs, 'cpu' for general-purpose "
            "processors, and 'mps' for Apple Silicon GPUs. Default is detected automatically"
        ),
    ] = None,
) -> str:
    """
    Generate heatmaps and tile scorings from WSIs using a trained model.

    Creates visual attention maps showing which regions the model focuses on for predictions.
    Works only with tile-level features. For each slide, generates:
    - Overview plots with complete heatmaps and class overlays
    - Raw data including thumbnails, class maps, and per-class heatmaps
    - Individual tile extractions (top/bottom scoring if specified)

    Output structure: Each slide gets its own folder
    (slide name without file extension)containing plots/, raw/, and tiles/ subdirectories.


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
                topk=3,
                bottomk=3
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
            "device": device,
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


def _resolve_path(subpath: str) -> Path:
    requested = Path(subpath).resolve()
    
    # Check if it's within workspace
    if WORKSPACE_PATH in requested.parents or requested == WORKSPACE_PATH:
        return requested
    
    # Check if it's in allowed external paths
    for allowed_path in ALLOWED_EXTERNAL_PATHS:
        allowed_path = Path(allowed_path).resolve()
        # Check both: exact match OR if allowed_path is a parent of requested
        if requested == allowed_path or allowed_path in requested.parents:
            return requested
    
    # If not allowed, raise error
    raise PermissionError(f"Access denied: {subpath}")


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
def list_files(subdir: str = "") -> str:
    """
    List all files and directories under the given subdirectory (default is root), recursively,
    returning paths relative to the base directory. If the list is too long, shows only directories
    with file type summaries. If still too long, shows a truncated message.

    Args:
        subdir (str): Relative subdirectory path to list files from.

    Returns:
        str: Formatted list of files/directories or summary information.
    """
    subdir_path = _resolve_path(subdir) if subdir else WORKSPACE_PATH
    if not subdir_path.is_dir():
        raise FileNotFoundError(f"Subdirectory does not exist: {subdir}")

    # Collect all files and directories
    all_items = []
    directories = {}
    base_len = len(str(WORKSPACE_PATH)) + 1  # To slice off base path + separator

    for root, dirs, files in os.walk(subdir_path):
        rel_root = str(root)[base_len:]  # relative path under base_dir

        # Track file types in each directory
        if rel_root not in directories:
            directories[rel_root] = {"subdirs": [], "file_types": {}, "file_count": 0}

        # Add subdirectories
        for d in dirs:
            path = os.path.join(rel_root, d)
            all_items.append(path + "/")
            directories[rel_root]["subdirs"].append(d)

        # Add files and track their extensions
        for f in files:
            path = os.path.join(rel_root, f)
            all_items.append(path)

            # Track file extension
            ext = Path(f).suffix.lower() or "no extension"
            directories[rel_root]["file_types"][ext] = (
                directories[rel_root]["file_types"].get(ext, 0) + 1
            )
            directories[rel_root]["file_count"] += 1

    # If the list is manageable, return the full list
    if len(all_items) <= MAX_ITEMS:
        return "\n".join(sorted(all_items))

    # Try directory summary instead
    dir_summary = []
    for dir_path, info in sorted(directories.items()):
        if not dir_path:  # Root directory
            dir_display = "/ (root)"
        else:
            dir_display = f"{dir_path}/"

        # File type summary
        if info["file_count"] > 0:
            file_types = []
            for ext, count in sorted(info["file_types"].items()):
                file_types.append(f"{count} {ext}")
            file_summary = f" [{', '.join(file_types)}]"
        else:
            file_summary = " [empty]"

        # Subdirectory info
        if info["subdirs"]:
            subdir_info = f" (contains {len(info['subdirs'])} subdirs)"
        else:
            subdir_info = ""

        dir_summary.append(f"{dir_display}{file_summary}{subdir_info}")

    # If directory summary is still too long, truncate
    if len(dir_summary) > MAX_ITEMS:
        total_dirs = len(directories)
        total_files = sum(info["file_count"] for info in directories.values())

        # Show first few directories and a summary
        shown_dirs = dir_summary[: MAX_ITEMS // 2]
        summary_text = (
            f"\n... (showing first {len(shown_dirs)} of {total_dirs} directories)\n\n"
            f"SUMMARY:\n"
            f"- Total directories: {total_dirs}\n"
            f"- Total files: {total_files}\n"
            f"- Directory '{subdir or '/'}' contains too many items to display completely.\n"
            f"- Use a more specific subdirectory path to see detailed listings."
        )

        # Get overall file type statistics
        all_extensions = {}
        for info in directories.values():
            for ext, count in info["file_types"].items():
                all_extensions[ext] = all_extensions.get(ext, 0) + count

        if all_extensions:
            ext_summary = []
            for ext, count in sorted(
                all_extensions.items(), key=lambda x: x[1], reverse=True
            )[:10]:
                ext_summary.append(f"  {ext}: {count} files")
            summary_text += "\n\nTop file types:\n" + "\n".join(ext_summary)
            if len(all_extensions) > 10:
                summary_text += (
                    f"\n  ... and {len(all_extensions) - 10} more file types"
                )

        return "\n".join(shown_dirs) + summary_text

    # Return directory summary
    header = f"Directory listing for '{subdir or '/'}' (showing directories with file type summaries):\n"
    return header + "\n".join(dir_summary)


@mcp.tool
def analyze_csv(path: str) -> str:
    """
    Analyze a CSV file and provide detailed information about its structure and contents.

    Args:
        path (str): Relative path to the CSV file.

    Returns:
        str: Detailed information about the CSV including dimensions, columns, and sample data.
    """
    safe_path = _resolve_path(path)

    if not safe_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    if safe_path.suffix.lower() not in [".csv", ".tsv"]:
        raise ValueError(f"File is not a CSV file: {path}")

    try:
        # Read the CSV file
        df = pd.read_csv(safe_path)

        # Get basic information
        num_rows, num_columns = df.shape
        column_names = df.columns.tolist()

        # Get first 3 rows as examples
        sample_rows = df.head(3).to_string(index=True, max_cols=None)

        # Format the output
        result = f"""CSV File Analysis: {path}
        
Dimensions:
- Number of rows: {num_rows:,}
- Number of columns: {num_columns}

Column Names:
{", ".join([f'"{col}"' for col in column_names])}

First 3 rows (sample data):
{sample_rows}

Data Types:
{df.dtypes.to_string()}
        """

        return result.strip()

    except pd.errors.EmptyDataError:
        return f"CSV file is empty: {path}"
    except pd.errors.ParserError as e:
        return f"Error parsing CSV file {path}: {str(e)}"
    except Exception as e:
        return f"Error analyzing CSV file {path}: {str(e)}"


@mcp.tool
def list_column_values(path: str, column_name: str) -> str:
    """
    List all unique values in a specific column of a CSV file.

    Args:
        path (str): Relative path to the CSV file.
        column_name (str): Name of the column to analyze.

    Returns:
        str: Information about the unique values in the specified column.
    """
    safe_path = _resolve_path(path)

    if not safe_path.exists():
        raise FileNotFoundError(f"CSV file does not exist: {path}")

    if safe_path.suffix.lower() not in [".csv", ".tsv"]:
        raise ValueError(f"File is not a CSV file: {path}")

    try:
        # Read the CSV file
        df = pd.read_csv(safe_path)

        # Check if column exists
        if column_name not in df.columns:
            available_columns = ", ".join([f'"{col}"' for col in df.columns])
            return f"Column '{column_name}' not found in CSV file: {path}\nAvailable columns: {available_columns}"

        # Get unique values
        unique_values = df[column_name].unique()

        # Count occurrences of each value
        value_counts = df[column_name].value_counts().sort_index()

        # Handle missing values
        null_count = df[column_name].isnull().sum()

        # Format the output
        result = f"""Column Analysis for '{column_name}' in {path}

Total rows: {len(df):,}
Unique values: {len(unique_values):,}
Missing/null values: {null_count:,}

Value distribution:
{value_counts.to_string()}
        """

        # If there are many unique values, show a sample
        if len(unique_values) > 20:
            result += f"""

First 20 unique values:
{", ".join([str(val) for val in unique_values[:20]])}
... and {len(unique_values) - 20} more values
            """
        else:
            result += f"""

All unique values:
{", ".join([str(val) for val in unique_values if pd.notna(val)])}
            """

        return result.strip()

    except pd.errors.EmptyDataError:
        return f"CSV file is empty: {path}"
    except pd.errors.ParserError as e:
        return f"Error parsing CSV file {path}: {str(e)}"
    except Exception as e:
        return f"Error analyzing column in CSV file {path}: {str(e)}"


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
