# STAMP: A Protocol for Solid Tumor Associative Modeling in Pathology

<img src="docs/STAMP_logo.svg" width="250px" align="right"></img>

![CI](https://github.com/KatherLab/STAMP/actions/workflows/ci.yml/badge.svg)
[![STAMP • Nature Protocols](https://img.shields.io/badge/Nature%20Protocols%20Paper-gray.svg)](https://www.nature.com/articles/s41596-024-01047-2)

*An efficient, ready‑to‑use workflow from whole‑slide image to biomarker prediction.*

STAMP is an **end‑to‑end, weakly‑supervised deep‑learning pipeline** that helps discover and evaluate candidate image‑based biomarkers from gigapixel histopathology slides, no pixel‑level annotations required. Backed by a peer‑reviewed protocol and used in multi‑center studies across several tumor types, STAMP lets clinical researchers and machine‑learning engineers collaborate on reproducible computational‑pathology projects with a clear, structured workflow.

**Want to start now?** [Jump to Installation](#installation) or [walk through our Getting Started guide](getting-started.md) for a hands-on tutorial.

## **Why choose STAMP?**

* 🚀 **Scalable**: Run locally or on HPC (SLURM) with the same CLI; built to handle multi‑center cohorts and large WSI collections.  
* 🎓 **Beginner‑friendly & expert‑ready**: Zero‑code CLI and YAML config for routine use; optional code‑level customization for advanced research.  
* 🧩 **Model‑rich**: Out‑of‑the‑box support for **+20 foundation models** at [tile level](getting-started.md#feature-extraction) (e.g., *Virchow‑v2*, *UNI‑v2*) and [slide level](getting-started.md#slide-level-encoding) (e.g., *TITAN*, *COBRA*).  
* 🔬 **Weakly‑supervised**: End‑to‑end MIL with Transformer aggregation for training, cross‑validation and external deployment; no pixel‑level labels required.  
* 🧮 **Multi-task learning**: Unified framework for **classification**, **multi-target classification**, **regression**, and **cox-based survival analysis**.
* 📊 **Stats & results**: Built‑in metrics and patient‑level predictions, ready for analysis and reporting.  
* 🖼️ **Explainable**: Generates heatmaps and top‑tile exports out‑of‑the‑box for transparent model auditing and publication‑ready figures.  
* 🤝 **Collaborative by design**: Clinicians drive hypothesis & interpretation while engineers handle compute; STAMP’s modular CLI mirrors real‑world workflows and tracks every step for full reproducibility.  
* 📑 **Peer‑reviewed**: Protocol published in [*Nature Protocols*](https://www.nature.com/articles/s41596-024-01047-2) and validated across multiple tumor types and centers.  
* **🔗 MCP Support**: Compatible with Model Context Protocol (MCP) via the \`mcp/\` module, ready for integration into next-gen agentic AI workflows.

## **Real-World Examples of STAMP in Action**

- **[Squamous Tumors & Survival](https://www.sciencedirect.com/science/article/pii/S0893395225001425):** In a multi-cohort study spanning four squamous carcinoma types (head & neck, esophageal, lung, cervical), STAMP was used to extract slide-level features for a deep learning model that predicted patient survival directly from H&E whole-slide images.  

- **[Inflammatory Bowel Disease Atlas](https://www.researchsquare.com/article/rs-6443303/v1):** In a 1,002-patient multi-center IBD study, all histology slides were processed with the STAMP workflow, enabling a weakly-supervised MIL model to accurately predict histologic disease activity scores from H&E tissue sections.  

- **[Foundation Model Benchmarking](https://arxiv.org/pdf/2408.15823):** A large-scale evaluation of 19 pathology foundation models built its pipeline on STAMP (v1.1.0) for standardized WSI tiling and feature extraction, demonstrating STAMP’s utility as an open-source framework for reproducible model training across diverse cancer biomarkers.  

- **[Breast Cancer Risk Stratification](https://doi.org/10.1038/s41467-025-57283-x):** In an international early breast cancer study, STAMP performed slide tessellation and color normalization (e.g. 1.14 µm/px resolution, Macenko norm) as part of a multimodal transformer pipeline to predict recurrence risk (Oncotype DX scores) from pathology images.  

- **[Endometrial Cancer Subtyping](https://www.actscience.org/Portals/0/Translational%20Science%202025/Top%2050%20Posters/TS25_VincentWagner_OralPosterSession.pdf):** A recent endometrial cancer project employed a modified STAMP pipeline with a pre-trained vision transformer (Virchow2) to predict molecular tumor subtypes directly from H&E slides, achieving strong diagnostic performance in cross-validation.  


## Installation
To setup STAMP you need [uv](https://docs.astral.sh/uv/) 0.12 or newer.

### Supported platforms

| | Python | Platforms |
|---|---|---|
| CPU (`--extra cpu`) | 3.13, 3.14 | Linux x86_64/aarch64, Windows AMD64, Apple Silicon macOS |
| CUDA (`--extra gpu`, `--extra gpu_all`) | 3.13, 3.14 | Linux x86_64, Linux aarch64 |

Python **3.14 is recommended** and is what `.python-version` selects; 3.13 remains
supported. The CUDA builds are pinned to **CUDA 13.0 with PyTorch 2.11.0 and
TorchVision 0.26.0**. There is no CUDA build for macOS or Windows — use
`--extra cpu` there.

> [!IMPORTANT]
> Install with uv, not pip. The wheel indexes STAMP needs are configured in
> `[tool.uv.sources]` in `pyproject.toml`, which pip does not read.

### Install or Update uv:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Update uv
uv self update
```

### Install STAMP from the Repository:

```bash
git clone https://github.com/KatherLab/STAMP.git
cd STAMP
```

```bash
# GPU (CUDA) Installation (excluding conchv1_5, gigapath and musk)

uv sync --extra gpu
source .venv/bin/activate
```

```bash
# CPU-only Installation (excluding conchv1_5, gigapath and musk)

uv sync --extra cpu
source .venv/bin/activate
```

For the full GPU stack, add `conchv1_5`, `gigapath` and `musk`:

```bash
# GPU (CUDA) Installation - everything
uv sync --extra gpu_all
source .venv/bin/activate
```

Both GPU extras install pre-built wheels, so nothing is compiled locally and no
CUDA toolkit is needed to install STAMP. You still need an NVIDIA driver to run
on a GPU.

> [!NOTE]
> `--extra gpu_prebuilt` is a deprecated alias for `--extra gpu_all` and will be
> removed in the next breaking release.

If you encounter errors during installation please read Installation Troubleshooting [below](#installation-troubleshooting).

### Additional Dependencies

> [!IMPORTANT]
> STAMP additionally requires OpenCV dependencies to be installed.
>

> For Ubuntu < 23.10:
> ```bash
> apt update && apt install -y libgl1-mesa-glx
> ```
>
> For Ubuntu >= 23.10:
> ```bash
> apt update && apt install -y libgl1 libglx-mesa0 libglib2.0-0
> ```

## Basic Usage

If the installation was successful, running `stamp` in your terminal should yield the following output:
```
$ stamp
usage: stamp [-h] [--config CONFIG_FILE_PATH] {init,preprocess,encode_slides,encode_patients,train,crossval,deploy,statistics,config,heatmaps} ...

STAMP: Solid Tumor Associative Modeling in Pathology

positional arguments:
  {init,preprocess,encode_slides,encode_patients,train,crossval,deploy,statistics,config,heatmaps}
    init                Create a new STAMP configuration file at the path specified by --config
    preprocess          Preprocess whole-slide images into feature vectors
    encode_slides       Encode patch-level features into slide-level embeddings
    encode_patients     Encode features into patient-level embeddings
    train               Train a Vision Transformer model
    crossval            Train a Vision Transformer model with cross validation for modeling.n_splits folds
    deploy              Deploy a trained Vision Transformer model
    statistics          Generate AUROCs and AUPRCs with 95%CI for a trained Vision Transformer model
    config              Print the loaded configuration
    heatmaps            Generate heatmaps for a trained model

options:
  -h, --help            show this help message and exit
  --config CONFIG_FILE_PATH, -c CONFIG_FILE_PATH
                        Path to config file. Default: config.yaml
```

## Getting Started Guide

For a quick introduction how to run stamp,
check out our [getting started guide](getting-started.md).

## Reference

If you find our work useful in your research
or if you use parts of this code
please consider citing our [Nature Protocols publication](https://www.nature.com/articles/s41596-024-01047-2):
```
@Article{ElNahhas2024,
  author={El Nahhas, Omar S. M. and van Treeck, Marko and W{\"o}lflein, Georg and Unger, Michaela and Ligero, Marta and Lenz, Tim and Wagner, Sophia J. and Hewitt, Katherine J. and Khader, Firas and Foersch, Sebastian and Truhn, Daniel and Kather, Jakob Nikolas},
  title={From whole-slide image to biomarker prediction: end-to-end weakly supervised deep learning in computational pathology},
  journal={Nature Protocols},
  year={2024},
  month={Sep},
  day={16},
  issn={1750-2799},
  doi={10.1038/s41596-024-01047-2},
  url={https://doi.org/10.1038/s41596-024-01047-2}
}
```

> [!NOTE]
> This repo contains an updated version of the codebase.
> For a version compatible with the instructions in the paper,
> please check out [version 1 of STAMP][stamp v1].

[stamp paper]: https://www.nature.com/articles/s41596-024-01047-2 "From whole-slide image to biomarker prediction: end-to-end weakly supervised deep learning in computational pathology"
[stamp v1]: https://github.com/KatherLab/STAMP/tree/v1

## Installation Troubleshooting

#### `flash-attn` / `mamba-ssm` / `causal-conv1d` cannot be installed

These only exist as pre-built wheels for Linux x86_64 and Linux aarch64 on
Python 3.13/3.14. On any other platform uv reports that no compatible version
was found. Use `--extra cpu` on macOS and Windows.

#### Triton Errors

If you encounter errors related to the [Triton package like the following](https://github.com/pytorch/pytorch/issues/153737):

```bash
SystemError: PY_SSIZE_T_CLEAN macro must be defined for '#' formats
``` 

Try to delete the triton cache: 

```bash
rm -r ~/.triton
```

A re-installation might be necessary afterwards.

#### Undefined Symbol Error

An error like the following when importing `flash_attn`, `mamba_ssm` or
`causal_conv1d` means the installed torch does not match the torch the extension
was compiled against:

```
>       import flash_attn_2_cuda as flash_attn_gpu
E       ImportError: [...]/.venv/lib/python3.14/site-packages/flash_attn_2_cuda.cpython-314-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE
```

This usually means the environment is left over from an older STAMP release.
Check what is installed:

```bash
uv run python scripts/verify_installed_stack.py --expect-cuda
```

and rebuild the environment from the lockfile:

```bash
rm -rf .venv
uv sync --locked --extra gpu_all
```

## Reproducibility
> [!NOTE]
> We use a central `Seed` utility to set seeds for PyTorch, NumPy, and Python’s `random`. This makes data loading and model initialization reproducible. Always call `Seed.set(seed)` once at startup.
> We do not enable [`torch.use_deterministic_algorithms()`](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility) because it can cause large performance drops. Expect runs with the same seed to follow the same training trajectory, but not bit-for-bit identical low-level kernels.
