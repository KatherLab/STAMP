# STAMP: A Protocol for Solid Tumor Associative Modeling in Pathology

<img src="docs/STAMP_logo.svg" width="250px" align="right"></img>

![CI](https://github.com/KatherLab/STAMP/actions/workflows/ci.yml/badge.svg)

This repository contains the accompanying code for the steps described in the [Nature Protocols paper][stamp paper]:
"From Whole Slide Image to Biomarker Prediction:
A Protocol for End-to-End Deep Learning in Computational Pathology".

> [!NOTE]
> This repo contains an updated version of the codebase.
> For a version compatible with the instructions in the paper,
> please check out [version 1 of STAMP][stamp v1].

[stamp paper]: https://www.nature.com/articles/s41596-024-01047-2 "From whole-slide image to biomarker prediction: end-to-end weakly supervised deep learning in computational pathology"
[stamp v1]: https://github.com/KatherLab/STAMP/tree/v1

## Installation

We recommend installing STAMP with [uv](https://docs.astral.sh/uv/):

### Install or Update uv:

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Update uv
uv self update
```

### Install STAMP in a Virtual Environment:

```bash
uv venv --python=3.12
source .venv/bin/activate

# For a CPU-only installation:
uv pip install "git+https://github.com/KatherLab/STAMP.git@fix/build[cpu]" --torch-backend=cpu

# For a GPU (CUDA) installation:
uv pip install "git+https://github.com/KatherLab/STAMP.git@fix/build[build]"
uv pip install "git+https://github.com/KatherLab/STAMP.git@fix/build[build,gpu]" --no-build-isolation

# Note: You must run one after the other, the build dependencies must be installed first!
```

### Install STAMP from the Repository:

```bash
git clone https://github.com/KatherLab/STAMP.git
cd STAMP
```


```bash
# CPU-only Installation (excluding COBRA, Gigapath (and flash-attn))
uv sync --extra cpu

source .venv/bin/activate
```

```bash
# GPU (CUDA) Installation (Using flash-attn on CUDA systems for gigapath and other models)

# First run this!!
uv sync --extra build

# And then this for all models:
uv sync --extra build --extra gpu

# Alternatively, you can install only a specific model:
uv sync --extra build --extra uni


# In case building flash-attn uses too much memory, you can limit the number of parallel compilation jobs:
MAX_JOBS=4 uv sync --extra build --extra gpu
```

### Additional Dependencies

> [!IMPORTANT]
> STAMP additionally requires OpenCV dependencies to be installed. If you want to use `flash-attn`, you also need to install the `clang` compiler and a [CUDA toolkit](https://developer.nvidia.com/cuda-downloads).
>

> For Ubuntu < 23.10:
> ```bash
> apt update && apt install -y libgl1-mesa-glx clang
> ```
>
> For Ubuntu >= 23.10:
> ```bash
> apt update && apt install -y libgl1 libglx-mesa0 libglib2.0-0 clang
> ```

### Basic Usage

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

## Running STAMP

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
