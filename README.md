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
uv pip install "git+https://github.com/KatherLab/STAMP.git[cpu]" --torch-backend=cpu

# For a GPU (CUDA) installation:
uv pip install "git+https://github.com/KatherLab/STAMP.git[build]"
uv pip install "git+https://github.com/KatherLab/STAMP.git[build,gpu]" --no-build-isolation

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


### Installation Troubleshooting

> [!NOTE]
> Installing the GPU version of STAMP will force the compilation of the `flash-attn` package (as well as `mamba-ssm` and `causal_conv1d`). This can take a long time and requires a lot of memory. You can limit the number of parallel compilation jobs by setting the `MAX_JOBS` environment variable before running the installation command, e.g. `MAX_JOBS=4 uv sync --extra build --extra gpu`.


#### Undefined Symbol Error

If you encounter an error similar to the following when importing flash_attn, mamba or causal_conv1d on a GPU system, it usually indicates that the torch version in your environment does not match the torch version used to build the flash-attn, mamba or causal_conv1d package. This can happen if you already built these packages for another environment or if for any reason between the installation commands with only `--extra build` and `--extra gpu` the torch version was changed.

```
>       import flash_attn_2_cuda as flash_attn_gpu
E       ImportError: [...]/.venv/lib/python3.12/site-packages/flash_attn_2_cuda.cpython-312-x86_64-linux-gnu.so: undefined symbol: _ZN3c105ErrorC2ENS_14SourceLocationENSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEE

.venv/lib/python3.12/site-packages/flash_attn/flash_attn_interface.py:15: ImportError
```

In case you encounter this error on a gpu installation, you can fix it by going back to the environment just with `--extra build`, clearing the uv cache and then reinstalling the `--extra gpu` packages:

```bash
uv cache clean flash_attn
uv cache clean mamba-ssm
uv cache clean causal_conv1d

# Now it should re-build the packages with the correct torch version

# With uv pip install
uv pip install "git+https://github.com/KatherLab/STAMP.git[build]"
uv pip install "git+https://github.com/KatherLab/STAMP.git[build,gpu] --no-build-isolation"

# With uv sync in the cloned repository
uv sync --extra build
uv sync --extra build --extra gpu
```


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
