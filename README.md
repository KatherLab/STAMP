[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/from-whole-slide-image-to-biomarker/classification-on-tcga)](https://paperswithcode.com/sota/classification-on-tcga?p=from-whole-slide-image-to-biomarker)

> [!Important]
> STAMP v1.0.3 now has built-in support for the [UNI Feature extractor](https://www.nature.com/articles/s41591-024-02857-3). Using it will require a Hugging Face account with granted access to the UNI model. For details on fair use, licensing and accessing the UNI model weights, refer to the [UNI GitHub repository](https://www.github.com/mahmoodlab/UNI.git). Note that the installation instructions within the [STAMP protocol paper](https://arxiv.org/abs/2312.10944v1) refer to v1.0.1 of the software, and that v1.0.3 has updated installation steps, see below. The README file will always contain the most up-to-date installation instructions.

# STAMP protocol <img src="docs/STAMP_logo.svg" width="250px" align="right" />
A protocol for Solid Tumor Associative Modeling in Pathology. This repository contains the accompanying code for the steps described in the [preprint](https://arxiv.org/abs/2312.10944v1): 

>From Whole Slide Image to Biomarker Prediction: A Protocol for End-to-End Deep Learning in Computational Pathology 

The code can be executed either in a local environment, or in a containerized environment (preferred in clusters).

## Using a local environment
For setting up a local environment, note that the following steps are for Ubuntu Linux systems. For other operating systems such as Windows, MacOS or other Linux distributions, it is recommend to use the containerized environment as described below.

First, install OpenSlide using either the command below or the [official installation instructions](https://openslide.org/download/#distribution-packages):
```bash
apt update && apt install -y openslide-tools libgl1-mesa-glx # libgl1-mesa-glx is needed for OpenCV
```

Second, [install conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html) on your local computer, create an environment with Python 3.10, and activate it:

```bash
conda create -n stamp python=3.10
conda activate stamp
conda install -c conda-forge libstdcxx-ng=12
```

Then, install the STAMP package via `pip`:
```bash
pip install git+https://github.com/KatherLab/STAMP
```

Once installed, you will be able to run the command line interface directly using the `stamp` command.

Next, initialize STAMP and obtain the required configuration file, config.yaml, in your current working directory, by running the following command:

```bash
stamp init
```

To download required resources such as the weights of the feature extractor, run the following command:
```bash
stamp setup
```

> [!Note]
> If you select a different feature extractor withing the configuration file, such as UNI, you will need to re-run the previous setup command to initiate the downloading step of the UNI feature extractor weights. This will trigger a prompt asking for your Hugging Face access key for the UNI model weights.

## Using the container
First, install Go and Singularity on your local machine using the [official installation instructions](https://docs.sylabs.io/guides/3.0/user-guide/installation.html). Note that the High-Performance Cluster (HPC) has Go and Singularity pre-installed, and do not require installation.

### Build container from scratch (requires root)
Second, build the container first on your local machine with (fake) root access:
```bash
sudo singularity build STAMP_container.sif setup/container.def
```
Note that the container is approximately 6 GB in size.

### Download pre-built container
Alternatively, lab members with access to the ZIH server can download the pre-built container into the base STAMP directory from:

```bash
/glw/ekfz_proj/STAMP_container.sif
```

Finally, to download required resources such as the weights of the CTransPath feature extractor, run the following command in the base directory of the protocol:
```bash
singularity run --nv -B /mnt:/mnt STAMP_container.sif "stamp --config /path/to/config.yaml setup"
```
Note that the binding of filesystems (-B) should be adapted to your own system. GPU acceleration (--nv) should be enabled if GPUs are available in the system, but is optional.

## Running
Available commands are:
```bash
stamp init       # create a new configuration file in the current directory
stamp setup      # download required resources
stamp config     # print resolved configuration
stamp preprocess # normalization and feature extraction with CTransPath
stamp crossval   # train n_splits models using cross-validation
stamp train      # train single model
stamp deploy     # deploy a model on another test set
stamp statistics # compute stats including ROC curves
stamp heatmaps   # generate heatmaps
```

> [!NOTE]  
> By default, STAMP will use the configuration file `config.yaml` in the current working directory (or, if that does not exist, it will use the [default STAMP configuration file](stamp/config.yaml) shipped with this package). If you want to use a different configuration file, use the `--config` command line option, i.e. `stamp --config some/other/file.yaml train`. You may also run `stamp init` to create a local `config.yaml` in the current working directory initialized to the default settings.

## Reference

If you find our work useful in your research or if you use parts of this code please consider citing our [preprint](https://arxiv.org/abs/2312.10944v1):

```
@misc{nahhas2023wholeslide,
      title={From Whole-slide Image to Biomarker Prediction: A Protocol for End-to-End Deep Learning in Computational Pathology}, 
      author={Omar S. M. El Nahhas and Marko van Treeck and Georg Wölflein and Michaela Unger and Marta Ligero and Tim Lenz and Sophia J. Wagner and Katherine J. Hewitt and Firas Khader and Sebastian Foersch and Daniel Truhn and Jakob Nikolas Kather},
      year={2023},
      eprint={2312.10944},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
