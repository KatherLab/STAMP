# STAMP: A Protocol for Solid Tumor Associative Modeling in Pathology

<img src="docs/STAMP_logo.svg" width="250px" align="right"></img>
This repository contains the accompanying code for the steps described in the [preprint](https://arxiv.org/abs/2312.10944v1):

> From Whole Slide Image to Biomarker Prediction:
> A Protocol for End-to-End Deep Learning in Computational Pathology

## Installing stamp

To install stamp, run
```bash
# We recommend using a virtual environment to install stamp
python -m venv .venv
. .venv/bin/activate

pip install "git+https://github.com/KatherLab/stamp@v2[all]"
```

If the installation was successful, running `stamp` in your terminal should yield the following output:

```
$ stamp
usage: stamp [-h] [--config CONFIG_FILE_PATH] {init,setup,preprocess,train,crossval,deploy,statistics,config,heatmaps} ...

STAMP: Solid Tumor Associative Modeling in Pathology

positional arguments:
  {init,setup,preprocess,train,crossval,deploy,statistics,config,heatmaps}
    init                Create a new STAMP configuration file at the path specified by --config
    preprocess          Preprocess whole-slide images into feature vectors
    train               Train a Vision Transformer model
    crossval            Train a Vision Transformer model with cross validation
    deploy              Deploy a trained Vision Transformer model
    statistics          Generate AUROCs and AUPRCs with 95%CI for a trained Vision Transformer model
    config              Print the loaded configuration
    heatmaps            Generate heatmaps for a trained model

options:
  -h, --help            show this help message and exit
  --config CONFIG_FILE_PATH, -c CONFIG_FILE_PATH
                        Path to config file. Default: config.yaml
```

## Running stamp

For a quick introduction how to run stamp,
check out our [getting started gudie](getting-started.md).


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
