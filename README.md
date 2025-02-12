# STAMP: A Protocol for Solid Tumor Associative Modeling in Pathology

<img src="docs/STAMP_logo.svg" width="250px" align="right"></img>
This repository contains the accompanying code for the steps described in the [Nature Protocols paper][stamp paper]:
"From Whole Slide Image to Biomarker Prediction:
A Protocol for End-to-End Deep Learning in Computational Pathology".

> **Note**
> This repo contains an updated version of the codebase.
> For a version compatible with the instructions in the paper,
> please check out [version 1 of STAMP][stamp v1].

[stamp paper]: https://www.nature.com/articles/s41596-024-01047-2 "From whole-slide image to biomarker prediction: end-to-end weakly supervised deep learning in computational pathology"
[stamp v1]: https://github.com/KatherLab/STAMP/tree/v1

## Installing stamp

To install stamp, run:
```bash
# We recommend using a virtual environment to install stamp
python -m venv .venv
. .venv/bin/activate

pip install "stamp[all] @ git+https://github.com/KatherLab/stamp"
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
