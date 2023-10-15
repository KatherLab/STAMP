# STAMP protocol
A protocol for Solid Tumor Associative Modeling in Pathology

## Installation
First, install OpenSlide using either the command below or the [official installation instructions](https://openslide.org/download/#distribution-packages):
```bash
apt update && apt install -y openslide-tools libgl1-mesa-glx # libgl1-mesa-glx is needed for OpenCV
```

Then, install this package via `pip` (NOTE: will be updated once this is a PyPI package):
```bash
pip install git+https://github.com/Avic3nna/STAMP
```

Once installed, you will be able to run the command line interface directly using the `stamp` command.

Finally, to download required resources such as the weights of the CTransPath feature extractor, run the following command:
```bash
stamp setup
```

## Running
Available commands are:
```bash
stamp setup    # download required resources
stamp config   # print resolved configuration
stamp train    # train single model
stamp crossval # train n_splits models using cross-validation
stamp deploy   # deploy a model on another test set
stamp stats    # compute stats including ROC curves
stamp heatmaps # generate heatmaps
```

By default, stamp will use the configuration file `config.yaml` in the current working directory. If you want to use a different configuration file use the `--config` command line option, i.e. `stamp --config some/other/file.yaml train`.