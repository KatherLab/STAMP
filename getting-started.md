# Getting Started with Stamp

This guide is designed to help you with your first steps using the stamp pipeline
to predict biomarkers and other attributes from whole slide images (WSIs).
To follow along,
you will need some WSIs,
a table mapping each of these slides to a patient
as well as some ground truth we will eventually train a neural network on.

### Whole Slide Images

The whole slide images have to be in any of the formats [supported by OpenSlide][openslide].
For the next steps we assume that all these WSIs are stored in the same directory.
We will call this directory the _WSI directory_.

[openslide]: https://openslide.org/#about-openslide "About Openslide"

## Creating a Configuration File

Stamp is configured using configuration files.
We recommend creating one configuration file per experiment
and storing in the same folder as the eventual results,
as this makes it easier to reconstruct which data and parameters a model was trained with later.

The `stamp init` command creates a new configuration file with dummy values.
By default, it is created in `$PWD/config.yaml`,
but we can use the `--config` option to specify its location:
```bash
# Create a directory to save our experiment results to
mkdir stamp-test-experiment
# Create a new config file in said directory
stamp --config stamp-test-experiment/config.yaml init
```


### Feature Extraction

To do any kind of training on our data, we first have to convert it into a form
more easily usable by neural networks.
We do this using a _feature extractor_.
A feature extractor is a neural network has been trained on a large amount of WSIs
to extract extract the information relevant for our domain from images.
This way, we can compress WSIs into a more compact representation,
which in turn allows us to efficiently train machine learning models with them.

Stamp currently supports three feature extractors, [ctranspath][ctranspath],
[UNI][uni] and [CONCH][conch].
The latter two require you to request access to the model on huggingface,
so we will stick with ctranspath for this example.

In order to use a feature extractor,
you also have to install their respective dependencies.
You can do so by specifying the feature extractor you want to use
when installing stamp:
```sh
# Install stamp including all depencencies for stamp and uni
pip install "git+https://github.com/KatherLab/stamp@v2[ctranspath,uni]"
```

Open the `stamp-test-experiment/config.yaml` we created in the last step
and modify the `output_dir`, `wsi_dir` and `cache_dir` entries
in the `preprocessing` section
to contain the absolute paths of the directory the configuration file resides in.
`wsi_dir` Needs to point to a path containing the WSIs you want to extract features from.

The `cache_dir` will be used to save intermediate data.
Should you decide to try another feature extractor later,
using the same cache dir again will significantly speed up the extraction process.
If you will only extract features once, it can be set to `none`.

```yaml
# stamp-test-experiment/config.yaml

preprocessing:
  output_dir: "/absolute/path/to/stamp-test-experiment"
  wsi_dir: "/absolute/path/to/wsi_dir"

  # Other possible values are "mahmood-uni" and "mahmood-conch"
  extractor: "ctranspath"

  # Having a cache dir will speed up extracting features multiple times,
  # e.g. with different feature extractors.
  # Optional.
  cache_dir: "/absolute/path/to/stamp-test-experiment/../cache"
  # If you do not want to use a cache,
  # change the cache dir to the following:
  # cache_dir: null

  # Device to run feature extraction on.
  # Set this to "cpu" if you do not have a CUDA-capable GPU.
  device: "cuda"

  # How many workers to use for tile extraction.  Should be less or equal to
  # the number of cores of your system.
  max_workers: 8
```

Extracting the features is then as easy as running
```sh
stamp --config stamp-test-experiment/config.yaml preprocess
```
Depending on the size of your dataset and your hardware,
this process may take anything between a few hours and days.

You can interrupt this process at any time.
It will continue where you stopped it the next time you run `stamp preprocess`.

> **If you are using the UNI or CONCH models**
> and working in an environment where your home directory storage is limited,
> you may want to also specify your huggingface storage directory
> by setting the `HF_HOME` environment variable:
> ```sh
> export HF_HOME=/path/to/directory/to/store/huggingface/data/in
> huggingface-cli login   # only needs to be done once per $HF_HOME
> stamp -c stamp-test-experiment/config.yaml preprocess
> ```


[ctranspath]: https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043 "Transformer-based unsupervised contrastive learning for histopathological image classification"
[uni]: https://www.nature.com/articles/s41591-024-02857-3 "Towards a general-purpose foundation model for computational pathology"
[conch]: https://www.nature.com/articles/s41591-024-02856-4 "A visual-language foundation model for computational pathology"