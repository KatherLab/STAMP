# Getting Started with Stamp

This guide is designed to help you with your first steps using the stamp pipeline
to predict biomarkers and other attributes from whole slide images (WSIs).
To follow along,
you will need some WSIs,
a table mapping each of these slides to a patient
as well as some ground truth we will eventually train a neural network on.

## Whole Slide Images

The whole slide images have to be in any of the formats [supported by OpenSlide][openslide].
For the next steps we assume that all these WSIs are stored in the same directory.
We will call this directory the _WSI directory_.

[openslide]: https://openslide.org/#about-openslide "About OpenSlide"

## Creating a Configuration File

Stamp is configured using configuration files.
We recommend creating one configuration file per experiment
and storing in the same folder as the eventual results,
as this makes it easier to reconstruct which data and parameters a model was trained with later.

The `stamp init` command creates a new configuration file with dummy values.
By default, it is created in `$PWD/config.yaml`,
but we can use the `--config` option to specify its location:
```sh
# Create a directory to save our experiment results to
mkdir stamp-test-experiment
# Create a new config file in said directory
stamp --config stamp-test-experiment/config.yaml init
```

## Feature Extraction

To do any kind of training on our data, we first have to convert it into a form
more easily usable by neural networks.
We do this using a _feature extractor_.
A feature extractor is a neural network has been trained on a large amount of WSIs
to extract extract the information relevant for our domain from images.
This way, we can compress WSIs into a more compact representation,
which in turn allows us to efficiently train machine learning models with them.

Stamp currently supports the following feature extractors:
  - [ctranspath][ctranspath]
  - [DinoBloom][dinobloom]
  - [CONCH][conch]
  - [UNI][uni]
  - [Virchow2][virchow2]

As some of the above require you to request access to the model on huggingface,
we will stick with ctranspath for this example.

In order to use a feature extractor,
you also have to install their respective dependencies.
You can do so by specifying the feature extractor you want to use
when installing stamp:
```sh
# Install stamp including the dependencies for all feature extractors
pip install "git+https://github.com/KatherLab/stamp[all]"
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

As the preprocessing is running,
you can see the output directory fill up with the features, saved in `.h5` files,
as well as `.jpg`s showing from which parts of the slide features are extracted.
Most of the background should be marked in red,
meaning ignored that it was ignored during feature extraction.

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
[dinobloom]: https://github.com/marrlab/DinoBloom "DinoBloom: A Foundation Model for Generalizable Cell Embeddings in Hematology"
[uni]: https://www.nature.com/articles/s41591-024-02857-3 "Towards a general-purpose foundation model for computational pathology"
[conch]: https://www.nature.com/articles/s41591-024-02856-4 "A visual-language foundation model for computational pathology"
[virchow2]: https://huggingface.co/paige-ai/Virchow2

## Doing Cross-Validation on the Data Set

One way to quickly ascertain if a neural network can be trained to recognize a specific pattern
without the need to source a separate testing set
is to perform a cross-validation on it.
During a cross validation,
we train multiple models on a subset of the data,
testing its effectiveness on the held-out part of the data not used during training.
To perform a cross-validation, add the following lines to your `stamp-test-experiment/config.yaml`,
with `feature_dir` adapted to match the directory the `.h5` files were output to in the last step.
`clini_table` and `slide_table` both need to point to tables,
either in excel or `.csv` format,
with contents as described below.
Finally, `ground_truth_label` needs to contain the column name
of the data we want to train our model on.
Stamp only can be used to train neural networks for categorical targets.
We recommend explicitly setting the possible classes using the `categories` field. <br>
You can choose between training a Transformer from scratch or utilizing the pretrained [COBRA](https://github.com/KatherLab/COBRA) model by setting the **use_cobra** flag in the config file.

```yaml
# stamp-test-experiment/config.yaml

crossval:
  output_dir: "/absolute/path/to/stamp-test-experiment"

  # An excel (.xlsx) or CSV (.csv) table containing the clinical information of
  # patients.  Patients not present in this file will be ignored during training.
  # Has to contain at least two columns, one titled "PATIENT", containing a patient ID,
  # and a second column containing the categorical ground truths for that patient.
  clini_table: "metadata-CRC/TCGA-CRC-DX_CLINI.xlsx"

  # Directory the extracted features are saved in.
  feature_dir: "/absolute/path/to/stamp-test-experiment/xiyuewang-ctranspath-7c998680-112fc79c"

  # A table (.xlsx or .csv) relating every patient to their feature files.
  # The table must contain at least two columns, one titled "PATIENT",
  # containing the patient ID (matching those in the `clini_table`), and one
  # called "FILENAME", containing the feature file path relative to `feature_dir`.
  # Patient IDs not present in the clini table as well as non-existent feature
  # paths are ignored.
  slide_table: "slide.csv"

  # Name of the column from the clini table to train on.
  ground_truth_label: "isMSIH"

  # Optional settings:

  # The categories occurring in the target label column of the clini table.
  # If unspecified, they will be inferred from the table itself.
  categories: ["yes", "no"]

  # Number of folds to split the data into for cross-validation
  #n_splits: 5
  # use_cobra = False
  # freeze_base = True
  # freeze_cobra = False
```

After specifying all the parameters of our cross-validation,
we can run it by invoking:
```sh
stamp --config stamp-test-experiment/config.yaml crossval
```

## Generating Statistics

After training and validating your model, you may want to generate statistics to evaluate its performance.
This can be done by adding a `statistics` section to your `stamp-test-experiment/config.yaml` file.
The configuration should look like this:

```yaml
# stamp-test-experiment/config.yaml

statistics:
  output_dir: "/absolute/path/to/stamp-test-experiment/statistics"

  # Name of the target label.
  ground_truth_label: "isMSIH"

  # A lot of the statistics are computed "one-vs-all", i.e. there needs to be
  # a positive class to calculate the statistics for.
  true_class: "yes"

  pred_csvs:
  - "/absolute/path/to/stamp-test-experiment/split-0/patient-preds.csv"
  - "/absolute/path/to/stamp-test-experiment/split-1/patient-preds.csv"
  - "/absolute/path/to/stamp-test-experiment/split-2/patient-preds.csv"
  - "/absolute/path/to/stamp-test-experiment/split-3/patient-preds.csv"
  - "/absolute/path/to/stamp-test-experiment/split-4/patient-preds.csv"
```

To generate the statistics, run the following command:
```sh
stamp --config stamp-test-experiment/config.yaml statistics
```

Afterwards, the `output_dir` should contain the following files:
  - `isMSIH-categorical-stats-individual.csv` contains statistical scores
    for each individual split.
  - `isMSIH-categorical-stats-aggregated.csv` contains the mean
    as well as the 95% confidence interval for the statistical scores
    for the splits.
  - `roc-curve_isMSIH=yes.svg` and `pr-curve_isMSIH=yes.svg`
    contain the ROC and precision recall curves of the splits.