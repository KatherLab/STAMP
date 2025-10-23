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
  - [chief_ctranspath][chief_ctranspath]
  - [DinoBloom][dinobloom]
  - [CONCH][conch]
  - [CONCHv1.5][conch1_5]
  - [UNI][uni]
  - [UNI2][uni2]
  - [Virchow][virchow]
  - [Virchow2][virchow2]
  - [Gigapath][gigapath]
  - [H-optimus-0][h_optimus_0]
  - [H-optimus-1][h_optimus_1]
  - [mSTAR][mstar]
  - [MUSK][musk]
  - [PLIP][plip]


As some of the above require you to request access to the model on huggingface,
we will stick with ctranspath for this example.

In order to use a feature extractor,
you also have to install their respective dependencies.
You can do so by specifying the feature extractor you want to use
when installing stamp. Please refer to the [installation instructions](README.md#installation)

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

> In case you want to use a gated model (e.g. Virchow2), you need to login in your console using:
> ```
>huggingface-cli login
> ```
> More info about this [here](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

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
[uni2]: https://huggingface.co/MahmoodLab/UNI2-h
[conch]: https://www.nature.com/articles/s41591-024-02856-4 "A visual-language foundation model for computational pathology"
[conch1_5]: https://huggingface.co/MahmoodLab/conchv1_5
[virchow]: https://huggingface.co/paige-ai/Virchow "A foundation model for clinical-grade computational pathology and rare cancers detection"
[virchow2]: https://huggingface.co/paige-ai/Virchow2
[chief_ctranspath]: https://github.com/hms-dbmi/CHIEF
[gigapath]: https://huggingface.co/prov-gigapath/prov-gigapath
[h_optimus_0]: https://huggingface.co/bioptimus/H-optimus-0
[h_optimus_1]: https://huggingface.co/bioptimus/H-optimus-1
[mstar]: https://huggingface.co/Wangyh/mSTAR
[musk]: https://huggingface.co/xiangjx/musk
[plip]: https://github.com/PathologyFoundation/plip
[TITAN]: https://huggingface.co/MahmoodLab/TITAN
[COBRA2]: https://huggingface.co/KatherLab/COBRA
[EAGLE]: https://github.com/KatherLab/EAGLE
[MADELEINE]: https://huggingface.co/MahmoodLab/madeleine
[PRISM]: https://huggingface.co/paige-ai/Prism



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
We recommend explicitly setting the possible classes using the `categories` field.

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

## Slide-Level Encoding 
Tile-Level features can be enconded into a single feature per slide, this is useful
when trying to capture global patterns across whole slides.

STAMP currently supports the following encoders:
- [CHIEF][CHIEF_CTRANSPATH]
- [TITAN]
- [GIGAPATH]
- [COBRA2]
- [EAGLE]
- [MADELEINE]
- [PRISM]

Slide encoders take as input the already extracted tile-level features in the 
preprocessing step. Each encoder accepts only certain extractors and most
work only on CUDA devices:

| Encoder | Required Extractor | Compatible Devices | Notes
|--|--|--|--|
| CHIEF | CHIEF-CTRANSPATH | CUDA only | Text encoding removed
| TITAN | CONCH1.5 | CUDA, cpu, mps | 
| GIGAPATH | GIGAPATH | CUDA only
| COBRA2 | CONCH, UNI, VIRCHOW2 or H-OPTIMUS-0 | CUDA only
| EAGLE | CTRANSPATH, CHIEF-CTRANSPATH | CUDA only
| MADELEINE | CONCH | CUDA only
| PRISM | VIRCHOW_FULL | CUDA only

> **Note:** Slide-level features cannot be used directly for modeling because the clinical labels are at the patient level. However, if only one slide is available per patient, using **[Patient-Level Encoding](#patient-level-encoding)** will produce the same representation as slide-level encoding—but supports downstream modeling.

As with feature extractors, most of these models require you to request
access. The following example uses CHIEF, which is available if you installed 
STAMP with `uv sync --all-extras`. The configuration should look like this:

```yaml
# stamp-test-experiment/config.yaml

slide_encoding:
  # Encoder to use for slide encoding. Possible options are "cobra",
  # "eagle", "titan", "gigapath", "chief", "prism", "madeleine".
  encoder: "chief"
  
  # Directory to save the output files.
  output_dir: "/path/to/save/files/to"
  
  # Directory where the extracted features are stored.
  feat_dir: "/path/your/extracted/features/are/stored/in"
  
  # Device to run slide encoding on ("cpu", "cuda", "cuda:0", etc.)
  device: "cuda"

  # Optional settings:
  # Directory where the aggregated features are stored. Needed for
  # some encoders such as eagle (it requires virchow2 features).
  #agg_feat_dir: "/path/your/aggregated/features/are/stored/in"

  # Add a hash of the entire preprocessing codebase in the feature folder name.
  #generate_hash: True
  ```

Don't forget to put in `feat_dir` a path containing, in this case, `ctranspath` or
`chief-ctranspath` tile-level features. Once everything is set, you can simply run:

```sh
stamp --config stamp-test-experiment/config.yaml encode_slides
```
The output will be one `.h5` file per slide. 

## Patient-Level Encoding
Even though the available encoders are designed for slide-level use, this
option concatenates the slides of a patient along the x-axis, creating a single
"virtual" slide that contains two blocks of tissue. The configuration is the same
except for `slide_table` which is required to link slides with patients.
```yaml
# stamp-test-experiment/config.yaml

patient_encoding:
  # Encoder to use for patient encoding. Possible options are "cobra",
  # "eagle", "titan", "gigapath", "chief", "prism", "madeleine".
  encoder: "eagle"
  
  # Directory to save the output files.
  output_dir: "/path/to/save/files/to"
  
  # Directory where the extracted features are stored.
  feat_dir: "/path/your/extracted/features/are/stored/in"
  
  # A table (.xlsx or .csv) relating every slide to their feature files.
  # The table must contain at least two columns, one titled "SLIDE",
  # containing the slide ID, and one called "FILENAME", containing the feature file path relative to `feat_dir`.
  slide_table: "/path/of/slide.csv"
  
  # Device to run slide encoding on ("cpu", "cuda", "cuda:0", etc.)
  device: "cuda"

  # Optional settings:
  patient_label: "PATIENT"
  filename_label: "FILENAME"
  
  # Directory where the aggregated features are stored. Needed for
  # some encoders such as eagle (it requires virchow2 features).
  #agg_feat_dir: "/path/your/aggregated/features/are/stored/in"

  # Add a hash of the entire preprocessing codebase in the feature folder name.
  #generate_hash: True
  ```

  Then run:
  ```sh
stamp --config stamp-test-experiment/config.yaml encode_patients
```

The output `.h5` features will have the patient's id as name. 

## Training with Patient-Level Features

Once you have patient-level features, 
you can train models directly on these features. This is useful because:
- **Efficient with Limited Data**: Patient-level modeling often performs better when data is scarce, since pretrained encoders can extract robust features from each slide as a whole.
- **Faster Training & Reduced Overfitting**: With fewer parameters to train compared to tile-level models, patient-level models train more quickly and are less prone to overfitting.
- **Enables Interpretable Cohort Analysis**: Patient-level features can be used for unsupervised analyses, such as clustering, making it easier to interpret and explore patient subgroups within your cohort.

To train a model using patient-level features, you can use the same command as before:
```sh
stamp --config stamp-test-experiment/config.yaml crossval
```

The key differences for patient-level modeling are:
- The `feature_dir` should contain patient-level `.h5` files (one per patient).
- The `slide_table` is not needed since there's a direct mapping from patient ID to feature file.
- STAMP will automatically detect that these are patient-level features and use a MultiLayer Perceptron (MLP) classifier instead of the Vision Transformer.

You can then run statistics as done with tile-level features.

## Heatmaps and Top Tiles
<img src="docs/overlay-heatmap.png" width="500px" align="center"></img>

The `stamp heatmaps` command generates visualization outputs to help interpret model predictions and identify which regions of the slide contribute most to the classification decision. This command creates:

- **Attention heatmaps**: Show which tiles the model focuses on for each class
- **Overlay visualizations**: Combine heatmaps with slide thumbnails for better spatial context
- **Class maps**: Display which class each tile is most associated with
- **Top/bottom tiles**: Extract the most and least predictive image patches from the predicted class. 

To generate heatmaps, you need a trained model checkpoint from either the train or crossval commands. The configuration file should look like this:

```yaml
# stamp-test-experiment/config.yaml

heatmaps:
  output_dir: "/absolute/path/to/stamp-test-experiment/heatmaps"

  # Directory where the extracted tile-level features are stored
  feature_dir: "/absolute/path/to/stamp-test-experiment/xiyuewang-ctranspath-7c998680-112fc79c"

  # Directory containing the original whole slide images
  wsi_dir: "/absolute/path/to/wsi_dir"

  # Path to the trained model checkpoint
  checkpoint_path: "/absolute/path/to/stamp-test-experiment/split-0/checkpoints/epoch=15-step=123.ckpt"

  # Optional settings:

  # Overlay plot opacity (0 = transparent, 1 = opaque)
  opacity: 0.6

  # Number of top-scoring tiles to extract for each slide
  topk: 5

  # Number of bottom-scoring tiles to extract for each slide  
  bottomk: 5

  # Specific slides to process (relative to wsi_dir)
  # If not specified, all slides in wsi_dir will be processed
  slide_paths:
  - slide1.svs
  - slide2.mrxs

  # Device to run heatmap generation on
  device: "cuda"
  ```

  > **Note:** Heatmaps currently only work with tile-level features. If you have slide-level or patient-level features, you'll need to use the original tile-level features for heatmap generation.

  Generate the heatmaps by running:

  ```sh
  stamp --config stamp-test-experiment/config.yaml heatmaps
  ```

  The heatmap command creates an organized folder structure for each slide:

  ```sh
  heatmaps/
└── slide-name/
    ├── plots/
    │   ├── overview-slide-name.png     # Complete overview with all classes
    │   └── overlay-slide-name-class.png # Individual class overlays
    ├── raw/             # Raw data files
    │   ├── thumbnail-slide-name.png         # Slide thumbnail
    │   ├── classmap-slide-name.png          # Class assignment map
    │   ├── slide-name-class=score.png       # Raw heatmap per class
    │   └── raw-overlay-slide-name-class.png # Overlay without legends
    └── tiles/           # Individual tile extractions
        ├── top_01-slide-name-class=score.jpg    # Highest scoring tiles
        ├── top_02-slide-name-class=score.jpg
        └── bottom_01-slide-name-class=score.jpg # Lowest scoring tiles
  ```


## Advanced configuration

Advanced experiment settings can be specified under the `advanced_config` section in your configuration file.
This section lets you control global training parameters, model type, and the target task (classification, regression, or survival).

```yaml
# stamp-test-experiment/config.yaml

advanced_config:
  seed: 42
  task: "classification" # or regression/survial
  max_epochs: 32
  patience: 16
  batch_size: 64
  # Only for tile-level training. Reducing its amount could affect
  # model performance. Reduces memory consumption. Default value works
  # fine for most cases.
  bag_size: 512
  #num_workers: 16 # Default chosen by cpu cores
  # One Cycle Learning Rate Scheduler parameters. Check docs for more info.
  # Determines the initial learning rate via initial_lr = max_lr/div_factor
  max_lr: 1e-4
  div_factor: 25. 
  # Select a model regardless of task
  # Available models are: vit, trans_mil, mlp
  model_name: "vit"

  model_params:
    vit: # Vision Transformer
      dim_model: 512
      dim_feedforward: 512
      n_heads: 8
      n_layers: 2
      dropout: 0.25
      use_alibi: false
```

STAMP automatically adapts its **model architecture**, **loss function**, and **evaluation metrics** based on the task specified in the configuration file.
 
**Regression** tasks only require `ground_truth_label`.  
**Survival analysis** tasks require `time_label` (follow-up time) and `status_label` (event indicator).  
These requirements apply consistently across cross-validation, training, deployment, and statistics.