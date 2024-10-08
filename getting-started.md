# Getting Started with STAMP

This guide is designed to help you with your first steps using the STAMP pipeline
to predict biomarkers and other attributes from whole slide images (WSIs).


## Cross-Validation


### Feature Extraction

To do any kind of training on our data, we first have to convert it into a form
more easily usable for neural networks.
We do this using a feature extraction network.
This network has been already been trained on a large amount of WSIs
to extract extract the information relevant for our domain from images.
This way, we can compress WSIs into a more compact representation,
which in turn allows us to efficiently train machine learning models with them.

STAMP currently supports three feature extractors, [ctranspath][ctranspath],
[UNI][uni] and [CONCH][conch].
The latter two require you to request access to the model on huggingface,
so we will stick with ctranspath for this example.

Create a file named `example-crossval.yaml` and add the following lines,
with the `output_dir`, `wsi_dir` and `cache_dir` entries adapted so
`output_dir` points to a directory you want to save the extracted features to
and `wsi_dir` is the path to a directory containing the WSIs you want to extract features from.
The `cache_dir` will be used to save intermediate data.
Should you decide to try another feature extractor later,
using the same cache dir again will significantly speed up the extraction process.

```yaml
# Cross-validation example STAMP config file

preprocessing:
  output_dir: "/path/to/save/files/to"
  wsi_dir: "/path/containing/whole/slide/images/to/extract/features/from"

  # Other possible values are "mahmood-uni" and "mahmood-conch"
  extractor: "ctranspath"

  # Having a cache dir will speed up extracting features multiple times,
  # e.g. with different feature extractors.  Optional.
  cache_dir: "/path/to/save/cache/files/to"

  # Device to run feature extraction on.
  # Set this to "cpu" if you do not have a CUDA-capable GPU.
  device: "cuda"

  # How many workers to use for tile extraction.  Should be less or equal to
  # the number of cores of your system.
  max_workers: 8
```

Extracting the features is then as easy as running
```sh
stamp -c example-crossval.yaml preprocess
```
Depending on the size of your dataset and your hardware,
this process may take anything between a few hours and days.

> You can interrupt this process at any time.
> It will continue where you stopped it the next time you run `stamp preprocess`.

**If you are using the UNI or CONCH models** and working in an environment
where your home directory storage is limited, you may want to also specify
your huggingface storage directory by setting the `HF_HOME` environment variable:
```sh
export HF_HOME=/path/to/directory/to/store/huggingface/data/in
huggingface-cli login   # only necessary once
stamp -c example-crossval.yaml preprocess
```


[ctranspath]: https://www.sciencedirect.com/science/article/abs/pii/S1361841522002043 "Transformer-based unsupervised contrastive learning for histopathological image classification"
[uni]: https://www.nature.com/articles/s41591-024-02857-3 "Towards a general-purpose foundation model for computational pathology"
[conch]: https://www.nature.com/articles/s41591-024-02856-4 "A visual-language foundation model for computational pathology"