# Barspoon: A Transformer Architecture for Multilabel Predictions

Barspoon transformers are a transformer architecture for multilabel prediction
tasks for application in histopathological problems, but easily adaptable to
other domains.

## User Guide

In the following, we will give examples of how to use barspoon to do some
common-place prediction tasks in histopathology.  We assume our dataset to
consist of multiple _patients_, each of which has zero or more histopathological
_slides_ assigned to them.  For each patient, we have a series of _target
labels_ we want to train the network to predict.

We initially need the following:

 1. A table containing clinical information, henceforth the _clini table_.  This
    table has to be in either csv or excel format.  It has to have at least one
    column `patient`, which contains an ID identifying each patient, and other
    columns matching clinical information to that patient.
 2. Features extracted from each slide, generated using e.g. [KatherLab's
    STAMP feature extraction][1].
 3. A table matching each patient to their slides, the _slide table_.  The slide
    table has two columns, `patient` and `filename`.  The `patient` column has
    to contain the same patient IDs found in the clini table.  The `filename`
    column contains the file paths to features belonging to that patient.  Each
    `filename` has to be unique, but one `patient` can be mapped to multiple
    `filename`s.

[1]: hhttps://github.com/KatherLab/STAMP/blob/main/getting-started.md

### Generating the Target File

```
python target_file.py 
```

### Training a Model

```
python train.py
```
