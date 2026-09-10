# Independent KRONOS2 marker features and MarkerFusion

The `kronos2_per_marker` extractor encodes each configured channel separately with
KRONOS2, using the channel's marker name and upstream normalization. All channels
share the same SPImage tile grid, dtype scaling, and tile order. Normalization and
forward passes receive one channel at a time. The pretrained extractor is frozen;
MarkerFusion is trained subsequently using clinical labels.

Per-slide HDF5 output:

- `marker_embeddings`: `(tiles, markers, 768)`, in the `marker_names` attribute order.
- `coord_x`, `coord_y`: the shared tile coordinates, in pixels.
- `feats` / `patch_embeddings`: mean of the independent marker vectors (hard links).
- `marker_embedding_method`: `independent_single_channel_cls`.
- `patch_embedding_method`: `mean_of_marker_embeddings`.

The mean is not KRONOS2's joint-panel CLS representation. MarkerFusion training
must select `marker_embeddings`. There are no native per-marker spatial token
outputs. The existing `kronos2` extractor continues to produce joint-panel features.

## Prepared Tumorpanel experiment

Files outside the STAMP checkout, relative to its parent workspace:

- `data/config/kronos2_per_marker_tumorpanel.yaml`: cohort configuration.
- `data/config/kronos2_per_marker_tumorpanel_smoke.yaml`: one-slide extraction.
- `data/helper_files/kronos2_per_marker_slides.txt`: 50 unmixed QPTIFF paths,
  matched to the existing slide table, excluding masks and other ancillary files.
- `data/helper_files/run_kronos2_per_marker.py`: validation and workflow commands.
- `data/slurm_files/kronos2_per_marker.sh`: GPU Slurm entry point.

The configuration retains the existing seven-channel mapping:
DAPI, cMET, panCK, DLL3, Prame, HER2, Trop2. The sampled QPTIFF has eight channels:
DAPI, Opal 480, Opal 520, Opal 570, Opal 620, Opal 690, Opal 780, Sample AF.
The existing reader takes the first seven and omits Sample AF. The biological
marker assignment follows the existing full-panel YAML; the dye names alone do
not independently establish that assignment. Update both YAMLs together if the
laboratory channel map differs.

Outputs go to:

```text
/mnt/nova-curie/arndtwagner/stamp_multiplex_implementation/data/results/Vorversuche_2_Tumorpanel/kronos2/per_marker_v1/
  preprocess/                    # shared by smoke and cohort extraction
  crossval/splits.json            # copied from the existing crossval_v2 baseline
  crossval/split-{0,1,2}/model.ckpt
  explainability/split-{0,1,2}/   # each checkpoint explains only its held-out slides
```

The target remains `Her2`, with three folds and seed 42. Training uses seven ViT
branches (192 hidden units, two layers) and batch size 4, with 512 tiles per bag.
Batch size was reduced from 32 to accommodate the seven branches; GPU memory and
runtime for full-cohort training have not yet been measured.

## Run next

From the workspace:

```bash
cd /mnt/nova-curie/arndtwagner/stamp_multiplex_implementation/data/slurm_files
sbatch kronos2_per_marker.sh smoke
```

This extracts the real first slide and then checks the shape, marker order,
coordinates, extractor identity, and finite values. Check the job finishes with
`COMPLETED` / exit code `0:0`, and that its output contains `Validated 1 slides`.
Logs are in `data/slurm_files/logs/<jobid>_kronos2_marker.{out,err}`.

After the smoke job succeeds, submit the cohort extraction:

```bash
sbatch kronos2_per_marker.sh preprocess
```

The smoke slide's completed features are reused. The final validation must report
`Validated 50 slides`. Preprocessing may skip individual failures internally;
the runner makes a missing or invalid feature file fail the job rather than
silently proceeding to training. A failed/expired extraction can be resubmitted:
completed HDF5 files are skipped, and unfinished temporary files are not treated
as completed features. If a *completed* HDF5 file fails validation, move that file
aside and rerun extraction after resolving the cause.

After successful extraction, run:

```bash
sbatch kronos2_per_marker.sh crossval
```

This validates the cohort, copies the baseline patient assignments from
`kronos2/single_channel/crossval_v2/splits.json`, and trains three new checkpoints.
It refuses to overwrite differing split assignments. Do not use a different
patient cohort with this runner without updating the split policy.

After successful training, run:

```bash
sbatch kronos2_per_marker.sh explainability
```

The runner selects each fold's held-out slides from the patient/slide table and
creates a separate output directory for that fold. Reports include
`marker_summary.csv`, `marker_contribution_barplot.png`, and
`tile_saliency_overview.png` per slide. The prepared explainability configuration
uses CPU; the generic Slurm wrapper reserves a GPU for every stage.

Stages must run in order. To check job status:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
```

The standard CLI also accepts the new extractor. The prepared runner adds cohort
completeness checks, baseline split reuse, and held-out-only explanation selection.
Running `stamp ... explainability` directly with the main YAML would instead use
split 0 on every slide, which is useful for exploration but is not held-out-only.

## Validation performed

- 35 targeted tests passed across preprocessing, data loading, models, and the new
  integration test. The selection excluded the gated download test and the
  previously documented unrelated bundled-marker normalization failure.
- The new test checks separate per-marker normalization and forward calls,
  non-default marker order, multiple tile batches, shared coordinates, HDF5
  provenance, a downstream optimizer step with gradients in both branches,
  checkpoint reload, marker contributions, and tile saliency.
- A GPU forward using locally cached real KRONOS2 weights, the configured additional
  marker CSV, and one synthetic seven-channel 224-pixel tile returned finite
  `(1, 7, 768)` marker embeddings. This verifies real-model single-channel support;
  full-slide extraction and cohort training remain to be run with the commands above.
- Both YAMLs validate against STAMP's schema; touched code passes Ruff, and the
  Slurm script passes `bash -n`.

Marker contributions describe the downstream predictor's use of independently
encoded markers. They do not decompose joint-panel KRONOS2 features or establish
biological causality. Compare predictive performance using the same patient splits.
