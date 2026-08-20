# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this is

STAMP (Solid Tumor Associative Modeling in Pathology) is an end-to-end, weakly
supervised deep-learning pipeline for computational pathology: gigapixel
whole-slide images (WSIs) in, biomarker predictions out. It ships as a single
CLI (`stamp`) driven by one YAML config file, plus an optional MCP server.

The pipeline stages, in order, map 1:1 onto CLI subcommands and onto top-level
keys in the config file:

```
WSIs → preprocess → [encode_slides | encode_patients] → train / crossval
                                                       → deploy → statistics
                                                       → heatmaps
```

Feature files are HDF5 (`.h5`) with a `feats` dataset plus metadata attributes
(`extractor`, `encoder`, `stamp_version`, `code_hash`, `feat_type`). Downstream
stages read those attributes to validate that features came from a compatible
extractor.

## Branches and PRs

**Never commit directly to `main`.** STAMP is developed branch-first: `main` is
the integration branch and only receives changes through pull requests.

- Branch off an up-to-date `main` (`git fetch origin && git switch -c <branch>
  origin/main`).
- Name branches with a topic prefix, as in the existing history:
  `fix/…`, `feature/…`, `dev/…` (e.g. `fix/build-6`, `dev/multitask`,
  `feature/random-seed`).
- Open a PR against `main`. CI is wired to `push` and `pull_request` on `main`
  only, so a branch gets its first full check when the PR opens — expect the
  dependency-config, ruff and Linux test jobs to gate the merge.
- Recent history is squash-merged, giving one commit per PR with a `(#NNN)`
  suffix. Write the commit message accordingly: imperative mood, one line
  describing the change ("Pin every dependency to an exact version", "Run the
  test suite on Linux only").

If you are an agent asked to "commit" work, check what branch you are on first
and create one off `main` if the answer is `main`.

## Repository layout

```
src/stamp/
  __main__.py            CLI entry point (`stamp`); argparse + dispatch on config
  config.yaml            Factory-settings config, copied by `stamp init`
  types.py               Shared type aliases (Microns, SlidePixels, Task, Bags, …)
  utils/                 config.py (StampConfig root model), seed.py, cache.py, path.py
  preprocessing/         Tiling and tile-level feature extraction
    __init__.py            `extract_()` — dispatches ExtractorName → extractor
    config.py              `ExtractorName` enum + `PreprocessingConfig`
    tiling.py              WSI → tiles, background rejection
    extractor/             One module per foundation model (uni2, virchow2, …)
  encoding/              Tile features → slide-/patient-level embeddings
    config.py              `EncoderName` enum + encoding configs
    encoder/__init__.py    `Encoder` ABC; subclasses in encoder/{titan,cobra,…}.py
  modeling/              Training, cross-validation, deployment
    data.py                Dataloaders, clini/slide table parsing, feature typing
    train.py, crossval.py, deploy.py
    registry.py            `ModelName` enum + (feature_type, task) → Lightning class
    models/                VisionTransformer, MLP, TransMIL, Barspoon, Cox heads
  statistics/            AUROC/AUPRC/regression/survival metrics and plots
  heatmaps/              Attention heatmaps and top/bottom-scoring tile exports
tests/                   pytest suite; `random_data.py` builds synthetic inputs
scripts/                 check_dependency_config.py, verify_installed_stack.py,
                         remote_gpu_check.sh (manual GPU-host harness, not CI)
mcp/                     FastMCP server exposing the CLI stages as MCP tools
.github/workflows/build.yml   CI
```

`README.md` covers installation, `getting-started.md` is the user-facing tutorial.

## Environment and commands

The project uses **uv** (pinned to `>=0.12,<0.13`) and Python 3.14 (3.13 also
supported). uv is not optional — the CUDA wheel indexes and build exclusions
live in `[tool.uv.*]` and are invisible to pip. See "Dependency rules" below.

```bash
uv sync --extra cpu --dev          # what you normally want locally
uv sync --locked --extra cpu --dev # what CI runs; never rewrites uv.lock
uv sync --extra gpu                # Linux + CUDA 13.0
uv sync --extra gpu_all            # + conchv1_5, gigapath, musk
```

`cpu`, `gpu` and `gpu_all` are declared as mutually conflicting extras; pick one.

```bash
# Tests. The full suite is slow (model downloads, real training runs).
uv run pytest tests/ --ignore=tests/test_feature_extractors.py   # what CI runs
uv run pytest -m "not slow" tests/                               # quick pass
uv run pytest tests/test_data.py -k some_case --verbose          # one case
uv run pytest tests/test_feature_extractors.py -k uni2           # one extractor

# Lint, format, types
uv run ruff check --target-version=py313
uv run ruff format --target-version=py313      # add --diff to only check
uv run pyright

# Dependency-configuration checks (stdlib only, no install needed)
python scripts/check_dependency_config.py
uv run python scripts/verify_installed_stack.py --expect-no-cuda
```

Extractor tests download gated model weights from Hugging Face and need
`HF_TOKEN` in the environment; without it they are expected to fail, not to be
"fixed". `tests/conftest.py` adds a `--extractor` option to select which ones
run.

CI (`.github/workflows/build.yml`) runs: dependency-config check, ruff
lint+format, the test suite on Linux (3.13 and 3.14), a macOS install/import
smoke test, GPU-extra installs on a machine with no CUDA toolkit (x86_64 and
aarch64), and one job per extractor. macOS runs installation and imports only —
it is a supported development platform, not a tested one.

## Dependency rules

These are enforced by `scripts/check_dependency_config.py` (and mirrored in
`tests/test_dependency_config.py`). Breaking one turns CI red:

- **Every requirement is pinned with `==`.** No floors, no ranges. A loose
  specifier lets an unreviewed release into a fresh resolution.
- **Git dependencies must be HTTPS + a full 40-char commit SHA.** No tags, no
  branches.
- **`flash-attn`, `mamba-ssm` and `causal-conv1d` come from the Astral index**
  (`https://wheels.astral.sh/simple/cu130/`) as pre-built wheels, never from a
  pinned wheel URL and never from source. `[tool.uv.no-build-package]` forbids
  building them; their local versions (`+cu.13.0.torch.2.11`) must agree with
  the PyTorch minor and CUDA channel the project selects.
- **CPU installs must contain zero CUDA packages.** That is what
  `[tool.uv.exclude-dependencies]` is for — upstream forks (UNI, GigaPath,
  COBRA) declare CUDA extras unconditionally, so STAMP drops them and
  re-declares them with platform markers in its own extras.
- Several pins carry a comment explaining *why* they are held back (numpy 2.4
  vs beartype, transformers 4.57 vs `trust_remote_code` models). Read the
  comment before bumping; if you bump anyway, update or remove the comment.
- Changing a dependency means regenerating `uv.lock` (`uv lock`) in the same
  commit.

## Code conventions

- **Runtime type checking is on.** `src/stamp/__init__.py` calls
  `beartype_this_package()`, so annotations are checked at runtime (violations
  surface as `UserWarning`). Annotate accurately; a sloppy annotation is a bug,
  not cosmetic.
- **Array shapes are annotated with jaxtyping**, e.g.
  `Float[Tensor, "batch tile feature"]`. Ruff's `F722` is globally ignored
  because of this — don't re-enable it.
- **Domain units are `NewType`s** in `stamp/types.py`: `Microns`,
  `SlidePixels`, `TilePixels`, `SlideMPP`. Use them rather than bare `float`/
  `int` so slide-space and tile-space pixels can't be mixed up.
- **A trailing underscore means the function has side effects** (writes files,
  mutates its arguments): `extract_()`, `train_categorical_model_()`,
  `encode_slides_()`, `filter_complete_patient_data_()`. Keep the convention
  when adding functions.
- **Configs are Pydantic models with `extra="forbid"`**, composed under
  `StampConfig` in `stamp/utils/config.py`. A new CLI option means: add the
  field to the right config model, document it in `src/stamp/config.yaml`, and
  pass it through in `__main__.py`.
- **Heavy imports are deferred.** `__main__.py` imports torch/pydantic-heavy
  modules inside the `match` arms so `stamp init` and `stamp --help` stay fast;
  `extract_()` imports each extractor inside its `case`. Keep new imports local
  in the same way.
- **Optional backends fail loudly at import.** Every `extractor/*.py` and
  `encoder/*.py` wraps its third-party imports in `try: … except
  ModuleNotFoundError` and re-raises with the extra to install
  (`pip install 'stamp[uni2]'`). Follow that pattern; the macOS CI job relies on
  it, treating only `cobra` and `gigapath` as legitimately absent.
- **Logging, not printing.** Use `_logger = logging.getLogger("stamp")`. Each
  command also attaches a file handler writing `logfile.log` into its
  `output_dir`.
- Ruff is the formatter and linter (line length and style are its defaults).
  Import sorting is on in the VS Code config; run `ruff format` before
  committing.

## Adding things

**A tile-level feature extractor**: add a module under
`preprocessing/extractor/` returning an `Extractor(model=…, transform=…,
identifier=…)`; add a variant to `ExtractorName`; add a `case` to `extract_()`;
declare its dependencies as an optional extra in `pyproject.toml` and add it to
the `cpu`/`gpu`/`gpu_all` blanket extras as appropriate; list it in
`src/stamp/config.yaml`'s comment and in the `test_extractors` CI matrix.

**A slide/patient encoder**: subclass `Encoder` in `encoding/encoder/`,
implementing `_generate_slide_embedding` and `_generate_patient_embedding`
(override `encode_slides_`/`encode_patients_` only if you need tile
coordinates); add a variant to `EncoderName` and wire it into
`init_slide_encoder_`/`init_patient_encoder_`. `required_extractors` gates which
tile features the encoder accepts.

**A model**: add it under `modeling/models/`, add a `ModelName` variant, and
wire it into `load_model_class()` in `modeling/registry.py`. The Lightning
wrapper is chosen by `(feature_type, task)` from `MODEL_REGISTRY`; feature type
is detected from the `.h5` metadata, task is one of `classification`,
`regression`, `survival`. Only `barspoon` supports multi-target classification.

## Gotchas

- `identifier` on an `Extractor` must uniquely identify model *and* weights —
  it is written into every feature file and checked downstream.
- `generate_hash: True` appends a hash of the preprocessing source (see
  `get_processing_code_hash`) to output directory names. Editing any file in
  `preprocessing/` therefore changes where features land; that is deliberate.
- Reproducibility goes through `Seed.set(seed)`, called once from `__main__.py`.
  `torch.use_deterministic_algorithms()` is intentionally *not* enabled (large
  performance cost), so same-seed runs follow the same trajectory but are not
  bit-identical. `Seed.get_loader_worker_init()` raises if the seed was never
  set.
- Tests force the `spawn` multiprocessing start method (`tests/conftest.py`).
  Don't switch to `fork`; it warns and misbehaves with the threaded loaders.
- `deploy` accepts multiple checkpoints and majority-votes across them — that's
  how cross-validation folds are combined.
- Slides that raise during feature extraction are skipped, not fatal;
  `extract_()` is deliberately fail-safe. Check `logfile.log` when output looks
  short.
- Don't commit `config.yaml` at the repo root — it's gitignored on purpose
  (it's a user's local, path-laden copy of `src/stamp/config.yaml`).
