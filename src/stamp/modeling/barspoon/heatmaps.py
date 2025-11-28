"""Generate heatmaps for barspoon transformer models

# Heatmap Computation

We use gradcam to generate the heatmaps.  This has the advantage of being
relatively model-agnostic and thus, we hope, more "objective".  It furthermore
simplifies implementation, as most of the primitives needed for gradcam come
included in pytorch.

We calculate the gradcam g of a slide consisting of tile feature vectors x as
follows:

```asciimath
g(x) = sum_i dy/dx_i * x_i
```

where `x_i` is the `i`-th tile's feature vector and `y` is the output we want to
compute a gradcam heatmap for.

The intuition about this is as follows:  `dy/dx_i` tells us how sensitive the
network is to changes in the feature `x_i`.  Since `x_i` which are large in
magnitude affect the output of the network stronger small ones, `dy/dx_i * x_i`
gives us an overall measure of how strongly `x_i` contributed to `y`.  Positive
`dy/dx_i * x_i` imply that this feature positively affected `y`, while negative
ones point towards this feature reducing the value of `y` in our result.  By
summing over all `x_i`, we get the overall contribution of the tile to the final
result `y`.

# "Attention" vs "Contribution"

We output two kinds of heatmaps:

 1. The Attention Map shows which part of the input image had an effect on the
    final result.  It is computed as `|g(x)|`, i.e. the absolute value of the
    gradcam scores.
 2. The Contribution Map is simply `g(x)`, with tiles positively contributing to
    the result being displayed as red ("hot") and those negatively contributing
    to it as blue ("cold").  Tiles with less contribution are also rendered in
    see-through optic, such that if rendered on top of a white background their
    brightness is roughly equivalent to their contribtion.
"""

import argparse
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from jaxtyping import Float, Int
from packaging.specifiers import SpecifierSet
from PIL import Image
from PIL.PngImagePlugin import PngInfo
from torch import Tensor
from tqdm import tqdm

from stamp.modeling.barspoon.model import LitEncDecTransformer, TargetLabel

CategoryLabel = str


def main():
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.top and not args.wsi_dir:
        parser.error("--top requires --wsi-dir")

    # Load model and ensure its version is compatible
    # We do all computations in bfloat16, as it needs way less VRAM and performs
    # virtually identical to float32 in inference tasks.
    model = LitEncDecTransformer.load_from_checkpoint(
        checkpoint_path=args.checkpoint_path
    ).to(device=args.device, dtype=torch.bfloat16)
    name, version = model.hparams.get("version", "undefined 0").split(" ")
    if not (
        name == "barspoon-transformer"
        and (spec := SpecifierSet(">=3.0,<4")).contains(version)
    ):
        raise ValueError(
            f"model not compatible. Found {name} {version}, expected barspoon-transformer {spec}",
            model.hparams["version"],
        )

    h5_paths = [h5 for d in args.feature_dirs for h5 in d.glob("*.h5")]
    if args.slides:
        h5_paths = [h5 for h5 in h5_paths if h5.stem in args.slides]

    for h5_path in tqdm(h5_paths):
        # Load features
        with h5py.File(h5_path) as h5_file:
            feats: Float[Tensor, "n_tiles d_feats"] = torch.tensor(
                h5_file["feats"][:], dtype=torch.bfloat16, device=args.device
            )
            coords: Int[Tensor, "n_tiles 2"] = torch.tensor(h5_file["coords"][:])
            xs = np.sort(np.unique(coords[:, 0]))
            stride_wsi_px = np.min(xs[1:] - xs[:-1])

        categories = model.hparams["categories"]

        if len(args.targets):
            categories = {k: v for k, v in categories.items() if k in args.targets}

        # Generate the gradcams
        # Skip the slide if we are out of memory
        try:
            with torch.inference_mode():
                overall_scores = model.predict_step(
                    (
                        feats.unsqueeze(0),
                        coords.type_as(feats).unsqueeze(0),
                    ),
                    batch_idx=0,
                )
                scores: Mapping[TargetLabel, Float[Tensor, "n_tiles cat_classes"]] = {}
                for i in range(0, feats.shape[0], args.batch_size):
                    scores = recursive_union(
                        scores,
                        model.predict_step(
                            (
                                feats[i : i + args.batch_size].unsqueeze(-2),
                                # coords.type_as(feats),
                                torch.zeros_like(coords).type_as(feats),
                            ),
                            batch_idx=0,
                        ),
                    )
            scores: Mapping[
                TargetLabel, Mapping[CategoryLabel, Float[Tensor, "n_tiles"]]
            ] = {
                target_label: {
                    category_label: scores[target_label][:, category_index].cpu()
                    for category_index, category_label in enumerate(category_labels)
                }
                for target_label, category_labels in categories.items()
            }

            gradcams: Mapping[
                TargetLabel, Mapping[CategoryLabel, Float[Tensor, "n_tiles"]]
            ] = compute_gradcams(
                model=model,
                feats=feats,
                coords=coords,
                categories=categories,
                batch_size=args.batch_size,
                device="cpu",
            )
        except torch.cuda.OutOfMemoryError as oom_error:  # type: ignore
            logging.error(f"error while processing {h5_path}: {oom_error})")
            continue

        for target_label, category_labels in categories.items():
            topk = torch.stack(list(scores[target_label].values())).topk(2, dim=-2)
            score_stack = torch.stack(list(scores[target_label].values()))
            for cat_idx, pos_category_label in enumerate(category_labels):
                category_support = torch.where(
                    topk.indices[0] == cat_idx,
                    score_stack[cat_idx] - topk.values[1],
                    score_stack[cat_idx] - topk.values[0],
                )
                attention = torch.where(
                    topk.indices[0] == cat_idx,
                    (gradcam := gradcams[target_label][pos_category_label].abs())
                    / gradcam.max(),
                    (
                        gradcam := torch.stack(
                            [
                                gradcams[target_label][category_label].abs()
                                for category_label in category_labels
                                if category_label != pos_category_label
                            ]
                        )
                        .max(dim=0)
                        .values
                    )
                    / gradcam.max(),
                )

                category_support_2d = (
                    vals_to_im(category_support, coords // stride_wsi_px)
                    .detach()
                    .cpu()
                    .float()
                )
                attention_2d = (
                    vals_to_im(attention, coords // stride_wsi_px)
                    .detach()
                    .cpu()
                    .float()
                )

                score_im = plt.get_cmap("RdBu")(
                    -category_support_2d * attention_2d / 2 + 0.5
                )
                score_im[..., -1] = attention_2d > 0

                target_dir = Path(args.output_dir) / str(target_label)
                target_dir.mkdir(exist_ok=True, parents=True)
                sanitized_pos_cat_label = re.sub(
                    r"[^()\-.0-9;A-Z\[\]^_a-z]", "_", pos_category_label
                )
                filename = f"{h5_path.stem}-{target_label}-{sanitized_pos_cat_label}={overall_scores[target_label][0, cat_idx]:1.2f}"
                Image.fromarray(np.uint8(score_im * 255)).resize(
                    np.array(score_im.shape[:2][::-1]) * 8, resample=Image.NEAREST
                ).save(target_dir / f"scores-{filename}.png")

                if args.top:
                    slide_url = args.wsi_dir.glob(f"{h5_path.stem}.*")
                    if not (slide_url := next(slide_url, None)):
                        logging.error(
                            f"could not find slide for {h5_path.stem}, skipping top tiles extraction..."
                        )
                        continue
                    logging.info(f"extracting top tiles for {h5_path.stem}...")
                    slide = openslide.open_slide(slide_url)
                    get_n_toptiles(
                        slide=slide,
                        stride_wsi_px=stride_wsi_px,
                        output_dir=target_dir / f"top-{filename}",
                        scores=scores[target_label][pos_category_label],
                        coords_wsi_px=coords,
                        n=args.top,
                    )


def get_n_toptiles(
    slide,
    stride_wsi_px: int,
    output_dir: Path,
    scores: Mapping[TargetLabel, Float[Tensor, "n_tiles cat_classes"]],
    coords_wsi_px: Int[Tensor, "n_tiles 2"],
    n: int,
):
    output_dir.mkdir(exist_ok=True, parents=True)
    top_score, bottom_score = scores.topk(n), scores.topk(n, largest=False)

    for label, score in zip(["top", "bottom"], [top_score, bottom_score]):
        # OPTIONAL: if the score is not larger than 0.5, it's indecisive on directionality
        # then add [top_score.values > 0.5]
        top_coords_px = coords_wsi_px[score.indices]

        # NOTE: target size (stride, stride) only works for NON-OVERLAPPING tiles
        # that were extracted in previous steps.
        for score_idx, pos in enumerate(top_coords_px):
            tile = slide.read_region(
                (pos[0], pos[1]),
                0,
                (stride_wsi_px, stride_wsi_px),
            ).convert("RGB")
            tile.save(
                (
                    output_dir
                    / f"{label}_{score_idx + 1}_{score.values[score_idx]:.2f}_{(pos[0].item(), pos[1].item())}.jpg"
                )
            )


def compute_gradcams(
    *,
    model: LitEncDecTransformer,
    feats: Float[Tensor, "n_tiles d_feats"],
    coords: Int[Tensor, "n_tiles 2"],
    categories: Mapping[TargetLabel, Sequence[CategoryLabel]],
    batch_size: int,
    device=None,
) -> Mapping[TargetLabel, Mapping[CategoryLabel, Float[Tensor, "n_tiles"]]]:
    """Computes a stack of attention maps"""
    n_outputs = sum(len(c) for c in categories.values())
    flattened_categories = [
        (target, cat_idx, category)
        for target, cs in categories.items()
        for cat_idx, category in enumerate(cs)
    ]

    # We need one gradcam per output class.  We thus replicate the feature
    # tensor `n_outputs` times so we can obtain a gradient map for each of them.
    # Since this would be infeasible for models with many targets, we don't
    # calculate all gradcams at once, but do it in batches instead.
    gradcams = []
    batch_size = min(batch_size, n_outputs)
    feats_batch: Float[Tensor, "batch_size n_tiles d_feats"] = feats.unsqueeze(
        0
    ).repeat(batch_size, 1, 1)
    model = model.eval()
    for idx in range(0, n_outputs, batch_size):
        feats_batch = feats_batch.detach()  # Zero grads of input features
        feats_batch.requires_grad = True
        model.zero_grad()
        scores = model.predict_step(
            (feats_batch, coords.type_as(feats_batch)), batch_idx=0
        )

        # TODO update this comment; the sentiment is all true
        # Now we have a stack of predictions for each class.  All the rows
        # should be exactly the same, as they only depend on the (repeated and
        # thus identical) tile features.  If we now take the diagonal values
        # `y_i`, the output of the n-th entry _exclusively_ depends on the n-th
        # feature repetition.  If we now calculate `(d sum y_i)/dx`, the n-th
        # repetition of tile features' tensor's gradient will contain exactly
        # `dy_i/dx`.
        sum(
            scores[target_label][i, cat_idx]
            for i, (target_label, cat_idx, _) in enumerate(
                flattened_categories[idx : idx + batch_size]
            )
        ).backward()

        gradcam: Float[Tensor, "batch_size n_tiles"] = (
            feats_batch.grad * feats_batch
        ).sum(-1)
        gradcams.append(gradcam.detach().to(device=device))

    # If n_outs isn't divisible by batch_size, we'll have some superfluous
    # output maps which we have to drop
    gradcams: Float[Tensor, "n_outputs n_tiles"] = torch.cat(gradcams)[:n_outputs]
    unflattened_gradcams_2d = defaultdict(dict)
    for (target_label, _, category), gradcam_2d in zip(flattened_categories, gradcams):
        unflattened_gradcams_2d[target_label][category] = gradcam_2d

    return unflattened_gradcams_2d


def vals_to_im(
    scores: Float[Tensor, "n_tiles *d_feats"],
    norm_coords: Int[Tensor, "n_tiles *d_feats"],
) -> Float[Tensor, "i j *d_feats"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = norm_coords.max(0).values.flip(0) + 1
    im = torch.zeros((*size, *scores.shape[1:])).type_as(scores)

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = norm_coords[:, 1] * im.shape[1] + norm_coords[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    return im


def recursive_union(*dicts):
    res = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, dict):
                res[k] = recursive_union(res[k], v) if k in res else v
            elif isinstance(v, Tensor):
                res[k] = torch.cat((res[k], v)) if k in res else v
            else:
                raise RuntimeError()

    return res


def make_metadata(**kwargs) -> PngInfo:
    """Helper function to generate a PNG metadata dict"""
    metadata = PngInfo()
    for k, v in kwargs.items():
        metadata.add_text(k, v)
    return metadata


def make_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="Directory path for the output",
    )

    parser.add_argument(
        "-f",
        "--feature-dir",
        metavar="FEATURE_DIR",
        dest="feature_dirs",
        type=Path,
        required=True,
        action="append",
        help="Path containing the slide features as `h5` files. Can be specified multiple times",
    )

    parser.add_argument(
        "-m",
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to the checkpoint file",
    )

    parser.add_argument(
        "-t",
        "--targets",
        action="append",
        default=[],
        help="Targets to generate heatmaps for",
    )

    parser.add_argument(
        "-s",
        "--slides",
        action="append",
        default=[],
        help="Slides to generate heatmaps for (stem of the h5 file)",
    )

    parser.add_argument(
        "-w",
        "--wsi-dir",
        type=Path,
        help="Path containing WSI images for top tile extraction",
    )

    parser.add_argument(
        "--top",
        default=0,
        type=int,
        help="How many top / bottom tiles to generate (default: 0)",
    )

    parser.add_argument("--batch-size", type=int, default=0x20)

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    return parser


if __name__ == "__main__":
    main()
