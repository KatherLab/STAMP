import argparse
from collections.abc import Collection
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from fastai.vision.learner import Learner, load_learner
from jaxtyping import Float, Int
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from PIL import Image
from torch import Tensor


def get_stride(coords: Tensor) -> int:
    xs = coords[:, 0].unique(sorted=True)
    stride = (xs[1:] - xs[:-1]).min()
    return stride


def gradcam_per_category(
    learn: Learner, feats: Tensor, categories: Collection
) -> tuple[Tensor, Tensor]:
    feats_batch = feats.expand((len(categories), *feats.shape)).detach()
    feats_batch.requires_grad = True
    preds = torch.softmax(
        learn.model(feats_batch, torch.tensor([len(feats)] * len(categories))),
        dim=1,
    )
    preds.trace().backward()
    gradcam = (feats_batch * feats_batch.grad).mean(-1).abs()
    return preds, gradcam


def vals_to_im(
    scores: Float[Tensor, "n_tiles *d_feats"],
    norm_coords: Int[Tensor, "n_tiles *d_feats"],
) -> Float[Tensor, "i j *d_feats"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = norm_coords.max(0).values.flip(0) + 1
    im = torch.zeros((*size, *scores.shape[1:]))

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = norm_coords[:, 1] * im.shape[1] + norm_coords[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    return im


def show_thumb(thumb_ax: Axes, svs_dir: Path, h5_path: Path, attention: Tensor) -> None:
    slide = openslide.open_slide(svs_dir / h5_path.with_suffix(".svs").name)
    mpp = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    dims_um = np.array(slide.dimensions) * mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb_ax.imshow(np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8])


def show_class_map(
    class_ax: Axes, top_scores: Tensor, gradcam_2d, categories: Collection[str]
) -> None:
    cmap = plt.get_cmap("Pastel1")
    classes = cmap(top_scores.indices[:, :, 0])
    classes[..., -1] = (gradcam_2d.sum(-1) > 0).detach().cpu() * 1.0
    class_ax.imshow(classes)
    class_ax.legend(
        handles=[
            Patch(facecolor=cmap(i), label=cat) for i, cat in enumerate(categories)
        ]
    )


def main(feature_dir: Path, svs_dir: Path, model_path: Path, output_dir: Path) -> None:
    learn = load_learner(model_path)
    learn.model.eval()
    categories: Collection[str] = learn.dls.train.dataset._datasets[
        -1
    ].encode.categories_[0]

    output_dir.mkdir(exist_ok=True, parents=True)

    for h5_path in feature_dir.glob(f"**/*.h5"):
        with h5py.File(h5_path) as h5:
            feats = torch.tensor(h5["feats"][:]).float()
            coords = torch.tensor(h5["coords"][:], dtype=torch.int)

        stride = get_stride(coords)

        preds, gradcam = gradcam_per_category(
            learn=learn, feats=feats, categories=categories
        )
        gradcam_2d = vals_to_im(gradcam.permute(-1, -2), coords // stride).detach()

        scores = torch.softmax(
            learn.model(feats.unsqueeze(-2), torch.ones((len(feats)))), dim=1
        )
        scores_2d = vals_to_im(scores, coords // stride).detach()

        fig, axs = plt.subplots(nrows=2, ncols=min(2, len(categories)), figsize=(12, 8))

        show_class_map(
            class_ax=axs[0, 1],
            top_scores=scores_2d.topk(2),
            gradcam_2d=gradcam_2d,
            categories=categories,
        )

        for ax, (pos_idx, category) in zip(axs[1, :], enumerate(categories)):
            ax: Axes
            topk = scores_2d.topk(2)
            category_support = torch.where(
                topk.indices[..., 0] == pos_idx,
                scores_2d[..., pos_idx] - topk.values[..., 1],
                scores_2d[..., pos_idx] - topk.values[..., 0],
            )
            attention = torch.where(
                topk.indices[..., 0] == pos_idx,
                gradcam_2d[..., pos_idx] / gradcam_2d.max(),
                (
                    others := gradcam_2d[
                        ..., list(set(range(len(categories))) - {pos_idx})
                    ]
                    .max(-1)
                    .values
                )
                / others.max(),
            )

            score_im = plt.get_cmap("RdBu")(
                -category_support * attention / attention.max() / 2 + 0.5
            )
            score_im[..., -1] = attention > 0

            ax.imshow(score_im)
            ax.set_title(f"{category} {preds[0,pos_idx]:1.2f}")

            Image.fromarray(np.uint8(score_im * 255)).resize(
                np.array(score_im.shape[:2][::-1]) * 8, resample=Image.NEAREST
            ).save(
                output_dir
                / f"scores-{h5_path.stem}--score_{category}={preds[0][pos_idx]:0.2f}.png"
            )

        show_thumb(
            thumb_ax=ax[0, 0], svs_dir=svs_dir, h5_path=h5_path, attention=attention
        )

        for ax in axs.ravel():
            ax.axis("off")

        fig.savefig(output_dir / f"overview-{h5_path.stem}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("heatmaps")
    parser.add_argument(
        "--svs-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory containing the SVSs",
    )
    parser.add_argument(
        "--feature-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory containing the slides' features",
    )
    parser.add_argument(
        "--model",
        metavar="PATH",
        dest="model_path",
        type=Path,
        required=True,
        help="Path to the trained model's export.pkl",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        type=Path,
        required=True,
        help="Directory to save the heatmaps to",
    )
    args = parser.parse_args()
    main(**vars(args))
