import logging
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import cast, no_type_check

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from jaxtyping import Float, Integer
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from PIL import Image
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.func import jacrev  # pyright: ignore[reportPrivateImportUsage]

from stamp.modeling.data import get_coords, get_stride
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.vision_transformer import VisionTransformer
from stamp.preprocessing import supported_extensions
from stamp.preprocessing.tiling import Microns, SlideMPP, TilePixels, get_slide_mpp_

_logger = logging.getLogger("stamp")


def _gradcam_per_category(
    model: VisionTransformer,
    feats: Float[Tensor, "tile feat"],
    coords: Float[Tensor, "tile 2"],
) -> Float[Tensor, "tile category"]:
    feat = -1  # feats dimension

    return (
        (
            feats
            * jacrev(
                lambda bags: torch.softmax(
                    model.forward(
                        bags=bags.unsqueeze(0),
                        coords=coords.unsqueeze(0),
                        mask=torch.zeros(
                            1, len(bags), dtype=torch.bool, device=bags.device
                        ),
                    ),
                    dim=1,
                ).squeeze(0)
            )(feats)
        )
        .mean(feat)
        .abs()
    ).permute(-1, -2)


def _vals_to_im(
    scores: Float[Tensor, "tile feat"],
    coords_norm: Integer[Tensor, "tile coord"],
) -> Float[Tensor, "width height category"]:
    """Arranges scores in a 2d grid according to coordinates"""
    size = coords_norm.max(0).values.flip(0) + 1
    im = torch.zeros((*size.tolist(), *scores.shape[1:])).type_as(scores)

    flattened_im = im.flatten(end_dim=1)
    flattened_coords = coords_norm[:, 1] * im.shape[1] + coords_norm[:, 0]
    flattened_im[flattened_coords] = scores

    im = flattened_im.reshape_as(im)

    return im


def _show_thumb(
    slide, thumb_ax: Axes, attention: Tensor, default_slide_mpp: SlideMPP | None
) -> np.ndarray:
    mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
    dims_um = np.array(slide.dimensions) * mpp
    thumb = slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))
    thumb_ax.imshow(np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8])
    return np.array(thumb)[: attention.shape[0] * 8, : attention.shape[1] * 8]


@no_type_check  # beartype<=0.19.0 breaks here for some reason
def _show_class_map(
    class_ax: Axes,
    top_score_indices: Integer[Tensor, "width height"],
    gradcam_2d: Float[Tensor, "width height category"],
    categories: Collection[str],
) -> None:
    cmap = plt.get_cmap("Pastel1")
    classes = cast(np.ndarray, cmap(top_score_indices.cpu().numpy()))
    classes[..., -1] = (gradcam_2d.sum(-1) > 0).detach().cpu().numpy() * 1.0
    class_ax.imshow(classes)
    class_ax.legend(
        handles=[
            Patch(facecolor=cmap(i), label=cat) for i, cat in enumerate(categories)
        ]
    )


def heatmaps_(
    *,
    feature_dir: Path,
    wsi_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    slide_paths: Iterable[Path] | None,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
    # top tiles
    topk: int,
    bottomk: int,
) -> None:
    model = LitVisionTransformer.load_from_checkpoint(checkpoint_path).to(device).eval()

    # Collect slides to generate heatmaps for
    if slide_paths is not None:
        wsis_to_process = (wsi_dir / slide for slide in slide_paths)
    else:
        wsis_to_process = (
            p for ext in supported_extensions for p in wsi_dir.glob(f"**/*{ext}")
        )

    for wsi_path in wsis_to_process:
        h5_path = feature_dir / wsi_path.with_suffix(".h5").name

        if not h5_path.exists():
            _logger.info(f"could not find matching h5 file at {h5_path}. Skipping...")
            continue

        slide_output_dir = output_dir / h5_path.stem
        slide_output_dir.mkdir(exist_ok=True, parents=True)
        _logger.info(f"creating heatmaps for {wsi_path.name}")

        slide = openslide.open_slide(wsi_path)
        slide_mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
        assert slide_mpp is not None, "could not determine slide MPP"

        with h5py.File(h5_path) as h5:
            feats = (
                torch.tensor(
                    h5["feats"][:]  # pyright: ignore[reportIndexIssue]
                )
                .float()
                .to(device)
            )
            coords_um = get_coords(h5).coords_um
            stride_um = Microns(get_stride(coords_um))

            if h5.attrs.get("unit") == "um":
                tile_size_slide_px = TilePixels(
                    int(round(cast(float, h5.attrs["tile_size"]) / slide_mpp))
                )
            else:
                tile_size_slide_px = TilePixels(int(round(256 / slide_mpp)))

        # grid coordinates, i.e. the top-left most tile is (0, 0), the one to its right (0, 1) etc.
        coords_norm = (coords_um / stride_um).round().long()

        # coordinates as used by OpenSlide
        coords_tile_slide_px = torch.round(coords_um / slide_mpp).long()

        # Score for the entire slide
        slide_score = (
            model.vision_transformer(
                bags=feats.unsqueeze(0),
                coords=coords_um.unsqueeze(0),
                mask=torch.zeros(1, len(feats), dtype=torch.bool, device=device),
            )
            .squeeze(0)
            .softmax(0)
        )

        gradcam = _gradcam_per_category(
            model=model.vision_transformer,
            feats=feats,
            coords=coords_um,
        )  # shape: [tile, category]
        gradcam_2d = _vals_to_im(
            gradcam,
            coords_norm,
        ).detach()  # shape: [width, height, category]

        scores = torch.softmax(
            model.vision_transformer.forward(
                bags=feats.unsqueeze(-2),
                coords=coords_um.unsqueeze(-2),
                mask=torch.zeros(len(feats), 1, dtype=torch.bool, device=device),
            ),
            dim=1,
        )  # shape: [tile, category]
        scores_2d = _vals_to_im(
            scores, coords_norm
        ).detach()  # shape: [width, height, category]

        fig, axs = plt.subplots(
            nrows=2, ncols=max(2, len(model.categories)), figsize=(12, 8)
        )

        _show_class_map(
            class_ax=axs[0, 1],
            top_score_indices=scores_2d.topk(2).indices[:, :, 0],
            gradcam_2d=gradcam_2d,
            categories=model.categories,
        )

        attention = None
        for ax, (pos_idx, category) in zip(axs[1, :], enumerate(model.categories)):
            ax: Axes
            top2 = scores.topk(2)
            # Calculate the distance of the "hot" class
            # to the class with the highest score apart from the hot class
            category_support = torch.where(
                top2.indices[..., 0] == pos_idx,
                scores[..., pos_idx] - top2.values[..., 1],
                scores[..., pos_idx] - top2.values[..., 0],
            )  # shape: [tile]
            assert ((category_support >= -1) & (category_support <= 1)).all()

            # So, if we have a pixel with scores (.4, .4, .2) and would want to get the heat value for the first class,
            # we would get a neutral color, because it is matched with the second class
            # But if our scores were (.4, .3, .3), it would be red,
            # because now our class is .1 above its nearest competitor

            attention = torch.where(
                top2.indices[..., 0] == pos_idx,
                gradcam[..., pos_idx] / gradcam.max(),
                (
                    others := gradcam[
                        ..., list(set(range(len(model.categories))) - {pos_idx})
                    ]
                    .max(-1)
                    .values
                )
                / others.max(),
            )  # shape: [tile]

            category_score = (
                category_support * attention / attention.max()
            )  # shape: [tile]

            score_im = cast(
                np.ndarray,
                plt.get_cmap("RdBu_r")(
                    _vals_to_im(category_score.unsqueeze(-1) / 2 + 0.5, coords_norm)
                    .squeeze(-1)
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )

            score_im[..., -1] = (
                (_vals_to_im(attention.unsqueeze(-1), coords_norm).squeeze(-1) > 0)
                .cpu()
                .numpy()
            )

            ax.imshow(score_im)
            ax.set_title(f"{category} {slide_score[pos_idx].item():1.2f}")
            target_size = np.array(score_im.shape[:2][::-1]) * 8

            Image.fromarray(np.uint8(score_im * 255)).resize(
                tuple(target_size), resample=Image.Resampling.NEAREST
            ).save(
                slide_output_dir
                / f"{h5_path.stem}-{category}={slide_score[pos_idx]:0.2f}.png"
            )

            # Top tiles
            for score, index in zip(*category_score.topk(topk)):
                (
                    slide.read_region(
                        tuple(coords_tile_slide_px[index].tolist()),
                        0,
                        (tile_size_slide_px, tile_size_slide_px),
                    )
                    .convert("RGB")
                    .save(
                        slide_output_dir
                        / f"top-{h5_path.stem}-{category}={score:0.2f}.jpg"
                    )
                )
            for score, index in zip(*(-category_score).topk(bottomk)):
                (
                    slide.read_region(
                        tuple(coords_tile_slide_px[index].tolist()),
                        0,
                        (tile_size_slide_px, tile_size_slide_px),
                    )
                    .convert("RGB")
                    .save(
                        slide_output_dir
                        / f"bottom-{h5_path.stem}-{category}={-score:0.2f}.jpg"
                    )
                )

        assert attention is not None, (
            "attention should have been set in the for loop above"
        )

        # Generate overview
        thumb = _show_thumb(
            slide=slide,
            thumb_ax=axs[0, 0],
            attention=_vals_to_im(
                attention.unsqueeze(-1),
                coords_norm,  # pyright: ignore[reportPossiblyUnboundVariable]
            ).squeeze(-1),
            default_slide_mpp=default_slide_mpp,
        )
        Image.fromarray(thumb).save(slide_output_dir / f"thumbnail-{h5_path.stem}.png")

        for ax in axs.ravel():
            ax.axis("off")

        fig.savefig(slide_output_dir / f"overview-{h5_path.stem}.png")
        plt.close(fig)


def attention_ui_(
    *,
    feature_dir: Path,
    wsi_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    slide_paths: Iterable[Path] | None,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
) -> None:
    try:
        from stamp.heatmaps.attention_ui import show_attention_ui
    except ImportError as e:
        raise ImportError(
            "Attention UI dependencies not installed. "
            "Please reinstall stamp using `pip install 'stamp[attentionui]'`"
        ) from e

    with torch.no_grad():
        # Collect slides to generate attention maps for
        if slide_paths is not None:
            wsis_to_process_all = (wsi_dir / slide for slide in slide_paths)
        else:
            wsis_to_process_all = (
                p for ext in supported_extensions for p in wsi_dir.glob(f"**/*{ext}")
            )

        # Check of a corresponding feature file exists
        wsis_to_process = []
        for wsi_path in wsis_to_process_all:
            h5_path = feature_dir / wsi_path.with_suffix(".h5").name

            if not h5_path.exists():
                _logger.info(
                    f"could not find matching h5 file at {h5_path}. Skipping..."
                )
                continue

            wsis_to_process.append(str(wsi_path))

        # Launch the UI
        show_attention_ui(
            feature_dir,
            wsis_to_process,
            checkpoint_path,
            output_dir,
            slide_paths,
            device,
            default_slide_mpp,
        )
