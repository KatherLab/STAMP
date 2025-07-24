from __future__ import annotations

import logging
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import cast

import h5py
import matplotlib.pyplot as plt
import numpy as np
import openslide
import torch
from jaxtyping import Float, Integer
from matplotlib.axes import Axes
from matplotlib.patches import Patch
from matplotlib.figure import Figure
from PIL import Image
from torch import Tensor

from stamp.modeling.data import get_coords, get_stride
from stamp.modeling.lightning_model import LitVisionTransformer
from stamp.modeling.vision_transformer import VisionTransformer
from stamp.preprocessing import supported_extensions
from stamp.preprocessing.tiling import get_slide_mpp_
from stamp.types import DeviceLikeType, Microns, SlideMPP, TilePixels

_logger = logging.getLogger("stamp")


def _gradcam_per_category(
    model: VisionTransformer,
    feats: Float[Tensor, "tile feat"],
    coords: Float[Tensor, "tile 2"],
) -> Float[Tensor, "tile category"]:
    feats = feats.detach()
    feats.requires_grad_(True)

    logits = model(
        bags=feats.unsqueeze(0), coords=coords.unsqueeze(0), mask=None
    ).squeeze(0)  # shape [C]

    cams = []
    for c in range(logits.numel()):
        (grad,) = torch.autograd.grad(
            logits[c], feats, retain_graph=True, allow_unused=False
        )

        # channel‑wise weights, spatially pooled
        w = grad.mean(dim=0)  # [feat]
        # apply ReLU to focus on features that have a positive influence on the class
        cam = torch.relu((feats * w).sum(dim=1))  # [tile]

        # normalize
        cam = (cam - cam.min()) / (cam.max() + 1e-8)
        cams.append(cam)

    return torch.stack(cams, dim=1)  # [tile, C]


def _vals_to_im(
    scores: Float[Tensor, "tile *"],
    coords_norm: Integer[Tensor, "tile coord"],
) -> Float[Tensor, "width height *"]:
    """Scatter *scores* into image grid using integer *coords_norm*."""
    size = coords_norm.max(0).values.flip(0) + 1  # (H, W)
    im = torch.zeros(
        (*size.tolist(), *scores.shape[1:]), dtype=scores.dtype, device=scores.device
    )

    idx_flat = coords_norm[:, 1] * im.shape[1] + coords_norm[:, 0]
    im_flat = im.flatten(end_dim=1)
    im_flat[idx_flat] = scores
    return im_flat.reshape_as(im)


def _show_class_map(
    class_ax: Axes,
    top_score_indices: Integer[Tensor, "W H"],
    gradcam_2d: Float[Tensor, "W H C"],
    categories: Collection[str],
) -> None:
    """Render per‑tile class index on a white background, only where CAM>0."""
    cmap = plt.get_cmap("Pastel1")
    w, h = top_score_indices.shape

    # Start with white canvas alpha=1
    classes = np.ones((w, h, 4), dtype=float)

    mask = gradcam_2d.sum(-1).cpu().numpy() > 0  # tiles with any activation
    coloured = cmap(top_score_indices.cpu().numpy())  # (w,h,4)
    classes[mask] = coloured[mask]

    class_ax.imshow(classes)
    class_ax.set_title("Class map")
    class_ax.axis("off")

    class_ax.legend(
        handles=[
            Patch(facecolor=cmap(i), label=cat) for i, cat in enumerate(categories)
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=min(4, len(categories)),
        frameon=False,
    )


def _plot_thumbnail_comparison(
    *,
    thumb: np.ndarray,
    overlay: np.ndarray,
    title_left: str,
    title_right: str,
) -> Figure:
    """Side-by-side figure helper."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(thumb)
    axs[0].set_title(title_left)
    axs[0].axis("off")

    axs[1].imshow(overlay)
    axs[1].set_title(title_right)
    axs[1].axis("off")

    fig.tight_layout()
    return fig


def heatmaps_(
    *,
    feature_dir: Path,
    wsi_dir: Path,
    checkpoint_path: Path,
    output_dir: Path,
    slide_paths: Iterable[Path] | None,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
    # tile selection
    topk: int = 10,
    bottomk: int = 10,
) -> None:
    """Generate Grad-CAM heat-maps / plots / tiles for WSIs under *wsi_dir*."""

    model = LitVisionTransformer.load_from_checkpoint(checkpoint_path).to(device).eval()

    wsis_to_process: Iterable[Path]
    if slide_paths is not None:
        wsis_to_process = (wsi_dir / p for p in slide_paths)
    else:
        wsis_to_process = (
            p for ext in supported_extensions for p in wsi_dir.glob(f"**/*{ext}")
        )

    # ---------------------------------------------------------------------
    for wsi_path in wsis_to_process:
        h5_path = feature_dir / wsi_path.with_suffix(".h5").name
        if not h5_path.exists():
            _logger.warning("Missing h5 for %s - skipped", wsi_path.name)
            continue

        # ── Prepare output folders ───────────────────────────────────────
        slide_dir = output_dir / h5_path.stem
        raw_dir = slide_dir / "raw"
        plots_dir = slide_dir / "plots"
        tiles_dir = slide_dir / "tiles"
        for d in (raw_dir, plots_dir, tiles_dir):
            d.mkdir(parents=True, exist_ok=True)

        _logger.info("Processing %s", wsi_path.name)

        # ── Load slide & features ────────────────────────────────────────
        slide = openslide.open_slide(str(wsi_path))
        slide_mpp = get_slide_mpp_(slide, default_mpp=default_slide_mpp)
        if slide_mpp is None:
            _logger.error("Could not determine MPP for %s - skipped", wsi_path.name)
            continue

        with h5py.File(h5_path) as h5:
            feats = torch.tensor(h5["feats"][:]).float().to(device)  # type: ignore # [tile, feat]
            coords_info = get_coords(h5)
            coords_um = torch.from_numpy(coords_info.coords_um).float()  # [tile,2]
            stride_um = Microns(get_stride(coords_um))
            tile_size_slide_px = TilePixels(
                int(round(float(coords_info.tile_size_um) / slide_mpp))
            )

        coords_norm = (coords_um / stride_um).round().long()  # grid coords
        coords_tile_slide_px = torch.round(coords_um / slide_mpp).long()

        # ── Inference ────────────────────────────────────────────────────
        with torch.no_grad():
            slide_logits = model.vision_transformer(
                bags=feats.unsqueeze(0), coords=coords_um.unsqueeze(0), mask=None
            ).squeeze(0)
        slide_score = slide_logits.softmax(0)  # [C]
        predicted_idx = int(slide_score.argmax())

        gradcam = _gradcam_per_category(
            model=model.vision_transformer, feats=feats, coords=coords_um
        )
        gradcam_2d = _vals_to_im(gradcam, coords_norm).detach()  # (W,H,C)

        # Per‑tile soft‑max scores
        with torch.no_grad():
            scores = model.vision_transformer.forward(
                bags=feats.unsqueeze(-2), coords=coords_um.unsqueeze(-2), mask=None
            )
            scores = scores.softmax(dim=1)
        scores_2d = _vals_to_im(scores, coords_norm).detach()  # (W,H,C)

        # ── Overview (unchanged except path) ─────────────────────────────
        fig_over, axs_over = plt.subplots(
            nrows=2, ncols=max(2, len(model.categories)), figsize=(14, 9)
        )

        # Class map (first row, second col)
        _show_class_map(
            class_ax=axs_over[0, 1],
            top_score_indices=scores_2d.topk(2).indices[:, :, 0],
            gradcam_2d=gradcam_2d,
            categories=model.categories,
        )

        last_attention: Tensor | None = None  # for thumbnail later
        for ax, (pos_idx, category) in zip(axs_over[1, :], enumerate(model.categories)):
            # support measure (distance to runner‑up)
            top2 = scores.topk(2)
            category_support = torch.where(
                top2.indices[..., 0] == pos_idx,
                scores[..., pos_idx] - top2.values[..., 1],
                scores[..., pos_idx] - top2.values[..., 0],
            )

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
            )
            category_score = category_support * attention / (attention.max() + 1e-8)
            last_attention = attention  # keep last for thumbnail scaling

            # ─── Heat‑map image (RdBu) – saved to raw ─────────────────--
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
                raw_dir / f"{h5_path.stem}-{category}={slide_score[pos_idx]:0.2f}.png"
            )

            # ─── Top and bottom tiles (predicted class only) ──────────────────
            if pos_idx == predicted_idx:
                for score_val, index in zip(*category_score.topk(topk)):
                    slide.read_region(
                        tuple(coords_tile_slide_px[index].tolist()),
                        0,
                        (tile_size_slide_px, tile_size_slide_px),
                    ).convert("RGB").save(
                        tiles_dir / f"top-{h5_path.stem}-{category}={score_val:.2f}.jpg"
                    )

                for score_val, index in zip(*(-category_score).topk(bottomk)):
                    slide.read_region(
                        tuple(coords_tile_slide_px[index].tolist()),
                        0,
                        (tile_size_slide_px, tile_size_slide_px),
                    ).convert("RGB").save(
                        tiles_dir
                        / f"bottom-{h5_path.stem}-{category}={-score_val:.2f}.jpg"
                    )

        assert last_attention is not None

        # Thumbnail extraction (H&E)
        dims_um = np.array(slide.dimensions) * slide_mpp
        thumb_raw = np.array(
            slide.get_thumbnail(np.round(dims_um * 8 / 256).astype(int))  # type: ignore
        )
        att_im = _vals_to_im(last_attention.unsqueeze(-1), coords_norm).squeeze(-1)

        # Crop thumbnail to slide footprint (8 px per tile)
        thumb_cropped = thumb_raw[: att_im.shape[0] * 8, : att_im.shape[1] * 8]
        Image.fromarray(thumb_cropped).save(raw_dir / f"thumbnail-{h5_path.stem}.png")

        # Place H&E thumbnail into overview (first row, first col)
        axs_over[0, 0].imshow(thumb_cropped)
        axs_over[0, 0].set_title("Thumbnail")
        axs_over[0, 0].axis("off")

        # Save overview
        for ax in axs_over.ravel():
            ax.axis("off")
        fig_over.tight_layout()
        fig_over.savefig(plots_dir / f"overview-{h5_path.stem}.png")
        plt.close(fig_over)

        # ── Comparison plot: H&E vs class‑map ───────────────────────────
        # Build RGB class‑map with same palette & white background
        cmap = plt.get_cmap("Pastel1")
        top_class_idx = scores_2d.topk(1).indices.squeeze(-1)  # (W,H)
        mask_activate = gradcam_2d.sum(-1) > 0
        classmap_rgb = np.ones((*top_class_idx.shape, 3), dtype=float)  # white base
        classmap_rgb[mask_activate.cpu().numpy()] = cmap(top_class_idx.cpu().numpy())[
            mask_activate.cpu().numpy()
        ][:, :3]
        fig_cmp2 = _plot_thumbnail_comparison(
            thumb=thumb_cropped,
            overlay=(classmap_rgb * 255).astype(np.uint8),
            title_left="H&E thumbnail",
            title_right="Class-map (Pastel1)",
        )
        fig_cmp2.savefig(plots_dir / f"classmap-{h5_path.stem}.png")
        plt.close(fig_cmp2)
