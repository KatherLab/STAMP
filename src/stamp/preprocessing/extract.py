# %%
import hashlib
import logging
from collections.abc import Callable
from functools import cache
from pathlib import Path
from random import shuffle
from tempfile import NamedTemporaryFile
from typing import Iterator, Literal, assert_never

import h5py
import numpy as np
import numpy.typing as npt
import openslide
import torch
from PIL import Image
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from stamp.preprocessing.extractor import Extractor
from stamp.preprocessing.tiling import (
    Microns,
    MPPExtractionError,
    SlidePixels,
    TilePixels,
    get_slide_mpp_,
    tiles_with_cache,
)

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2024 Marko van Treeck"
__license__ = "MIT"


supported_extensions = {
    ".czi",
    ".svs",
    ".tif",
    ".vms",
    ".vmu",
    ".ndpi",
    ".scn",
    ".mrxs",
    ".tiff",
    ".svslide",
    ".bif",
    ".qptiff",
}

logger = logging.getLogger("stamp")


@cache
def _get_preprocessing_code_hash() -> str:
    """The hash of the entire preprocessing codebase.

    It is used to assure that features extracted with different versions of this code base
    can be identified as such after the fact.
    """
    hasher = hashlib.sha256()
    for file_path in sorted(Path(__file__).parent.glob("*.py")):
        with open(file_path, "rb") as fp:
            hasher.update(fp.read())
    return hasher.hexdigest()


class _TileDataset(IterableDataset):
    def __init__(
        self,
        slide_path: Path,
        cache_dir: Path | None,
        transform: Callable[[Image.Image], torch.Tensor],
        tile_size_um: Microns,
        tile_size_px: TilePixels,
        max_supertile_size_slide_px: SlidePixels,
        max_workers: int,
        brightness_cutoff: int | None,
    ) -> None:
        self.slide_path = slide_path
        self.cache_dir = cache_dir
        self.transform = transform
        self.tile_size_um = tile_size_um
        self.tile_size_px = tile_size_px
        self.max_supertile_size_slide_px = max_supertile_size_slide_px
        self.max_workers = max_workers
        self.brightness_cutoff = brightness_cutoff

        # Already check if we can extract the MPP here.
        # We don't want to kill our dataloader later,
        # because that leads to _a lot_ of error messages which are difficult to read
        if get_slide_mpp_(openslide.OpenSlide(slide_path)) is None:
            raise MPPExtractionError()

    def __iter__(self) -> Iterator[tuple[Tensor, Microns, Microns]]:
        return (
            (self.transform(tile.image), tile.coordinates.x, tile.coordinates.y)
            for tile in tiles_with_cache(
                self.slide_path,
                cache_dir=self.cache_dir,
                tile_size_um=self.tile_size_um,
                tile_size_px=self.tile_size_px,
                max_supertile_size_slide_px=self.max_supertile_size_slide_px,
                max_workers=self.max_workers,
                brightness_cutoff=self.brightness_cutoff,
            )
        )


def extract_(
    *,
    wsi_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    extractor: (
        Literal[
            "ctranspath",
            "mahmood-conch",
            "mahmood-uni",
            "dino-bloom",
            "virchow2",
            "empty",
        ]
        | Extractor
    ),
    tile_size_px: TilePixels,
    tile_size_um: Microns,
    max_workers: int,
    accelerator: DeviceLikeType,
    brightness_cutoff: int | None,
) -> None:
    if extractor == "ctranspath":
        from stamp.preprocessing.extractor.ctranspath import ctranspath

        extractor = ctranspath()

    elif extractor == "mahmood-conch":
        from stamp.preprocessing.extractor.conch import conch

        extractor = conch()

    elif extractor == "mahmood-uni":
        from stamp.preprocessing.extractor.uni import uni

        extractor = uni()

    elif extractor == "dino-bloom":
        from stamp.preprocessing.extractor.dinobloom import dino_bloom

        extractor = dino_bloom()

    elif extractor == "virchow2":
        from stamp.preprocessing.extractor.virchow2 import virchow2

        extractor = virchow2()

    elif extractor == "empty":
        from stamp.preprocessing.extractor.empty import empty

        extractor = empty()

    elif isinstance(extractor, Extractor):
        extractor = extractor

    else:
        assert_never(extractor)  # This should be unreachable

    model = extractor.model.to(accelerator).eval()
    extractor_id = f"{extractor.identifier}-{_get_preprocessing_code_hash()[:8]}"

    logger.info(f"Using extractor {extractor.identifier}")

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    feat_output_dir = output_dir / extractor_id

    slide_paths = [
        slide_path
        for extension in supported_extensions
        for slide_path in wsi_dir.glob(f"**/*{extension}")
    ]
    # We shuffle so if we run multiple jobs on multiple computers at the same time,
    # They won't interfere with each other too much
    shuffle(slide_paths)

    for slide_path in (progress := tqdm(slide_paths)):
        progress.set_description(str(slide_path.relative_to(wsi_dir)))
        logger.debug(f"processing {slide_path}")

        feature_output_path = feat_output_dir / slide_path.relative_to(
            wsi_dir
        ).with_suffix(".h5")
        if feature_output_path.exists():
            logger.debug(
                f"skipping {slide_path} because {feature_output_path} already exists"
            )
            continue

        feature_output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            ds = _TileDataset(
                slide_path=slide_path,
                cache_dir=cache_dir,
                transform=extractor.transform,
                tile_size_um=tile_size_um,
                tile_size_px=tile_size_px,
                max_supertile_size_slide_px=SlidePixels(2**10),
                max_workers=max_workers,
                brightness_cutoff=brightness_cutoff,
            )
            # Parallelism is implemented in the dataset iterator already, so one worker is enough!
            dl = DataLoader(ds, batch_size=64, num_workers=1, drop_last=False)

            feats, xs_um, ys_um = [], [], []
            for tiles, xs, ys in tqdm(dl, leave=False):
                with torch.inference_mode():
                    feats.append(model(tiles.to(accelerator)).detach().half().cpu())
                xs_um.append(xs.float())
                ys_um.append(ys.float())
        except Exception:
            logger.exception(f"error while extracting features from {slide_path}")
            continue

        if len(feats) == 0:
            logger.info(f"no tiles found in {slide_path}, skipping")
            continue

        coords = torch.stack([torch.concat(xs_um), torch.concat(ys_um)], dim=1).numpy()

        # Save the file under an intermediate name to prevent half-written files
        with (
            NamedTemporaryFile(dir=output_dir, delete=False) as tmp_h5_file,
            h5py.File(tmp_h5_file, "w") as h5_fp,
        ):
            try:
                h5_fp["coords"] = coords
                h5_fp["feats"] = torch.concat(feats).numpy()

                h5_fp.attrs["extractor"] = extractor_id
                h5_fp.attrs["unit"] = "um"
                h5_fp.attrs["tile_size"] = tile_size_um
            except Exception:
                logger.exception(f"error while writing {feature_output_path}")
                if tmp_h5_file is not None:
                    Path(tmp_h5_file.name).unlink(missing_ok=True)
                continue

            Path(tmp_h5_file.name).rename(feature_output_path)
            logger.debug(f"saved features to {feature_output_path}")

        # Save rejection thumbnail
        thumbnail_path = feat_output_dir / slide_path.relative_to(wsi_dir).with_suffix(
            ".jpg"
        )
        thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
        _get_rejection_thumb(
            openslide.OpenSlide(str(slide_path)),
            size=(512, 512),
            coords_um=coords,
            tile_size_um=tile_size_um,
        ).convert("RGB").save(thumbnail_path)


def _get_rejection_thumb(
    slide: openslide.OpenSlide,
    *,
    size: tuple[int, int],
    coords_um: npt.NDArray,
    tile_size_um: Microns,
) -> Image.Image:
    """Creates a thumbnail of the slide"""

    inclusion_map = np.zeros(
        np.uint32(
            np.ceil(np.array(slide.dimensions) * get_slide_mpp_(slide) / tile_size_um)
        ),
        dtype=bool,
    )

    for y, x in np.round(coords_um / tile_size_um).astype(np.uint32):
        inclusion_map[y, x] = True

    thumb = slide.get_thumbnail(size).convert("RGBA")
    discarded_im = Image.fromarray(
        np.where(
            inclusion_map.transpose()[:, :, None], [0, 0, 0, 0], [255, 0, 0, 128]
        ).astype(np.uint8)
    ).resize(thumb.size, resample=Image.Resampling.NEAREST)

    thumb.paste(discarded_im, mask=discarded_im)
    return thumb
