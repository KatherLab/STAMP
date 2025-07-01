import logging
from collections.abc import Callable, Iterator
from pathlib import Path
from random import shuffle
from tempfile import NamedTemporaryFile
from typing import assert_never

import h5py
import numpy as np
import numpy.typing as npt
import openslide
import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

import stamp
from stamp.cache import get_processing_code_hash
from stamp.preprocessing.config import ExtractorName
from stamp.preprocessing.extractor import Extractor
from stamp.preprocessing.tiling import (
    MPPExtractionError,
    get_slide_mpp_,
    tiles_with_cache,
)
from stamp.types import (
    DeviceLikeType,
    ImageExtension,
    Microns,
    SlideMPP,
    SlidePixels,
    TilePixels,
)

__author__ = "Marko van Treeck"
__copyright__ = "Copyright (C) 2022-2024 Marko van Treeck"
__license__ = "MIT"

# Our images can be rather large, so let's remove the decompression bomb warning
Image.MAX_IMAGE_PIXELS = None

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

_logger = logging.getLogger("stamp")


class _TileDataset(IterableDataset):
    def __init__(
        self,
        slide_path: Path,
        cache_dir: Path | None,
        cache_tiles_ext: ImageExtension,
        transform: Callable[[Image.Image], torch.Tensor],
        tile_size_um: Microns,
        tile_size_px: TilePixels,
        max_supertile_size_slide_px: SlidePixels,
        max_workers: int,
        brightness_cutoff: int | None,
        canny_cutoff: float | None,
        default_slide_mpp: SlideMPP | None,
    ) -> None:
        self.slide_path = slide_path
        self.cache_dir = cache_dir
        self.cache_tiles_ext: ImageExtension = cache_tiles_ext
        self.transform = transform
        self.tile_size_um = tile_size_um
        self.tile_size_px = tile_size_px
        self.max_supertile_size_slide_px = max_supertile_size_slide_px
        self.max_workers = max_workers
        self.brightness_cutoff = brightness_cutoff
        self.canny_cutoff = canny_cutoff
        self.default_slide_mpp = default_slide_mpp

        # Already check if we can extract the MPP here.
        # We don't want to kill our dataloader later,
        # because that leads to _a lot_ of error messages which are difficult to read
        if (
            get_slide_mpp_(
                openslide.open_slide(slide_path), default_mpp=default_slide_mpp
            )
            is None
        ):
            raise MPPExtractionError()

    def __iter__(self) -> Iterator[tuple[Tensor, Microns, Microns]]:
        return (
            (self.transform(tile.image), tile.coordinates.x, tile.coordinates.y)
            for tile in tiles_with_cache(
                self.slide_path,
                cache_dir=self.cache_dir,
                cache_tiles_ext=self.cache_tiles_ext,
                tile_size_um=self.tile_size_um,
                tile_size_px=self.tile_size_px,
                max_supertile_size_slide_px=self.max_supertile_size_slide_px,
                max_workers=self.max_workers,
                brightness_cutoff=self.brightness_cutoff,
                canny_cutoff=self.canny_cutoff,
                default_slide_mpp=self.default_slide_mpp,
            )
        )


def extract_(
    *,
    wsi_dir: Path,
    output_dir: Path,
    cache_dir: Path | None,
    cache_tiles_ext: ImageExtension,
    extractor: ExtractorName | Extractor,
    tile_size_px: TilePixels,
    tile_size_um: Microns,
    max_workers: int,
    device: DeviceLikeType,
    default_slide_mpp: SlideMPP | None,
    brightness_cutoff: int | None,
    canny_cutoff: float | None,
    generate_hash: bool,
) -> None:
    """
    Extracts features from slides.
    Build in a fail-safe way, i.e. slides for which feature extraction triggers an exception
    are skipped.

    Args:
        default_slide_mpp:
            If not `None`, ignore the slide metadata MPP, instead replacing it with this value.
            Useful for slides without metadata.
    """
    match extractor:
        case ExtractorName.CTRANSPATH:
            from stamp.preprocessing.extractor.ctranspath import ctranspath

            extractor = ctranspath()

        case ExtractorName.CHIEF_CTRANSPATH:
            from stamp.preprocessing.extractor.chief_ctranspath import chief_ctranspath

            extractor = chief_ctranspath()

        case ExtractorName.CONCH:
            from stamp.preprocessing.extractor.conch import conch

            extractor = conch()

        case ExtractorName.CONCH1_5:
            from stamp.preprocessing.extractor.conch1_5 import conch1_5

            extractor = conch1_5()

        case ExtractorName.UNI:
            from stamp.preprocessing.extractor.uni import uni

            extractor = uni()

        case ExtractorName.UNI2:
            from stamp.preprocessing.extractor.uni2 import uni2

            extractor = uni2()

        case ExtractorName.DINO_BLOOM:
            from stamp.preprocessing.extractor.dinobloom import dino_bloom

            extractor = dino_bloom()

        case ExtractorName.VIRCHOW:
            from stamp.preprocessing.extractor.virchow import virchow

            extractor = virchow()

        case ExtractorName.VIRCHOW_FULL:
            from stamp.preprocessing.extractor.virchow_full import virchow_full

            extractor = virchow_full()

        case ExtractorName.VIRCHOW2:
            from stamp.preprocessing.extractor.virchow2 import virchow2

            extractor = virchow2()

        case ExtractorName.H_OPTIMUS_0:
            from stamp.preprocessing.extractor.h_optimus_0 import h_optimus_0

            extractor = h_optimus_0()

        case ExtractorName.H_OPTIMUS_1:
            from stamp.preprocessing.extractor.h_optimus_1 import h_optimus_1

            extractor = h_optimus_1()

        case ExtractorName.GIGAPATH:
            from stamp.preprocessing.extractor.gigapath import gigapath

            extractor = gigapath()

        case ExtractorName.MUSK:
            from stamp.preprocessing.extractor.musk import musk

            extractor = musk()

        case ExtractorName.MSTAR:
            from stamp.preprocessing.extractor.mstar import mstar

            extractor = mstar()

        case ExtractorName.PLIP:
            from stamp.preprocessing.extractor.plip import plip

            extractor = plip()

        case ExtractorName.EMPTY:
            from stamp.preprocessing.extractor.empty import empty

            extractor = empty()

        case Extractor():
            extractor = extractor

        case _ as unreachable:
            assert_never(unreachable)

    model = extractor.model.to(device).eval()

    code_hash = get_processing_code_hash(Path(__file__))[:8]

    extractor_id = extractor.identifier

    if generate_hash:
        extractor_id += f"-{code_hash}"

    _logger.info(f"Using extractor {extractor.identifier}")

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
        _logger.debug(f"processing {slide_path}")

        feature_output_path = feat_output_dir / slide_path.relative_to(
            wsi_dir
        ).with_suffix(".h5")
        if feature_output_path.exists():
            _logger.debug(
                f"skipping {slide_path} because {feature_output_path} already exists"
            )
            continue

        feature_output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            ds = _TileDataset(
                slide_path=slide_path,
                cache_dir=cache_dir,
                cache_tiles_ext=cache_tiles_ext,
                transform=extractor.transform,
                tile_size_um=tile_size_um,
                tile_size_px=tile_size_px,
                max_supertile_size_slide_px=SlidePixels(2**10),
                max_workers=max_workers,
                brightness_cutoff=brightness_cutoff,
                canny_cutoff=canny_cutoff,
                default_slide_mpp=default_slide_mpp,
            )
            # Parallelism is implemented in the dataset iterator already, so one worker is enough!
            dl = DataLoader(ds, batch_size=64, num_workers=1, drop_last=False)

            feats, xs_um, ys_um = [], [], []
            for tiles, xs, ys in tqdm(dl, leave=False):
                with torch.inference_mode():
                    feats.append(model(tiles.to(device)).detach().half().cpu())
                xs_um.append(xs.float())
                ys_um.append(ys.float())
        except MPPExtractionError:
            _logger.exception(
                "failed to extract MPP from slide. "
                "You can try manually setting it by adding `preprocessing.default_slide_mpp = <MPP>` "
            )
            continue
        except Exception:
            _logger.exception(f"error while extracting features from {slide_path}")
            continue

        if len(feats) == 0:
            _logger.info(f"no tiles found in {slide_path}, skipping")
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

                h5_fp.attrs["stamp_version"] = stamp.__version__
                h5_fp.attrs["extractor"] = str(extractor.identifier)
                h5_fp.attrs["unit"] = "um"
                h5_fp.attrs["tile_size_um"] = tile_size_um  # changed in v2.1.0
                h5_fp.attrs["tile_size_px"] = tile_size_px
                h5_fp.attrs["code_hash"] = code_hash
            except Exception:
                _logger.exception(f"error while writing {feature_output_path}")
                if tmp_h5_file is not None:
                    Path(tmp_h5_file.name).unlink(missing_ok=True)
                continue

            Path(tmp_h5_file.name).rename(feature_output_path)
            _logger.debug(f"saved features to {feature_output_path}")

        # Save rejection thumbnail
        thumbnail_path = feat_output_dir / slide_path.relative_to(wsi_dir).with_suffix(
            ".jpg"
        )
        thumbnail_path.parent.mkdir(exist_ok=True, parents=True)
        _get_rejection_thumb(
            openslide.open_slide(str(slide_path)),
            size=(512, 512),
            coords_um=coords,
            tile_size_um=tile_size_um,
            default_slide_mpp=default_slide_mpp,
        ).convert("RGB").save(thumbnail_path)


def _get_rejection_thumb(
    slide: openslide.AbstractSlide,
    *,
    size: tuple[int, int],
    coords_um: npt.NDArray,
    tile_size_um: Microns,
    default_slide_mpp: SlideMPP | None,
) -> Image.Image:
    """Creates a thumbnail of the slide"""

    inclusion_map = np.zeros(
        np.uint32(
            np.ceil(
                np.array(slide.dimensions)
                * get_slide_mpp_(slide, default_mpp=default_slide_mpp)
                / tile_size_um
            )
        ),
        dtype=bool,
    )

    for y, x in np.floor(coords_um / tile_size_um).astype(np.uint32):
        inclusion_map[y, x] = True

    thumb = slide.get_thumbnail(size).convert("RGBA")
    discarded_im = Image.fromarray(
        np.where(
            inclusion_map.transpose()[:, :, None], [0, 0, 0, 0], [255, 0, 0, 128]
        ).astype(np.uint8)
    ).resize(thumb.size, resample=Image.Resampling.NEAREST)

    thumb.paste(discarded_im, mask=discarded_im)
    return thumb
