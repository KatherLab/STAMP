# %%
import hashlib
import logging
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Iterator, Literal, assert_never

import h5py
import torch
from PIL import Image
from torch import Tensor
from torch._prims_common import DeviceLikeType
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm

from stamp.preprocessing.extractor import Extractor
from stamp.preprocessing.tiling import (
    Microns,
    SlidePixels,
    TilePixels,
    tiles_with_cache,
)

supported_extensions = {
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
def get_preprocessing_code_hash() -> str:
    """The hash of the entire preprocessing codebase.

    It is used to assure that features extracted with different versions of this code base
    can be identified as such after the fact.
    """
    hasher = hashlib.sha256()
    for file_path in sorted(Path(__file__).parent.glob("**/*.py")):
        with open(file_path, "rb") as fp:
            hasher.update(fp.read())
    return hasher.hexdigest()


class TileDataset(IterableDataset):
    def __init__(
        self,
        slide_path: Path,
        cache_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        tile_size_um: Microns,
        tile_size_px: TilePixels,
        max_supertile_size_slide_px: SlidePixels,
        max_workers: int,
    ) -> None:
        self.slide_path = slide_path
        self.cache_dir = cache_dir
        self.transform = transform
        self.tile_size_um = tile_size_um
        self.tile_size_px = tile_size_px
        self.max_supertile_size_slide_px = max_supertile_size_slide_px
        self.max_workers = max_workers

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
            )
        )


def extract_(
    *,
    wsi_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    extractor: (
        Literal[
            "ctranspath",
            "mahmood-conch",
            "mahmood-uni",
        ]
        | Extractor
    ),
    tile_size_px: TilePixels = TilePixels(224),
    tile_size_um: Microns = Microns(256),
    max_workers: int = 32,
    device: DeviceLikeType = "cuda" if torch.cuda.is_available() else "cpu",
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
    elif isinstance(extractor, Extractor):
        extractor = extractor
    else:
        assert_never(extractor)  # This should be unreachable

    model = extractor.model.to(device).eval()
    extractor_id = f"{extractor.identifier}-{get_preprocessing_code_hash()[:8]}"

    logger.info(f"Using extractor {extractor.identifier}")

    output_dir.mkdir(exist_ok=True)
    output_dir = output_dir / extractor_id
    output_dir.mkdir(exist_ok=True)
    cache_dir.mkdir(exist_ok=True)

    for slide_path in (
        progress := tqdm(
            list(
                slide_path
                for extension in supported_extensions
                for slide_path in wsi_dir.glob(f"**/*{extension}")
            )
        )
    ):
        progress.set_description(str(slide_path.relative_to(wsi_dir)))
        logger.debug(f"processing {slide_path}")

        feature_output_path = output_dir / slide_path.relative_to(wsi_dir).with_suffix(
            ".h5"
        )
        tmp_feature_output_path = feature_output_path.with_suffix(".tmp")
        if feature_output_path.exists() or tmp_feature_output_path.exists():
            logger.debug(
                f"skipping {slide_path} because {feature_output_path.exists()=} or {tmp_feature_output_path.exists()}"
            )
            continue

        ds = TileDataset(
            slide_path=slide_path,
            cache_dir=cache_dir,
            transform=extractor.transform,
            tile_size_um=tile_size_um,
            tile_size_px=tile_size_px,
            max_supertile_size_slide_px=SlidePixels(2**10),
            max_workers=max_workers,
        )
        # Parallelism is implemented in the dataset iterator already, so one worker is enough!
        dl = DataLoader(ds, batch_size=64, num_workers=1, drop_last=False)

        feats, xs_um, ys_um = [], [], []
        for tiles, xs, ys in tqdm(dl, leave=False):
            with torch.inference_mode():
                feats.append(model(tiles.to(device)).detach().half().cpu())
            xs_um.append(xs.float())
            ys_um.append(ys.float())

        try:
            # Save the file under an intermediate name to prevent half-written files
            with h5py.File(tmp_feature_output_path, "w") as h5_fp:
                h5_fp["coords"] = torch.stack(
                    [torch.concat(xs_um), torch.concat(ys_um)], dim=1
                ).numpy()
                h5_fp["feats"] = torch.concat(feats).numpy()
                h5_fp.attrs["extractor"] = extractor_id
        except Exception:
            logger.exception(f"error while writing {tmp_feature_output_path}")
            tmp_feature_output_path.unlink(missing_ok=True)
        else:  # no exception
            # We have written the entire file, time to rename it to its final name.
            tmp_feature_output_path.rename(feature_output_path)
            logger.debug(f"saving to {feature_output_path}")
