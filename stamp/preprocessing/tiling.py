import hashlib
import json
import logging
import re
import xml.dom.minidom as minidom
from collections.abc import Iterator
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, NamedTuple, NewType, TypedDict, TypeVar
from zipfile import ZipFile

import cv2
import numpy as np
import openslide
from PIL import Image

logger = logging.getLogger("stamp")

# Digest of _this_ file, used for unambiguously identifying the tiling procedure.
# As a consequence, all details pertaining to tiling should be limited to _this_ file.
with open(__file__, "rb") as this_file_fp:
    code_hash = hashlib.file_digest(this_file_fp, "sha256").hexdigest()


Microns = NewType("Microns", float)
"""Micrometers, usually referring to the tissue on the slide"""

SlidePixels = NewType("SlidePixels", int)
"""Pixels of the WSI scan at largest magnification"""

TilePixels = NewType("TilePixels", int)
"""Pixels on the Tile"""

Unit = TypeVar("Unit")


@dataclass
class XYCoords(Generic[Unit]):
    x: Unit
    y: Unit


class Tile(NamedTuple, Generic[Unit]):
    """A tile with associated metadata"""

    image: Image.Image
    """The actual image data"""
    coordinates: XYCoords[Unit]
    """Position from the top-left corner of the WSI scan"""
    size: Unit
    """Length of the tile's sides"""


def tiles_with_cache(
    slide_path: Path,
    *,
    cache_dir: Path | None,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    max_supertile_size_slide_px: SlidePixels,
    max_workers: int,
) -> Iterator[Tile[Microns]]:
    """Iterates over the tiles in a WSI, using or saving a cached version if applicable"""

    if cache_dir is None:
        # If we have no cache dir, fall back to normal tile extraction.
        yield from tiles_with_tissue(
            slide=openslide.OpenSlide(slide_path),
            tile_size_um=tile_size_um,
            tile_size_px=tile_size_px,
            max_supertile_size_slide_px=max_supertile_size_slide_px,
            max_workers=max_workers,
        )
        return

    tiler_params: TilerParams = {
        "slide_path": str(slide_path),
        "tile_size_um": tile_size_um,
        "tile_size_px": tile_size_px,
        "max_supertile_size_slide_px": max_supertile_size_slide_px,
        "code_sha256": code_hash,
    }
    tiler_params_hash = hashlib.sha256(
        json.dumps(tiler_params, sort_keys=True).encode()
    ).hexdigest()
    cache_file_path = (
        cache_dir / slide_path.with_suffix(f".{tiler_params_hash}.zip").name
    )
    if cache_file_path.exists():
        # If we have a cached version of the tiles
        # which were extracted with the same params / code,
        # we will use those.
        yield from tiles_from_cache_file(cache_file_path)

    else:
        # Extract the features and save them to a cache file for later retreival.

        # We first open a temporary file and then rename it at the end.
        # Since renaming is an atomic operation on most file systems,
        # this will ensure that our cache zips will always be consistent.
        tmp_cache_file_path = cache_file_path.with_suffix(".tmp")
        try:
            with ZipFile(tmp_cache_file_path, "w") as zip:
                with zip.open("tiler_params.json", "w") as tiler_params_json_fp:
                    tiler_params_json_fp.write(json.dumps(tiler_params).encode())

                for tile in tiles_with_tissue(
                    openslide.OpenSlide(slide_path),
                    tile_size_um=tile_size_um,
                    tile_size_px=tile_size_px,
                    max_supertile_size_slide_px=max_supertile_size_slide_px,
                    max_workers=max_workers,
                ):
                    with zip.open(
                        f"tile_({float(tile.coordinates.x)}, {float(tile.coordinates.y)}).jpg",
                        "w",
                    ) as tile_zip_fp:
                        tile.image.save(tile_zip_fp, format="jpeg")

                    yield tile
        except Exception as e:
            logger.exception(f"error while processing {slide_path}")
            tmp_cache_file_path.unlink(missing_ok=True)
            raise e

        # We have written the entire file, time to rename it to its final name.
        tmp_cache_file_path.rename(cache_file_path)


def tiles_with_tissue(
    slide: openslide.OpenSlide,
    *,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    max_supertile_size_slide_px: SlidePixels,
    max_workers: int,
) -> Iterator[Tile[Microns]]:
    """Yields all tiels from a WSI which (probably) show tissue"""
    for tile in tiles(
        slide=slide,
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
        max_supertile_size_slide_px=max_supertile_size_slide_px,
        max_workers=max_workers,
    ):
        if has_enough_texture(tile.image):
            yield tile


def tiles(
    slide: openslide.OpenSlide,
    *,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    max_supertile_size_slide_px: SlidePixels,
    max_workers: int,
) -> Iterator[Tile[Microns]]:
    """Yields tiles, excluding background.

    Some background may still be included,
    as the background rejection only happens on supertile level.
    """
    # With openslide extracting medium-sized tiles is faster
    # than extracting very small or large tiles.
    # We thus first extract these larger "supertiles".
    for supertile, supertile_coords_um, supertile_size_um in supertiles(
        slide=slide,
        tile_size_um=tile_size_um,
        tile_size_px=tile_size_px,
        max_supertile_size_slide_px=max_supertile_size_slide_px,
        max_workers=max_workers,
    ):
        assert supertile.size[0] == supertile.size[1], "supertile needs to be square"
        assert (
            supertile.size[0] % tile_size_px == 0
        ), "supertile needs to perfectly divide into tiles"
        no_tiles = supertile.size[0] // tile_size_px
        assert round(supertile_size_um / no_tiles - tile_size_um) == 0

        for y in range(0, no_tiles):
            for x in range(0, no_tiles):
                tile = supertile.crop(
                    (
                        x * tile_size_px,
                        y * tile_size_px,
                        (x + 1) * tile_size_px,
                        (y + 1) * tile_size_px,
                    )
                )
                yield Tile(
                    image=tile,
                    coordinates=XYCoords(
                        x=Microns(supertile_coords_um.x + x * tile_size_um),
                        y=Microns(supertile_coords_um.y + y * tile_size_um),
                    ),
                    size=tile_size_um,
                )


def foreground_coords(
    slide: openslide.OpenSlide, tile_size_slide_px: SlidePixels
) -> Iterator[XYCoords[SlidePixels]]:
    """Yields coordinates of tiles which aren't too bright and thus probably not background"""
    supertile_thumb_size = np.ceil(
        np.array(slide.dimensions) / tile_size_slide_px
    ).astype(np.uint32)
    # We resize a second time because differences in rounding
    thumb_grayscale = np.array(
        slide.get_thumbnail(tuple(supertile_thumb_size.astype(np.uint32) * 2))
        .resize(tuple(supertile_thumb_size))
        .convert("I")
    )
    is_foreground = thumb_grayscale < 240

    for y_slide_px in range(0, slide.dimensions[1], tile_size_slide_px):
        for x_slide_px in range(0, slide.dimensions[0], tile_size_slide_px):
            if is_foreground[
                y_slide_px // tile_size_slide_px, x_slide_px // tile_size_slide_px
            ]:
                yield XYCoords(SlidePixels(x_slide_px), SlidePixels(y_slide_px))


def has_enough_texture(tile: Image.Image) -> bool:
    """`True` if the image has a bunch of edges,
    i.e. if the image is likely to contain tissue"""
    # L mode converts the image to grayscale with values from 0...255
    tile_grayscale = tile.convert("L")
    # hardcoded thresholds
    edges = cv2.Canny(np.array(tile_grayscale), 40, 100)
    edge_score = np.mean(edges) / 255

    # if "at least two precent of our image are edges",
    # we deem it to have enough texture
    return bool(edge_score > 0.02)


def supertiles(
    slide: openslide.OpenSlide,
    *,
    tile_size_um: Microns,
    tile_size_px: TilePixels,
    max_supertile_size_slide_px: SlidePixels,
    max_workers: int,
) -> Iterator[Tile[Microns]]:
    slide_mpp = get_slide_mpp(slide)
    if slide_mpp is None:
        raise MPPExtractionError()

    # We calculate the `supertile_slide_px` such that they can hold a whole number of tiles
    # which, before scaling down, is still less than `max_supertile_slide_px`
    max_supertile_um = max_supertile_size_slide_px * slide_mpp
    len_of_supertile_in_tiles = max(int(max_supertile_um // tile_size_um), 1)

    tile_size_slide_px = int(np.ceil(tile_size_um / slide_mpp))
    supertile_size_slide_px = SlidePixels(
        tile_size_slide_px * len_of_supertile_in_tiles
    )
    supertile_size_tile_px = TilePixels(tile_size_px * len_of_supertile_in_tiles)

    with futures.ThreadPoolExecutor(max_workers) as executor:
        futs = []
        for coords_slide_px in foreground_coords(
            slide=slide, tile_size_slide_px=supertile_size_slide_px
        ):
            future = executor.submit(
                lambda x_slide_px, y_slide_px: Tile(
                    image=slide.read_region(
                        (x_slide_px, y_slide_px),
                        0,
                        (supertile_size_slide_px, supertile_size_slide_px),
                    )
                    .resize((supertile_size_tile_px, supertile_size_tile_px))
                    .convert("RGB"),
                    coordinates=XYCoords(
                        x=Microns(x_slide_px * slide_mpp),
                        y=Microns(y_slide_px * slide_mpp),
                    ),
                    size=Microns(supertile_size_slide_px * slide_mpp),
                ),
                x_slide_px=coords_slide_px.x,
                y_slide_px=coords_slide_px.y,
            )
            futs.append(future)

        for future in futures.as_completed(futs):
            yield future.result()


class MPPExtractionError(Exception):
    """Raised when the Microns Per Pixel (MPP) extraction from the slide's metadata fails"""

    pass


class TilerParams(TypedDict):
    """The parameters used during tiling / background rejection"""

    slide_path: str
    """The path of the WSI the features were extracted from"""
    tile_size_um: Microns
    """Length of each tile in microns"""
    tile_size_px: TilePixels
    """Length of each tile in pixels"""
    max_supertile_size_slide_px: SlidePixels

    code_sha256: str
    """The hash of this file at the time of extraction"""
    # Including this ensures that,
    # if we change the tile rejection strategy,
    # the cache also gets invalidated


def get_cache_file_tiler_params(cache_file_path: Path) -> TilerParams | None:
    with ZipFile(cache_file_path, "r") as zip_fp:
        if "tiler_params.json" in zip_fp.namelist():
            return json.loads(zip_fp.read("tiler_params.json").decode())
        else:
            return None


def tiles_from_cache_file(cache_file_path: Path) -> Iterator[Tile]:
    with ZipFile(cache_file_path, "r") as zip_fp:
        tiler_params = json.loads(zip_fp.read("tiler_params.json").decode())

        for name in zip_fp.namelist():
            match = re.match(r"tile_\((\d+\.\d+), (\d+\.\d+)\).jpg", name)
            if match is None:
                continue

            # extract coordinates
            x_um_str, y_um_str = match.groups()
            x_um, y_um = Microns(float(x_um_str)), Microns(float(y_um_str))

            with zip_fp.open(name, "r") as tile_fp:
                yield Tile(
                    image=Image.open(tile_fp),
                    coordinates=XYCoords(x_um, y_um),
                    size=tiler_params["tile_size_um"],
                )


def get_slide_mpp(slide: openslide.OpenSlide) -> float | None:
    """Returns the Microns per Slide Pixel of the WSI, or None if none could be found."""
    if openslide.PROPERTY_NAME_MPP_X in slide.properties:
        return float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
    elif slide_mpp := extract_mpp_from_comments(slide):
        return slide_mpp
    elif slide_mpp := extract_mpp_from_metadata(slide):
        return slide_mpp
    else:
        return None


def extract_mpp_from_comments(slide: openslide.OpenSlide) -> float | None:
    slide_properties = slide.properties.get("openslide.comment", default="")
    pattern = r"<PixelSizeMicrons>(.*?)</PixelSizeMicrons>"
    match = re.search(pattern, slide_properties)
    if match is not None and (mpp := match.group(1)) is not None:
        return float(mpp)
    else:
        return None


def extract_mpp_from_metadata(slide: openslide.OpenSlide) -> float | None:
    try:
        xml_path = slide.properties["tiff.ImageDescription"]
        doc = minidom.parseString(xml_path)
        collection = doc.documentElement
        images = collection.getElementsByTagName("Image")
        pixels = images[0].getElementsByTagName("Pixels")
        mpp = float(pixels[0].getAttribute("PhysicalSizeX"))
    except Exception as e:
        logger.exception("failed to extract MPP from image description")
        return None
    return mpp
