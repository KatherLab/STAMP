#!/usr/bin/env python3
"""Build before/after preprocessing visualizations and extract rejected tiles.

For each target slide:
  - "before.jpg" : plain WSI thumbnail
  - "after.jpg"  : thumbnail with KEPT tile positions highlighted (green grid)
  - "rejected_grid.jpg" : grid of example tiles that were filtered out
  - "kept_grid.jpg"     : grid of example tiles that were kept
  - "stats.json"        : tile counts + retention rate + per-reason counts
"""

import json
import zipfile
from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image, ImageDraw

WSI_DIR = Path("/mnt/nvme0n1p1/Jeff_projects/B01/data/AG Janssen")
CACHE_DIR = Path("/mnt/nvme0n1p1/Jeff_projects/B01/cache")
OUT_DIR = Path("/home/jeff/Projects/STAMP/validation_report/preprocessing")
OUT_DIR.mkdir(parents=True, exist_ok=True)

BRIGHTNESS_CUTOFF = 240
CANNY_CUTOFF = 0.02
TILE_SIZE_UM = 256.0
TILE_SIZE_PX = 224

SLIDES = [
    "116 26 B Score 49 - 10 Fokuspunkte - entfettet - 2026-03-04 18.49.41",
    "127 26 C Score 0 - 5 Fokuspunkte - entfettet - 2026-03-04 17.28.20",
    "333 19 A Score 29 - 5 Fokuspunkte - entfettet - 2026-03-03 18.42.23",
    "931 19 A Score 30 - 5 Fokuspunkte - entfettet - 2026-03-04 21.23.41",
]


def find_cache_zip(stem: str) -> Path | None:
    for p in CACHE_DIR.glob(f"{stem}.*.zip"):
        return p
    return None


def load_kept_coords(zip_path: Path) -> set[tuple[int, int]]:
    """Return set of (x_um_int, y_um_int) rounded for kept tiles."""
    coords = set()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not name.startswith("tile_("):
                continue
            body = name[len("tile_(") :].rsplit(")", 1)[0]
            xs, ys = body.split(",")
            coords.add((round(float(xs.strip())), round(float(ys.strip()))))
    return coords


def classify_reject(tile_img: Image.Image) -> str:
    """Return reason a tile would be rejected, or 'kept'."""
    arr_gray = np.array(tile_img.convert("L"))
    mean_brightness = arr_gray.mean()
    if mean_brightness >= BRIGHTNESS_CUTOFF:
        return "brightness"
    edges = cv2.Canny(arr_gray, 40, 100)
    edge_score = edges.mean() / 255
    if edge_score < CANNY_CUTOFF:
        return "canny"
    return "kept"


def grid_image(
    tiles: list[Image.Image],
    cols: int = 4,
    size_px: int = 160,
    gap: int = 4,
    bg=(255, 255, 255),
) -> Image.Image:
    if not tiles:
        return Image.new("RGB", (size_px, size_px), bg)
    rows = (len(tiles) + cols - 1) // cols
    W = cols * size_px + (cols + 1) * gap
    H = rows * size_px + (rows + 1) * gap
    canvas = Image.new("RGB", (W, H), bg)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        t2 = t.resize((size_px, size_px), Image.LANCZOS)
        canvas.paste(t2, (gap + c * (size_px + gap), gap + r * (size_px + gap)))
    return canvas


def process_slide(stem: str) -> dict:
    print(f"\n=== {stem} ===")
    ndpi = WSI_DIR / (stem + ".ndpi")
    cache_zip = find_cache_zip(stem)
    if not cache_zip:
        raise FileNotFoundError(f"no cache zip for {stem}")

    slide = openslide.OpenSlide(str(ndpi))
    mpp = float(slide.properties.get("openslide.mpp-x", 0.221))
    tile_px_slide = int(np.ceil(TILE_SIZE_UM / mpp))
    dims = slide.dimensions
    n_tiles_x = dims[0] // tile_px_slide
    n_tiles_y = dims[1] // tile_px_slide
    theoretical = n_tiles_x * n_tiles_y

    kept_coords = load_kept_coords(cache_zip)
    n_kept = len(kept_coords)

    # Thumbnail at ~1000px max side
    target = 1200
    scale = target / max(dims)
    thumb_size = (int(dims[0] * scale), int(dims[1] * scale))
    thumb = slide.get_thumbnail(thumb_size).convert("RGB")
    tile_thumb_px = max(1, int(tile_px_slide * scale))

    # BEFORE
    thumb.save(OUT_DIR / f"{stem}__before.jpg", quality=85)

    # AFTER (shade rejected tiles)
    after = thumb.copy()
    overlay = Image.new("RGBA", thumb.size, (0, 0, 0, 0))
    dr = ImageDraw.Draw(overlay)
    kept_grid_mask = np.zeros((n_tiles_y, n_tiles_x), dtype=bool)
    for x_um, y_um in kept_coords:
        # Tile (xi, yi) on the grid. The tiler uses supertile-aligned coords,
        # but individual tile coord = xi*TILE_SIZE_UM
        xi = int(round(x_um / TILE_SIZE_UM))
        yi = int(round(y_um / TILE_SIZE_UM))
        if 0 <= xi < n_tiles_x and 0 <= yi < n_tiles_y:
            kept_grid_mask[yi, xi] = True

    # Paint rejected tiles (translucent red) and kept tiles (light green outline)
    for yi in range(n_tiles_y):
        for xi in range(n_tiles_x):
            x0 = xi * tile_thumb_px
            y0 = yi * tile_thumb_px
            x1 = min(x0 + tile_thumb_px, thumb.size[0])
            y1 = min(y0 + tile_thumb_px, thumb.size[1])
            if kept_grid_mask[yi, xi]:
                dr.rectangle(
                    [x0, y0, x1 - 1, y1 - 1], outline=(0, 200, 0, 200), width=1
                )
            else:
                dr.rectangle([x0, y0, x1 - 1, y1 - 1], fill=(200, 0, 0, 90))

    after = Image.alpha_composite(after.convert("RGBA"), overlay).convert("RGB")
    ImageDraw.Draw(after).text(
        (10, 10),
        f"kept={n_kept}/{theoretical} ({100 * n_kept / max(theoretical, 1):.1f}%)  |  rejected in red, kept outlined green",
        fill=(255, 255, 255),
    )
    after.save(OUT_DIR / f"{stem}__after.jpg", quality=85)

    # Sample some rejected positions, extract tiles from WSI at low-res, classify reason
    rejected_positions = []
    for yi in range(n_tiles_y):
        for xi in range(n_tiles_x):
            if not kept_grid_mask[yi, xi]:
                rejected_positions.append((xi, yi))

    kept_sample_zip_names = []
    with zipfile.ZipFile(cache_zip) as zf:
        for n in zf.namelist():
            if n.startswith("tile_(") and n.endswith(".jpg"):
                kept_sample_zip_names.append(n)

    # Render example KEPT tiles from cache
    kept_examples = []
    rng = np.random.default_rng(0)
    if kept_sample_zip_names:
        sampled = rng.choice(
            kept_sample_zip_names,
            size=min(12, len(kept_sample_zip_names)),
            replace=False,
        )
        with zipfile.ZipFile(cache_zip) as zf:
            for name in sampled:
                with zf.open(name) as fp:
                    kept_examples.append(Image.open(fp).convert("RGB").copy())

    # Render example REJECTED tiles by reading from slide
    rejected_examples = []
    reasons = {"brightness": 0, "canny": 0, "unknown": 0}
    if rejected_positions:
        sampled_rej = rng.choice(
            len(rejected_positions),
            size=min(32, len(rejected_positions)),
            replace=False,
        )
        for idx in sampled_rej:
            xi, yi = rejected_positions[idx]
            x_slide = xi * tile_px_slide
            # read at level 0 (full res) at tile_px_slide size
            region = slide.read_region(
                (x_slide, yi * tile_px_slide), 0, (tile_px_slide, tile_px_slide)
            ).convert("RGB")
            small = region.resize((TILE_SIZE_PX, TILE_SIZE_PX), Image.LANCZOS)
            reason = classify_reject(small)
            if reason == "kept":
                # inconsistency: position wasn't in cache but passes both tests.
                # likely a boundary / supertile rounding edge case. count as 'unknown'.
                reasons["unknown"] += 1
            else:
                reasons[reason] += 1
            if len(rejected_examples) < 12:
                # Draw a label on the tile showing reason
                labelled = small.copy()
                d2 = ImageDraw.Draw(labelled)
                d2.rectangle([0, 0, TILE_SIZE_PX - 1, 18], fill=(0, 0, 0))
                d2.text((4, 2), reason, fill=(255, 255, 255))
                rejected_examples.append(labelled)

    # Also classify ALL rejected positions to get reason stats (sample up to 300 for speed)
    reason_counts = {"brightness": 0, "canny": 0, "unknown": 0}
    sample_n = min(300, len(rejected_positions))
    if sample_n > 0:
        idx_sample = rng.choice(len(rejected_positions), size=sample_n, replace=False)
        for idx in idx_sample:
            xi, yi = rejected_positions[idx]
            region = slide.read_region(
                (xi * tile_px_slide, yi * tile_px_slide),
                0,
                (tile_px_slide, tile_px_slide),
            ).convert("RGB")
            small = region.resize((TILE_SIZE_PX, TILE_SIZE_PX), Image.LANCZOS)
            r = classify_reject(small)
            if r == "kept":
                reason_counts["unknown"] += 1
            else:
                reason_counts[r] += 1

    slide.close()

    # Save grids
    grid_image(kept_examples, cols=4, size_px=160).save(
        OUT_DIR / f"{stem}__kept_grid.jpg", quality=88
    )
    grid_image(rejected_examples, cols=4, size_px=160).save(
        OUT_DIR / f"{stem}__rejected_grid.jpg", quality=88
    )

    stats = {
        "stem": stem,
        "dims": list(dims),
        "mpp_x": mpp,
        "tile_size_um": TILE_SIZE_UM,
        "tile_slide_px": tile_px_slide,
        "grid": [n_tiles_x, n_tiles_y],
        "theoretical_tiles": theoretical,
        "kept_tiles": n_kept,
        "retention_pct": 100 * n_kept / max(theoretical, 1),
        "rejected_sampled": sample_n,
        "rejection_reasons_sampled": reason_counts,
        "inferred_fraction_brightness": reason_counts["brightness"] / max(sample_n, 1),
        "inferred_fraction_canny": reason_counts["canny"] / max(sample_n, 1),
    }
    with open(OUT_DIR / f"{stem}__stats.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    out = {}
    for stem in SLIDES:
        out[stem] = process_slide(stem)
    with open(OUT_DIR / "_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote summary to {OUT_DIR / '_summary.json'}")
