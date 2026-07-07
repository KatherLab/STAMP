#!/usr/bin/env python3
"""Conservative QC analysis for the STAMP cell-resolution pilot.

This script measures image quality and cell-like object visibility in the
existing pilot review crops. It does not classify cells and does not call blasts.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "validation_report" / "pilot_cell_res" / "pilot_manifest.json"
DEFAULT_OUT = ROOT / "validation_report" / "pilot_cell_qc"

LANCZOS = Image.Resampling.LANCZOS


@dataclass(frozen=True)
class CandidateSummary:
    contours: list[np.ndarray]
    count: int
    area_fraction: float
    mask_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conservative QC on STAMP cell-resolution pilot crops."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as fp:
        return json.load(fp)


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def quality_metrics(rgb: np.ndarray, prefix: str) -> dict[str, float]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    tenengrad = float(np.mean(sobel_x * sobel_x + sobel_y * sobel_y))

    edges = cv2.Canny(gray, 40, 100)
    background = (gray > 235) & (saturation < 25)
    tissue = (gray < 242) & (saturation > 10)

    return {
        f"{prefix}_brightness_mean": float(gray.mean()),
        f"{prefix}_brightness_std": float(gray.std()),
        f"{prefix}_saturation_mean": float(saturation.mean()),
        f"{prefix}_laplacian_var": laplacian_var,
        f"{prefix}_tenengrad": tenengrad,
        f"{prefix}_edge_density": float((edges > 0).mean()),
        f"{prefix}_background_fraction": float(background.mean()),
        f"{prefix}_tissue_fraction": float(tissue.mean()),
    }


def segment_cell_like_candidates(rgb: np.ndarray) -> CandidateSummary:
    """Detect stained round-ish objects in zoom crops.

    The result is intentionally conservative and non-diagnostic. It should be
    interpreted as "cell-like candidates", not as cells of a known type.
    """

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    saturation = hsv[:, :, 1]

    tissue_mask = ((gray < 242) & (saturation > 10)).astype(np.uint8) * 255
    stain_mask = ((gray < 210) & (saturation > 25)).astype(np.uint8) * 255

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]
    mask = cv2.bitwise_and(otsu, stain_mask)
    mask = cv2.bitwise_and(mask, tissue_mask)

    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    accepted: list[np.ndarray] = []
    accepted_area = 0.0
    image_area = float(rgb.shape[0] * rgb.shape[1])
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < 60 or area > 3500:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width < 6 or height < 6 or width > 100 or height > 100:
            continue
        aspect = width / max(height, 1)
        if aspect < 0.25 or aspect > 4.0:
            continue
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            continue
        circularity = 4 * math.pi * area / (perimeter * perimeter)
        if circularity < 0.15:
            continue
        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = area / max(hull_area, 1.0)
        if solidity < 0.35:
            continue

        accepted.append(contour)
        accepted_area += area

    return CandidateSummary(
        contours=accepted,
        count=len(accepted),
        area_fraction=accepted_area / max(image_area, 1.0),
        mask_fraction=float((mask > 0).mean()),
    )


def save_overlay(
    *,
    rgb: np.ndarray,
    candidates: CandidateSummary,
    out_path: Path,
) -> None:
    overlay = rgb.copy()
    cv2.drawContours(overlay, candidates.contours, -1, (0, 220, 0), 2)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(
        overlay,
        f"cell-like candidates: {candidates.count} (not blasts)",
        (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path, quality=92)


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarize_tiles(
    tile_rows: list[dict[str, Any]], slides: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, float]]:
    by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tile_rows:
        by_sample[row["sample_id"]].append(row)

    prelim: list[dict[str, Any]] = []
    for slide in slides:
        meta = slide["metadata"]
        rows = by_sample[meta["sample_id"]]
        top_rows = [row for row in rows if row["tile_kind"] == "top"]
        bottom_rows = [row for row in rows if row["tile_kind"] == "bottom"]

        top_candidate_density = mean(
            [float(row["candidate_count"]) for row in top_rows]
        )
        bottom_candidate_density = mean(
            [float(row["candidate_count"]) for row in bottom_rows]
        )
        all_candidate_density = mean(
            [float(row["candidate_count"]) for row in rows]
        )
        zoom_laplacian_mean = mean(
            [float(row["zoom_laplacian_var"]) for row in rows]
        )
        zoom_background_mean = mean(
            [float(row["zoom_background_fraction"]) for row in rows]
        )
        zoom_tissue_mean = mean([float(row["zoom_tissue_fraction"]) for row in rows])
        top_bottom_candidate_ratio = (
            top_candidate_density / bottom_candidate_density
            if bottom_candidate_density > 0
            else 0.0
        )

        prelim.append(
            {
                "group": meta["group"],
                "sample_id": meta["sample_id"],
                "stem": meta["stem"],
                "experiment": meta["experiment"],
                "split": meta["split"],
                "blast_percent_gt": meta.get("blast_percent_gt", ""),
                "high_blast_gt": meta.get("high_blast_gt", ""),
                "blast_percent_pred": meta.get("blast_percent_pred", ""),
                "blast_percent_abs_err": meta.get("blast_percent_abs_err", ""),
                "high_blast_yes": meta.get("high_blast_yes", ""),
                "tile_count": len(rows),
                "top_candidate_count_mean": top_candidate_density,
                "bottom_candidate_count_mean": bottom_candidate_density,
                "all_candidate_count_mean": all_candidate_density,
                "top_bottom_candidate_ratio": top_bottom_candidate_ratio,
                "zoom_laplacian_var_mean": zoom_laplacian_mean,
                "zoom_background_fraction_mean": zoom_background_mean,
                "zoom_tissue_fraction_mean": zoom_tissue_mean,
                "overlay_contact_sheet": "",
            }
        )

    thresholds = {
        "low_focus_laplacian_p25": percentile(
            [float(row["zoom_laplacian_var_mean"]) for row in prelim], 25
        ),
        "low_candidate_count_p25": percentile(
            [float(row["all_candidate_count_mean"]) for row in prelim], 25
        ),
        "high_background_fraction_p75": percentile(
            [float(row["zoom_background_fraction_mean"]) for row in prelim], 75
        ),
    }

    slide_rows: list[dict[str, Any]] = []
    for row in prelim:
        low_focus = (
            float(row["zoom_laplacian_var_mean"])
            <= thresholds["low_focus_laplacian_p25"]
        )
        low_candidates = (
            float(row["all_candidate_count_mean"])
            <= thresholds["low_candidate_count_p25"]
        )
        high_background = (
            float(row["zoom_background_fraction_mean"])
            >= thresholds["high_background_fraction_p75"]
        )
        ratio = float(row["top_bottom_candidate_ratio"])
        top_bottom_shift = ratio < 0.5 or ratio > 2.0
        concern_score = int(low_focus) + int(low_candidates) + int(high_background)

        row = {
            **row,
            "flag_low_focus": int(low_focus),
            "flag_low_candidate_density": int(low_candidates),
            "flag_high_background": int(high_background),
            "flag_top_bottom_candidate_shift": int(top_bottom_shift),
            "technical_concern_score": concern_score,
        }
        slide_rows.append(row)

    group_rows = summarize_groups(slide_rows)
    return slide_rows, group_rows, thresholds


def summarize_groups(slide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in slide_rows:
        grouped[row["group"]].append(row)

    summaries: list[dict[str, Any]] = []
    for group, rows in grouped.items():
        summaries.append(
            {
                "group": group,
                "slide_count": len(rows),
                "candidate_count_mean": mean(
                    [float(row["all_candidate_count_mean"]) for row in rows]
                ),
                "top_candidate_count_mean": mean(
                    [float(row["top_candidate_count_mean"]) for row in rows]
                ),
                "bottom_candidate_count_mean": mean(
                    [float(row["bottom_candidate_count_mean"]) for row in rows]
                ),
                "zoom_laplacian_var_mean": mean(
                    [float(row["zoom_laplacian_var_mean"]) for row in rows]
                ),
                "background_fraction_mean": mean(
                    [float(row["zoom_background_fraction_mean"]) for row in rows]
                ),
                "technical_concern_score_mean": mean(
                    [float(row["technical_concern_score"]) for row in rows]
                ),
                "slides_with_any_concern": sum(
                    1 for row in rows if int(row["technical_concern_score"]) > 0
                ),
            }
        )
    return sorted(summaries, key=lambda row: row["group"])


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def load_thumb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        thumb = image.convert("RGB")
        thumb.thumbnail((size, size), LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        x = (size - thumb.size[0]) // 2
        y = (size - thumb.size[1]) // 2
        canvas.paste(thumb, (x, y))
        return canvas


def make_overlay_contact_sheet(
    *,
    sample_id: str,
    group: str,
    overlays: list[Path],
    out_path: Path,
) -> None:
    margin = 28
    gap = 10
    tile_size = 168
    cols = 4
    rows = math.ceil(len(overlays) / cols)
    width = margin * 2 + cols * tile_size + (cols - 1) * gap
    height = margin * 2 + 70 + rows * (tile_size + 24)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    draw.text(
        (margin, margin),
        f"{group.upper()} | {sample_id} | cell-like candidates only",
        fill=(0, 0, 0),
        font=font,
    )
    draw.text(
        (margin, margin + 22),
        "Green contours are non-diagnostic object candidates, not blast calls.",
        fill=(120, 50, 20),
        font=font,
    )

    start_y = margin + 58
    for idx, path in enumerate(overlays):
        row, col = divmod(idx, cols)
        x = margin + col * (tile_size + gap)
        y = start_y + row * (tile_size + 24)
        thumb = load_thumb(path, tile_size)
        canvas.paste(thumb, (x, y))
        draw.rectangle(
            [x, y, x + tile_size - 1, y + tile_size - 1],
            outline=(180, 180, 180),
        )
        draw.text((x + 4, y + tile_size + 4), path.stem, fill=(60, 60, 60))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def analyze_tile(
    *,
    slide_meta: dict[str, Any],
    tile: dict[str, Any],
    out_dir: Path,
) -> dict[str, Any]:
    native_path = Path(tile["native_path"])
    zoom_path = Path(tile["zoom_path"])
    native_rgb = read_rgb(native_path)
    zoom_rgb = read_rgb(zoom_path)

    row: dict[str, Any] = {
        "group": slide_meta["group"],
        "sample_id": slide_meta["sample_id"],
        "stem": slide_meta["stem"],
        "experiment": slide_meta["experiment"],
        "split": slide_meta["split"],
        "blast_percent_gt": slide_meta.get("blast_percent_gt", ""),
        "high_blast_gt": slide_meta.get("high_blast_gt", ""),
        "blast_percent_pred": slide_meta.get("blast_percent_pred", ""),
        "blast_percent_abs_err": slide_meta.get("blast_percent_abs_err", ""),
        "high_blast_yes": slide_meta.get("high_blast_yes", ""),
        "tile_kind": tile["kind"],
        "tile_rank": int(tile["rank"]),
        "tile_score": tile.get("score", ""),
        "native_path": str(native_path),
        "zoom_path": str(zoom_path),
        "native_width": int(native_rgb.shape[1]),
        "native_height": int(native_rgb.shape[0]),
        "zoom_width": int(zoom_rgb.shape[1]),
        "zoom_height": int(zoom_rgb.shape[0]),
    }
    row.update(quality_metrics(native_rgb, "native"))
    row.update(quality_metrics(zoom_rgb, "zoom"))

    candidates = segment_cell_like_candidates(zoom_rgb)
    row.update(
        {
            "candidate_count": candidates.count,
            "candidate_area_fraction": candidates.area_fraction,
            "candidate_mask_fraction": candidates.mask_fraction,
            "candidate_count_per_megapixel": candidates.count
            / max((zoom_rgb.shape[0] * zoom_rgb.shape[1]) / 1_000_000, 1e-8),
        }
    )

    overlay_path = (
        out_dir
        / slide_meta["group"]
        / slide_meta["sample_id"]
        / "overlays"
        / f"{tile['kind']}_{int(tile['rank']):02d}_overlay.jpg"
    )
    save_overlay(rgb=zoom_rgb, candidates=candidates, out_path=overlay_path)
    row["overlay_path"] = str(overlay_path)
    return row


def format_number(value: Any, digits: int = 2) -> str:
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def html_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "<p>No rows.</p>"
    parts = ["<table>", "<thead><tr>"]
    for field in fields:
        parts.append(f"<th>{html.escape(field)}</th>")
    parts.append("</tr></thead><tbody>")
    for row in rows:
        parts.append("<tr>")
        for field in fields:
            value = row.get(field, "")
            if isinstance(value, float):
                value = format_number(value, 3)
            parts.append(f"<td>{html.escape(str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def write_html_report(
    *,
    out_dir: Path,
    tile_rows: list[dict[str, Any]],
    slide_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> None:
    ranked = sorted(
        slide_rows,
        key=lambda row: (
            int(row["technical_concern_score"]),
            float(row["zoom_background_fraction_mean"]),
            -float(row["zoom_laplacian_var_mean"]),
        ),
        reverse=True,
    )
    slide_fields = [
        "group",
        "sample_id",
        "blast_percent_gt",
        "blast_percent_pred",
        "all_candidate_count_mean",
        "zoom_laplacian_var_mean",
        "zoom_background_fraction_mean",
        "technical_concern_score",
        "flag_low_focus",
        "flag_low_candidate_density",
        "flag_high_background",
        "flag_top_bottom_candidate_shift",
    ]
    group_fields = [
        "group",
        "slide_count",
        "candidate_count_mean",
        "top_candidate_count_mean",
        "bottom_candidate_count_mean",
        "zoom_laplacian_var_mean",
        "background_fraction_mean",
        "technical_concern_score_mean",
        "slides_with_any_concern",
    ]

    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>STAMP Cell-Resolution QC</title>",
        """
<style>
body { font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }
.warning { background: #fff3cd; border: 1px solid #e0b95c; padding: 12px; }
.readiness { background: #eef6ff; border: 1px solid #9cc4e4; padding: 12px; }
table { border-collapse: collapse; margin: 12px 0 24px; width: 100%; }
th, td { border: 1px solid #d9e2ec; padding: 5px 7px; font-size: 13px; }
th { background: #f5f7fa; text-align: left; }
.slide { border-top: 2px solid #d9e2ec; padding: 18px 0 28px; }
.contact { max-width: 980px; width: 100%; border: 1px solid #d9e2ec; }
.small { color: #52606d; }
</style>
""",
        "</head><body>",
        "<h1>STAMP Cell-Resolution QC</h1>",
        (
            "<div class='warning'><b>Important:</b> green contours are "
            "cell-like object candidates only. They are not validated cells, "
            "not blast calls, and not diagnostic output.</div>"
        ),
        "<h2>DeepHeme readiness</h2>",
        (
            "<div class='readiness'><ul>"
            "<li>Raw WSIs: available locally.</li>"
            "<li>Tile and UNI2 feature data: available locally.</li>"
            "<li>Cell-level annotations: missing.</li>"
            "<li>DeepHeme pretrained detector/classifier: not available from "
            "the public repo/release checked during planning.</li>"
            "<li>Conclusion: ready for QC and annotation preparation, not "
            "ready for true DeepHeme inference.</li>"
            "</ul></div>"
        ),
        "<h2>Thresholds</h2>",
        html_table([thresholds], list(thresholds)),
        "<h2>Group Summary</h2>",
        html_table(group_rows, group_fields),
        "<h2>Slides Ranked By Technical Concern</h2>",
        html_table(ranked, slide_fields),
        "<h2>Overlay Contact Sheets</h2>",
    ]

    for row in ranked:
        sample_id = row["sample_id"]
        contact = Path(row["overlay_contact_sheet"])
        parts.append("<section class='slide'>")
        parts.append(
            f"<h3>{html.escape(row['group'])}: {html.escape(sample_id)}</h3>"
        )
        parts.append(
            "<p class='small'>"
            f"Candidate mean: {format_number(row['all_candidate_count_mean'])}; "
            f"focus mean: {format_number(row['zoom_laplacian_var_mean'])}; "
            f"background mean: {format_number(row['zoom_background_fraction_mean'])}; "
            f"concern score: {row['technical_concern_score']}"
            "</p>"
        )
        parts.append(
            f"<a href='{html.escape(rel(contact, out_dir))}'>"
            f"<img class='contact' src='{html.escape(rel(contact, out_dir))}' "
            f"alt='overlay contact sheet for {html.escape(sample_id)}'></a>"
        )
        parts.append("</section>")

    parts.append("<h2>Per-Tile Metrics</h2>")
    parts.append(
        "<p class='small'>Full table is in <code>tile_metrics.csv</code>; "
        "showing first 40 rows here.</p>"
    )
    tile_fields = [
        "group",
        "sample_id",
        "tile_kind",
        "tile_rank",
        "candidate_count",
        "zoom_laplacian_var",
        "zoom_background_fraction",
        "zoom_edge_density",
    ]
    parts.append(html_table(tile_rows[:40], tile_fields))
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts))


def run_analysis(
    manifest: dict[str, Any], manifest_path: Path, out_dir: Path, limit: int | None
) -> None:
    slides = manifest["slides"]
    if limit is not None:
        if limit < 1:
            raise SystemExit("--limit must be at least 1")
        slides = slides[:limit]

    out_dir.mkdir(parents=True, exist_ok=True)

    tile_rows: list[dict[str, Any]] = []
    overlay_paths_by_sample: dict[str, list[Path]] = defaultdict(list)

    for slide in slides:
        meta = slide["metadata"]
        for tile in slide["tiles"]:
            row = analyze_tile(slide_meta=meta, tile=tile, out_dir=out_dir)
            tile_rows.append(row)
            overlay_paths_by_sample[meta["sample_id"]].append(Path(row["overlay_path"]))

    slide_rows, group_rows, thresholds = summarize_tiles(tile_rows, slides)
    for slide in slides:
        meta = slide["metadata"]
        sample_id = meta["sample_id"]
        contact_path = out_dir / meta["group"] / sample_id / "overlay_contact_sheet.jpg"
        make_overlay_contact_sheet(
            sample_id=sample_id,
            group=meta["group"],
            overlays=overlay_paths_by_sample[sample_id],
            out_path=contact_path,
        )
        for row in slide_rows:
            if row["sample_id"] == sample_id:
                row["overlay_contact_sheet"] = str(contact_path)

    tile_fields = [
        "group",
        "sample_id",
        "stem",
        "experiment",
        "split",
        "blast_percent_gt",
        "high_blast_gt",
        "blast_percent_pred",
        "blast_percent_abs_err",
        "high_blast_yes",
        "tile_kind",
        "tile_rank",
        "tile_score",
        "native_width",
        "native_height",
        "zoom_width",
        "zoom_height",
        "native_brightness_mean",
        "native_brightness_std",
        "native_saturation_mean",
        "native_laplacian_var",
        "native_tenengrad",
        "native_edge_density",
        "native_background_fraction",
        "native_tissue_fraction",
        "zoom_brightness_mean",
        "zoom_brightness_std",
        "zoom_saturation_mean",
        "zoom_laplacian_var",
        "zoom_tenengrad",
        "zoom_edge_density",
        "zoom_background_fraction",
        "zoom_tissue_fraction",
        "candidate_count",
        "candidate_area_fraction",
        "candidate_mask_fraction",
        "candidate_count_per_megapixel",
        "native_path",
        "zoom_path",
        "overlay_path",
    ]
    slide_fields = [
        "group",
        "sample_id",
        "stem",
        "experiment",
        "split",
        "blast_percent_gt",
        "high_blast_gt",
        "blast_percent_pred",
        "blast_percent_abs_err",
        "high_blast_yes",
        "tile_count",
        "top_candidate_count_mean",
        "bottom_candidate_count_mean",
        "all_candidate_count_mean",
        "top_bottom_candidate_ratio",
        "zoom_laplacian_var_mean",
        "zoom_background_fraction_mean",
        "zoom_tissue_fraction_mean",
        "flag_low_focus",
        "flag_low_candidate_density",
        "flag_high_background",
        "flag_top_bottom_candidate_shift",
        "technical_concern_score",
        "overlay_contact_sheet",
    ]
    group_fields = [
        "group",
        "slide_count",
        "candidate_count_mean",
        "top_candidate_count_mean",
        "bottom_candidate_count_mean",
        "zoom_laplacian_var_mean",
        "background_fraction_mean",
        "technical_concern_score_mean",
        "slides_with_any_concern",
    ]

    write_csv(out_dir / "tile_metrics.csv", tile_rows, tile_fields)
    write_csv(out_dir / "slide_summary.csv", slide_rows, slide_fields)
    write_csv(out_dir / "group_summary.csv", group_rows, group_fields)

    qc_manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "input_manifest": str(manifest_path),
        "total_slides": len(slides),
        "total_tiles": len(tile_rows),
        "thresholds": thresholds,
        "outputs": {
            "tile_metrics": str(out_dir / "tile_metrics.csv"),
            "slide_summary": str(out_dir / "slide_summary.csv"),
            "group_summary": str(out_dir / "group_summary.csv"),
            "html": str(out_dir / "index.html"),
        },
        "warning": (
            "Cell-like candidates are conservative image objects, not validated "
            "cells, not blasts, and not diagnostic output."
        ),
    }
    (out_dir / "qc_manifest.json").write_text(json.dumps(qc_manifest, indent=2))
    write_html_report(
        out_dir=out_dir,
        tile_rows=tile_rows,
        slide_rows=slide_rows,
        group_rows=group_rows,
        thresholds=thresholds,
    )

    print(f"Wrote {len(slides)} slide summaries and {len(tile_rows)} tile rows")
    print(f"Output directory: {out_dir}")
    print(f"HTML: {out_dir / 'index.html'}")
    print(f"Slide summary: {out_dir / 'slide_summary.csv'}")


def main() -> None:
    args = parse_args()
    manifest = load_json(args.manifest)
    run_analysis(manifest, args.manifest, args.out, args.limit)


if __name__ == "__main__":
    main()
