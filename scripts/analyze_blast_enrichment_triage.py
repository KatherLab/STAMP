#!/usr/bin/env python3
"""Non-diagnostic blast-enrichment triage for AML STAMP heatmap tiles.

This script does not detect blasts. It combines existing STAMP heatmap tile
scores with conservative image-quality and cell-like object visibility metrics
to rank slides/tiles for review while pathologist feedback is pending.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen")
DEFAULT_OUT = ROOT / "validation_report" / "blast_enrichment_triage"
SLIDE_TABLE = ROOT / "tables" / "stamp_slide.csv"
CLINI_TABLE = ROOT / "tables" / "stamp_clini.csv"

EXPERIMENTS = {
    "blast_percent": {
        "display_name": "BLAST_PERCENT regression",
        "base_dir": DATA_ROOT / "stamp_aml_blast_percent_uni2",
        "task": "regression",
    },
    "high_blast": {
        "display_name": "HIGH_BLAST classification",
        "base_dir": DATA_ROOT / "stamp_aml_high_blast_uni2",
        "task": "classification",
    },
}

TILE_RE = re.compile(r"^(top|bottom)_(\d+)-(.+)\.jpg$")
LANCZOS = Image.Resampling.LANCZOS


@dataclass(frozen=True)
class SourceTile:
    kind: str
    rank: int
    attention_label: str
    attention_score: float
    path: Path


@dataclass(frozen=True)
class SlideSpec:
    experiment: str
    split: str
    stem: str
    sample_id: str
    filename: str
    clinical: dict[str, str]
    prediction: dict[str, str]
    tile_dir: Path
    top_tiles: list[SourceTile]
    bottom_tiles: list[SourceTile]

    @property
    def slide_key(self) -> str:
        return f"{self.experiment}|{self.split}|{self.stem}"


@dataclass(frozen=True)
class CandidateSummary:
    contours: list[np.ndarray]
    count: int
    area_fraction: float
    mask_fraction: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run conservative blast-enrichment triage on STAMP heatmap tiles."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--experiments",
        default="blast_percent,high_blast",
        help="Comma-separated subset of: blast_percent,high_blast",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bottom-k", type=int, default=8)
    parser.add_argument("--zoom-px", type=int, default=512)
    parser.add_argument("--overlay-slides", type=int, default=30)
    parser.add_argument("--limit-slides", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--write-all-overlays", action="store_true")
    return parser.parse_args()


def parse_experiments(raw: str) -> list[str]:
    experiments = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(experiments) - set(EXPERIMENTS))
    if unknown:
        raise SystemExit(f"unknown experiment(s): {', '.join(unknown)}")
    ordered = [key for key in EXPERIMENTS if key in experiments]
    if not ordered:
        raise SystemExit("no experiments selected")
    return ordered


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def load_slide_table() -> dict[str, dict[str, str]]:
    by_stem: dict[str, dict[str, str]] = {}
    for row in read_csv_rows(SLIDE_TABLE):
        stem = Path(row["FILENAME"]).stem
        by_stem[stem] = row
    return by_stem


def load_clinical_table() -> dict[str, dict[str, str]]:
    return {row["SAMPLE_ID"]: row for row in read_csv_rows(CLINI_TABLE)}


def load_predictions(experiment: str) -> dict[tuple[str, str], dict[str, str]]:
    base_dir = EXPERIMENTS[experiment]["base_dir"]
    predictions: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted((base_dir / "crossval").glob("split-*/patient-preds.csv")):
        split = path.parent.name
        for row in read_csv_rows(path):
            predictions[(split, row["SAMPLE_ID"])] = row
    return predictions


def parse_tile(path: Path) -> SourceTile | None:
    match = TILE_RE.match(path.name)
    if not match:
        return None
    kind, rank_text, rest = match.groups()
    try:
        before_score, score_text = rest.rsplit("=", 1)
        _, label = before_score.rsplit("-", 1)
        score = float(score_text)
    except ValueError:
        return None
    return SourceTile(
        kind=kind,
        rank=int(rank_text),
        attention_label=label,
        attention_score=score,
        path=path,
    )


def collect_ranked_tiles(
    tile_dir: Path, kind: str, count: int
) -> tuple[list[SourceTile], list[int]]:
    by_rank: dict[int, SourceTile] = {}
    for path in tile_dir.glob(f"{kind}_*.jpg"):
        tile = parse_tile(path)
        if tile is None or tile.kind != kind:
            continue
        by_rank[tile.rank] = tile
    missing = [rank for rank in range(1, count + 1) if rank not in by_rank]
    return [by_rank[rank] for rank in range(1, count + 1) if rank in by_rank], missing


def discover_slides_for_experiment(
    *,
    experiment: str,
    top_k: int,
    bottom_k: int,
    limit_slides: int | None,
    stem_to_slide: dict[str, dict[str, str]],
    clinical_by_sample: dict[str, dict[str, str]],
    predictions: dict[tuple[str, str], dict[str, str]],
) -> tuple[list[SlideSpec], list[dict[str, Any]]]:
    base_dir = EXPERIMENTS[experiment]["base_dir"]
    heatmap_root = base_dir / "heatmaps"
    slides: list[SlideSpec] = []
    skipped: list[dict[str, Any]] = []

    tile_dirs = sorted(
        heatmap_root.glob("split-*/*/tiles"),
        key=lambda path: (path.parent.parent.name, path.parent.name),
    )
    for tile_dir in tile_dirs:
        split = tile_dir.parent.parent.name
        stem = tile_dir.parent.name
        slide_row = stem_to_slide.get(stem)
        if slide_row is None:
            skipped.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "stem": stem,
                    "reason": "missing slide-table mapping",
                }
            )
            continue

        top_tiles, missing_top = collect_ranked_tiles(tile_dir, "top", top_k)
        bottom_tiles, missing_bottom = collect_ranked_tiles(tile_dir, "bottom", bottom_k)
        if missing_top or missing_bottom:
            skipped.append(
                {
                    "experiment": experiment,
                    "split": split,
                    "stem": stem,
                    "sample_id": slide_row["SAMPLE_ID"],
                    "reason": "incomplete heatmap tile ranks",
                    "missing_top": missing_top,
                    "missing_bottom": missing_bottom,
                }
            )
            continue

        sample_id = slide_row["SAMPLE_ID"]
        slides.append(
            SlideSpec(
                experiment=experiment,
                split=split,
                stem=stem,
                sample_id=sample_id,
                filename=slide_row["FILENAME"],
                clinical=clinical_by_sample.get(sample_id, {}),
                prediction=predictions.get((split, sample_id), {}),
                tile_dir=tile_dir,
                top_tiles=top_tiles,
                bottom_tiles=bottom_tiles,
            )
        )

    if limit_slides is not None:
        if limit_slides < 1:
            raise SystemExit("--limit-slides must be at least 1")
        slides = slides[:limit_slides]
    return slides, skipped


def read_center_crop(path: Path, zoom_px: int) -> tuple[np.ndarray, tuple[int, int]]:
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        width, height = rgb.size
        if width < zoom_px or height < zoom_px:
            raise ValueError(f"{path} is {width}x{height}, smaller than {zoom_px}px")
        left = (width - zoom_px) // 2
        top = (height - zoom_px) // 2
        crop = rgb.crop((left, top, left + zoom_px, top + zoom_px))
        return np.asarray(crop), (width, height)


def quality_metrics(rgb: np.ndarray) -> dict[str, float]:
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
        "zoom_brightness_mean": float(gray.mean()),
        "zoom_brightness_std": float(gray.std()),
        "zoom_saturation_mean": float(saturation.mean()),
        "zoom_laplacian_var": laplacian_var,
        "zoom_tenengrad": tenengrad,
        "zoom_edge_density": float((edges > 0).mean()),
        "zoom_background_fraction": float(background.mean()),
        "zoom_tissue_fraction": float(tissue.mean()),
    }


def segment_cell_like_candidates(rgb: np.ndarray) -> CandidateSummary:
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


def positive_attention_score(experiment: str, tile: SourceTile) -> float:
    score = min(max(tile.attention_score, 0.0), 1.0)
    if experiment == "blast_percent":
        return score
    if experiment == "high_blast":
        if tile.attention_label == "yes":
            return score
        if tile.attention_label == "no":
            return 1.0 - score
    return score


def float_or_blank(value: Any) -> float | str:
    if value in ("", None):
        return ""
    try:
        return float(value)
    except (TypeError, ValueError):
        return ""


def clinical_float(clinical: dict[str, str], key: str) -> float | str:
    return float_or_blank(clinical.get(key, ""))


def analyze_tile(
    *,
    slide: SlideSpec,
    tile: SourceTile,
    zoom_px: int,
) -> dict[str, Any]:
    zoom_rgb, (native_width, native_height) = read_center_crop(tile.path, zoom_px)
    metrics = quality_metrics(zoom_rgb)
    candidates = segment_cell_like_candidates(zoom_rgb)
    blast_percent_gt = clinical_float(slide.clinical, "BLAST_PERCENT")

    row: dict[str, Any] = {
        "experiment": slide.experiment,
        "split": slide.split,
        "slide_key": slide.slide_key,
        "sample_id": slide.sample_id,
        "stem": slide.stem,
        "filename": slide.filename,
        "tile_kind": tile.kind,
        "tile_rank": tile.rank,
        "stamp_attention_label": tile.attention_label,
        "stamp_attention_score": tile.attention_score,
        "positive_attention_score": positive_attention_score(slide.experiment, tile),
        "native_width": native_width,
        "native_height": native_height,
        "zoom_px": zoom_px,
        "tile_path": str(tile.path),
        "blast_percent_gt": blast_percent_gt,
        "blast_severity_gt": slide.clinical.get("BLAST_SEVERITY", ""),
        "high_blast_gt": slide.clinical.get("HIGH_BLAST", ""),
        "blast_percent_pred": float_or_blank(slide.prediction.get("pred", ""))
        if slide.experiment == "blast_percent"
        else "",
        "blast_percent_abs_err": float_or_blank(slide.prediction.get("loss", ""))
        if slide.experiment == "blast_percent"
        else "",
        "high_blast_pred_label": slide.prediction.get("pred", "")
        if slide.experiment == "high_blast"
        else "",
        "high_blast_yes_pred": float_or_blank(
            slide.prediction.get("HIGH_BLAST_yes", "")
        )
        if slide.experiment == "high_blast"
        else "",
        "high_blast_no_pred": float_or_blank(slide.prediction.get("HIGH_BLAST_no", ""))
        if slide.experiment == "high_blast"
        else "",
        "candidate_count": candidates.count,
        "candidate_area_fraction": candidates.area_fraction,
        "candidate_mask_fraction": candidates.mask_fraction,
        "candidate_count_per_megapixel": candidates.count
        / max((zoom_px * zoom_px) / 1_000_000, 1e-8),
    }
    row.update(metrics)
    return row


def mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.median(np.asarray(values, dtype=float)))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=float), q))


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return numerator / denominator


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def normalize_tile_rows(tile_rows: list[dict[str, Any]]) -> None:
    by_slide: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tile_rows:
        by_slide[row["slide_key"]].append(row)

    for rows in by_slide.values():
        scores = [as_float(row["positive_attention_score"]) for row in rows]
        min_score = min(scores) if scores else 0.0
        max_score = max(scores) if scores else 0.0
        score_range = max_score - min_score
        max_log_candidate = max(
            [math.log1p(as_float(row["candidate_count"])) for row in rows] or [0.0]
        )
        max_focus = max([as_float(row["zoom_laplacian_var"]) for row in rows] or [0.0])

        for row in rows:
            positive_score = as_float(row["positive_attention_score"])
            if score_range > 1e-8:
                positive_norm = (positive_score - min_score) / score_range
            else:
                positive_norm = 0.0

            candidate_norm = (
                math.log1p(as_float(row["candidate_count"])) / max_log_candidate
                if max_log_candidate > 0
                else 0.0
            )
            focus_norm = (
                as_float(row["zoom_laplacian_var"]) / max_focus if max_focus > 0 else 0.0
            )
            tissue_fraction = as_float(row["zoom_tissue_fraction"])
            cellular_visibility = candidate_norm * math.sqrt(focus_norm) * tissue_fraction

            row["positive_attention_score_slide_norm"] = positive_norm
            row["cellular_visibility_proxy"] = cellular_visibility
            row["blast_enrichment_proxy"] = positive_norm * cellular_visibility
            row["attention_visibility_proxy"] = (
                as_float(row["stamp_attention_score"]) * cellular_visibility
            )


def rank_blast_percent_failures(tile_rows: list[dict[str, Any]]) -> set[str]:
    by_sample: dict[str, float] = {}
    for row in tile_rows:
        if row["experiment"] != "blast_percent":
            continue
        err = row.get("blast_percent_abs_err", "")
        if err == "":
            continue
        by_sample[row["sample_id"]] = max(by_sample.get(row["sample_id"], 0.0), float(err))
    ranked = sorted(by_sample.items(), key=lambda item: item[1], reverse=True)
    return {sample_id for sample_id, _ in ranked[:10]}


def summarize_slides(
    tile_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    normalize_tile_rows(tile_rows)
    failure_top10 = rank_blast_percent_failures(tile_rows)

    by_slide: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tile_rows:
        by_slide[row["slide_key"]].append(row)

    prelim: list[dict[str, Any]] = []
    for slide_key, rows in by_slide.items():
        first = rows[0]
        top_rows = [row for row in rows if row["tile_kind"] == "top"]
        bottom_rows = [row for row in rows if row["tile_kind"] == "bottom"]

        top_enrichment = mean([as_float(row["blast_enrichment_proxy"]) for row in top_rows])
        bottom_enrichment = mean(
            [as_float(row["blast_enrichment_proxy"]) for row in bottom_rows]
        )
        top_visibility = mean(
            [as_float(row["cellular_visibility_proxy"]) for row in top_rows]
        )
        bottom_visibility = mean(
            [as_float(row["cellular_visibility_proxy"]) for row in bottom_rows]
        )
        top_candidates = mean([as_float(row["candidate_count"]) for row in top_rows])
        bottom_candidates = mean(
            [as_float(row["candidate_count"]) for row in bottom_rows]
        )
        focus_mean = mean([as_float(row["zoom_laplacian_var"]) for row in rows])
        background_mean = mean(
            [as_float(row["zoom_background_fraction"]) for row in rows]
        )
        tissue_mean = mean([as_float(row["zoom_tissue_fraction"]) for row in rows])

        blast_percent_gt = first["blast_percent_gt"]
        high_blast_gt = first["high_blast_gt"]
        blast_percent_pred = first["blast_percent_pred"]
        blast_percent_abs_err = first["blast_percent_abs_err"]
        high_blast_yes_pred = first["high_blast_yes_pred"]

        clinical_band = "missing"
        if blast_percent_gt != "":
            blast_value = float(blast_percent_gt)
            if blast_value < 5:
                clinical_band = "low_blast_lt5"
            elif blast_value < 20:
                clinical_band = "intermediate_blast_5_19"
            else:
                clinical_band = "high_blast_ge20"

        prelim.append(
            {
                "experiment": first["experiment"],
                "split": first["split"],
                "slide_key": slide_key,
                "sample_id": first["sample_id"],
                "stem": first["stem"],
                "filename": first["filename"],
                "clinical_blast_band": clinical_band,
                "blast_percent_gt": blast_percent_gt,
                "blast_severity_gt": first["blast_severity_gt"],
                "high_blast_gt": high_blast_gt,
                "blast_percent_pred": blast_percent_pred,
                "blast_percent_abs_err": blast_percent_abs_err,
                "high_blast_pred_label": first["high_blast_pred_label"],
                "high_blast_yes_pred": high_blast_yes_pred,
                "high_blast_no_pred": first["high_blast_no_pred"],
                "is_blast_percent_top10_failure": int(
                    first["experiment"] == "blast_percent"
                    and first["sample_id"] in failure_top10
                ),
                "is_high_blast_confident_yes": int(
                    first["experiment"] == "high_blast"
                    and high_blast_gt == "yes"
                    and high_blast_yes_pred != ""
                    and float(high_blast_yes_pred) >= 0.9
                ),
                "is_high_blast_confident_no": int(
                    first["experiment"] == "high_blast"
                    and high_blast_gt == "no"
                    and high_blast_yes_pred != ""
                    and float(high_blast_yes_pred) <= 0.1
                ),
                "tile_count": len(rows),
                "top_enrichment_proxy_mean": top_enrichment,
                "bottom_enrichment_proxy_mean": bottom_enrichment,
                "top_minus_bottom_enrichment_proxy": top_enrichment
                - bottom_enrichment,
                "top_bottom_enrichment_ratio": safe_ratio(
                    top_enrichment, bottom_enrichment
                ),
                "top_cellular_visibility_mean": top_visibility,
                "bottom_cellular_visibility_mean": bottom_visibility,
                "top_candidate_count_mean": top_candidates,
                "bottom_candidate_count_mean": bottom_candidates,
                "all_candidate_count_mean": mean(
                    [as_float(row["candidate_count"]) for row in rows]
                ),
                "top_bottom_candidate_ratio": safe_ratio(
                    top_candidates, bottom_candidates
                ),
                "zoom_laplacian_var_mean": focus_mean,
                "zoom_laplacian_var_median": median(
                    [as_float(row["zoom_laplacian_var"]) for row in rows]
                ),
                "zoom_background_fraction_mean": background_mean,
                "zoom_tissue_fraction_mean": tissue_mean,
                "overlay_contact_sheet": "",
            }
        )

    thresholds = {
        "low_focus_laplacian_p25": percentile(
            [row["zoom_laplacian_var_mean"] for row in prelim], 25
        ),
        "low_candidate_count_p25": percentile(
            [row["all_candidate_count_mean"] for row in prelim], 25
        ),
        "high_background_fraction_p75": percentile(
            [row["zoom_background_fraction_mean"] for row in prelim], 75
        ),
    }

    slide_rows: list[dict[str, Any]] = []
    for row in prelim:
        low_focus = (
            row["zoom_laplacian_var_mean"] <= thresholds["low_focus_laplacian_p25"]
        )
        low_candidates = (
            row["all_candidate_count_mean"] <= thresholds["low_candidate_count_p25"]
        )
        high_background = (
            row["zoom_background_fraction_mean"]
            >= thresholds["high_background_fraction_p75"]
        )
        ratio = row["top_bottom_candidate_ratio"]
        candidate_shift = ratio < 0.5 or ratio > 2.0
        concern_score = (
            int(low_focus)
            + int(low_candidates)
            + int(high_background)
            + int(candidate_shift)
        )
        slide_rows.append(
            {
                **row,
                "flag_low_focus": int(low_focus),
                "flag_low_candidate_density": int(low_candidates),
                "flag_high_background": int(high_background),
                "flag_top_bottom_candidate_shift": int(candidate_shift),
                "technical_concern_score": concern_score,
            }
        )

    return slide_rows, thresholds


def summarize_group_rows(slide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    group_defs: list[tuple[str, str, Any]] = [
        (
            "clinical_blast_band",
            "low_blast_lt5",
            lambda row: row["clinical_blast_band"] == "low_blast_lt5",
        ),
        (
            "clinical_blast_band",
            "intermediate_blast_5_19",
            lambda row: row["clinical_blast_band"] == "intermediate_blast_5_19",
        ),
        (
            "clinical_blast_band",
            "high_blast_ge20",
            lambda row: row["clinical_blast_band"] == "high_blast_ge20",
        ),
        (
            "model_error_control",
            "blast_percent_top10_failures",
            lambda row: row["is_blast_percent_top10_failure"] == 1,
        ),
        (
            "confidence_control",
            "high_blast_confident_yes",
            lambda row: row["is_high_blast_confident_yes"] == 1,
        ),
        (
            "confidence_control",
            "high_blast_confident_no",
            lambda row: row["is_high_blast_confident_no"] == 1,
        ),
    ]

    rows_out: list[dict[str, Any]] = []
    for experiment in sorted({row["experiment"] for row in slide_rows}):
        exp_rows = [row for row in slide_rows if row["experiment"] == experiment]
        for group_type, group_name, predicate in group_defs:
            rows = [row for row in exp_rows if predicate(row)]
            if not rows:
                continue
            rows_out.append(
                {
                    "experiment": experiment,
                    "group_type": group_type,
                    "group_name": group_name,
                    "slide_count": len(rows),
                    "unique_sample_count": len({row["sample_id"] for row in rows}),
                    "tile_count": sum(int(row["tile_count"]) for row in rows),
                    "blast_percent_gt_mean": mean(
                        [
                            float(row["blast_percent_gt"])
                            for row in rows
                            if row["blast_percent_gt"] != ""
                        ]
                    ),
                    "top_enrichment_proxy_mean": mean(
                        [float(row["top_enrichment_proxy_mean"]) for row in rows]
                    ),
                    "bottom_enrichment_proxy_mean": mean(
                        [float(row["bottom_enrichment_proxy_mean"]) for row in rows]
                    ),
                    "top_minus_bottom_enrichment_proxy_mean": mean(
                        [
                            float(row["top_minus_bottom_enrichment_proxy"])
                            for row in rows
                        ]
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
    return rows_out


def metric_or_blank(fn: Any, y_true: list[int], y_score: list[float]) -> float | str:
    try:
        if len(set(y_true)) < 2:
            return ""
        return float(fn(y_true, y_score))
    except ValueError:
        return ""


def summarize_experiments(slide_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for experiment in sorted({row["experiment"] for row in slide_rows}):
        rows = [row for row in slide_rows if row["experiment"] == experiment]
        labeled = [row for row in rows if row["blast_percent_gt"] != ""]
        high_blast_rows = [
            row for row in rows if row["high_blast_gt"] in {"yes", "no"}
        ]

        spearman_top = ""
        spearman_delta = ""
        spearman_top_p = ""
        spearman_delta_p = ""
        if len(labeled) >= 3:
            y = [float(row["blast_percent_gt"]) for row in labeled]
            top_scores = [float(row["top_enrichment_proxy_mean"]) for row in labeled]
            delta_scores = [
                float(row["top_minus_bottom_enrichment_proxy"]) for row in labeled
            ]
            top_result = spearmanr(y, top_scores)
            delta_result = spearmanr(y, delta_scores)
            spearman_top = float(top_result.statistic)
            spearman_top_p = float(top_result.pvalue)
            spearman_delta = float(delta_result.statistic)
            spearman_delta_p = float(delta_result.pvalue)

        y_true = [1 if row["high_blast_gt"] == "yes" else 0 for row in high_blast_rows]
        y_top = [float(row["top_enrichment_proxy_mean"]) for row in high_blast_rows]
        y_delta = [
            float(row["top_minus_bottom_enrichment_proxy"]) for row in high_blast_rows
        ]
        failures = [row for row in rows if row["is_blast_percent_top10_failure"] == 1]

        summaries.append(
            {
                "experiment": experiment,
                "display_name": EXPERIMENTS[experiment]["display_name"],
                "slide_count": len(rows),
                "unique_sample_count": len({row["sample_id"] for row in rows}),
                "tile_count": sum(int(row["tile_count"]) for row in rows),
                "spearman_blast_percent_vs_top_enrichment": spearman_top,
                "spearman_blast_percent_vs_top_enrichment_p": spearman_top_p,
                "spearman_blast_percent_vs_delta_enrichment": spearman_delta,
                "spearman_blast_percent_vs_delta_enrichment_p": spearman_delta_p,
                "high_blast_auroc_top_enrichment": metric_or_blank(
                    roc_auc_score, y_true, y_top
                ),
                "high_blast_auprc_top_enrichment": metric_or_blank(
                    average_precision_score, y_true, y_top
                ),
                "high_blast_auroc_delta_enrichment": metric_or_blank(
                    roc_auc_score, y_true, y_delta
                ),
                "high_blast_auprc_delta_enrichment": metric_or_blank(
                    average_precision_score, y_true, y_delta
                ),
                "top_enrichment_proxy_mean": mean(
                    [float(row["top_enrichment_proxy_mean"]) for row in rows]
                ),
                "bottom_enrichment_proxy_mean": mean(
                    [float(row["bottom_enrichment_proxy_mean"]) for row in rows]
                ),
                "technical_concern_score_mean": mean(
                    [float(row["technical_concern_score"]) for row in rows]
                ),
                "blast_percent_top10_failure_count": len(failures),
                "blast_percent_top10_failure_top_enrichment_mean": mean(
                    [float(row["top_enrichment_proxy_mean"]) for row in failures]
                ),
                "blast_percent_top10_failure_delta_mean": mean(
                    [float(row["top_minus_bottom_enrichment_proxy"]) for row in failures]
                ),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_name(value: str, max_len: int = 110) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return cleaned[:max_len] or "unnamed"


def rel(path: Path, base: Path) -> str:
    try:
        return path.relative_to(base).as_posix()
    except ValueError:
        return path.as_posix()


def save_overlay(
    *,
    rgb: np.ndarray,
    candidates: CandidateSummary,
    row: dict[str, Any],
    out_path: Path,
) -> None:
    overlay = rgb.copy()
    cv2.drawContours(overlay, candidates.contours, -1, (0, 220, 0), 2)
    cv2.rectangle(overlay, (0, 0), (overlay.shape[1], 34), (0, 0, 0), -1)
    text = (
        f"{row['tile_kind']}_{int(row['tile_rank']):02d} | "
        f"candidates {candidates.count} | proxy "
        f"{as_float(row.get('blast_enrichment_proxy')):.3f} | not blasts"
    )
    cv2.putText(
        overlay,
        text,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path, quality=92)


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
    slide: dict[str, Any],
    overlays: list[Path],
    out_path: Path,
) -> None:
    margin = 28
    gap = 10
    tile_size = 168
    cols = 4
    rows = math.ceil(len(overlays) / cols)
    width = margin * 2 + cols * tile_size + (cols - 1) * gap
    height = margin * 2 + 84 + rows * (tile_size + 24)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = (
        f"{slide['experiment']} | {slide['sample_id']} | "
        "non-diagnostic blast-enrichment triage"
    )
    draw.text((margin, margin), title, fill=(0, 0, 0), font=font)
    draw.text(
        (margin, margin + 22),
        "Green contours are cell-like candidates only, not validated cells or blasts.",
        fill=(120, 50, 20),
        font=font,
    )
    draw.text(
        (margin, margin + 44),
        (
            f"GT blast %: {slide['blast_percent_gt']} | "
            f"top proxy: {as_float(slide['top_enrichment_proxy_mean']):.3f} | "
            f"delta: {as_float(slide['top_minus_bottom_enrichment_proxy']):.3f}"
        ),
        fill=(60, 60, 60),
        font=font,
    )

    start_y = margin + 74
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


def select_overlay_slides(
    slide_rows: list[dict[str, Any]], overlay_slides: int, write_all: bool
) -> list[dict[str, Any]]:
    if write_all:
        return slide_rows
    if overlay_slides <= 0:
        return []

    selected: dict[str, dict[str, Any]] = {}

    def add(rows: list[dict[str, Any]], limit: int) -> None:
        for row in rows:
            if len(selected) >= overlay_slides:
                return
            if len(selected) >= limit and limit < overlay_slides:
                return
            selected.setdefault(row["slide_key"], row)

    concern_ranked = sorted(
        slide_rows,
        key=lambda row: (
            int(row["technical_concern_score"]),
            float(row["zoom_background_fraction_mean"]),
            -float(row["zoom_laplacian_var_mean"]),
        ),
        reverse=True,
    )
    failure_ranked = sorted(
        [row for row in slide_rows if row["is_blast_percent_top10_failure"] == 1],
        key=lambda row: as_float(row["blast_percent_abs_err"]),
        reverse=True,
    )
    confident_yes = sorted(
        [row for row in slide_rows if row["is_high_blast_confident_yes"] == 1],
        key=lambda row: as_float(row["high_blast_yes_pred"]),
        reverse=True,
    )
    confident_no = sorted(
        [row for row in slide_rows if row["is_high_blast_confident_no"] == 1],
        key=lambda row: as_float(row["high_blast_yes_pred"]),
    )
    high_delta = sorted(
        slide_rows,
        key=lambda row: abs(float(row["top_minus_bottom_enrichment_proxy"])),
        reverse=True,
    )

    quotas = [
        (concern_ranked, min(10, overlay_slides)),
        (failure_ranked, min(20, overlay_slides)),
        (confident_yes, min(25, overlay_slides)),
        (confident_no, min(30, overlay_slides)),
        (high_delta, overlay_slides),
    ]
    for rows, cumulative_limit in quotas:
        add(rows, cumulative_limit)
        if len(selected) >= overlay_slides:
            break

    return list(selected.values())


def write_selected_overlays(
    *,
    out_dir: Path,
    slide_rows: list[dict[str, Any]],
    tile_rows: list[dict[str, Any]],
    zoom_px: int,
    overlay_slides: int,
    write_all: bool,
) -> list[dict[str, Any]]:
    selected_slides = select_overlay_slides(slide_rows, overlay_slides, write_all)
    if not selected_slides:
        return []

    overlay_root = out_dir / "overlays"
    if overlay_root.exists():
        shutil.rmtree(overlay_root)

    rows_by_slide: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tile_rows:
        rows_by_slide[row["slide_key"]].append(row)

    selected_keys = {row["slide_key"] for row in selected_slides}
    selected_slide_by_key = {row["slide_key"]: row for row in selected_slides}
    overlay_records: list[dict[str, Any]] = []

    for slide_key in selected_keys:
        slide = selected_slide_by_key[slide_key]
        overlay_dir = (
            out_dir
            / "overlays"
            / slide["experiment"]
            / f"{safe_name(slide['sample_id'])}__{safe_name(slide['stem'], 70)}"
        )
        overlay_paths: list[Path] = []
        rows = sorted(
            rows_by_slide[slide_key],
            key=lambda row: (row["tile_kind"] != "top", int(row["tile_rank"])),
        )
        for row in rows:
            zoom_rgb, _ = read_center_crop(Path(row["tile_path"]), zoom_px)
            candidates = segment_cell_like_candidates(zoom_rgb)
            overlay_path = (
                overlay_dir
                / "tiles"
                / f"{row['tile_kind']}_{int(row['tile_rank']):02d}_overlay.jpg"
            )
            save_overlay(
                rgb=zoom_rgb,
                candidates=candidates,
                row=row,
                out_path=overlay_path,
            )
            overlay_paths.append(overlay_path)

        contact_path = overlay_dir / "overlay_contact_sheet.jpg"
        make_overlay_contact_sheet(
            slide=slide,
            overlays=overlay_paths,
            out_path=contact_path,
        )
        slide["overlay_contact_sheet"] = str(contact_path)
        overlay_records.append(
            {
                "slide_key": slide_key,
                "experiment": slide["experiment"],
                "sample_id": slide["sample_id"],
                "contact_sheet": str(contact_path),
                "tile_overlay_count": len(overlay_paths),
            }
        )

    return overlay_records


def format_number(value: Any, digits: int = 3) -> str:
    if value == "":
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return ""
    return f"{numeric:.{digits}f}"


def format_p_value(value: Any) -> str:
    if value == "":
        return ""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(numeric):
        return ""
    if numeric < 0.001:
        return f"{numeric:.2e}"
    return f"{numeric:.3f}"


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
                value = format_number(value)
            parts.append(f"<td>{html.escape(str(value))}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def html_list(items: list[str]) -> str:
    return "<ul>" + "".join(f"<li>{html.escape(item)}</li>" for item in items) + "</ul>"


def markdown_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def row_by_key(rows: list[dict[str, Any]], key: str, value: str) -> dict[str, Any]:
    for row in rows:
        if row.get(key) == value:
            return row
    return {}


def group_row(
    rows: list[dict[str, Any]], experiment: str, group_name: str
) -> dict[str, Any]:
    for row in rows:
        if row.get("experiment") == experiment and row.get("group_name") == group_name:
            return row
    return {}


def mean_from_rows(rows: list[dict[str, Any]], key: str) -> float:
    values = [as_float(row.get(key, "")) for row in rows if row.get(key, "") != ""]
    return mean(values)


def build_report_summary(
    *,
    tile_rows: list[dict[str, Any]],
    slide_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    overlay_records: list[dict[str, Any]],
) -> dict[str, Any]:
    blast_percent = row_by_key(experiment_rows, "experiment", "blast_percent")
    high_blast = row_by_key(experiment_rows, "experiment", "high_blast")
    high_blast_high = group_row(group_rows, "high_blast", "high_blast_ge20")
    high_blast_low = group_row(group_rows, "high_blast", "low_blast_lt5")
    high_blast_conf_yes = group_row(group_rows, "high_blast", "high_blast_confident_yes")
    high_blast_conf_no = group_row(group_rows, "high_blast", "high_blast_confident_no")
    blast_percent_high = group_row(group_rows, "blast_percent", "high_blast_ge20")
    blast_percent_low = group_row(group_rows, "blast_percent", "low_blast_lt5")
    blast_percent_failures = group_row(
        group_rows, "blast_percent", "blast_percent_top10_failures"
    )

    top_failures = sorted(
        [row for row in slide_rows if row["is_blast_percent_top10_failure"] == 1],
        key=lambda row: as_float(row["blast_percent_abs_err"]),
        reverse=True,
    )

    slide_count = len(slide_rows)
    tile_count = len(tile_rows)
    unique_samples = len({row["sample_id"] for row in slide_rows})
    overlay_count = len(overlay_records)
    failure_gt_mean = as_float(blast_percent_failures.get("blast_percent_gt_mean", ""))
    failure_pred_mean = mean_from_rows(top_failures, "blast_percent_pred")
    failure_err_mean = mean_from_rows(top_failures, "blast_percent_abs_err")

    methods = [
        (
            f"Analyzed {slide_count} experiment-slide records from {unique_samples} "
            f"unique sample IDs using {tile_count} existing heatmap-selected tile JPGs."
        ),
        (
            "For each complete slide/experiment pair, the script used the existing "
            "top_01..top_08 and bottom_01..bottom_08 heatmap tiles, without rereading "
            "raw WSIs or rerunning preprocessing."
        ),
        (
            "Each tile was center-cropped to 512 x 512 px and scored for focus, "
            "background, tissue fraction, edge content, and conservative cell-like "
            "candidate visibility."
        ),
        (
            "The blast-enrichment proxy combines positive-class STAMP attention with "
            "cellular visibility. It is a triage signal only, not a blast detector."
        ),
    ]

    key_findings = [
        (
            f"Data coverage was strong: {slide_count} slide records and {tile_count} "
            f"tile records were analyzed; {len(skipped)} incomplete heatmap folders "
            "were skipped."
        ),
        (
            "The HIGH_BLAST heatmap signal aligned with clinical blast burden: "
            f"Spearman rho for clinical BLAST_PERCENT vs top enrichment was "
            f"{format_number(high_blast.get('spearman_blast_percent_vs_top_enrichment'))} "
            f"(p={format_p_value(high_blast.get('spearman_blast_percent_vs_top_enrichment_p'))}), "
            f"and HIGH_BLAST AUROC using top enrichment was "
            f"{format_number(high_blast.get('high_blast_auroc_top_enrichment'))}."
        ),
        (
            "The BLAST_PERCENT regression heatmap signal did not align with clinical "
            f"blast burden: Spearman rho was "
            f"{format_number(blast_percent.get('spearman_blast_percent_vs_top_enrichment'))}, "
            f"and HIGH_BLAST AUROC from that proxy was "
            f"{format_number(blast_percent.get('high_blast_auroc_top_enrichment'))}."
        ),
        (
            "In HIGH_BLAST, high-blast slides had higher top enrichment than low-blast "
            f"slides ({format_number(high_blast_high.get('top_enrichment_proxy_mean'))} "
            f"vs {format_number(high_blast_low.get('top_enrichment_proxy_mean'))}); "
            "confident HIGH_BLAST positives were also much higher than confident "
            f"negatives ({format_number(high_blast_conf_yes.get('top_enrichment_proxy_mean'))} "
            f"vs {format_number(high_blast_conf_no.get('top_enrichment_proxy_mean'))})."
        ),
        (
            "In BLAST_PERCENT, high-blast slides were not enriched above low-blast "
            f"slides by this proxy ({format_number(blast_percent_high.get('top_enrichment_proxy_mean'))} "
            f"vs {format_number(blast_percent_low.get('top_enrichment_proxy_mean'))}), "
            "which points away from using the regression heatmaps as the main interim "
            "visual triage signal."
        ),
        (
            "The top 10 BLAST_PERCENT failure slides had very high clinical blast "
            f"burden (mean GT {format_number(failure_gt_mean)}%) but low regression "
            f"predictions (mean {format_number(failure_pred_mean)}%, mean absolute "
            f"error {format_number(failure_err_mean)}%). Their mean top enrichment "
            f"proxy was {format_number(blast_percent_failures.get('top_enrichment_proxy_mean'))}, "
            "not clearly elevated compared with the overall BLAST_PERCENT run."
        ),
        (
            f"The report includes {overlay_count} selected overlay contact sheets for "
            "technical concerns, BLAST_PERCENT failures, and HIGH_BLAST confidence "
            "controls. These overlays are intended for pathologist review and visual "
            "sanity checking only."
        ),
    ]

    limitations = [
        "The analysis does not classify individual cells and does not identify blasts.",
        "Green contours are conservative cell-like/foreground candidates, not validated cell detections.",
        "Clinical BLAST_PERCENT is slide-level metadata; it is not a cell-level or tile-level annotation.",
        "DeepHeme-style inference is not currently feasible because public DeepHeme assets do not provide a ready pretrained detector/classifier for these STAMP tiles.",
        "The enrichment proxy can be biased by focus, staining, background, RBC-rich areas, and where STAMP attention lands.",
    ]

    next_steps = [
        "Use HIGH_BLAST heatmap tiles, not BLAST_PERCENT regression heatmaps, as the primary interim source for blast-rich candidate regions.",
        "Ask pathologists to review the selected overlay/contact-sheet examples plus the existing 12-slide pilot deck, with emphasis on whether top-attended regions are cell-rich, blast-rich, or technically misleading.",
        "Create a small annotation set from high-blast controls, low-blast controls, and BLAST_PERCENT failure slides. Minimum useful labels: interpretable/uninterpretable region, cell-rich yes/no, blast-rich estimate, and artifact/focus notes.",
        "After pathologist feedback, compare their top-tile blast estimates against the HIGH_BLAST enrichment proxy and clinical BLAST_PERCENT to decide whether to scale annotation.",
        "If annotations confirm visible blast enrichment, train or calibrate a lightweight cell/region classifier on local annotations. Keep DeepHeme as a reference method unless usable pretrained weights or compatible annotations become available.",
        "Keep technical QC gates in later analyses: low focus, high background, low candidate density, and top/bottom candidate shifts should be flagged before interpreting model attention.",
    ]

    email_subject = "Interim STAMP blast-enrichment triage findings"
    email_body = [
        "Dear all,",
        "",
        "While we wait for the pathology review of the initial cell-resolution deck, I ran a conservative interim triage analysis on the existing STAMP heatmap-selected tiles.",
        "",
        (
            f"In total, this covered {slide_count} slide/experiment records "
            f"({tile_count} tiles) from {unique_samples} unique sample IDs. "
            "This is not a diagnostic blast detector: it combines STAMP heatmap "
            "attention with image-quality and cell-like object visibility metrics."
        ),
        "",
        "Main findings:",
        (
            f"- The HIGH_BLAST classifier heatmaps showed a meaningful association "
            f"with clinical blast burden (Spearman rho "
            f"{format_number(high_blast.get('spearman_blast_percent_vs_top_enrichment'))}; "
            f"HIGH_BLAST AUROC {format_number(high_blast.get('high_blast_auroc_top_enrichment'))})."
        ),
        (
            f"- The BLAST_PERCENT regression heatmaps did not show useful alignment "
            f"with clinical blast burden (Spearman rho "
            f"{format_number(blast_percent.get('spearman_blast_percent_vs_top_enrichment'))}; "
            f"AUROC {format_number(blast_percent.get('high_blast_auroc_top_enrichment'))})."
        ),
        (
            "- The worst BLAST_PERCENT failures had very high clinical blast counts "
            f"(mean GT {format_number(failure_gt_mean)}%) but low model predictions "
            f"(mean {format_number(failure_pred_mean)}%), suggesting the regression "
            "model is not reliably surfacing blast-rich visual regions."
        ),
        "",
        "Suggested next step: use the HIGH_BLAST heatmap tiles as the main interim source for candidate blast-rich regions, and ask pathologists to review whether the selected top-attended regions are actually cell-rich/blast-rich or technically misleading. If this is confirmed, we can build a small local annotation set and then train or calibrate a lightweight region/cell classifier.",
        "",
        "Best,",
        "Jeff",
    ]

    return {
        "methods": methods,
        "key_findings": key_findings,
        "limitations": limitations,
        "next_steps": next_steps,
        "email_subject": email_subject,
        "email_body": email_body,
        "top_failures": top_failures,
    }


def write_markdown_reports(out_dir: Path, summary: dict[str, Any]) -> None:
    findings_parts = [
        "# STAMP Blast-Enrichment Triage Findings",
        "",
        "## Methods",
        markdown_list(summary["methods"]),
        "",
        "## Key Findings",
        markdown_list(summary["key_findings"]),
        "",
        "## Limitations",
        markdown_list(summary["limitations"]),
        "",
        "## Recommended Next Evaluations",
        markdown_list(summary["next_steps"]),
        "",
        "## Top BLAST_PERCENT Failure Slides",
    ]
    for row in summary["top_failures"]:
        findings_parts.append(
            "- "
            f"{row['sample_id']}: GT {format_number(row['blast_percent_gt'])}%, "
            f"pred {format_number(row['blast_percent_pred'])}%, "
            f"abs err {format_number(row['blast_percent_abs_err'])}%, "
            f"top proxy {format_number(row['top_enrichment_proxy_mean'])}, "
            f"technical concern {row['technical_concern_score']}"
        )
    findings_parts.append("")
    findings_parts.append(
        "Note: all contour overlays are non-diagnostic cell-like candidates, not blast calls."
    )

    email_parts = [
        f"Subject: {summary['email_subject']}",
        "",
        *summary["email_body"],
    ]
    (out_dir / "findings_summary.md").write_text("\n".join(findings_parts))
    (out_dir / "email_draft.md").write_text("\n".join(email_parts))


def write_html_report(
    *,
    out_dir: Path,
    tile_rows: list[dict[str, Any]],
    slide_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    experiment_rows: list[dict[str, Any]],
    thresholds: dict[str, float],
    skipped: list[dict[str, Any]],
    overlay_records: list[dict[str, Any]],
    report_summary: dict[str, Any],
) -> None:
    ranked = sorted(
        slide_rows,
        key=lambda row: (
            float(row["top_enrichment_proxy_mean"]),
            float(row["top_minus_bottom_enrichment_proxy"]),
        ),
        reverse=True,
    )
    concern_ranked = sorted(
        slide_rows,
        key=lambda row: (
            int(row["technical_concern_score"]),
            float(row["zoom_background_fraction_mean"]),
            -float(row["zoom_laplacian_var_mean"]),
        ),
        reverse=True,
    )

    experiment_fields = [
        "experiment",
        "slide_count",
        "unique_sample_count",
        "tile_count",
        "spearman_blast_percent_vs_top_enrichment",
        "spearman_blast_percent_vs_delta_enrichment",
        "high_blast_auroc_top_enrichment",
        "high_blast_auprc_top_enrichment",
        "high_blast_auroc_delta_enrichment",
        "high_blast_auprc_delta_enrichment",
        "technical_concern_score_mean",
    ]
    group_fields = [
        "experiment",
        "group_type",
        "group_name",
        "slide_count",
        "unique_sample_count",
        "top_enrichment_proxy_mean",
        "bottom_enrichment_proxy_mean",
        "top_minus_bottom_enrichment_proxy_mean",
        "top_candidate_count_mean",
        "technical_concern_score_mean",
        "slides_with_any_concern",
    ]
    slide_fields = [
        "experiment",
        "sample_id",
        "clinical_blast_band",
        "blast_percent_gt",
        "blast_percent_pred",
        "blast_percent_abs_err",
        "high_blast_yes_pred",
        "top_enrichment_proxy_mean",
        "bottom_enrichment_proxy_mean",
        "top_minus_bottom_enrichment_proxy",
        "top_candidate_count_mean",
        "bottom_candidate_count_mean",
        "technical_concern_score",
    ]
    top_failure_fields = [
        "sample_id",
        "blast_percent_gt",
        "blast_percent_pred",
        "blast_percent_abs_err",
        "top_enrichment_proxy_mean",
        "technical_concern_score",
    ]

    overlay_by_key = {
        record["slide_key"]: record["contact_sheet"] for record in overlay_records
    }
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>STAMP Blast-Enrichment Triage</title>",
        """
<style>
body { font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }
.warning { background: #fff3cd; border: 1px solid #e0b95c; padding: 12px; }
.readiness { background: #eef6ff; border: 1px solid #9cc4e4; padding: 12px; }
table { border-collapse: collapse; margin: 12px 0 24px; width: 100%; }
th, td { border: 1px solid #d9e2ec; padding: 5px 7px; font-size: 12px; }
th { background: #f5f7fa; text-align: left; }
.slide { border-top: 2px solid #d9e2ec; padding: 18px 0 28px; }
.contact { max-width: 980px; width: 100%; border: 1px solid #d9e2ec; }
.small { color: #52606d; }
</style>
""",
        "</head><body>",
        "<h1>STAMP Blast-Enrichment Triage</h1>",
        (
            "<div class='warning'><b>Important:</b> This is non-diagnostic "
            "blast-enrichment triage, not validated blast detection. Green "
            "contours are cell-like candidates only. DeepHeme remains "
            "unavailable for faithful inference without pretrained weights or "
            "cell-level annotations.</div>"
        ),
        "<h2>DeepHeme readiness</h2>",
        (
            "<div class='readiness'><ul>"
            "<li>Raw WSIs: available locally.</li>"
            "<li>UNI2 features and STAMP heatmap tiles: available locally.</li>"
            "<li>Cell-level blast annotations: missing.</li>"
            "<li>DeepHeme pretrained detector/classifier: not available from "
            "the public repo/release checked during planning.</li>"
            "<li>Sources: "
            "<a href='https://github.com/GoldgofLab/DeepHeme'>DeepHeme repo</a>, "
            "<a href='https://github.com/GoldgofLab/DeepHeme/releases/tag/v1.1'>"
            "v1.1 release</a>.</li>"
            "</ul></div>"
        ),
        "<h2>Run Summary</h2>",
        f"<p>Tile records: {len(tile_rows)}. Slide records: {len(slide_rows)}. "
        f"Skipped heatmap slide folders: {len(skipped)}.</p>",
        "<h2>Methods In Brief</h2>",
        html_list(report_summary["methods"]),
        "<h2>Key Findings</h2>",
        html_list(report_summary["key_findings"]),
        "<h2>Top BLAST_PERCENT Failure Slides</h2>",
        html_table(report_summary["top_failures"], top_failure_fields),
        "<h2>Interpretation Limits</h2>",
        html_list(report_summary["limitations"]),
        "<h2>Recommended Next Evaluations</h2>",
        html_list(report_summary["next_steps"]),
        "<h2>Email Draft</h2>",
        f"<p><b>Subject:</b> {html.escape(report_summary['email_subject'])}</p>",
        "<pre>"
        + html.escape("\n".join(report_summary["email_body"]))
        + "</pre>",
        "<h2>QC Thresholds</h2>",
        html_table([thresholds], list(thresholds)),
        "<h2>Experiment Summary</h2>",
        html_table(experiment_rows, experiment_fields),
        "<h2>Group Summary</h2>",
        html_table(group_rows, group_fields),
        "<h2>Highest Enrichment Proxy Slides</h2>",
        html_table(ranked[:60], slide_fields),
        "<h2>Highest Technical Concern Slides</h2>",
        html_table(concern_ranked[:60], slide_fields),
        "<h2>Selected Overlay Contact Sheets</h2>",
    ]

    for row in [r for r in concern_ranked if r["slide_key"] in overlay_by_key]:
        contact = Path(overlay_by_key[row["slide_key"]])
        parts.append("<section class='slide'>")
        parts.append(
            f"<h3>{html.escape(row['experiment'])}: "
            f"{html.escape(row['sample_id'])}</h3>"
        )
        parts.append(
            "<p class='small'>"
            f"GT blast %: {html.escape(str(row['blast_percent_gt']))}; "
            f"top proxy: {format_number(row['top_enrichment_proxy_mean'])}; "
            f"delta: {format_number(row['top_minus_bottom_enrichment_proxy'])}; "
            f"technical concern: {row['technical_concern_score']}"
            "</p>"
        )
        parts.append(
            f"<a href='{html.escape(rel(contact, out_dir))}'>"
            f"<img class='contact' src='{html.escape(rel(contact, out_dir))}' "
            f"alt='overlay contact sheet for {html.escape(row['sample_id'])}'></a>"
        )
        parts.append("</section>")

    tile_fields = [
        "experiment",
        "sample_id",
        "tile_kind",
        "tile_rank",
        "stamp_attention_label",
        "stamp_attention_score",
        "positive_attention_score_slide_norm",
        "candidate_count",
        "cellular_visibility_proxy",
        "blast_enrichment_proxy",
        "zoom_laplacian_var",
        "zoom_background_fraction",
    ]
    parts.append("<h2>Per-Tile Preview</h2>")
    parts.append(
        "<p class='small'>Full table is in <code>tile_metrics.csv</code>; "
        "showing first 100 rows here.</p>"
    )
    parts.append(html_table(tile_rows[:100], tile_fields))
    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts))


def collect_all_specs(
    selected_experiments: list[str], top_k: int, bottom_k: int, limit_slides: int | None
) -> tuple[list[SlideSpec], list[dict[str, Any]], dict[str, Any]]:
    stem_to_slide = load_slide_table()
    clinical_by_sample = load_clinical_table()
    all_specs: list[SlideSpec] = []
    all_skipped: list[dict[str, Any]] = []
    dry_counts: dict[str, Any] = {}

    for experiment in selected_experiments:
        predictions = load_predictions(experiment)
        base_dir = EXPERIMENTS[experiment]["base_dir"]
        discovered_dirs = list((base_dir / "heatmaps").glob("split-*/*/tiles"))
        specs, skipped = discover_slides_for_experiment(
            experiment=experiment,
            top_k=top_k,
            bottom_k=bottom_k,
            limit_slides=limit_slides,
            stem_to_slide=stem_to_slide,
            clinical_by_sample=clinical_by_sample,
            predictions=predictions,
        )
        all_specs.extend(specs)
        all_skipped.extend(skipped)
        dry_counts[experiment] = {
            "heatmap_slide_dirs": len(discovered_dirs),
            "complete_slide_dirs": len(specs)
            if limit_slides is None
            else len(specs) + max(0, len(discovered_dirs) - len(skipped) - len(specs)),
            "selected_slide_dirs": len(specs),
            "skipped_slide_dirs": len(skipped),
            "selected_source_tiles": len(specs) * (top_k + bottom_k),
            "prediction_rows": len(predictions),
        }

    return all_specs, all_skipped, dry_counts


def dry_run_report(dry_counts: dict[str, Any], skipped: list[dict[str, Any]]) -> None:
    for experiment, counts in dry_counts.items():
        print(f"{experiment}:")
        for key, value in counts.items():
            print(f"  {key}: {value}")
    if skipped:
        print("Skipped examples:")
        for row in skipped[:8]:
            print(f"  {row}")


def run_analysis(args: argparse.Namespace) -> None:
    selected_experiments = parse_experiments(args.experiments)
    specs, skipped, dry_counts = collect_all_specs(
        selected_experiments=selected_experiments,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        limit_slides=args.limit_slides,
    )
    if args.dry_run:
        dry_run_report(dry_counts, skipped)
        return

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    tile_rows: list[dict[str, Any]] = []
    for idx, slide in enumerate(specs, start=1):
        for tile in [*slide.top_tiles, *slide.bottom_tiles]:
            tile_rows.append(analyze_tile(slide=slide, tile=tile, zoom_px=args.zoom_px))
        if idx % 50 == 0:
            print(f"Analyzed {idx}/{len(specs)} slide records")

    slide_rows, thresholds = summarize_slides(tile_rows)
    group_rows = summarize_group_rows(slide_rows)
    experiment_rows = summarize_experiments(slide_rows)
    overlay_records = write_selected_overlays(
        out_dir=out_dir,
        slide_rows=slide_rows,
        tile_rows=tile_rows,
        zoom_px=args.zoom_px,
        overlay_slides=args.overlay_slides,
        write_all=args.write_all_overlays,
    )
    report_summary = build_report_summary(
        tile_rows=tile_rows,
        slide_rows=slide_rows,
        group_rows=group_rows,
        experiment_rows=experiment_rows,
        skipped=skipped,
        overlay_records=overlay_records,
    )

    tile_fields = [
        "experiment",
        "split",
        "slide_key",
        "sample_id",
        "stem",
        "filename",
        "tile_kind",
        "tile_rank",
        "stamp_attention_label",
        "stamp_attention_score",
        "positive_attention_score",
        "positive_attention_score_slide_norm",
        "cellular_visibility_proxy",
        "blast_enrichment_proxy",
        "attention_visibility_proxy",
        "native_width",
        "native_height",
        "zoom_px",
        "blast_percent_gt",
        "blast_severity_gt",
        "high_blast_gt",
        "blast_percent_pred",
        "blast_percent_abs_err",
        "high_blast_pred_label",
        "high_blast_yes_pred",
        "high_blast_no_pred",
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
        "tile_path",
    ]
    slide_fields = [
        "experiment",
        "split",
        "slide_key",
        "sample_id",
        "stem",
        "filename",
        "clinical_blast_band",
        "blast_percent_gt",
        "blast_severity_gt",
        "high_blast_gt",
        "blast_percent_pred",
        "blast_percent_abs_err",
        "high_blast_pred_label",
        "high_blast_yes_pred",
        "high_blast_no_pred",
        "is_blast_percent_top10_failure",
        "is_high_blast_confident_yes",
        "is_high_blast_confident_no",
        "tile_count",
        "top_enrichment_proxy_mean",
        "bottom_enrichment_proxy_mean",
        "top_minus_bottom_enrichment_proxy",
        "top_bottom_enrichment_ratio",
        "top_cellular_visibility_mean",
        "bottom_cellular_visibility_mean",
        "top_candidate_count_mean",
        "bottom_candidate_count_mean",
        "all_candidate_count_mean",
        "top_bottom_candidate_ratio",
        "zoom_laplacian_var_mean",
        "zoom_laplacian_var_median",
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
        "experiment",
        "group_type",
        "group_name",
        "slide_count",
        "unique_sample_count",
        "tile_count",
        "blast_percent_gt_mean",
        "top_enrichment_proxy_mean",
        "bottom_enrichment_proxy_mean",
        "top_minus_bottom_enrichment_proxy_mean",
        "top_candidate_count_mean",
        "bottom_candidate_count_mean",
        "zoom_laplacian_var_mean",
        "background_fraction_mean",
        "technical_concern_score_mean",
        "slides_with_any_concern",
    ]
    experiment_fields = [
        "experiment",
        "display_name",
        "slide_count",
        "unique_sample_count",
        "tile_count",
        "spearman_blast_percent_vs_top_enrichment",
        "spearman_blast_percent_vs_top_enrichment_p",
        "spearman_blast_percent_vs_delta_enrichment",
        "spearman_blast_percent_vs_delta_enrichment_p",
        "high_blast_auroc_top_enrichment",
        "high_blast_auprc_top_enrichment",
        "high_blast_auroc_delta_enrichment",
        "high_blast_auprc_delta_enrichment",
        "top_enrichment_proxy_mean",
        "bottom_enrichment_proxy_mean",
        "technical_concern_score_mean",
        "blast_percent_top10_failure_count",
        "blast_percent_top10_failure_top_enrichment_mean",
        "blast_percent_top10_failure_delta_mean",
    ]

    write_csv(out_dir / "tile_metrics.csv", tile_rows, tile_fields)
    write_csv(out_dir / "slide_summary.csv", slide_rows, slide_fields)
    write_csv(out_dir / "group_summary.csv", group_rows, group_fields)
    write_csv(out_dir / "experiment_summary.csv", experiment_rows, experiment_fields)
    write_markdown_reports(out_dir, report_summary)

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "warning": (
            "This is non-diagnostic blast-enrichment triage, not validated blast "
            "detection. Cell-like candidates are not blast calls."
        ),
        "arguments": {
            "experiments": selected_experiments,
            "top_k": args.top_k,
            "bottom_k": args.bottom_k,
            "zoom_px": args.zoom_px,
            "overlay_slides": args.overlay_slides,
            "limit_slides": args.limit_slides,
            "write_all_overlays": args.write_all_overlays,
        },
        "inputs": {
            "slide_table": str(SLIDE_TABLE),
            "clinical_table": str(CLINI_TABLE),
            "experiment_dirs": {
                key: str(EXPERIMENTS[key]["base_dir"]) for key in selected_experiments
            },
        },
        "counts": {
            "slide_records": len(slide_rows),
            "unique_samples": len({row["sample_id"] for row in slide_rows}),
            "tile_records": len(tile_rows),
            "skipped_heatmap_slide_dirs": len(skipped),
            "overlay_contact_sheets": len(overlay_records),
        },
        "dry_run_counts": dry_counts,
        "thresholds": thresholds,
        "outputs": {
            "tile_metrics": str(out_dir / "tile_metrics.csv"),
            "slide_summary": str(out_dir / "slide_summary.csv"),
            "group_summary": str(out_dir / "group_summary.csv"),
            "experiment_summary": str(out_dir / "experiment_summary.csv"),
            "findings_summary": str(out_dir / "findings_summary.md"),
            "email_draft": str(out_dir / "email_draft.md"),
            "html": str(out_dir / "index.html"),
        },
        "summary": {
            "methods": report_summary["methods"],
            "key_findings": report_summary["key_findings"],
            "limitations": report_summary["limitations"],
            "next_steps": report_summary["next_steps"],
            "email_subject": report_summary["email_subject"],
        },
        "skipped": skipped,
        "overlays": overlay_records,
        "deepheme_readiness": {
            "raw_wsis": "available locally",
            "uni2_features": "available locally",
            "stamp_heatmap_tiles": "available locally",
            "cell_level_annotations": "missing",
            "deepheme_pretrained_detector_classifier": "not available from public repo/release checked during planning",
            "sources": [
                "https://github.com/GoldgofLab/DeepHeme",
                "https://github.com/GoldgofLab/DeepHeme/releases/tag/v1.1",
            ],
        },
    }
    (out_dir / "triage_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_html_report(
        out_dir=out_dir,
        tile_rows=tile_rows,
        slide_rows=slide_rows,
        group_rows=group_rows,
        experiment_rows=experiment_rows,
        thresholds=thresholds,
        skipped=skipped,
        overlay_records=overlay_records,
        report_summary=report_summary,
    )

    print(f"Wrote {len(slide_rows)} slide records and {len(tile_rows)} tile records")
    print(f"Overlay contact sheets: {len(overlay_records)}")
    print(f"Output directory: {out_dir}")
    print(f"HTML: {out_dir / 'index.html'}")


def main() -> None:
    args = parse_args()
    run_analysis(args)


if __name__ == "__main__":
    main()
