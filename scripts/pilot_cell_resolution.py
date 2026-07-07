#!/usr/bin/env python3
"""Build a pathologist review deck for the STAMP cell-resolution pilot.

This script reuses existing STAMP heatmap tile exports. It does not recompute
heatmaps, rerun preprocessing, or require DeepHeme assets.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
CANDIDATES_PATH = (
    ROOT / "validation_report" / "annotation_candidates" / "candidates.json"
)
CLINI_PATH = ROOT / "tables" / "stamp_clini.csv"
DEFAULT_OUT = ROOT / "validation_report" / "pilot_cell_res"

EXPERIMENT_BASES = {
    "blast_percent": Path(
        "/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_blast_percent_uni2"
    ),
    "high_blast": Path(
        "/mnt/nvme0n1p1/Jeff_projects/B01/AG Janssen/stamp_aml_high_blast_uni2"
    ),
}

GROUP_ORDER = ("failure", "positive", "negative")
GROUPS = {
    "failure": {
        "title": "BLAST_PERCENT failures",
        "experiment": "blast_percent",
        "candidate_path": ("blast_percent", "largest_errors"),
        "sample_ids": (
            "SAMPLE_396_803_19_B",
            "SAMPLE_396_803_19_A",
            "SAMPLE_408_1411_19_A",
            "SAMPLE_404_1261_19_C",
            "SAMPLE_408_1411_19_C",
        ),
    },
    "positive": {
        "title": "High-blast positive controls",
        "experiment": "high_blast",
        "candidate_path": ("high_blast", "high_confidence_yes"),
        "sample_ids": (
            "SAMPLE_392_585_19_C",
            "SAMPLE_406_1309_19_C",
            "SAMPLE_414_968_19_C",
            "SAMPLE_395_486_19_A",
        ),
    },
    "negative": {
        "title": "Low-blast negative controls",
        "experiment": "high_blast",
        "candidate_path": ("high_blast", "high_confidence_no"),
        "sample_ids": (
            "SAMPLE_309_665_19_C",
            "SAMPLE_379_1373_19_A",
            "SAMPLE_368_755_19_A",
        ),
    },
}

TILE_RE = re.compile(r"^(top|bottom)_(\d+)-.*=([-+]?\d+(?:\.\d+)?)\.jpg$")
LANCZOS = Image.Resampling.LANCZOS


@dataclass(frozen=True)
class SourceTile:
    kind: str
    rank: int
    score: float | None
    path: Path


@dataclass(frozen=True)
class SlideSpec:
    group: str
    sample_id: str
    stem: str
    experiment: str
    split: str
    candidate: dict[str, Any]
    clinical: dict[str, str]
    source_tile_dir: Path
    top_tiles: list[SourceTile]
    bottom_tiles: list[SourceTile]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the STAMP cell-resolution pilot review deck."
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--bottom-k", type=int, default=8)
    parser.add_argument("--zoom-px", type=int, default=512)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--groups",
        default=",".join(GROUP_ORDER),
        help="Comma-separated subset of: failure,positive,negative",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open() as fp:
        return json.load(fp)


def load_clinical(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as fp:
        reader = csv.DictReader(fp)
        return {row["SAMPLE_ID"]: row for row in reader}


def rows_at_path(data: dict[str, Any], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    value: Any = data
    for key in keys:
        value = value[key]
    if not isinstance(value, list):
        raise TypeError(f"candidate path {'.'.join(keys)} is not a list")
    return value


def parse_groups(raw: str) -> list[str]:
    groups = [g.strip() for g in raw.split(",") if g.strip()]
    unknown = sorted(set(groups) - set(GROUP_ORDER))
    if unknown:
        raise SystemExit(f"unknown group(s): {', '.join(unknown)}")
    ordered = [g for g in GROUP_ORDER if g in groups]
    if not ordered:
        raise SystemExit("no groups selected")
    return ordered


def collect_ranked_tiles(tile_dir: Path, kind: str, count: int) -> list[SourceTile]:
    by_rank: dict[int, SourceTile] = {}
    for path in tile_dir.glob(f"{kind}_*.jpg"):
        match = TILE_RE.match(path.name)
        if not match:
            continue
        matched_kind, rank_text, score_text = match.groups()
        if matched_kind != kind:
            continue
        rank = int(rank_text)
        try:
            score = float(score_text)
        except ValueError:
            score = None
        by_rank[rank] = SourceTile(kind=kind, rank=rank, score=score, path=path)

    missing = [rank for rank in range(1, count + 1) if rank not in by_rank]
    if missing:
        raise FileNotFoundError(
            f"{tile_dir} is missing {kind} ranks: "
            + ", ".join(f"{rank:02d}" for rank in missing)
        )
    return [by_rank[rank] for rank in range(1, count + 1)]


def build_slide_specs(
    *,
    candidates: dict[str, Any],
    clinical: dict[str, dict[str, str]],
    selected_groups: list[str],
    top_k: int,
    bottom_k: int,
    limit: int | None,
) -> list[SlideSpec]:
    slides: list[SlideSpec] = []
    for group in selected_groups:
        group_cfg = GROUPS[group]
        candidate_rows = rows_at_path(candidates, group_cfg["candidate_path"])
        rows_by_sample: dict[str, dict[str, Any]] = {}
        for row in candidate_rows:
            # Keep the first candidate occurrence to preserve the locked ranking
            # from candidates.json when multiple slides share a SAMPLE_ID.
            rows_by_sample.setdefault(row["SAMPLE_ID"], row)

        for sample_id in group_cfg["sample_ids"]:
            if sample_id not in rows_by_sample:
                raise KeyError(
                    f"{sample_id} not found at {'.'.join(group_cfg['candidate_path'])}"
                )
            if sample_id not in clinical:
                raise KeyError(f"{sample_id} not found in {CLINI_PATH}")

            row = rows_by_sample[sample_id]
            experiment = str(group_cfg["experiment"])
            split = str(row["split"])
            stem = str(row["stem"])
            source_tile_dir = (
                EXPERIMENT_BASES[experiment] / "heatmaps" / split / stem / "tiles"
            )
            if not source_tile_dir.exists():
                raise FileNotFoundError(f"missing tile directory: {source_tile_dir}")

            slides.append(
                SlideSpec(
                    group=group,
                    sample_id=sample_id,
                    stem=stem,
                    experiment=experiment,
                    split=split,
                    candidate=row,
                    clinical=clinical[sample_id],
                    source_tile_dir=source_tile_dir,
                    top_tiles=collect_ranked_tiles(source_tile_dir, "top", top_k),
                    bottom_tiles=collect_ranked_tiles(
                        source_tile_dir, "bottom", bottom_k
                    ),
                )
            )

    if limit is not None:
        if limit < 1:
            raise SystemExit("--limit must be at least 1")
        slides = slides[:limit]
    return slides


def copy_and_zoom_tile(
    *,
    source: SourceTile,
    out_dir: Path,
    zoom_px: int,
) -> dict[str, Any]:
    native_dir = out_dir / "native"
    zoom_dir = out_dir / "zoom"
    native_dir.mkdir(parents=True, exist_ok=True)
    zoom_dir.mkdir(parents=True, exist_ok=True)

    native_name = f"{source.kind}_{source.rank:02d}_native.jpg"
    zoom_name = f"{source.kind}_{source.rank:02d}_zoom.jpg"
    native_path = native_dir / native_name
    zoom_path = zoom_dir / zoom_name

    shutil.copy2(source.path, native_path)

    with Image.open(source.path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        if width < zoom_px or height < zoom_px:
            raise ValueError(
                f"{source.path} is {width}x{height}, smaller than --zoom-px {zoom_px}"
            )
        left = (width - zoom_px) // 2
        top = (height - zoom_px) // 2
        rgb.crop((left, top, left + zoom_px, top + zoom_px)).save(zoom_path, quality=92)

    return {
        "kind": source.kind,
        "rank": source.rank,
        "score": source.score,
        "source_path": str(source.path),
        "native_path": str(native_path),
        "zoom_path": str(zoom_path),
    }


def text_size(
    draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont
) -> tuple[int, int]:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0], box[3] - box[1]


def draw_wrapped(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int],
    max_width: int,
    line_gap: int = 4,
) -> int:
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        if text_size(draw, candidate, font)[0] <= max_width:
            line = candidate
        else:
            if line:
                lines.append(line)
            line = word
    if line:
        lines.append(line)

    x, y = xy
    line_height = text_size(draw, "Ag", font)[1] + line_gap
    for line in lines:
        draw.text((x, y), line, fill=fill, font=font)
        y += line_height
    return y


def load_thumb(path: Path, size: int) -> Image.Image:
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        rgb.thumbnail((size, size), LANCZOS)
        canvas = Image.new("RGB", (size, size), "white")
        x = (size - rgb.size[0]) // 2
        y = (size - rgb.size[1]) // 2
        canvas.paste(rgb, (x, y))
        return canvas


def paste_grid(
    *,
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    title: str,
    paths: list[Path],
    x: int,
    y: int,
    tile_size: int,
    cols: int,
    gap: int,
    font: ImageFont.ImageFont,
) -> int:
    draw.text((x, y), title, fill=(25, 25, 25), font=font)
    y += 24
    for idx, path in enumerate(paths):
        row, col = divmod(idx, cols)
        px = x + col * (tile_size + gap)
        py = y + row * (tile_size + 26)
        thumb = load_thumb(path, tile_size)
        canvas.paste(thumb, (px, py))
        draw.rectangle(
            [px, py, px + tile_size - 1, py + tile_size - 1],
            outline=(180, 180, 180),
        )
        draw.text((px + 4, py + tile_size + 4), f"{idx + 1:02d}", fill=(60, 60, 60))
    rows = (len(paths) + cols - 1) // cols
    return y + rows * (tile_size + 26) + 18


def make_contact_sheet(
    *,
    slide: SlideSpec,
    tile_records: list[dict[str, Any]],
    out_path: Path,
) -> None:
    margin = 28
    gap = 12
    tile_size = 168
    cols = 4
    width = margin * 2 + cols * tile_size + (cols - 1) * gap
    header_height = 184
    section_height = 24 + 2 * (tile_size + 26) + 18
    height = header_height + 4 * section_height + margin

    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    title = f"{slide.group.upper()} | {slide.sample_id}"
    y = margin
    draw.text((margin, y), title, fill=(0, 0, 0), font=font)
    y += 22

    pred_bits = []
    if "pred" in slide.candidate:
        pred_bits.append(f"BLAST_PERCENT pred={float(slide.candidate['pred']):.2f}")
    if "abs_err" in slide.candidate:
        pred_bits.append(f"abs_err={float(slide.candidate['abs_err']):.2f}")
    if "HIGH_BLAST_yes" in slide.candidate:
        pred_bits.append(
            f"HIGH_BLAST_yes={float(slide.candidate['HIGH_BLAST_yes']):.4f}"
        )
    metadata = (
        f"GT BLAST_PERCENT={slide.clinical.get('BLAST_PERCENT', '')}; "
        f"GT HIGH_BLAST={slide.clinical.get('HIGH_BLAST', '')}; "
        f"experiment={slide.experiment}; split={slide.split}; " + "; ".join(pred_bits)
    )
    y = draw_wrapped(
        draw,
        (margin, y),
        metadata,
        font=font,
        fill=(30, 30, 30),
        max_width=width - margin * 2,
    )
    y += 6
    y = draw_wrapped(
        draw,
        (margin, y),
        f"Slide: {slide.stem}",
        font=font,
        fill=(60, 60, 60),
        max_width=width - margin * 2,
    )
    y += 14

    top_native = [Path(r["native_path"]) for r in tile_records if r["kind"] == "top"]
    top_zoom = [Path(r["zoom_path"]) for r in tile_records if r["kind"] == "top"]
    bottom_native = [
        Path(r["native_path"]) for r in tile_records if r["kind"] == "bottom"
    ]
    bottom_zoom = [Path(r["zoom_path"]) for r in tile_records if r["kind"] == "bottom"]

    y = paste_grid(
        canvas=canvas,
        draw=draw,
        title="Top-attended native tiles",
        paths=top_native,
        x=margin,
        y=y,
        tile_size=tile_size,
        cols=cols,
        gap=gap,
        font=font,
    )
    y = paste_grid(
        canvas=canvas,
        draw=draw,
        title="Top-attended centered zoom crops",
        paths=top_zoom,
        x=margin,
        y=y,
        tile_size=tile_size,
        cols=cols,
        gap=gap,
        font=font,
    )
    y = paste_grid(
        canvas=canvas,
        draw=draw,
        title="Bottom-attended native tiles",
        paths=bottom_native,
        x=margin,
        y=y,
        tile_size=tile_size,
        cols=cols,
        gap=gap,
        font=font,
    )
    paste_grid(
        canvas=canvas,
        draw=draw,
        title="Bottom-attended centered zoom crops",
        paths=bottom_zoom,
        x=margin,
        y=y,
        tile_size=tile_size,
        cols=cols,
        gap=gap,
        font=font,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def rel(path: Path, base: Path) -> str:
    return path.relative_to(base).as_posix()


def format_float(value: Any, digits: int = 2) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def slide_metadata(slide: SlideSpec) -> dict[str, str]:
    return {
        "group": slide.group,
        "sample_id": slide.sample_id,
        "stem": slide.stem,
        "experiment": slide.experiment,
        "split": slide.split,
        "blast_percent_gt": format_float(slide.clinical.get("BLAST_PERCENT")),
        "high_blast_gt": slide.clinical.get("HIGH_BLAST", ""),
        "blast_percent_pred": format_float(slide.candidate.get("pred")),
        "blast_percent_abs_err": format_float(slide.candidate.get("abs_err")),
        "high_blast_yes": format_float(slide.candidate.get("HIGH_BLAST_yes"), 4),
    }


def write_review_template(slides: list[dict[str, Any]], out_path: Path) -> None:
    fields = [
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
        "cells_identifiable",
        "top_grid_blast_percent_estimate",
        "artifact_wrong_tissue_focus_notes",
        "reviewer",
        "review_date",
    ]
    with out_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for slide in slides:
            row = {field: "" for field in fields}
            row.update(slide["metadata"])
            writer.writerow(row)


def write_pdf(slides: list[dict[str, Any]], out_path: Path) -> None:
    pages: list[Image.Image] = []
    for slide in slides:
        image = Image.open(slide["contact_sheet_path"]).convert("RGB")
        pages.append(image)

    if not pages:
        return
    first, rest = pages[0], pages[1:]
    first.save(out_path, save_all=True, append_images=rest, resolution=150)
    for page in pages:
        page.close()


def write_html(slides: list[dict[str, Any]], out_dir: Path) -> None:
    parts = [
        "<!doctype html>",
        "<html><head><meta charset='utf-8'>",
        "<title>STAMP Cell-Resolution Pilot</title>",
        """
<style>
body { font-family: Arial, sans-serif; margin: 24px; color: #1f2933; }
h1 { margin-bottom: 0; }
.summary { color: #52606d; margin: 6px 0 24px; }
.slide { border-top: 2px solid #d9e2ec; padding: 22px 0 30px; }
.meta { border-collapse: collapse; margin: 8px 0 14px; }
.meta td { border: 1px solid #d9e2ec; padding: 5px 8px; }
.questions { background: #f5f7fa; padding: 10px 14px; margin: 10px 0 14px; }
.contact { max-width: 980px; width: 100%; border: 1px solid #d9e2ec; }
.grid { display: grid; grid-template-columns: repeat(4, minmax(120px, 1fr)); gap: 8px; max-width: 980px; }
.tile img { width: 100%; border: 1px solid #d9e2ec; display: block; }
.tile span { font-size: 12px; color: #52606d; }
</style>
""",
        "</head><body>",
        "<h1>STAMP Cell-Resolution Pilot</h1>",
        (
            "<p class='summary'>Review top/bottom attended native tiles and "
            "centered zoom crops. Fill answers in review_answers_template.csv.</p>"
        ),
    ]

    for slide in slides:
        meta = slide["metadata"]
        parts.append("<section class='slide'>")
        parts.append(
            f"<h2>{html.escape(meta['group'])}: {html.escape(meta['sample_id'])}</h2>"
        )
        parts.append("<table class='meta'>")
        for key in [
            "stem",
            "experiment",
            "split",
            "blast_percent_gt",
            "high_blast_gt",
            "blast_percent_pred",
            "blast_percent_abs_err",
            "high_blast_yes",
        ]:
            parts.append(
                "<tr>"
                f"<td>{html.escape(key)}</td>"
                f"<td>{html.escape(str(meta.get(key, '')))}</td>"
                "</tr>"
            )
        parts.append("</table>")
        parts.append(
            "<div class='questions'><b>Pathologist questions:</b> "
            "1. Can individual cells be identified? yes / sometimes / no. "
            "2. Estimate blast percentage in the top-attended grid. "
            "3. Note artifacts, wrong tissue, focus problems, or uninterpretable regions."
            "</div>"
        )
        parts.append(
            f"<a href='{html.escape(rel(Path(slide['contact_sheet_path']), out_dir))}'>"
            f"<img class='contact' src='{html.escape(rel(Path(slide['contact_sheet_path']), out_dir))}' "
            "alt='contact sheet'></a>"
        )
        parts.append("<h3>Zoom crops</h3><div class='grid'>")
        for tile in slide["tiles"]:
            zoom_rel = rel(Path(tile["zoom_path"]), out_dir)
            native_rel = rel(Path(tile["native_path"]), out_dir)
            label = f"{tile['kind']} {int(tile['rank']):02d}"
            parts.append(
                "<div class='tile'>"
                f"<a href='{html.escape(native_rel)}'>"
                f"<img src='{html.escape(zoom_rel)}' alt='{html.escape(label)}'></a>"
                f"<span>{html.escape(label)}</span>"
                "</div>"
            )
        parts.append("</div></section>")

    parts.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(parts))


def build_outputs(
    *,
    slides: list[SlideSpec],
    out_dir: Path,
    zoom_px: int,
    top_k: int,
    bottom_k: int,
    selected_groups: list[str],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_slides: list[dict[str, Any]] = []

    for slide in slides:
        slide_out = out_dir / slide.group / slide.sample_id
        slide_out.mkdir(parents=True, exist_ok=True)
        tile_records = [
            copy_and_zoom_tile(source=tile, out_dir=slide_out, zoom_px=zoom_px)
            for tile in [*slide.top_tiles, *slide.bottom_tiles]
        ]
        contact_sheet_path = slide_out / "contact_sheet.jpg"
        make_contact_sheet(
            slide=slide, tile_records=tile_records, out_path=contact_sheet_path
        )

        metadata = slide_metadata(slide)
        manifest_slides.append(
            {
                "metadata": metadata,
                "source_tile_dir": str(slide.source_tile_dir),
                "output_dir": str(slide_out),
                "contact_sheet_path": str(contact_sheet_path),
                "tiles": tile_records,
            }
        )

    manifest = {
        "generated_at": datetime.now(UTC).isoformat(),
        "parameters": {
            "top_k": top_k,
            "bottom_k": bottom_k,
            "zoom_px": zoom_px,
            "groups": selected_groups,
        },
        "inputs": {
            "candidates": str(CANDIDATES_PATH),
            "clinical": str(CLINI_PATH),
            "experiments": {key: str(value) for key, value in EXPERIMENT_BASES.items()},
        },
        "total_slides": len(manifest_slides),
        "total_source_tiles": sum(len(s["tiles"]) for s in manifest_slides),
        "slides": manifest_slides,
    }

    (out_dir / "pilot_manifest.json").write_text(json.dumps(manifest, indent=2))
    write_review_template(manifest_slides, out_dir / "review_answers_template.csv")
    write_html(manifest_slides, out_dir)
    write_pdf(manifest_slides, out_dir / "pilot_cell_resolution_review.pdf")
    return manifest


def print_summary(slides: list[SlideSpec], *, dry_run: bool) -> None:
    total_tiles = sum(
        len(slide.top_tiles) + len(slide.bottom_tiles) for slide in slides
    )
    mode = "DRY RUN" if dry_run else "WROTE"
    print(f"{mode}: {len(slides)} slides, {total_tiles} source tiles")
    for group in GROUP_ORDER:
        group_slides = [slide for slide in slides if slide.group == group]
        if not group_slides:
            continue
        print(f"\n{GROUPS[group]['title']} ({len(group_slides)})")
        for slide in group_slides:
            meta = slide_metadata(slide)
            print(
                "  "
                f"{slide.sample_id} | {slide.split} | "
                f"GT_BLAST={meta['blast_percent_gt']} | "
                f"pred={meta['blast_percent_pred'] or meta['high_blast_yes']} | "
                f"{slide.stem}"
            )


def main() -> None:
    args = parse_args()
    if args.top_k < 1 or args.bottom_k < 1:
        raise SystemExit("--top-k and --bottom-k must be at least 1")
    if args.zoom_px < 1:
        raise SystemExit("--zoom-px must be at least 1")

    selected_groups = parse_groups(args.groups)
    candidates = load_json(CANDIDATES_PATH)
    clinical = load_clinical(CLINI_PATH)
    slides = build_slide_specs(
        candidates=candidates,
        clinical=clinical,
        selected_groups=selected_groups,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        limit=args.limit,
    )

    print_summary(slides, dry_run=args.dry_run)
    if args.dry_run:
        return

    manifest = build_outputs(
        slides=slides,
        out_dir=args.out,
        zoom_px=args.zoom_px,
        top_k=args.top_k,
        bottom_k=args.bottom_k,
        selected_groups=selected_groups,
    )
    print(f"\nOutput directory: {args.out}")
    print(f"Manifest: {args.out / 'pilot_manifest.json'}")
    print(f"HTML: {args.out / 'index.html'}")
    print(f"PDF: {args.out / 'pilot_cell_resolution_review.pdf'}")
    print(f"Review CSV: {args.out / 'review_answers_template.csv'}")
    print(
        f"Verified {manifest['total_slides']} slides and "
        f"{manifest['total_source_tiles']} copied source tiles."
    )


if __name__ == "__main__":
    main()
