from __future__ import annotations

import os
import platform
import shlex
import signal
import subprocess
import sys
import tempfile
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from shutil import which
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError

from .catalog import (
    TASK_CATALOG,
    TASK_COMMANDS,
    catalog_payload,
    default_advanced_config,
)

# The workbench should operate on the STAMP checkout the user launched it from,
# not on the installed package location inside site-packages.
REPO_ROOT = Path(os.environ.get("STAMP_WORKBENCH_ROOT", str(Path.cwd()))).expanduser().resolve()
MAX_LOG_LINES = 2000
MAX_TERMINAL_LINES = 600
ALLOWED_SHELL_COMMANDS = {
    "basename",
    "cat",
    "dirname",
    "du",
    "echo",
    "find",
    "grep",
    "head",
    "ls",
    "pwd",
    "realpath",
    "sed",
    "stat",
    "tail",
    "tree",
    "wc",
}
FINAL_RUN_STATUSES = {"completed", "failed", "terminated"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PipelineBlock(BaseModel):
    id: str
    section: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    ui: dict[str, Any] = Field(default_factory=dict)


class WorkbenchPayload(BaseModel):
    blocks: list[PipelineBlock] = Field(default_factory=list)
    advanced_config: dict[str, Any] | None = None


class ExportConfigPayload(WorkbenchPayload):
    selected_sections: list[str] = Field(default_factory=list)
    selected_block_ids: list[str] = Field(default_factory=list)
    output_dir: str
    filename: str = "config.yaml"


class AutoSlideTablePayload(BaseModel):
    clinical_table: str
    feature_dir: str
    patient_column: str = "PATIENT"
    filename_column: str = "FILENAME"


@dataclass
class StageRecord:
    section: str
    command: str
    status: str = "pending"
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    logs: list[str] = field(default_factory=list)


@dataclass
class RunRecord:
    run_id: str
    status: str
    created_at: str
    updated_at: str
    config_preview: str
    warnings: list[str]
    scope: str = "pipeline"
    block_id: str | None = None
    block_title: str | None = None
    errors: list[str] = field(default_factory=list)
    logs: list[str] = field(default_factory=list)
    stages: list[StageRecord] = field(default_factory=list)
    stop_requested: bool = False
    terminate_requested: bool = False


@dataclass
class TerminalRecord:
    cwd: str
    history: list[str]



def _clean_scalar(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        return None if stripped == "" else stripped
    return value



def _clean_list(value: Any, *, presentation: str | None) -> list[str] | None:
    if value is None:
        return None

    raw_items: list[str]
    if isinstance(value, str):
        separator = "\n" if presentation == "lines" else ","
        interim = value.replace("\r\n", "\n")
        if presentation == "csv":
            interim = interim.replace("\n", ",")
        raw_items = interim.split(separator)
    elif isinstance(value, list):
        raw_items = [str(item) for item in value]
    else:
        raw_items = [str(value)]

    cleaned = [item.strip() for item in raw_items if item.strip()]
    return cleaned or None



def _normalize_block(section: str, params: dict[str, Any]) -> dict[str, Any]:
    block_schema = TASK_CATALOG[section]
    normalized: dict[str, Any] = {}

    for field_spec in block_schema["fields"]:
        name = field_spec["name"]
        raw_value = params.get(name, field_spec.get("default"))

        if field_spec["kind"] == "list":
            cleaned = _clean_list(
                raw_value,
                presentation=field_spec.get("presentation"),
            )
            if cleaned is None:
                continue
            if field_spec.get("coerce_single") and len(cleaned) == 1:
                normalized[name] = cleaned[0]
            else:
                normalized[name] = cleaned
            continue

        cleaned_value = _clean_scalar(raw_value)
        if cleaned_value is None:
            continue

        normalized[name] = cleaned_value

    return normalized



def _normalize_advanced(advanced_config: dict[str, Any] | None) -> dict[str, Any]:
    defaults = default_advanced_config()
    if advanced_config is None:
        return defaults

    merged = dict(defaults)
    for key, value in advanced_config.items():
        cleaned = _clean_scalar(value)
        if cleaned is None:
            if key == "model_name":
                merged[key] = None
            continue
        merged[key] = cleaned
    return merged



def _validate_paths(blocks: list[PipelineBlock]) -> list[str]:
    errors: list[str] = []
    for block in blocks:
        schema = TASK_CATALOG[block.section]
        params = _normalize_block(block.section, block.params)
        for field_spec in schema["fields"]:
            if field_spec.get("path_role") != "input":
                continue

            name = field_spec["name"]
            path_type = field_spec.get("path_type")
            value = params.get(name)
            if value is None:
                continue

            values = value if isinstance(value, list) else [value]
            for candidate in values:
                path = Path(str(candidate)).expanduser()

                if name == "slide_paths" and block.section == "heatmaps":
                    continue

                if not path.exists():
                    errors.append(
                        f"{schema['title']} -> {field_spec['label']} does not exist: {path}"
                    )
                    continue

                if path_type == "dir" and not path.is_dir():
                    errors.append(
                        f"{schema['title']} -> {field_spec['label']} must be a directory: {path}"
                    )
                elif path_type == "file" and not path.is_file():
                    errors.append(
                        f"{schema['title']} -> {field_spec['label']} must be a file: {path}"
                    )
    return errors



def _pipeline_warnings(enabled_sections: list[str]) -> list[str]:
    warnings: list[str] = []
    if "training" in enabled_sections and "crossval" in enabled_sections:
        warnings.append(
            "Both train and crossval are enabled. The workbench will execute both in the order shown."
        )
    if "statistics" in enabled_sections and not (
        {"crossval", "deployment"} & set(enabled_sections)
    ):
        warnings.append(
            "Statistics is usually run after crossval or deployment. Make sure pred_csvs point to existing prediction files."
        )
    if "heatmaps" in enabled_sections and "preprocessing" not in enabled_sections:
        warnings.append(
            "Heatmaps relies on tile-level features and a trained checkpoint. This is fine if you already have those artifacts."
        )
    return warnings


def _auto_slide_table_enabled(block: PipelineBlock) -> bool:
    ui = block.ui or {}
    if not ui.get("autoCreateSlideTable"):
        return False
    param_names = {
        field_spec["name"]
        for field_spec in TASK_CATALOG.get(block.section, {}).get("fields", [])
    }
    return (
        "slide_table" in param_names
        and "clini_table" in param_names
        and ("feature_dir" in param_names or "feat_dir" in param_names)
    )


def _materialize_auto_slide_tables(
    payload: WorkbenchPayload,
    *,
    keep_files: bool,
) -> tuple[WorkbenchPayload, list[str], list[Path]]:
    blocks: list[PipelineBlock] = []
    warnings: list[str] = []
    temp_paths: list[Path] = []

    for block in payload.blocks:
        copied = block.model_copy(deep=True)
        if copied.enabled and _auto_slide_table_enabled(copied):
            result = create_temp_slide_table(
                {
                    "clinical_table": copied.params.get("clini_table"),
                    "feature_dir": copied.params.get("feature_dir") or copied.params.get("feat_dir"),
                    "patient_column": copied.params.get("patient_label") or "PATIENT",
                    "filename_column": copied.params.get("filename_label") or "FILENAME",
                }
            )
            copied.params["slide_table"] = result["path"]
            temp_path = Path(result["path"])
            temp_paths.append(temp_path)
            warnings.append(
                f"{TASK_CATALOG[copied.section]['title']}: auto-created slide table at {result['path']} ({result['rows']} rows)."
            )
            warnings.extend(result.get("warnings", []))
        blocks.append(copied)

    materialized = WorkbenchPayload(blocks=blocks, advanced_config=payload.advanced_config)

    if not keep_files:
        # The caller will clean these up after validation-only use.
        return materialized, warnings, temp_paths

    return materialized, warnings, temp_paths



def build_stage_configs(payload: WorkbenchPayload) -> tuple[list[dict[str, Any]], list[str]]:
    enabled_blocks = [block for block in payload.blocks if block.enabled]
    if not enabled_blocks:
        raise ValueError("Add at least one enabled block to the pipeline.")

    stage_configs: list[dict[str, Any]] = []
    enabled_sections: list[str] = []

    from stamp.utils.config import StampConfig

    for block in enabled_blocks:
        if block.section not in TASK_CATALOG:
            raise ValueError(f"Unknown block section: {block.section}")

        config: dict[str, Any] = {
            block.section: _normalize_block(block.section, block.params)
        }

        if block.section in {"training", "crossval"}:
            config["advanced_config"] = _normalize_advanced(payload.advanced_config)

        validated = StampConfig.model_validate(config)
        stage_configs.append(validated.model_dump(mode="json", exclude_none=True))
        enabled_sections.append(block.section)

    warnings = _pipeline_warnings(enabled_sections)
    return stage_configs, warnings


def _compose_config_preview(
    blocks: list[PipelineBlock],
    stage_configs: list[dict[str, Any]],
) -> str:
    enabled_blocks = [block for block in blocks if block.enabled]
    previews: list[str] = []

    for index, (block, config) in enumerate(zip(enabled_blocks, stage_configs, strict=True), start=1):
        header = f"# Cell {index}: {TASK_COMMANDS[block.section]}"
        previews.append(f"{header}\n{yaml.dump(config, sort_keys=False).rstrip()}")

    return "\n\n---\n\n".join(previews)



def validate_payload_dict(payload_dict: dict[str, Any]) -> dict[str, Any]:
    temp_paths: list[Path] = []
    auto_warnings: list[str] = []
    warnings: list[str] = []
    stage_configs: list[dict[str, Any]] = []
    materialized_payload: WorkbenchPayload | None = None
    try:
        payload = WorkbenchPayload.model_validate(payload_dict)
    except ValidationError as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": [], "config_preview": ""}

    try:
        materialized_payload, auto_warnings, temp_paths = _materialize_auto_slide_tables(
            payload,
            keep_files=False,
        )
        stage_configs, warnings = build_stage_configs(materialized_payload)
    except Exception as exc:
        return {"valid": False, "errors": [str(exc)], "warnings": [], "config_preview": ""}

    if materialized_payload is None:
        return {"valid": False, "errors": ["Failed to prepare pipeline payload."], "warnings": [], "config_preview": ""}

    config_preview = _compose_config_preview(materialized_payload.blocks, stage_configs)

    path_errors = _validate_paths([block for block in materialized_payload.blocks if block.enabled])
    for temp_path in temp_paths:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
    if path_errors:
        return {
            "valid": False,
            "errors": path_errors,
            "warnings": [*auto_warnings, *warnings],
            "config_preview": config_preview,
        }

    return {
        "valid": True,
        "errors": [],
        "warnings": [*auto_warnings, *warnings],
        "config_preview": config_preview,
        "stage_configs": stage_configs,
    }


def import_config_yaml(*, content: str | None = None, path_str: str | None = None, filename: str | None = None) -> dict[str, Any]:
    if content is None and path_str is None:
        raise ValueError("Provide YAML content or a path to config.yaml.")

    if path_str is not None:
        path = Path(path_str).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Config file does not exist: {path}")
        if not path.is_file():
            raise ValueError(f"Config path is not a file: {path}")
        raw_text = path.read_text(encoding="utf-8")
        source_name = str(path)
    else:
        raw_text = content or ""
        source_name = filename or "config.yaml"

    try:
        raw_payload = yaml.safe_load(raw_text) or {}
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML: {exc}") from exc

    if not isinstance(raw_payload, dict):
        raise ValueError("STAMP config must be a YAML mapping at the top level.")

    from stamp.utils.config import StampConfig

    validated = StampConfig.model_validate(raw_payload)
    normalized = validated.model_dump(mode="json", exclude_none=True)

    ordered_sections = [
        section
        for section in raw_payload.keys()
        if section in TASK_CATALOG and normalized.get(section) is not None
    ]
    for section in TASK_CATALOG:
        if section in normalized and section not in ordered_sections:
            ordered_sections.append(section)

    blocks: list[dict[str, Any]] = []
    for index, section in enumerate(ordered_sections, start=1):
        params = normalized.get(section) or {}
        ui = {
            "multiTarget": bool(
                params.get("task") == "classification"
                and isinstance(params.get("ground_truth_label"), list)
            )
        }
        blocks.append(
            {
                "id": f"import-{index}-{uuid.uuid4().hex[:8]}",
                "section": section,
                "enabled": True,
                "params": params,
                "ui": ui,
            }
        )

    advanced_config = _normalize_advanced(normalized.get("advanced_config"))

    warnings: list[str] = []
    if not blocks:
        warnings.append("The YAML loaded, but it did not contain any executable pipeline sections.")

    return {
        "source": source_name,
        "blocks": blocks,
        "advanced_config": advanced_config,
        "warnings": warnings,
    }


def create_temp_slide_table(payload_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = AutoSlideTablePayload.model_validate(payload_dict)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    clinical_table = Path(payload.clinical_table).expanduser().resolve()
    feature_dir = Path(payload.feature_dir).expanduser().resolve()
    if not clinical_table.exists():
        raise FileNotFoundError(f"Clinical table does not exist: {clinical_table}")
    if not clinical_table.is_file():
        raise ValueError(f"Clinical table is not a file: {clinical_table}")
    if not feature_dir.exists():
        raise FileNotFoundError(f"Feature directory does not exist: {feature_dir}")
    if not feature_dir.is_dir():
        raise ValueError(f"Feature directory is not a directory: {feature_dir}")

    from stamp.modeling.data import read_table
    import pandas as pd

    frame = read_table(clinical_table, usecols=[payload.patient_column], dtype=str)
    patient_ids = sorted(
        {
            str(value).strip()
            for value in frame[payload.patient_column].dropna()
            if str(value).strip()
        },
        key=len,
        reverse=True,
    )
    if not patient_ids:
        raise ValueError(
            f"No patient IDs found in clinical table column {payload.patient_column!r}."
        )

    records: list[dict[str, str]] = []
    unmatched: list[str] = []
    for feature_path in sorted(feature_dir.rglob("*.h5")):
        relative_name = feature_path.relative_to(feature_dir).as_posix()
        basename = feature_path.name
        patient_match = next(
            (patient_id for patient_id in patient_ids if basename.startswith(patient_id)),
            None,
        )
        if patient_match is None:
            unmatched.append(relative_name)
            continue
        records.append(
            {
                payload.filename_column: relative_name,
                payload.patient_column: patient_match,
            }
        )

    if not records:
        raise ValueError(
            "No .h5 feature files could be matched to patient IDs from the clinical table."
        )

    output_dir = Path(tempfile.gettempdir()) / "stamp-workbench"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"slide_table_{uuid.uuid4().hex[:10]}.csv"
    pd.DataFrame(records).to_csv(output_path, index=False)

    warnings: list[str] = []
    if unmatched:
        sample = ", ".join(unmatched[:5])
        suffix = " ..." if len(unmatched) > 5 else ""
        warnings.append(
            f"{len(unmatched)} feature file(s) could not be matched to a patient ID and were skipped: {sample}{suffix}"
        )

    return {
        "path": str(output_path),
        "rows": len(records),
        "patient_column": payload.patient_column,
        "filename_column": payload.filename_column,
        "warnings": warnings,
    }


def export_config_yaml(payload_dict: dict[str, Any]) -> dict[str, Any]:
    try:
        payload = ExportConfigPayload.model_validate(payload_dict)
    except ValidationError as exc:
        raise ValueError(str(exc)) from exc

    selected_block_ids = list(dict.fromkeys(payload.selected_block_ids))
    selected_sections = list(dict.fromkeys(payload.selected_sections))
    if not selected_block_ids and not selected_sections:
        raise ValueError("Select at least one task cell to save.")

    output_dir = Path(payload.output_dir).expanduser().resolve()
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory does not exist: {output_dir}")
    if not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")

    filename = payload.filename.strip() or "config.yaml"
    if not filename.endswith((".yaml", ".yml")):
        filename = f"{filename}.yaml"

    chosen_blocks: list[PipelineBlock] = []
    seen_sections: set[str] = set()
    seen_block_ids: set[str] = set()

    for block in payload.blocks:
        picked = False
        if selected_block_ids:
            picked = block.id in selected_block_ids
        elif selected_sections:
            picked = block.section in selected_sections and block.section not in seen_sections

        if not picked:
            continue

        if block.section in seen_sections:
            raise ValueError(
                f"Only one cell per task section can be saved. Multiple selections found for {TASK_CATALOG[block.section]['title']}."
            )

        chosen_blocks.append(block)
        seen_sections.add(block.section)
        seen_block_ids.add(block.id)

    if selected_block_ids:
        missing_block_ids = [block_id for block_id in selected_block_ids if block_id not in seen_block_ids]
        if missing_block_ids:
            raise ValueError(f"Selected cells missing from pipeline: {', '.join(missing_block_ids)}")
    elif selected_sections:
        missing_sections = [section for section in selected_sections if section not in seen_sections]
        if missing_sections:
            raise ValueError(f"Selected sections missing from pipeline: {', '.join(missing_sections)}")

    config_payload = WorkbenchPayload(
        blocks=[PipelineBlock.model_validate({
            "id": block.id,
            "section": block.section,
            "enabled": True,
            "params": block.params,
        }) for block in chosen_blocks],
        advanced_config=payload.advanced_config,
    )

    stage_configs, warnings = build_stage_configs(config_payload)
    ordered_blocks = [block for block in config_payload.blocks if block.enabled]
    yaml_payload: dict[str, Any] = {}
    for block, stage_config in zip(ordered_blocks, stage_configs, strict=True):
        yaml_payload[block.section] = stage_config.get(block.section, {})

    if {"training", "crossval"} & seen_sections:
        advanced = _normalize_advanced(payload.advanced_config)
        if advanced:
            yaml_payload["advanced_config"] = advanced

    from stamp.utils.config import StampConfig

    validated = StampConfig.model_validate(yaml_payload)
    normalized = validated.model_dump(mode="json", exclude_none=True)
    save_path = output_dir / filename
    yaml_text = yaml.safe_dump(normalized, sort_keys=False, allow_unicode=False)
    save_path.write_text(yaml_text, encoding="utf-8")

    return {
        "path": str(save_path),
        "filename": filename,
        "warnings": warnings,
        "sections": [block.section for block in ordered_blocks],
        "block_ids": [block.id for block in ordered_blocks],
    }



def list_directory(path_str: str | None = None) -> dict[str, Any]:
    base = Path(path_str or REPO_ROOT).expanduser().resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path does not exist: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {base}")

    entries = []
    for child in sorted(base.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
        entries.append(
            {
                "name": child.name,
                "path": str(child),
                "is_dir": child.is_dir(),
                "size": child.stat().st_size if child.is_file() else None,
            }
        )
    return {
        "path": str(base),
        "parent": str(base.parent),
        "entries": entries,
    }



def inspect_table(path_str: str) -> dict[str, Any]:
    import pandas as pd

    from stamp.modeling.data import read_table

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Table does not exist: {path}")

    frame = read_table(path)
    preview = frame.head(5).replace({pd.NA: None}).to_dict(orient="records")
    return {
        "path": str(path),
        "rows": int(frame.shape[0]),
        "columns": list(frame.columns),
        "dtypes": {column: str(dtype) for column, dtype in frame.dtypes.items()},
        "preview": preview,
    }



def inspect_column_values(path_str: str, column_name: str) -> dict[str, Any]:
    import pandas as pd

    from stamp.modeling.data import read_table

    path = Path(path_str).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Table does not exist: {path}")

    frame = read_table(path)
    if column_name not in frame.columns:
        raise KeyError(f"Column not found: {column_name}")

    counts = frame[column_name].value_counts(dropna=False)
    return {
        "path": str(path),
        "column": column_name,
        "values": [
            {
                "value": None if pd.isna(value) else str(value),
                "count": int(count),
            }
            for value, count in counts.items()
        ],
    }



def environment_summary() -> dict[str, Any]:
    devices = ["cpu"]
    if which("nvidia-smi"):
        devices.insert(0, "cuda")
    if platform.system() == "Darwin":
        devices.append("mps")

    return {
        "cwd": str(REPO_ROOT),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "devices": devices,
    }


class RunManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: dict[str, RunRecord] = {}
        self._stage_configs: dict[str, list[dict[str, Any]]] = {}
        self._temp_files: dict[str, list[Path]] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._processes: dict[str, subprocess.Popen[str]] = {}
        self._stop_flags: dict[str, threading.Event] = {}
        self._terminate_flags: dict[str, threading.Event] = {}

    @staticmethod
    def _console_echo(run_id: str, message: str) -> None:
        text = message.rstrip("\n")
        if not text:
            return
        print(f"[workbench:{run_id}] {text}", flush=True)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            runs = sorted(
                self._runs.values(),
                key=lambda run: run.created_at,
                reverse=True,
            )
            return [self._serialize(run) for run in runs]

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            run = self._runs.get(run_id)
            return None if run is None else self._serialize(run)

    def create(self, payload_dict: dict[str, Any]) -> dict[str, Any]:
        temp_paths: list[Path] = []
        materialized_payload: WorkbenchPayload | None = None
        auto_warnings: list[str] = []
        warnings: list[str] = []
        stage_configs: list[dict[str, Any]] = []
        try:
            payload = WorkbenchPayload.model_validate(payload_dict)
            materialized_payload, auto_warnings, temp_paths = _materialize_auto_slide_tables(
                payload,
                keep_files=True,
            )
            stage_configs, warnings = build_stage_configs(materialized_payload)
            path_errors = _validate_paths([block for block in materialized_payload.blocks if block.enabled])
            if path_errors:
                raise ValueError("\n".join(path_errors))
        except Exception:
            for temp_path in temp_paths:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise

        if materialized_payload is None:
            raise RuntimeError("Failed to prepare run payload.")

        enabled_blocks = [block for block in materialized_payload.blocks if block.enabled]
        run_id = uuid.uuid4().hex[:10]
        now = _now_iso()
        stages = [
            StageRecord(section=block.section, command=TASK_COMMANDS[block.section])
            for block in enabled_blocks
        ]
        run = RunRecord(
            run_id=run_id,
            status="queued",
            created_at=now,
            updated_at=now,
            config_preview=_compose_config_preview(materialized_payload.blocks, stage_configs),
            warnings=[*auto_warnings, *warnings],
            scope=str(payload_dict.get("scope") or "pipeline"),
            block_id=payload_dict.get("block_id"),
            block_title=payload_dict.get("block_title"),
            stages=stages,
        )

        with self._lock:
            self._runs[run_id] = run
            self._stage_configs[run_id] = stage_configs
            self._temp_files[run_id] = temp_paths
            self._stop_flags[run_id] = threading.Event()
            self._terminate_flags[run_id] = threading.Event()

        return self.start(run_id)

    def start(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError("Run not found.")
            existing_worker = self._threads.get(run_id)
            if existing_worker is not None and existing_worker.is_alive():
                return self._serialize(run)
            if run.status in {"running", "stopping", "terminating"}:
                return self._serialize(run)
            if run.status in FINAL_RUN_STATUSES:
                raise ValueError("This run has already finished. Create a new run to execute again.")

            self._stop_flags[run_id].clear()
            self._terminate_flags[run_id].clear()
            run.stop_requested = False
            run.terminate_requested = False
            run.status = "queued"
            run.updated_at = _now_iso()

            worker = threading.Thread(target=self._execute, args=(run_id,), daemon=True)
            self._threads[run_id] = worker
            worker.start()
            return self._serialize(run)

    def stop(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError("Run not found.")
            if run.status in FINAL_RUN_STATUSES | {"stopped"}:
                return self._serialize(run)

            self._stop_flags[run_id].set()
            run.stop_requested = True
            run.updated_at = _now_iso()
            if run.status == "queued":
                run.status = "stopped"
            elif run.status == "running":
                run.status = "stopping"
            return self._serialize(run)

    def terminate(self, run_id: str) -> dict[str, Any]:
        process: subprocess.Popen[str] | None = None
        with self._lock:
            run = self._runs.get(run_id)
            if run is None:
                raise KeyError("Run not found.")
            if run.status in FINAL_RUN_STATUSES | {"terminated"}:
                return self._serialize(run)

            self._stop_flags[run_id].set()
            self._terminate_flags[run_id].set()
            run.stop_requested = True
            run.terminate_requested = True
            run.updated_at = _now_iso()
            if run.status == "queued":
                run.status = "terminated"
            elif run.status not in {"terminated", "failed", "completed"}:
                run.status = "terminating"
            process = self._processes.get(run_id)

        if process is not None and process.poll() is None:
            self._terminate_process(process)

        return self.get_run(run_id) or {"run_id": run_id}

    @staticmethod
    def _terminate_process(process: subprocess.Popen[str]) -> None:
        if process.poll() is not None:
            return

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except Exception:
            try:
                process.terminate()
            except Exception:
                return

        try:
            process.wait(timeout=5)
            return
        except Exception:
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            try:
                process.kill()
            except Exception:
                return

        try:
            process.wait(timeout=2)
        except Exception:
            pass

    def _append_log(self, run_id: str, message: str, *, stage_index: int | None = None) -> None:
        clean_message = message.rstrip("\n")
        if not clean_message:
            return

        with self._lock:
            run = self._runs[run_id]
            run.updated_at = _now_iso()
            run.logs.append(clean_message)
            if len(run.logs) > MAX_LOG_LINES:
                run.logs = run.logs[-MAX_LOG_LINES:]
            if stage_index is not None:
                run.stages[stage_index].logs.append(clean_message)
                if len(run.stages[stage_index].logs) > MAX_LOG_LINES:
                    run.stages[stage_index].logs = run.stages[stage_index].logs[-MAX_LOG_LINES:]
        self._console_echo(run_id, clean_message)

    def _set_status(
        self,
        run_id: str,
        *,
        run_status: str | None = None,
        stage_index: int | None = None,
        stage_status: str | None = None,
        return_code: int | None = None,
    ) -> None:
        with self._lock:
            run = self._runs[run_id]
            run.updated_at = _now_iso()
            if run_status is not None:
                run.status = run_status
            if stage_index is not None:
                stage = run.stages[stage_index]
                if stage_status == "running" and stage.started_at is None:
                    stage.started_at = _now_iso()
                if stage_status in {"completed", "failed", "terminated"}:
                    stage.finished_at = _now_iso()
                if stage_status is not None:
                    stage.status = stage_status
                if return_code is not None:
                    stage.return_code = return_code
            stage_command = None if stage_index is None else run.stages[stage_index].command
        status_parts: list[str] = []
        if run_status is not None:
            status_parts.append(f"run={run_status}")
        if stage_command is not None and stage_status is not None:
            status_parts.append(f"{stage_command}={stage_status}")
        if return_code is not None:
            status_parts.append(f"exit={return_code}")
        if status_parts:
            self._console_echo(run_id, "[status] " + " | ".join(status_parts))

    def _add_error(self, run_id: str, message: str) -> None:
        with self._lock:
            run = self._runs[run_id]
            run.updated_at = _now_iso()
            run.errors.append(message)

    def _has_stop_request(self, run_id: str) -> bool:
        return self._stop_flags[run_id].is_set()

    def _has_terminate_request(self, run_id: str) -> bool:
        return self._terminate_flags[run_id].is_set()

    def _execute(self, run_id: str) -> None:
        self._set_status(run_id, run_status="running")

        try:
            with self._lock:
                stages = list(enumerate(self._runs[run_id].stages))
                stage_configs = list(self._stage_configs[run_id])

            for stage_index, stage in stages:
                if stage.status == "completed":
                    continue
                if self._has_terminate_request(run_id):
                    self._append_log(run_id, "[workbench] Run terminated before next stage.")
                    self._set_status(run_id, run_status="terminated")
                    return
                if self._has_stop_request(run_id):
                    self._append_log(run_id, "[workbench] Stop requested. Remaining stages left pending.")
                    self._set_status(run_id, run_status="stopped")
                    return

                temp_path: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as handle:
                        yaml.safe_dump(stage_configs[stage_index], handle, sort_keys=False)
                        temp_path = Path(handle.name)

                    command = [
                        sys.executable,
                        "-m",
                        "stamp",
                        "--config",
                        str(temp_path),
                        stage.command,
                    ]
                    self._append_log(run_id, f"$ {' '.join(command)}", stage_index=stage_index)
                    self._set_status(run_id, stage_index=stage_index, stage_status="running")

                    process = subprocess.Popen(
                        command,
                        cwd=REPO_ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        start_new_session=True,
                    )
                    with self._lock:
                        self._processes[run_id] = process

                    assert process.stdout is not None
                    for line in iter(process.stdout.readline, ""):
                        if not line and process.poll() is not None:
                            break
                        if line:
                            self._append_log(run_id, line, stage_index=stage_index)
                        if self._has_terminate_request(run_id) and process.poll() is None:
                            self._terminate_process(process)

                    return_code = process.wait()
                    with self._lock:
                        self._processes.pop(run_id, None)

                    if self._has_terminate_request(run_id):
                        self._append_log(run_id, "[workbench] Stage terminated by user.", stage_index=stage_index)
                        self._set_status(
                            run_id,
                            run_status="terminated",
                            stage_index=stage_index,
                            stage_status="terminated",
                            return_code=return_code,
                        )
                        return

                    if return_code != 0:
                        self._set_status(
                            run_id,
                            run_status="failed",
                            stage_index=stage_index,
                            stage_status="failed",
                            return_code=return_code,
                        )
                        self._add_error(
                            run_id,
                            f"{stage.command} failed with exit code {return_code}.",
                        )
                        return

                    self._set_status(
                        run_id,
                        stage_index=stage_index,
                        stage_status="completed",
                        return_code=return_code,
                    )

                    if self._has_stop_request(run_id):
                        self._append_log(run_id, "[workbench] Stop requested. Pipeline paused after current stage.")
                        self._set_status(run_id, run_status="stopped")
                        return
                finally:
                    if temp_path is not None and temp_path.exists():
                        temp_path.unlink()

            self._set_status(run_id, run_status="completed")
        except Exception:
            self._set_status(run_id, run_status="failed")
            self._add_error(run_id, traceback.format_exc())
        finally:
            with self._lock:
                self._processes.pop(run_id, None)
                run = self._runs.get(run_id)
                should_cleanup = run is not None and run.status in FINAL_RUN_STATUSES
                temp_files = self._temp_files.get(run_id, [])
            if should_cleanup:
                for temp_path in temp_files:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                with self._lock:
                    self._temp_files.pop(run_id, None)

    def _serialize(self, run: RunRecord) -> dict[str, Any]:
        return asdict(run)


class TerminalManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._record = TerminalRecord(
            cwd=str(REPO_ROOT),
            history=[f"Workbench navigator ready in {REPO_ROOT}"],
        )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cwd": self._record.cwd,
                "history": list(self._record.history[-MAX_TERMINAL_LINES:]),
            }

    def run(self, command: str) -> dict[str, Any]:
        cmd = (command or "").strip()
        if not cmd:
            return self.snapshot()

        if cmd == "clear":
            with self._lock:
                self._record.history = []
            return self.snapshot()

        with self._lock:
            cwd = Path(self._record.cwd)

        self._append(f"{cwd}$ {cmd}")

        if cmd == "cd" or cmd.startswith("cd "):
            self._change_directory(cwd, cmd[2:].strip() or "~")
            return self.snapshot()

        try:
            tokens = shlex.split(cmd)
        except ValueError as exc:
            self._append(f"shell parse error: {exc}")
            return self.snapshot()

        if not tokens:
            return self.snapshot()

        if tokens[0] not in ALLOWED_SHELL_COMMANDS:
            self._append(
                "only navigation commands are allowed here: "
                + ", ".join(sorted(ALLOWED_SHELL_COMMANDS))
                + ", cd, clear"
            )
            return self.snapshot()

        try:
            completed = subprocess.run(
                cmd,
                cwd=cwd,
                shell=True,
                executable="/bin/bash",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=30,
            )
        except subprocess.TimeoutExpired:
            self._append("command timed out after 30 seconds")
            return self.snapshot()
        except Exception as exc:
            self._append(f"command failed: {type(exc).__name__}: {exc}")
            return self.snapshot()

        output = (completed.stdout or "").rstrip()
        if output:
            for line in output.splitlines():
                self._append(line)
        self._append(f"[exit {completed.returncode}]")
        return self.snapshot()

    def _change_directory(self, cwd: Path, target_str: str) -> None:
        target = Path(target_str).expanduser()
        if not target.is_absolute():
            target = cwd / target
        try:
            resolved = target.resolve(strict=True)
        except FileNotFoundError:
            self._append(f"cd: no such file or directory: {target}")
            return
        except Exception as exc:
            self._append(f"cd: {type(exc).__name__}: {exc}")
            return

        if not resolved.is_dir():
            self._append(f"cd: not a directory: {resolved}")
            return

        with self._lock:
            self._record.cwd = str(resolved)
        self._append(str(resolved))

    def _append(self, line: str) -> None:
        if not line:
            return
        with self._lock:
            self._record.history.append(line)
            if len(self._record.history) > MAX_TERMINAL_LINES:
                self._record.history = self._record.history[-MAX_TERMINAL_LINES:]



def bootstrap_payload() -> dict[str, Any]:
    payload = catalog_payload()
    payload["environment"] = environment_summary()
    payload["runs"] = []
    return payload
