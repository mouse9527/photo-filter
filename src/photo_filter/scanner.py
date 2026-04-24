from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path

import structlog

from photo_filter.config import SourceConfig
from photo_filter.models import PhotoUnit

logger = structlog.get_logger()

RAW_EXTENSIONS = {".ARW", ".arw", ".DNG", ".dng", ".CR3", ".cr3", ".NEF", ".nef"}
JPEG_EXTENSIONS = {".JPG", ".jpg", ".JPEG", ".jpeg"}
REJECTED_DIR = "_rejected"
SKIP_DIRS = {REJECTED_DIR, "#recycle", "@eaDir"}


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _should_skip_dir(dir_path: Path) -> bool:
    return any(part in SKIP_DIRS for part in dir_path.parts)


def collect_directories(
    source: SourceConfig,
) -> dict[str, int]:
    root = Path(source.path)
    if not root.exists():
        logger.warning("source_dir_not_found", path=str(root))
        return {}

    allowed = {ext.upper() for ext in source.extensions}
    dir_counts: dict[str, int] = defaultdict(int)

    iterator = root.rglob("*") if source.recursive else root.glob("*")
    for file_path in iterator:
        if not file_path.is_file():
            continue
        if _should_skip_dir(file_path):
            continue
        if file_path.suffix.upper() not in allowed:
            continue
        dir_counts[str(file_path.parent)] += 1

    return dict(dir_counts)


def scan_source(
    source: SourceConfig,
    skip_dirs: set[str] | None = None,
) -> tuple[list[PhotoUnit], dict[str, int]]:
    dir_counts = collect_directories(source)
    if not dir_counts:
        logger.info("no_photos_found", source=source.path)
        return [], dir_counts

    skip = skip_dirs or set()
    skipped = sum(1 for d in dir_counts if d in skip)
    scan_dirs = {d for d in dir_counts if d not in skip}

    logger.info(
        "scan_directories",
        source=source.path,
        total_dirs=len(dir_counts),
        skipped=skipped,
        to_scan=len(scan_dirs),
    )

    allowed = {ext.upper() for ext in source.extensions}
    groups: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(
        lambda: {"raw": [], "jpg": [], "other": []}
    )

    for dir_path in sorted(scan_dirs):
        for file_path in Path(dir_path).iterdir():
            if not file_path.is_file():
                continue
            ext = file_path.suffix.upper()
            if ext not in allowed:
                continue

            stem = file_path.stem
            source_dir = str(file_path.parent)
            key = (stem, source_dir)

            if ext in {e.upper() for e in RAW_EXTENSIONS}:
                groups[key]["raw"].append(file_path)
            elif ext in {e.upper() for e in JPEG_EXTENSIONS}:
                groups[key]["jpg"].append(file_path)
            else:
                groups[key]["other"].append(file_path)

    units = []
    for (stem, source_dir), files in groups.items():
        jpg_path = files["jpg"][0] if files["jpg"] else None
        unit = PhotoUnit(
            stem=stem,
            source_dir=Path(source_dir),
            camera=source.camera,
            jpg_path=jpg_path,
            arw_path=files["raw"][0] if files["raw"] else None,
            extra_paths=files["jpg"][1:] + files["raw"][1:] + files["other"],
        )
        if unit.analysis_path is None:
            logger.debug("skipping_no_jpg", stem=stem, source_dir=source_dir)
            continue
        units.append(unit)

    units.sort(key=lambda u: (str(u.source_dir), u.stem))
    logger.info(
        "scan_complete",
        source=source.path,
        camera=source.camera,
        found=len(units),
    )
    return units, dir_counts


def filter_unprocessed(
    units: list[PhotoUnit],
    processed_keys: set[tuple[str, str]],
) -> list[PhotoUnit]:
    return [
        u for u in units
        if (u.stem, str(u.source_dir)) not in processed_keys
    ]
