from __future__ import annotations

import structlog
from collections import defaultdict
from pathlib import Path

from photo_filter.config import SourceConfig
from photo_filter.models import PhotoUnit

logger = structlog.get_logger()

RAW_EXTENSIONS = {".ARW", ".arw", ".DNG", ".dng", ".CR3", ".cr3", ".NEF", ".nef"}
JPEG_EXTENSIONS = {".JPG", ".jpg", ".JPEG", ".jpeg"}
REJECTED_DIR = "_rejected"


def scan_source(source: SourceConfig) -> list[PhotoUnit]:
    root = Path(source.path)
    if not root.exists():
        logger.warning("source_dir_not_found", path=str(root))
        return []

    allowed = {ext.upper() for ext in source.extensions}
    groups: dict[tuple[str, str], dict[str, list[Path]]] = defaultdict(
        lambda: {"raw": [], "jpg": [], "other": []}
    )

    iterator = root.rglob("*") if source.recursive else root.glob("*")
    for file_path in iterator:
        if not file_path.is_file():
            continue
        if file_path.parent.name == REJECTED_DIR:
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
        unit = PhotoUnit(
            stem=stem,
            source_dir=Path(source_dir),
            camera=source.camera,
            jpg_path=files["jpg"][0] if files["jpg"] else None,
            arw_path=files["raw"][0] if files["raw"] else None,
            extra_paths=files["jpg"][1:] + files["raw"][1:] + files["other"],
        )
        if unit.analysis_path is None:
            logger.debug("skipping_no_jpg", stem=stem, source_dir=source_dir)
            continue
        units.append(unit)

    units.sort(key=lambda u: (str(u.source_dir), u.stem))
    logger.info("scan_complete", source=source.path, camera=source.camera, found=len(units))
    return units


def filter_unprocessed(
    units: list[PhotoUnit], processed_stems: dict[str, set[str]]
) -> list[PhotoUnit]:
    unprocessed = []
    for unit in units:
        dir_key = str(unit.source_dir)
        stems = processed_stems.get(dir_key, set())
        if unit.stem not in stems:
            unprocessed.append(unit)
    return unprocessed
