from __future__ import annotations

import shutil
from pathlib import Path

import structlog

from photo_filter.models import PhotoUnit
from photo_filter.scanner import REJECTED_DIR

logger = structlog.get_logger()


def reject_photo(unit: PhotoUnit) -> list[Path]:
    rejected_dir = unit.source_dir / REJECTED_DIR
    rejected_dir.mkdir(exist_ok=True)

    moved = []
    for src in unit.all_paths:
        if not src.exists():
            logger.warning("file_not_found", path=str(src))
            continue
        dst = rejected_dir / src.name
        if dst.exists():
            logger.warning("destination_exists", src=str(src), dst=str(dst))
            continue
        shutil.move(str(src), str(dst))
        moved.append(dst)
        logger.info("file_moved", src=str(src), dst=str(dst))

    return moved


def undo_rejection(unit: PhotoUnit) -> list[Path]:
    rejected_dir = unit.source_dir / REJECTED_DIR
    restored = []
    for original_path in unit.all_paths:
        rejected_path = rejected_dir / original_path.name
        if not rejected_path.exists():
            continue
        if original_path.exists():
            logger.warning("original_exists", path=str(original_path))
            continue
        shutil.move(str(rejected_path), str(original_path))
        restored.append(original_path)
        logger.info("file_restored", src=str(rejected_path), dst=str(original_path))

    return restored
