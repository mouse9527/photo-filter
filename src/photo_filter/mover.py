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


RECYCLE_DIR = "#recycle"


def _find_share_root(file_path: Path, photo_dirs: list[str]) -> Path | None:
    for d in photo_dirs:
        root = Path(d)
        try:
            file_path.relative_to(root)
            return root
        except ValueError:
            continue
    return None


def delete_photo(unit: PhotoUnit, photo_dirs: list[str]) -> list[Path]:
    moved = []
    for src in unit.all_paths:
        actual = src
        rejected_path = unit.source_dir / REJECTED_DIR / src.name
        if not actual.exists() and rejected_path.exists():
            actual = rejected_path

        if not actual.exists():
            logger.warning("delete_file_not_found", path=str(src))
            continue

        share_root = _find_share_root(actual, photo_dirs)
        if share_root is None:
            logger.warning(
                "delete_no_share_root", path=str(actual),
            )
            continue

        relative = actual.relative_to(share_root)
        dest = share_root / RECYCLE_DIR / relative
        dest.parent.mkdir(parents=True, exist_ok=True)

        if dest.exists():
            stem = dest.stem
            suffix = dest.suffix
            counter = 1
            while dest.exists():
                dest = dest.with_name(f"{stem}_{counter}{suffix}")
                counter += 1

        shutil.move(str(actual), str(dest))
        moved.append(dest)
        logger.info("file_recycled", src=str(actual), dst=str(dest))

    return moved
