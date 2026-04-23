from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class PhotoResult:
    stem: str
    status: str
    confidence: float | None = None
    reasons: list[str] = field(default_factory=list)
    category: str | None = None
    error: str | None = None


@dataclass
class ReportData:
    command: str
    started_at: str
    finished_at: str
    duration_seconds: float
    dry_run: bool
    config_summary: dict
    counters: dict[str, int]
    photos: list[PhotoResult]
    total_photos: int


def build_config_summary(config) -> dict:
    return {
        "model": config.llm.model,
        "confidence_threshold": config.llm.confidence_threshold,
        "sources": [{"path": s.path, "camera": s.camera} for s in config.sources],
        "daily_max": config.quota.daily_max,
        "concurrency": config.processing.concurrency,
    }


def build_report_path(output_dir: str, command: str, timestamp: datetime) -> Path:
    ts = timestamp.strftime("%Y%m%dT%H%M%SZ")
    return Path(output_dir) / f"photo-filter-{command}-{ts}.json"


def write_report(data: ReportData, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(asdict(data), ensure_ascii=False)
    try:
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        closed = False
        try:
            os.write(fd, content.encode())
            os.close(fd)
            closed = True
            os.replace(tmp, path)
        except BaseException:
            if not closed:
                os.close(fd)
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except OSError:
        logger.warning("report_write_failed", path=str(path), exc_info=True)
        return path
    logger.info("report_written", path=str(path), size=len(content))
    return path
