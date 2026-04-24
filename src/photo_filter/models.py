from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path


class Verdict(StrEnum):
    REJECT = "reject"
    KEEP = "keep"
    REVIEW = "review"


class Status(StrEnum):
    PENDING = "pending"
    KEPT = "kept"
    REJECTED = "rejected"
    ERROR = "error"
    REVIEW = "review"
    DELETED = "deleted"


class IssueCategory(StrEnum):
    TECHNICAL = "technical"
    COMPOSITION = "composition"
    SUBJECT = "subject"
    ACCIDENTAL = "accidental"
    NONE = "none"


@dataclass
class PhotoUnit:
    stem: str
    source_dir: Path
    camera: str
    jpg_path: Path | None = None
    arw_path: Path | None = None
    extra_paths: list[Path] = field(default_factory=list)
    file_hash: str | None = None

    @property
    def analysis_path(self) -> Path | None:
        return self.jpg_path

    @property
    def all_paths(self) -> list[Path]:
        paths = []
        if self.jpg_path:
            paths.append(self.jpg_path)
        if self.arw_path:
            paths.append(self.arw_path)
        paths.extend(self.extra_paths)
        return paths


@dataclass
class AnalysisResult:
    verdict: Verdict
    confidence: float
    reasons: list[str]
    category: IssueCategory
    raw_response: str
