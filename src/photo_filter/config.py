from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


def _resolve_env_vars(value: str) -> str:
    """Replace ${ENV_VAR} patterns with environment variable values."""
    return re.sub(
        r"\$\{(\w+)\}",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        value,
    )


def _walk_and_resolve(obj: object) -> object:
    if isinstance(obj, str):
        return _resolve_env_vars(obj)
    if isinstance(obj, dict):
        return {k: _walk_and_resolve(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_and_resolve(v) for v in obj]
    return obj


class LLMConfig(BaseModel):
    base_url: str = "https://lite-llm.app:1443/v1"
    api_key: str = ""
    model: str = "claude-sonnet-4-20250514"
    confidence_threshold: float = 0.7
    max_tokens: int = 1024
    timeout: int = 60


class SourceConfig(BaseModel):
    path: str
    camera: str
    extensions: list[str] = Field(default_factory=lambda: [".ARW", ".JPG"])
    recursive: bool = True


class QuotaConfig(BaseModel):
    daily_max: int = 200


class DatabaseConfig(BaseModel):
    url: str = "postgresql+asyncpg://postgres@localhost:5432/photo_filter"


class ProcessingConfig(BaseModel):
    max_image_size: int = 3840
    jpeg_quality: int = 85
    max_retries: int = 3
    concurrency: int = 5


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"


class ReportConfig(BaseModel):
    output_dir: str = "."
    enabled: bool = True


class ProxyConfig(BaseModel):
    http: str | None = None
    https: str | None = None


class AppConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    sources: list[SourceConfig] = Field(default_factory=list)
    quota: QuotaConfig = Field(default_factory=QuotaConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)


def load_config(path: str | Path) -> AppConfig:
    path = Path(path)
    with path.open() as f:
        raw = yaml.safe_load(f)
    resolved = _walk_and_resolve(raw or {})
    return AppConfig.model_validate(resolved)
