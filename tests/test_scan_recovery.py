"""Run with TEST_DATABASE_URL pointing to a disposable PostgreSQL database."""

import os
from datetime import datetime, timezone
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
import pytest_asyncio
from PIL import Image
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import create_async_engine

from photo_filter import analyzer, db
from photo_filter.cli import _scan
from photo_filter.config import AppConfig, SourceConfig
from photo_filter.models import AnalysisResult, IssueCategory, Verdict


@pytest_asyncio.fixture
async def database(monkeypatch):
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL is required for PostgreSQL recovery tests")
    schema = "test_recovery_" + uuid4().hex
    admin = create_async_engine(url)
    async with admin.begin() as conn:
        await conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    engine = create_async_engine(url, connect_args={"server_settings": {"search_path": schema}})
    monkeypatch.setattr(db, "make_engine", lambda _: engine)
    try:
        await db.init_db(engine)
        yield engine, db.make_session_factory(engine)
    finally:
        await engine.dispose()
        async with admin.begin() as conn:
            await conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        await admin.dispose()


async def test_scan_recovers_after_model_error(database, tmp_path, monkeypatch):
    _, sessions = database
    Image.new("RGB", (2, 2)).save(tmp_path / "photo.JPG")
    config = AppConfig(sources=[SourceConfig(path=str(tmp_path), camera="test")])
    monkeypatch.setattr(analyzer, "make_client", lambda _: None)
    analyze = AsyncMock(side_effect=RuntimeError("model unavailable"))
    monkeypatch.setattr(analyzer, "analyze_photo", analyze)

    await _scan(config, False, None, None, True)
    async with sessions() as session:
        assert await db.get_processed_keys(session) == set()
        assert await db.count_processed_in_dir(session, str(tmp_path)) == 0
        assert await db.get_completed_dirs(session) == set()
        failed = (await db.get_error_records(session))[0]
        failed_id = failed.id

    analyze.side_effect = None
    analyze.return_value = AnalysisResult(
        verdict=Verdict.KEEP, confidence=1.0, reasons=[], category=IssueCategory.NONE,
        raw_response='{"verdict": "keep"}',
    )
    await _scan(config, False, None, None, True)
    assert analyze.await_count == 2
    async with sessions() as session:
        assert await db.get_error_records(session) == []
        assert (await session.get(db.PhotoRecord, failed_id)).status == "kept"
        assert await db.get_completed_dirs(session) == {str(tmp_path)}
    await _scan(config, False, None, None, True)
    assert analyze.await_count == 2


@pytest.mark.parametrize("status", ["error", "pending"])
async def test_legacy_directory_repair_is_idempotent(database, status):
    engine, sessions = database
    async with sessions() as session:
        for stem, directory, state in [
            ("failed", "/failed", status),
            ("old", "/failed", "deleted"),
            ("good", "/good", "kept"),
        ]:
            session.add(db.PhotoRecord(
                file_stem=stem, source_dir=directory, status=state,
                processed_at=datetime.now(timezone.utc),
            ))
        for directory in ("/failed", "/good"):
            await db.upsert_scanned_dir(session, directory, "test", 1, True)
        await session.commit()

    await db.init_db(engine)
    await db.init_db(engine)
    async with sessions() as session:
        assert await db.get_completed_dirs(session) == {"/good"}
        assert await db.count_processed_in_dir(session, "/failed") == 0
        assert await db.count_processed_in_dir(session, "/good") == 1
        assert ("failed", "/failed") not in await db.get_processed_keys(session)
        rows = list((await session.execute(select(db.PhotoRecord))).scalars())
        assert len(rows) == 3
        assert next(r for r in rows if r.file_stem == "failed").status == status
