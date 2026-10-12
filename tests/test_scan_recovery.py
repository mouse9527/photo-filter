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


def _keep_result():
    return AnalysisResult(
        verdict=Verdict.KEEP, confidence=1.0, reasons=[], category=IssueCategory.NONE,
        raw_response='{"verdict": "keep"}',
    )


async def test_byte_identical_copy_is_skipped_without_crashing(database, tmp_path, monkeypatch):
    # 生产事故 2026-10-12：Finder 生成的 "DSC09930 2.JPG" 与原图内容相同，
    # 写库撞 uq_photo_file_hash，error 记录也撞，异常冲出 gather 导致整个任务崩溃。
    _, sessions = database
    Image.new("RGB", (4, 4), (10, 20, 30)).save(tmp_path / "DSC09930.JPG")
    (tmp_path / "DSC09930 2.JPG").write_bytes((tmp_path / "DSC09930.JPG").read_bytes())
    Image.new("RGB", (4, 4), (90, 90, 90)).save(tmp_path / "other.JPG")
    config = AppConfig(sources=[SourceConfig(path=str(tmp_path), camera="test")])
    monkeypatch.setattr(analyzer, "make_client", lambda _: None)
    analyze = AsyncMock(return_value=_keep_result())
    monkeypatch.setattr(analyzer, "analyze_photo", analyze)

    await _scan(config, False, None, None, True)

    # 两份副本只调用一次 LLM
    assert analyze.await_count == 2
    async with sessions() as session:
        rows = {r.file_stem: r for r in (await session.execute(select(db.PhotoRecord))).scalars()}
        assert set(rows) == {"DSC09930", "DSC09930 2", "other"}
        statuses = sorted(r.status for r in rows.values())
        assert statuses == ["duplicate", "kept", "kept"]
        dup = next(r for r in rows.values() if r.status == "duplicate")
        assert dup.file_hash is None
        # 重复文件计入完成数，目录能正常完成
        assert await db.get_completed_dirs(session) == {str(tmp_path)}

    # 再跑一次：都已处理，不再调用 LLM
    await _scan(config, False, None, None, True)
    assert analyze.await_count == 2


async def test_copy_added_after_original_was_processed(database, tmp_path, monkeypatch):
    _, sessions = database
    Image.new("RGB", (4, 4), (10, 20, 30)).save(tmp_path / "a.JPG")
    config = AppConfig(sources=[SourceConfig(path=str(tmp_path), camera="test")])
    monkeypatch.setattr(analyzer, "make_client", lambda _: None)
    analyze = AsyncMock(return_value=_keep_result())
    monkeypatch.setattr(analyzer, "analyze_photo", analyze)
    await _scan(config, False, None, None, True)

    (tmp_path / "a 2.JPG").write_bytes((tmp_path / "a.JPG").read_bytes())
    async with sessions() as session:
        # 新文件让目录变回未完成，模拟后续导入
        await db.upsert_scanned_dir(session, str(tmp_path), "test", 2, False)
        await session.commit()
    await _scan(config, False, None, None, True)

    assert analyze.await_count == 1
    async with sessions() as session:
        records = (await session.execute(select(db.PhotoRecord))).scalars()
        assert {r.file_stem: r.status for r in records} == {"a": "kept", "a 2": "duplicate"}


async def test_rescanning_same_photo_is_not_a_duplicate(database, tmp_path, monkeypatch):
    # 同一张照片 error 后重试，hash 属于它自己的记录，不能误判为重复
    _, sessions = database
    Image.new("RGB", (4, 4)).save(tmp_path / "photo.JPG")
    config = AppConfig(sources=[SourceConfig(path=str(tmp_path), camera="test")])
    monkeypatch.setattr(analyzer, "make_client", lambda _: None)
    analyze = AsyncMock(side_effect=[RuntimeError("model down"), _keep_result()])
    monkeypatch.setattr(analyzer, "analyze_photo", analyze)

    await _scan(config, False, None, None, True)
    await _scan(config, False, None, None, True)

    async with sessions() as session:
        rows = list((await session.execute(select(db.PhotoRecord))).scalars())
        assert [r.status for r in rows] == ["kept"]
        assert rows[0].file_hash is not None
