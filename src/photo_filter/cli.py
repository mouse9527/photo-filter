from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

import click
import structlog

from photo_filter.config import load_config
from photo_filter.logging_config import setup_logging

logger = structlog.get_logger()


@click.group()
@click.option(
    "--config",
    "config_path",
    default="config.yaml",
    envvar="PHOTO_FILTER_CONFIG",
    type=click.Path(exists=True),
    help="Path to config YAML file.",
)
@click.pass_context
def main(ctx: click.Context, config_path: str) -> None:
    ctx.ensure_object(dict)
    cfg = load_config(config_path)
    setup_logging(cfg.logging.level, cfg.logging.format)
    ctx.obj["config"] = cfg


@main.command()
@click.option("--dry-run", is_flag=True, help="Analyze without moving files.")
@click.option("--limit", type=int, default=None, help="Override daily quota for this run.")
@click.option("--report-path", type=click.Path(), default=None, help="Path for JSON report file.")
@click.option("--no-report", is_flag=True, help="Disable report generation for this run.")
@click.pass_context
def scan(
    ctx: click.Context, dry_run: bool, limit: int | None,
    report_path: str | None, no_report: bool,
) -> None:
    """Scan photo directories, analyze with LLM, and reject bad photos."""
    asyncio.run(
        _scan(ctx.obj["config"], dry_run, limit, report_path, no_report)
    )


async def _process_one(
    unit, client, config, session_factory, semaphore,
    dry_run, counters, results,
):
    from photo_filter.analyzer import analyze_photo
    from photo_filter.db import PhotoRecord, upsert_record
    from photo_filter.models import Verdict
    from photo_filter.mover import reject_photo

    async with semaphore:
        try:
            result = await analyze_photo(unit, client, config)
            now = datetime.now(timezone.utc)

            should_reject = (
                result.verdict == Verdict.REJECT
                and result.confidence >= config.llm.confidence_threshold
            )
            if should_reject:
                status = "rejected"
                if not dry_run:
                    reject_photo(unit)
                counters["rejected"] += 1
            elif result.verdict == Verdict.REVIEW:
                status = "review"
                counters["kept"] += 1
            else:
                status = "kept"
                counters["kept"] += 1

            record = PhotoRecord(
                file_stem=unit.stem,
                source_dir=str(unit.source_dir),
                file_hash=unit.file_hash,
                jpg_path=str(unit.jpg_path) if unit.jpg_path else None,
                arw_path=str(unit.arw_path) if unit.arw_path else None,
                camera=unit.camera,
                status=status,
                confidence=result.confidence,
                verdict_reasons=json.dumps(result.reasons),
                llm_model=config.llm.model,
                llm_response=result.raw_response,
                file_size_bytes=(
                    unit.analysis_path.stat().st_size
                    if unit.analysis_path and unit.analysis_path.exists()
                    else None
                ),
                processed_at=now,
            )
            async with session_factory() as session:
                await upsert_record(session, record)
                await session.commit()
            counters["analyzed"] += 1
            results.append({
                "stem": unit.stem,
                "status": status,
                "confidence": result.confidence,
                "reasons": result.reasons,
                "category": result.category.value,
                "error": None,
            })

            logger.info(
                "photo_processed",
                stem=unit.stem,
                status=status,
                confidence=result.confidence,
                dry_run=dry_run,
            )

        except Exception as exc:
            logger.exception("analysis_failed", stem=unit.stem)
            async with session_factory() as session:
                record = PhotoRecord(
                    file_stem=unit.stem,
                    source_dir=str(unit.source_dir),
                    file_hash=unit.file_hash,
                    jpg_path=str(unit.jpg_path) if unit.jpg_path else None,
                    arw_path=str(unit.arw_path) if unit.arw_path else None,
                    camera=unit.camera,
                    status="error",
                    processed_at=datetime.now(timezone.utc),
                )
                await upsert_record(session, record)
                await session.commit()
            counters["errors"] += 1
            results.append({
                "stem": unit.stem,
                "status": "error",
                "confidence": None,
                "reasons": [],
                "category": None,
                "error": f"{type(exc).__name__}: {exc}",
            })


async def _scan(
    config, dry_run: bool, limit: int | None,
    report_path: str | None, no_report: bool,
) -> None:
    from photo_filter.analyzer import make_client
    from photo_filter.db import (
        get_daily_count,
        get_processed_hashes,
        init_db,
        make_engine,
        make_session_factory,
    )
    from photo_filter.report import (
        PhotoResult,
        ReportData,
        build_config_summary,
        build_report_path,
        write_report,
    )
    from photo_filter.scanner import filter_unprocessed, scan_source

    started_at = datetime.now(timezone.utc)
    engine = make_engine(config.database.url)
    await init_db(engine)
    session_factory = make_session_factory(engine)

    client = make_client(config)
    daily_max = limit if limit is not None else config.quota.daily_max
    semaphore = asyncio.Semaphore(config.processing.concurrency)
    counters = {"analyzed": 0, "rejected": 0, "kept": 0, "errors": 0}
    results: list[dict] = []

    try:
        async with session_factory() as session:
            processed_hashes = await get_processed_hashes(session)

        for source in config.sources:
            logger.info("scanning_source", path=source.path, camera=source.camera)
            units = scan_source(source)

            if not units:
                logger.info("no_photos_found", source=source.path)
                continue

            unprocessed = filter_unprocessed(units, processed_hashes)
            logger.info(
                "unprocessed_photos",
                source=source.path,
                total=len(units),
                unprocessed=len(unprocessed),
            )

            async with session_factory() as session:
                daily_count = await get_daily_count(session)
                remaining = max(0, daily_max - daily_count)

                if remaining == 0:
                    logger.info("daily_quota_reached", daily_max=daily_max)
                    break

            batch = unprocessed[:remaining]
            logger.info(
                "processing_batch",
                count=len(batch),
                remaining=remaining,
                concurrency=config.processing.concurrency,
            )

            tasks = [
                _process_one(
                    unit, client, config, session_factory,
                    semaphore, dry_run, counters, results,
                )
                for unit in batch
            ]
            await asyncio.gather(*tasks)
    finally:
        await engine.dispose()

    logger.info(
        "scan_complete",
        total_analyzed=counters["analyzed"],
        rejected=counters["rejected"],
        kept=counters["kept"],
        errors=counters["errors"],
        dry_run=dry_run,
    )
    click.echo(
        f"Done: {counters['analyzed']} analyzed, {counters['rejected']} rejected, "
        f"{counters['kept']} kept, {counters['errors']} errors"
    )

    should_report = not no_report and (report_path or config.report.enabled)
    if should_report:
        finished_at = datetime.now(timezone.utc)
        duration = (finished_at - started_at).total_seconds()
        path = (
            Path(report_path) if report_path
            else build_report_path(config.report.output_dir, "scan", started_at)
        )
        report = ReportData(
            command="scan",
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_seconds=duration,
            dry_run=dry_run,
            config_summary=build_config_summary(config),
            counters=counters,
            photos=[PhotoResult(**r) for r in results],
            total_photos=len(results),
        )
        written = write_report(report, path)
        click.echo(f"Report: {written}")


@main.command()
@click.pass_context
def stats(ctx: click.Context) -> None:
    """Show processing statistics."""
    asyncio.run(_stats(ctx.obj["config"]))


async def _stats(config) -> None:
    from photo_filter.db import get_stats, make_engine, make_session_factory

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)

    try:
        async with session_factory() as session:
            s = await get_stats(session)
    finally:
        await engine.dispose()

    click.echo("Photo Filter Statistics:")
    click.echo(f"  Total:    {s.get('total', 0)}")
    click.echo(f"  Kept:     {s.get('kept', 0)}")
    click.echo(f"  Rejected: {s.get('rejected', 0)}")
    click.echo(f"  Review:   {s.get('review', 0)}")
    click.echo(f"  Error:    {s.get('error', 0)}")
    click.echo(f"  Pending:  {s.get('pending', 0)}")


@main.command()
@click.option("--host", default=None, help="Bind host (overrides config).")
@click.option("--port", type=int, default=None, help="Bind port (overrides config).")
@click.pass_context
def web(ctx: click.Context, host: str | None, port: int | None) -> None:
    """Start the review web UI."""
    import uvicorn

    from photo_filter.web import create_app

    config = ctx.obj["config"]
    app = create_app(config)
    uvicorn.run(
        app,
        host=host or config.web.host,
        port=port or config.web.port,
        log_level="info",
    )


@main.command(name="init-db")
@click.pass_context
def init_db_cmd(ctx: click.Context) -> None:
    """Initialize database tables."""
    asyncio.run(_init_db(ctx.obj["config"]))


async def _init_db(config) -> None:
    from photo_filter.db import init_db, make_engine

    engine = make_engine(config.database.url)
    try:
        await init_db(engine)
    finally:
        await engine.dispose()
    click.echo("Database tables created.")


@main.command(name="retry-errors")
@click.option("--dry-run", is_flag=True, help="Analyze without moving files.")
@click.option("--report-path", type=click.Path(), default=None, help="Path for JSON report file.")
@click.option("--no-report", is_flag=True, help="Disable report generation for this run.")
@click.pass_context
def retry_errors(
    ctx: click.Context, dry_run: bool,
    report_path: str | None, no_report: bool,
) -> None:
    """Reprocess photos that previously failed."""
    asyncio.run(
        _retry_errors(ctx.obj["config"], dry_run, report_path, no_report)
    )


async def _retry_one(
    record, client, config, session_factory, semaphore,
    dry_run, counters, results,
):
    from photo_filter.analyzer import analyze_photo
    from photo_filter.models import PhotoUnit, Verdict
    from photo_filter.mover import reject_photo

    unit = PhotoUnit(
        stem=record.file_stem,
        source_dir=Path(record.source_dir),
        camera=record.camera or "",
        jpg_path=Path(record.jpg_path) if record.jpg_path else None,
        arw_path=Path(record.arw_path) if record.arw_path else None,
    )
    if unit.analysis_path is None or not unit.analysis_path.exists():
        logger.warning("retry_skip_no_file", stem=unit.stem)
        results.append({
            "stem": unit.stem,
            "status": "skipped",
            "confidence": None,
            "reasons": ["File not found for retry"],
            "category": None,
            "error": None,
        })
        return

    async with semaphore:
        try:
            result = await analyze_photo(unit, client, config)
            now = datetime.now(timezone.utc)
            should_reject = (
                result.verdict == Verdict.REJECT
                and result.confidence >= config.llm.confidence_threshold
            )
            if should_reject:
                status = "rejected"
                if not dry_run:
                    reject_photo(unit)
            elif result.verdict == Verdict.REVIEW:
                status = "review"
            else:
                status = "kept"

            async with session_factory() as session:
                from photo_filter.db import PhotoRecord

                existing = await session.get(PhotoRecord, record.id)
                if existing:
                    existing.status = status
                    existing.confidence = result.confidence
                    existing.verdict_reasons = json.dumps(result.reasons)
                    existing.llm_model = config.llm.model
                    existing.llm_response = result.raw_response
                    existing.processed_at = now
                    await session.commit()
            counters["retried"] += 1
            results.append({
                "stem": unit.stem,
                "status": status,
                "confidence": result.confidence,
                "reasons": result.reasons,
                "category": result.category.value,
                "error": None,
            })
        except Exception as exc:
            logger.exception("retry_failed", stem=unit.stem)
            results.append({
                "stem": unit.stem,
                "status": "error",
                "confidence": None,
                "reasons": [],
                "category": None,
                "error": f"{type(exc).__name__}: {exc}",
            })


async def _retry_errors(config, dry_run: bool, report_path: str | None, no_report: bool) -> None:
    from photo_filter.analyzer import make_client
    from photo_filter.db import get_error_records, make_engine, make_session_factory
    from photo_filter.report import (
        PhotoResult,
        ReportData,
        build_config_summary,
        build_report_path,
        write_report,
    )

    started_at = datetime.now(timezone.utc)
    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)
    client = make_client(config)
    semaphore = asyncio.Semaphore(config.processing.concurrency)
    counters = {"retried": 0}
    results: list[dict] = []

    try:
        async with session_factory() as session:
            errors = await get_error_records(session)

        if not errors:
            click.echo("No error records to retry.")
            return

        click.echo(f"Retrying {len(errors)} error records...")
        tasks = [
            _retry_one(
                record, client, config, session_factory,
                semaphore, dry_run, counters, results,
            )
            for record in errors
        ]
        await asyncio.gather(*tasks)
    finally:
        await engine.dispose()

    click.echo(f"Retried {counters['retried']}/{len(errors)} records.")

    should_report = not no_report and (report_path or config.report.enabled)
    if should_report:
        finished_at = datetime.now(timezone.utc)
        duration = (finished_at - started_at).total_seconds()
        path = (
            Path(report_path) if report_path
            else build_report_path(
                config.report.output_dir, "retry-errors", started_at,
            )
        )
        report = ReportData(
            command="retry-errors",
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            duration_seconds=duration,
            dry_run=dry_run,
            config_summary=build_config_summary(config),
            counters=counters,
            photos=[PhotoResult(**r) for r in results],
            total_photos=len(results),
        )
        written = write_report(report, path)
        click.echo(f"Report: {written}")
