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
@click.pass_context
def scan(ctx: click.Context, dry_run: bool, limit: int | None) -> None:
    """Scan photo directories, analyze with LLM, and reject bad photos."""
    asyncio.run(_scan(ctx.obj["config"], dry_run, limit))


async def _scan(config, dry_run: bool, limit: int | None) -> None:
    from photo_filter.analyzer import analyze_photo, make_client
    from photo_filter.db import (
        PhotoRecord,
        get_daily_count,
        get_processed_stems,
        init_db,
        make_engine,
        make_session_factory,
        upsert_record,
    )
    from photo_filter.models import Verdict
    from photo_filter.mover import reject_photo
    from photo_filter.scanner import filter_unprocessed, scan_source

    engine = make_engine(config.database.url)
    await init_db(engine)
    session_factory = make_session_factory(engine)

    client = make_client(config)
    daily_max = limit if limit is not None else config.quota.daily_max

    total_analyzed = 0
    total_rejected = 0
    total_kept = 0
    total_errors = 0

    try:
        for source in config.sources:
            logger.info("scanning_source", path=source.path, camera=source.camera)
            units = scan_source(source)

            if not units:
                logger.info("no_photos_found", source=source.path)
                continue

            async with session_factory() as session:
                processed = {}
                dirs = {str(u.source_dir) for u in units}
                for d in dirs:
                    processed[d] = await get_processed_stems(session, d)

                unprocessed = filter_unprocessed(units, processed)
                logger.info(
                    "unprocessed_photos",
                    source=source.path,
                    total=len(units),
                    unprocessed=len(unprocessed),
                )

                daily_count = await get_daily_count(session)
                remaining = max(0, daily_max - daily_count)

                if remaining == 0:
                    logger.info("daily_quota_reached", daily_max=daily_max)
                    break

                batch = unprocessed[:remaining]
                logger.info("processing_batch", count=len(batch), remaining=remaining)

                for unit in batch:
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
                            total_rejected += 1
                        elif result.verdict == Verdict.REVIEW:
                            status = "review"
                            total_kept += 1
                        else:
                            status = "kept"
                            total_kept += 1

                        record = PhotoRecord(
                            file_stem=unit.stem,
                            source_dir=str(unit.source_dir),
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
                        await upsert_record(session, record)
                        await session.commit()
                        total_analyzed += 1

                        logger.info(
                            "photo_processed",
                            stem=unit.stem,
                            status=status,
                            confidence=result.confidence,
                            dry_run=dry_run,
                        )

                    except Exception:
                        logger.exception("analysis_failed", stem=unit.stem)
                        record = PhotoRecord(
                            file_stem=unit.stem,
                            source_dir=str(unit.source_dir),
                            jpg_path=str(unit.jpg_path) if unit.jpg_path else None,
                            arw_path=str(unit.arw_path) if unit.arw_path else None,
                            camera=unit.camera,
                            status="error",
                            processed_at=datetime.now(timezone.utc),
                        )
                        await upsert_record(session, record)
                        await session.commit()
                        total_errors += 1
    finally:
        await engine.dispose()

    logger.info(
        "scan_complete",
        total_analyzed=total_analyzed,
        rejected=total_rejected,
        kept=total_kept,
        errors=total_errors,
        dry_run=dry_run,
    )
    click.echo(
        f"Done: {total_analyzed} analyzed, {total_rejected} rejected, "
        f"{total_kept} kept, {total_errors} errors"
    )


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
@click.pass_context
def retry_errors(ctx: click.Context, dry_run: bool) -> None:
    """Reprocess photos that previously failed."""
    asyncio.run(_retry_errors(ctx.obj["config"], dry_run))


async def _retry_errors(config, dry_run: bool) -> None:
    from photo_filter.analyzer import analyze_photo, make_client
    from photo_filter.db import (
        get_error_records,
        make_engine,
        make_session_factory,
    )
    from photo_filter.models import PhotoUnit, Verdict
    from photo_filter.mover import reject_photo

    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)
    client = make_client(config)

    retried = 0
    try:
        async with session_factory() as session:
            errors = await get_error_records(session)
            if not errors:
                click.echo("No error records to retry.")
                return

            click.echo(f"Retrying {len(errors)} error records...")
            for record in errors:
                unit = PhotoUnit(
                    stem=record.file_stem,
                    source_dir=Path(record.source_dir),
                    camera=record.camera or "",
                    jpg_path=Path(record.jpg_path) if record.jpg_path else None,
                    arw_path=Path(record.arw_path) if record.arw_path else None,
                )
                if unit.analysis_path is None or not unit.analysis_path.exists():
                    logger.warning("retry_skip_no_file", stem=unit.stem)
                    continue

                try:
                    result = await analyze_photo(unit, client, config)
                    now = datetime.now(timezone.utc)
                    should_reject = (
                        result.verdict == Verdict.REJECT
                        and result.confidence >= config.llm.confidence_threshold
                    )
                    if should_reject:
                        record.status = "rejected"
                        if not dry_run:
                            reject_photo(unit)
                    elif result.verdict == Verdict.REVIEW:
                        record.status = "review"
                    else:
                        record.status = "kept"

                    record.confidence = result.confidence
                    record.verdict_reasons = json.dumps(result.reasons)
                    record.llm_model = config.llm.model
                    record.llm_response = result.raw_response
                    record.processed_at = now
                    await session.commit()
                    retried += 1
                except Exception:
                    logger.exception("retry_failed", stem=unit.stem)
    finally:
        await engine.dispose()

    click.echo(f"Retried {retried}/{len(errors)} records.")
