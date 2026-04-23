from __future__ import annotations

from datetime import date, datetime, timezone

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    String,
    Text,
    UniqueConstraint,
    func,
    select,
)
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class PhotoRecord(Base):
    __tablename__ = "photo_records"
    __table_args__ = (
        UniqueConstraint("file_hash", name="uq_photo_file_hash"),
        Index("idx_photo_records_status", "status"),
        Index("idx_photo_records_source_dir", "source_dir"),
        Index("idx_photo_records_processed_at", "processed_at"),
        Index("idx_photo_records_camera", "camera"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    file_stem: Mapped[str] = mapped_column(String(255), nullable=False)
    source_dir: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_hash: Mapped[str | None] = mapped_column(String(64))
    jpg_path: Mapped[str | None] = mapped_column(String(1024))
    arw_path: Mapped[str | None] = mapped_column(String(1024))
    camera: Mapped[str | None] = mapped_column(String(50))
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    confidence: Mapped[float | None] = mapped_column(Float)
    verdict_reasons: Mapped[str | None] = mapped_column(Text)
    llm_model: Mapped[str | None] = mapped_column(String(100))
    llm_response: Mapped[str | None] = mapped_column(Text)
    file_size_bytes: Mapped[int | None] = mapped_column(BigInteger)
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )


def make_engine(database_url: str):
    return create_async_engine(database_url, echo=False)


def make_session_factory(engine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False)


async def init_db(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def get_processed_hashes(session: AsyncSession) -> set[str]:
    result = await session.execute(
        select(PhotoRecord.file_hash).where(PhotoRecord.file_hash.is_not(None))
    )
    return {row[0] for row in result.all()}


async def get_daily_count(session: AsyncSession, day: date | None = None) -> int:
    day = day or date.today()
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = datetime(day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc)
    result = await session.execute(
        select(func.count(PhotoRecord.id)).where(
            PhotoRecord.processed_at.between(start, end),
            PhotoRecord.status != "pending",
        )
    )
    return result.scalar_one()


async def upsert_record(session: AsyncSession, record: PhotoRecord) -> PhotoRecord:
    existing = None
    if record.file_hash:
        result = await session.execute(
            select(PhotoRecord).where(PhotoRecord.file_hash == record.file_hash)
        )
        existing = result.scalar_one_or_none()
    if existing:
        existing.file_stem = record.file_stem
        existing.source_dir = record.source_dir
        existing.jpg_path = record.jpg_path
        existing.arw_path = record.arw_path
        existing.status = record.status
        existing.confidence = record.confidence
        existing.verdict_reasons = record.verdict_reasons
        existing.llm_model = record.llm_model
        existing.llm_response = record.llm_response
        existing.processed_at = record.processed_at
        existing.updated_at = datetime.now(timezone.utc)
        return existing
    session.add(record)
    return record


async def get_error_records(session: AsyncSession) -> list[PhotoRecord]:
    result = await session.execute(
        select(PhotoRecord).where(PhotoRecord.status == "error")
    )
    return list(result.scalars().all())


async def get_review_photos(
    session: AsyncSession,
    status: str | None = None,
    camera: str | None = None,
    limit: int = 100,
    offset: int = 0,
) -> tuple[list[PhotoRecord], int]:
    base_filter = [
        PhotoRecord.status.in_(["rejected", "review"]),
        ~PhotoRecord.source_dir.contains("@eaDir"),
    ]
    query = select(PhotoRecord).where(*base_filter)
    count_query = select(func.count(PhotoRecord.id)).where(*base_filter)
    if status:
        query = query.where(PhotoRecord.status == status)
        count_query = count_query.where(PhotoRecord.status == status)
    if camera:
        query = query.where(PhotoRecord.camera == camera)
        count_query = count_query.where(PhotoRecord.camera == camera)

    total = (await session.execute(count_query)).scalar_one()
    query = query.order_by(PhotoRecord.processed_at.desc())
    query = query.offset(offset).limit(limit)
    result = await session.execute(query)
    return list(result.scalars().all()), total


async def get_photo_by_id(
    session: AsyncSession, photo_id: int,
) -> PhotoRecord | None:
    return await session.get(PhotoRecord, photo_id)


async def get_cameras(session: AsyncSession) -> list[str]:
    result = await session.execute(
        select(PhotoRecord.camera)
        .where(PhotoRecord.camera.is_not(None))
        .distinct()
    )
    return [row[0] for row in result.all()]


async def get_stats(session: AsyncSession) -> dict[str, int]:
    result = await session.execute(
        select(PhotoRecord.status, func.count(PhotoRecord.id)).group_by(PhotoRecord.status)
    )
    stats = {row[0]: row[1] for row in result.all()}
    stats["total"] = sum(stats.values())
    return stats
