from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse, Response
from PIL import Image

from photo_filter.config import AppConfig
from photo_filter.db import (
    get_cameras,
    get_photo_by_id,
    get_review_photos,
    make_engine,
    make_session_factory,
)
from photo_filter.models import PhotoUnit
from photo_filter.mover import delete_photo, undo_rejection

logger = structlog.get_logger()

STATIC_DIR = Path(__file__).parent / "static"


def create_app(config: AppConfig) -> FastAPI:
    app = FastAPI(title="Photo Filter Review")
    engine = make_engine(config.database.url)
    session_factory = make_session_factory(engine)
    photo_dirs = config.web.photo_dirs

    def _resolve_photo_path(jpg_path: str | None) -> Path | None:
        if not jpg_path:
            return None
        p = Path(jpg_path)
        if p.exists():
            return p
        rejected = p.parent / "_rejected" / p.name
        if rejected.exists():
            return rejected
        return None

    def _is_path_allowed(file_path: Path) -> bool:
        resolved = file_path.resolve()
        return any(
            resolved.is_relative_to(Path(d).resolve())
            for d in photo_dirs
        )

    @app.get("/", response_class=HTMLResponse)
    async def index():
        html_path = STATIC_DIR / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.get("/api/photos")
    async def list_photos(
        status: str | None = Query(None),
        camera: str | None = Query(None),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        async with session_factory() as session:
            photos, total = await get_review_photos(
                session, status=status, camera=camera,
                limit=limit, offset=offset,
            )
        items = []
        for p in photos:
            reasons = []
            if p.verdict_reasons:
                try:
                    reasons = json.loads(p.verdict_reasons)
                except (json.JSONDecodeError, TypeError):
                    reasons = [p.verdict_reasons]
            items.append({
                "id": p.id,
                "file_stem": p.file_stem,
                "source_dir": p.source_dir,
                "jpg_path": p.jpg_path,
                "arw_path": p.arw_path,
                "camera": p.camera,
                "status": p.status,
                "confidence": p.confidence,
                "reasons": reasons,
                "processed_at": (
                    p.processed_at.isoformat() if p.processed_at else None
                ),
            })
        return {"items": items, "total": total}

    @app.get("/api/cameras")
    async def list_cameras():
        async with session_factory() as session:
            cameras = await get_cameras(session)
        return {"cameras": cameras}

    def _compress_image(
        file_path: Path, max_size: int, quality: int,
    ) -> bytes:
        with Image.open(file_path) as img:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            return buf.getvalue()

    @app.get("/photos/{file_path:path}")
    async def serve_photo(
        file_path: str,
        w: int = Query(1200, ge=100, le=3840),
        q: int = Query(80, ge=10, le=100),
    ):
        p = Path("/") / file_path
        if not _is_path_allowed(p):
            raise HTTPException(403, "Access denied")
        actual = _resolve_photo_path(str(p))
        if actual is None or not actual.exists():
            raise HTTPException(404, "Photo not found")
        data = _compress_image(actual, max_size=w, quality=q)
        return Response(
            content=data,
            media_type="image/jpeg",
            headers={"Cache-Control": "public, max-age=86400"},
        )

    @app.post("/api/photos/{photo_id}/undo")
    async def undo_photo(photo_id: int):
        async with session_factory() as session:
            record = await get_photo_by_id(session, photo_id)
            if not record:
                raise HTTPException(404, "Record not found")
            if record.status not in ("rejected", "review"):
                raise HTTPException(
                    400, f"Cannot undo photo with status '{record.status}'",
                )

            unit = PhotoUnit(
                stem=record.file_stem,
                source_dir=Path(record.source_dir),
                camera=record.camera or "",
                jpg_path=(
                    Path(record.jpg_path) if record.jpg_path else None
                ),
                arw_path=(
                    Path(record.arw_path) if record.arw_path else None
                ),
            )
            restored = undo_rejection(unit)
            record.status = "kept"
            record.updated_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "photo_undo", photo_id=photo_id, stem=record.file_stem,
            restored=len(restored),
        )
        return {"status": "ok", "restored": len(restored)}

    @app.post("/api/photos/{photo_id}/delete")
    async def delete_photo_endpoint(photo_id: int):
        async with session_factory() as session:
            record = await get_photo_by_id(session, photo_id)
            if not record:
                raise HTTPException(404, "Record not found")
            if record.status == "deleted":
                raise HTTPException(400, "Already deleted")

            unit = PhotoUnit(
                stem=record.file_stem,
                source_dir=Path(record.source_dir),
                camera=record.camera or "",
                jpg_path=(
                    Path(record.jpg_path) if record.jpg_path else None
                ),
                arw_path=(
                    Path(record.arw_path) if record.arw_path else None
                ),
            )
            moved = delete_photo(unit, photo_dirs)
            record.status = "deleted"
            record.updated_at = datetime.now(timezone.utc)
            await session.commit()

        logger.info(
            "photo_deleted", photo_id=photo_id, stem=record.file_stem,
            recycled=len(moved),
        )
        return {"status": "ok", "recycled": len(moved)}

    return app
