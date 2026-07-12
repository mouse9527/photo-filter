from fastapi.testclient import TestClient
from starlette.routing import Match

from photo_filter.config import AppConfig
from photo_filter.web import create_app


def _first_full_match(app, path: str, method: str = "POST") -> str | None:
    scope = {"type": "http", "path": path, "method": method}
    for route in app.routes:
        match, _ = route.matches(scope)
        if match is Match.FULL:
            return route.name
    return None


def test_batch_routes_are_not_shadowed_by_photo_id(tmp_path, monkeypatch):
    monkeypatch.setattr("photo_filter.web.CACHE_DIR", tmp_path)
    app = create_app(AppConfig())
    client = TestClient(app)

    delete_response = client.post("/api/photos/batch/delete", json={"ids": []})
    undo_response = client.post("/api/photos/batch/undo", json={"ids": []})

    assert delete_response.status_code == 200
    assert delete_response.json() == {"ok": 0, "failed": 0}
    assert undo_response.status_code == 200
    assert undo_response.json() == {"ok": 0, "failed": 0}


def test_numeric_photo_routes_still_match(tmp_path, monkeypatch):
    monkeypatch.setattr("photo_filter.web.CACHE_DIR", tmp_path)
    app = create_app(AppConfig())

    assert _first_full_match(app, "/api/photos/123/delete") == "delete_photo_endpoint"
    assert _first_full_match(app, "/api/photos/123/undo") == "undo_photo"
