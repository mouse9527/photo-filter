import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photo_filter.config import SourceConfig
from photo_filter.imaging import is_raw_decodable, load_thumbnail
from photo_filter.models import PhotoUnit
from photo_filter.scanner import collect_directories, scan_source
from photo_filter.web import _preview_path


def _write_jpg(path: Path, size=(64, 48)) -> Path:
    Image.new("RGB", size, (200, 100, 50)).save(path, format="JPEG")
    return path


@pytest.fixture
def fake_rawpy(monkeypatch):
    """rawpy 无法凭空生成 DNG，这里替换成返回固定 RGB 数组的假实现。"""
    calls = []

    class _Raw:
        sizes = types.SimpleNamespace(width=4032, height=2268)

        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def postprocess(self, **kwargs):
            calls.append((self.path, kwargs))
            return np.zeros((30, 40, 3), dtype=np.uint8)

    module = types.SimpleNamespace(imread=lambda path: _Raw(path))
    monkeypatch.setitem(sys.modules, "rawpy", module)
    return calls


def test_is_raw_decodable_only_dng():
    assert is_raw_decodable(Path("a.DNG"))
    assert is_raw_decodable(Path("a.dng"))
    assert not is_raw_decodable(Path("a.ARW"))
    assert not is_raw_decodable(Path("a.JPG"))


def test_load_thumbnail_jpg_uses_pillow(tmp_path, fake_rawpy):
    img = load_thumbnail(_write_jpg(tmp_path / "a.JPG"), 1000)
    assert img.size == (64, 48)
    assert fake_rawpy == []


def test_load_thumbnail_jpg_downscales_large_image(tmp_path):
    img = load_thumbnail(_write_jpg(tmp_path / "big.JPG", size=(4000, 3000)), 400)
    assert img.size == (400, 300)
    assert img.mode == "RGB"


def test_load_thumbnail_applies_exif_orientation(tmp_path):
    path = tmp_path / "rotated.JPG"
    exif = Image.Exif()
    exif[0x0112] = 6  # 需要顺时针旋转 90 度
    Image.new("RGB", (800, 400)).save(path, format="JPEG", exif=exif)
    img = load_thumbnail(path, 400)
    assert img.size == (200, 400)  # 竖图


@pytest.mark.parametrize(
    ("max_size", "half_size"),
    [(400, True), (1440, True), (2016, True), (3840, False)],
)
def test_load_thumbnail_dng_uses_rawpy(tmp_path, fake_rawpy, max_size, half_size):
    path = tmp_path / "a.DNG"
    path.write_bytes(b"not-a-real-dng")
    img = load_thumbnail(path, max_size)
    assert img.mode == "RGB"
    # 目标不超过传感器长边一半（4032/2）时才用 half_size，否则会损失清晰度
    assert fake_rawpy[0][1] == {"use_camera_wb": True, "half_size": half_size}


@pytest.mark.parametrize(
    ("jpg", "raw", "expected"),
    [
        ("a.JPG", "a.DNG", "a.JPG"),  # Action4 JPG+DNG：仍然分析 JPG
        ("a.JPG", "a.ARW", "a.JPG"),  # Sony ARW+JPG：行为不变
        (None, "a.DNG", "a.DNG"),  # Mavic 只有 DNG：回退到 DNG
        (None, "a.ARW", None),  # 只有 ARW：不解码，仍然跳过
        (None, None, None),
    ],
)
def test_analysis_path_priority(jpg, raw, expected):
    unit = PhotoUnit(
        stem="a",
        source_dir=Path("/x"),
        camera="c",
        jpg_path=Path(jpg) if jpg else None,
        arw_path=Path(raw) if raw else None,
    )
    assert unit.analysis_path == (Path(expected) if expected else None)


def test_preview_path_matches_analysis_priority():
    assert _preview_path("/p/a.JPG", "/p/a.DNG") == "/p/a.JPG"
    assert _preview_path(None, "/p/a.DNG") == "/p/a.DNG"
    assert _preview_path(None, "/p/a.ARW") is None
    assert _preview_path(None, None) is None


def test_scan_includes_dng_only_and_skips_appledouble(tmp_path):
    shot = tmp_path / "DCIM"
    shot.mkdir()
    _write_jpg(shot / "pair.JPG")
    (shot / "pair.DNG").write_bytes(b"raw")
    (shot / "mavic.DNG").write_bytes(b"raw")
    (shot / "sony.ARW").write_bytes(b"raw")  # 只有 ARW：不应进入分析
    (shot / "._pair.JPG").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 4092)
    (shot / "._mavic 2.DNG").write_bytes(b"\x00\x05\x16\x07" + b"\x00" * 4092)

    source = SourceConfig(
        path=str(tmp_path), camera="dji", extensions=[".JPG", ".DNG", ".ARW"],
        recursive=True,
    )

    assert collect_directories(source) == {str(shot): 4}

    units, counts = scan_source(source)
    by_stem = {u.stem: u for u in units}
    assert set(by_stem) == {"pair", "mavic"}
    assert by_stem["pair"].analysis_path == shot / "pair.JPG"
    assert by_stem["mavic"].analysis_path == shot / "mavic.DNG"
    assert counts == {str(shot): 2}


SAMPLE_DNG = os.environ.get("PHOTO_FILTER_SAMPLE_DNG")


@pytest.mark.skipif(not SAMPLE_DNG, reason="设置 PHOTO_FILTER_SAMPLE_DNG 指向真实 DNG 时运行")
def test_real_dng_decodes_full_resolution():
    img = load_thumbnail(Path(SAMPLE_DNG), 3840)
    # Pillow 直接打开只能拿到内嵌的小缩略图，这里确认拿到的是完整画面
    assert min(img.size) > 1000
    assert img.mode == "RGB"
    small = load_thumbnail(Path(SAMPLE_DNG), 1440)
    assert max(small.size) == 1440
