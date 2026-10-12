from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

# 只对这些 RAW 扩展名做像素解码。Pillow 打开 DNG 不报错，但只能读到
# 内嵌的最小缩略图（Mavic 4 Pro 实测 160x90），必须交给 rawpy。
# ARW 等不在此列：Sony 都是 ARW+JPG 成对，分析一直用 JPG。
RAW_DECODE_EXTENSIONS = {".DNG"}


def is_raw_decodable(path: Path) -> bool:
    return path.suffix.upper() in RAW_DECODE_EXTENSIONS


def load_thumbnail(path: Path, max_size: int) -> Image.Image:
    """打开照片，校正方向，缩放到长边不超过 max_size。

    解码阶段就按目标尺寸缩小，避免先解出全尺寸图：A7C2 的 33MP JPEG
    全解约占 100MB，Web 预热时会把 1Gi 的容器撑到 OOM。
    """
    if is_raw_decodable(path):
        import rawpy

        with rawpy.imread(str(path)) as raw:
            # 目标不超过传感器一半时用 half_size，解码量减为 1/4
            half = max_size * 2 <= max(raw.sizes.width, raw.sizes.height)
            rgb = raw.postprocess(use_camera_wb=True, half_size=half)
        # rawpy 输出已按传感器方向旋转，不再做 exif_transpose
        img = Image.fromarray(rgb)
    else:
        with Image.open(path) as src:
            # JPEG 解码时按 1/2、1/4、1/8 缩小；其他格式无副作用
            src.draft("RGB", (max_size, max_size))
            img = ImageOps.exif_transpose(src)
            # exif_transpose 无需旋转时返回原对象，离开 with 前确保像素已读入
            img.load()
    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return img
