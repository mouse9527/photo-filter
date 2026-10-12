from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

# 只对这些 RAW 扩展名做像素解码。Pillow 打开 DNG 不报错，但只能读到
# 内嵌的最小缩略图（Mavic 4 Pro 实测 160x90），必须交给 rawpy。
# ARW 等不在此列：Sony 都是 ARW+JPG 成对，分析一直用 JPG。
RAW_DECODE_EXTENSIONS = {".DNG"}


def is_raw_decodable(path: Path) -> bool:
    return path.suffix.upper() in RAW_DECODE_EXTENSIONS


def open_image(path: Path) -> Image.Image:
    """打开照片并返回已校正方向的图像。

    JPEG 走 Pillow 并应用 EXIF 方向；DNG 用 rawpy 完整解码，
    rawpy 输出已按传感器方向旋转，不再做 exif_transpose。
    """
    if is_raw_decodable(path):
        import rawpy

        with rawpy.imread(str(path)) as raw:
            rgb = raw.postprocess(use_camera_wb=True)
        return Image.fromarray(rgb)

    with Image.open(path) as img:
        # exif_transpose 在无需旋转时返回原对象，copy 一份避免引用已关闭的文件
        return ImageOps.exif_transpose(img).copy()
