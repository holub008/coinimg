"""Concatenate obverse and reverse coin photos into a single image."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image, ImageFilter, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

type Rotation = Literal[0, 90, 180, 270]
type ImageInput = str | Path | Image.Image

# Clockwise degrees -> PIL transpose (PIL rotates counter-clockwise).
_CW_TRANSPOSE: dict[int, Image.Transpose | None] = {
    0: None,
    90: Image.Transpose.ROTATE_270,
    180: Image.Transpose.ROTATE_180,
    270: Image.Transpose.ROTATE_90,
}

# The coin is the darkest object; this percentile of the full frame locates it
# even when the paper has a lighting gradient or shadow.
_CORE_PERCENTILE = 1.5
_DETECT_MAX_SIDE = 384
_LOCAL_DARKNESS = 0.85


def concatenate(
    obverse: ImageInput,
    reverse: ImageInput,
    *,
    margin: float = 0.08,
    rotate: Rotation = 0,
    size: int | None = None,
) -> Image.Image:
    """Place the obverse on the left and reverse on the right, equally sized.

    Each photo is assumed to be one coin on a flat white background. The coin
    is located as the darkest region, then a square crop is taken around its
    center using the maximum radius (so chips and oblong flans still fit).
    Both crops are scaled to the same pixel size so the two faces match, then
    joined 50/50 horizontally.

    Args:
        obverse: Left-hand (obverse) image path or PIL image. First image.
        reverse: Right-hand (reverse) image path or PIL image.
        margin: Extra white space on each side, as a fraction of the coin's
            max diameter. ``0.08`` is 8% of the diameter on every side.
        rotate: Clockwise rotation applied to both input images, in degrees.
        size: Pixel width/height of each half of the output. Defaults to the
            larger of the two square crops (preserves the sharper photo).
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if size is not None and size <= 0:
        raise ValueError(f"size must be a positive pixel count, got {size}")

    left = _coin_square(_load(obverse, rotate), margin=margin)
    right = _coin_square(_load(reverse, rotate), margin=margin)

    panel = size if size is not None else max(left.size[0], right.size[0])
    left = left.resize((panel, panel), Image.Resampling.LANCZOS)
    right = right.resize((panel, panel), Image.Resampling.LANCZOS)

    out = Image.new("RGB", (panel * 2, panel), (255, 255, 255))
    out.paste(left, (0, 0))
    out.paste(right, (panel, 0))
    return out


def _load(source: ImageInput, rotate: Rotation) -> Image.Image:
    if rotate not in _CW_TRANSPOSE:
        raise ValueError(f"rotate must be 0, 90, 180, or 270; got {rotate}")
    if isinstance(source, Image.Image):
        image = source
    else:
        with Image.open(source) as opened:
            image = opened.copy()
    image = ImageOps.exif_transpose(image) or image
    if image.mode != "RGB":
        image = image.convert("RGB")
    transpose = _CW_TRANSPOSE[rotate]
    if transpose is not None:
        image = image.transpose(transpose)
    return image


def _coin_square(image: Image.Image, *, margin: float) -> Image.Image:
    cx, cy, radius = _detect_coin(image)
    half = radius * (1.0 + 2.0 * margin)
    side = max(int(np.ceil(2.0 * half)), 1)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))

    src_left = int(np.floor(cx - half))
    src_top = int(np.floor(cy - half))
    crop = image.crop(
        (
            max(src_left, 0),
            max(src_top, 0),
            min(src_left + side, image.width),
            min(src_top + side, image.height),
        )
    )
    canvas.paste(crop, (max(-src_left, 0), max(-src_top, 0)))
    return canvas


def _detect_coin(image: Image.Image) -> tuple[float, float, float]:
    """Return (cx, cy, radius) of the coin in the original image."""
    gray = image.convert("L").filter(ImageFilter.GaussianBlur(radius=1))
    full = np.asarray(gray, dtype=np.float32)
    scale = max(full.shape) / _DETECT_MAX_SIDE
    if scale > 1:
        small_img = gray.resize(
            (max(int(image.width / scale), 1), max(int(image.height / scale), 1)),
            Image.Resampling.BILINEAR,
        )
        small = np.asarray(small_img, dtype=np.float32)
    else:
        scale = 1.0
        small = full

    core = small <= np.percentile(small, _CORE_PERCENTILE)
    if not np.any(core):
        raise ValueError("could not find a coin: image has no dark pixels")
    core = _largest_component(core)
    ys, xs = np.nonzero(core)
    cx = float(xs.mean())
    cy = float(ys.mean())
    radius = float(np.max(np.hypot(xs - cx, ys - cy)))
    if radius < 1:
        raise ValueError("could not find a coin: detected region is too small")

    # Re-threshold in a window around the dark core so the full flan is
    # included (worn silver rims) without picking up distant paper shadows.
    window = max(int(np.ceil(radius * 2.5)), 4)
    x0 = max(int(cx - window), 0)
    y0 = max(int(cy - window), 0)
    x1 = min(int(cx + window) + 1, small.shape[1])
    y1 = min(int(cy + window) + 1, small.shape[0])
    local = small[y0:y1, x0:x1]
    background = float(np.percentile(local, 90))
    if background <= 0:
        raise ValueError("image is too dark to separate a coin from the background")
    local_mask = local < background * _LOCAL_DARKNESS
    if np.any(local_mask):
        local_mask = _component_containing(
            local_mask, int(cy) - y0, int(cx) - x0
        )
        ys, xs = np.nonzero(local_mask)
        cx = float(xs.mean()) + x0
        cy = float(ys.mean()) + y0
        radius = float(np.max(np.hypot(xs - (cx - x0), ys - (cy - y0))))

    return cx * scale, cy * scale, radius * scale


def _largest_component(mask: np.ndarray) -> np.ndarray:
    labels, counts = _label(mask)
    if not counts:
        return mask
    best = max(counts, key=lambda label: counts[label])
    return labels == best


def _component_containing(mask: np.ndarray, row: int, col: int) -> np.ndarray:
    labels, _counts = _label(mask)
    row = min(max(row, 0), mask.shape[0] - 1)
    col = min(max(col, 0), mask.shape[1] - 1)
    seed = int(labels[row, col])
    if seed == 0:
        return _largest_component(mask)
    return labels == seed


def _label(mask: np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    """4-connected component labels (1..) and pixel counts."""
    h, w = mask.shape
    labels = np.zeros(mask.shape, dtype=np.int32)
    counts: dict[int, int] = {}
    next_label = 0
    ys, xs = np.nonzero(mask)
    for y, x in zip(ys.tolist(), xs.tolist(), strict=True):
        if labels[y, x] != 0:
            continue
        next_label += 1
        count = 0
        queue: deque[tuple[int, int]] = deque([(y, x)])
        labels[y, x] = next_label
        while queue:
            cy, cx = queue.popleft()
            count += 1
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and labels[ny, nx] == 0:
                    labels[ny, nx] = next_label
                    queue.append((ny, nx))
        counts[next_label] = count
    return labels, counts
