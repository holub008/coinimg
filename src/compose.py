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

# Detect at a fixed resolution so edge thresholds and gap closing behave
# consistently across phone photos, without processing full-resolution arrays.
_DETECT_MAX_SIDE = 512
_EDGE_THRESHOLDS = (2.0, 4.0, 8.0, 16.0)


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
    is located by its closed, roughly circular outline, then a square crop
    is taken around its center using the maximum radius (so chips and oblong
    flans still fit).
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
    """Return (cx, cy, enclosing radius) in original-image coordinates.

    Look for enclosed, compact shapes at several edge strengths. RGB gradients
    respond to either contrast polarity and to colored coins with similar
    luminance to the paper. Broad lighting gradients do not form closed rims;
    frame-touching shadows and elongated objects are rejected by shape.
    """
    small = image.convert("RGB")
    small.thumbnail((_DETECT_MAX_SIDE, _DETECT_MAX_SIDE), Image.Resampling.LANCZOS)
    if min(small.size) < 3:
        raise ValueError("could not find a coin: image is too small")
    rgb = np.asarray(small.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32)
    dy, dx = np.gradient(rgb, axis=(0, 1))
    strength = np.max(np.hypot(dx, dy), axis=2)

    best: tuple[float, float, float] | None = None
    best_score = 0.0
    for threshold in _EDGE_THRESHOLDS:
        # Bridge small rim gaps, fill enclosed interiors (including bright
        # highlights), then undo the dilation to retain the flan's dimensions.
        edges = Image.fromarray((strength > threshold).astype(np.uint8) * 255)
        thick = np.asarray(edges.filter(ImageFilter.MaxFilter(3))) > 0
        labels, _ = _label(~thick)
        border = np.unique(
            np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1]))
        )
        enclosed = thick | ~np.isin(labels, border)
        filled = Image.fromarray(enclosed.astype(np.uint8) * 255)
        mask = np.asarray(filled.filter(ImageFilter.MinFilter(3))) > 0
        labels, counts = _label(mask)

        for label, area in counts.items():
            if area < 40 or area > mask.size * 0.8:
                continue
            ys, xs = np.nonzero(labels == label)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            if x0 == 0 or y0 == 0 or x1 == small.width - 1 or y1 == small.height - 1:
                continue
            width, height = x1 - x0 + 1, y1 - y0 + 1
            aspect = min(width, height) / max(width, height)
            extent = area / (width * height)
            # Allow oval/chipped flans, but exclude lines, rectangles and
            # sparse fragments of lettering or paper texture.
            if aspect < 0.6 or not 0.55 <= extent <= 0.9:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            radius = float(np.hypot(xs - cx, ys - cy).max())
            compactness = area / (np.pi * radius * radius)
            if compactness < 0.45:
                continue
            score = area * compactness * aspect
            if score > best_score:
                best_score = score
                best = cx, cy, radius

    if best is None:
        raise ValueError("could not find a coin: no closed, roughly circular outline")
    cx, cy, radius = best
    sx, sy = image.width / small.width, image.height / small.height
    # Map pixel centers, allowing half a detection pixel for the outer edge.
    return (cx + 0.5) * sx - 0.5, (cy + 0.5) * sy - 0.5, (radius + 0.5) * max(sx, sy)


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
