"""Deterministic image-tiling geometry + cross-tile detection merging for
EXP-0013 (image tiling for small/distant Person recovery). Experimental
benchmark infrastructure ONLY -- never imported by benchmark/model.py's
production predict() path, never touches benchmark/config.py or
benchmark/results/baseline/.

MECHANISM (verified against the actual pipeline, not assumed -- see
research/_exp0013_preregister.py section 1 for the full writeup):
ultralytics' model.predict(imgsz=640, ...) applies a LetterBox transform
(ultralytics/data/augment.py::LetterBox, confirmed by reading that source
directly) to whatever source image it is given -- full frame or a crop --
resizing it to fit within 640x640 while preserving aspect ratio (padding
the short side). All 380 eval images are larger than 640x640 on at least
one axis (verified via PIL: min 311x446, max 1024x1024, median 1024x768,
zero images with both dimensions <=640) -- so the full-image pass already
downscales every object, including small/distant Person instances, before
the network ever sees it. A CROP containing that same Person instance,
letterboxed to the SAME fixed 640x640 input, presents it at a LARGER
effective scale (less information lost to downscaling), because tiling
never changes the network's own input resolution -- only what portion of
the original image occupies that fixed frame. This is the key mechanistic
difference from EXP-0002 (which changed imgsz itself, 640->960/1280, and
made Person recall WORSE: 0.211->0.165 -- a likely out-of-distribution
scale mismatch for a model calibrated at 640). Tiling keeps imgsz=640
fixed throughout.

Deterministic tile geometry: given (img_w, img_h), a fixed (n_cols, n_rows)
grid, and a fixed overlap_fraction, tiles are generated in a fixed,
reproducible order (full image first if included, then row-major:
row 0 left-to-right, row 1 left-to-right, ...), each defined by pixel
bounds clipped to the image. The PRIMARY, preregistered configuration
additionally includes the FULL, unmodified image as tile 0 -- a standard
"slicing aided hyper inference" (SAHI-style) design that catches large or
boundary-crossing objects a small tile alone might split, and gives a
direct way to measure boundary-splitting effects.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from benchmark.metrics import Detection, iou_xywh


@dataclass(frozen=True)
class Tile:
    tile_id: int
    x0: int  # pixel bounds within the original image, [x0, x1) x [y0, y1)
    y0: int
    x1: int
    y1: int
    is_full_image: bool = False

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


def generate_tiles(
    img_w: int, img_h: int, n_cols: int, n_rows: int, overlap_fraction: float, include_full_image: bool = True,
) -> list:
    """Deterministic tile layout for one image. Row-major order (full image
    first, if included). Tiles are clipped to [0, img_w] x [0, img_h] -- an
    "edge tile" near the image boundary is simply narrower/shorter than the
    nominal tile size, never extends past the image, and is never dropped
    (a valid, if slightly smaller, tile at every grid position)."""
    if img_w <= 0 or img_h <= 0:
        raise ValueError(f"img_w/img_h must be positive, got {img_w}x{img_h}")
    if n_cols < 1 or n_rows < 1:
        raise ValueError(f"n_cols/n_rows must be >= 1, got {n_cols}x{n_rows}")
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError(f"overlap_fraction must be in [0, 1), got {overlap_fraction}")

    tiles: list = []
    next_id = 0
    if include_full_image:
        tiles.append(Tile(tile_id=next_id, x0=0, y0=0, x1=img_w, y1=img_h, is_full_image=True))
        next_id += 1

    tile_w = img_w / (n_cols - overlap_fraction * (n_cols - 1)) if n_cols > 1 else float(img_w)
    tile_h = img_h / (n_rows - overlap_fraction * (n_rows - 1)) if n_rows > 1 else float(img_h)
    stride_w = tile_w * (1 - overlap_fraction) if n_cols > 1 else float(img_w)
    stride_h = tile_h * (1 - overlap_fraction) if n_rows > 1 else float(img_h)

    for row in range(n_rows):
        for col in range(n_cols):
            x0 = max(0, min(img_w, round(col * stride_w)))
            y0 = max(0, min(img_h, round(row * stride_h)))
            x1 = max(0, min(img_w, round(col * stride_w + tile_w)))
            y1 = max(0, min(img_h, round(row * stride_h + tile_h)))
            if x1 <= x0 or y1 <= y0:
                continue  # degenerate (shouldn't occur for valid img_w/h > 0), skip defensively
            tiles.append(Tile(tile_id=next_id, x0=x0, y0=y0, x1=x1, y1=y1, is_full_image=False))
            next_id += 1
    return tiles


def remap_bbox_tile_to_full(bbox_tile_norm: tuple, tile: Tile, img_w: int, img_h: int) -> tuple:
    """A prediction's bbox, as returned by the model for a TILE (normalized
    [0,1] relative to that tile's OWN width/height), remapped to
    full-image-normalized [x, y, w, h] (top-left origin), clipped to the
    original image bounds. Confidence and class are untouched by the
    caller -- this function only ever touches geometry."""
    px, py, pw, ph = bbox_tile_norm
    x_full_px = tile.x0 + px * tile.width
    y_full_px = tile.y0 + py * tile.height
    x1_full_px = x_full_px + pw * tile.width
    y1_full_px = y_full_px + ph * tile.height

    # Clip to the original image's pixel bounds.
    x_full_px = max(0.0, min(float(img_w), x_full_px))
    y_full_px = max(0.0, min(float(img_h), y_full_px))
    x1_full_px = max(0.0, min(float(img_w), x1_full_px))
    y1_full_px = max(0.0, min(float(img_h), y1_full_px))

    w_full_px = max(0.0, x1_full_px - x_full_px)
    h_full_px = max(0.0, y1_full_px - y_full_px)

    return (x_full_px / img_w, y_full_px / img_h, w_full_px / img_w, h_full_px / img_h)


def merge_detections_nms(detections: list, iou_threshold: float) -> list:
    """Cross-tile duplicate merge: standard greedy, confidence-descending,
    SAME-CLASS-ONLY NMS (mirrors the model's own internal NMS algorithm,
    applied here across detections gathered from multiple separate
    tile/full-image inference passes, which ultralytics' own per-call NMS
    cannot see across). Detections of DIFFERENT classes never suppress each
    other, matching standard per-class NMS semantics and this lab's own
    evaluation convention (benchmark/metrics.py matches per class).
    Deterministic output order: sorted by (class_name, confidence
    descending) so repeated runs on identical input always produce an
    identical sequence, independent of dict/set iteration order."""
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    by_class: dict = defaultdict(list)
    for d in detections:
        by_class[d.class_name].append(d)

    merged: list = []
    for dets in by_class.values():
        dets_sorted = sorted(dets, key=lambda d: -d.confidence)
        kept: list = []
        for d in dets_sorted:
            if not any(iou_xywh(d.bbox, k.bbox) >= iou_threshold for k in kept):
                kept.append(d)
        merged.extend(kept)

    merged.sort(key=lambda d: (d.class_name, -d.confidence))
    return merged
