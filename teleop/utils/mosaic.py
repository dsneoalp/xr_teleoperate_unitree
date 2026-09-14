"""Compose teleimager RGB tiles onto a reused mosaic canvas.

`contain` aspect-fits with letterbox; `fill` stretches; `cover` center-crops.
Only dirty tiles are written; the canvas is not cleared every frame.
When the crop already matches the tile, this is a memcpy (no cv2.resize).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import yaml

_FIT_MODES = ("contain", "fill", "cover")


@dataclass(frozen=True)
class MosaicTile:
    slot: str
    x: int
    y: int
    w: int
    h: int
    fit: str = "contain"


@dataclass(frozen=True)
class MosaicLayout:
    width: int
    height: int
    tiles: tuple[MosaicTile, ...]

    @property
    def slots(self) -> tuple[str, ...]:
        return tuple(tile.slot for tile in self.tiles)


def load_mosaic_layout(path: str) -> MosaicLayout:
    """Load canvas + tiles. All extents must be even and on-canvas."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    canvas = raw.get("canvas") or {}
    width = int(canvas.get("width") or 0)
    height = int(canvas.get("height") or 0)
    if width <= 0 or height <= 0 or (width | height) & 1:
        raise ValueError(f"mosaic canvas must be positive even WxH, got {width}x{height}")
    tiles: list[MosaicTile] = []
    for item in raw.get("tiles") or []:
        fit = str(item.get("fit") or "contain").strip().lower()
        if fit not in _FIT_MODES:
            raise ValueError(f"mosaic tile '{item.get('slot')}' fit must be contain|fill|cover")
        tile = MosaicTile(
            slot=str(item["slot"]),
            x=int(item["x"]),
            y=int(item["y"]),
            w=int(item["w"]),
            h=int(item["h"]),
            fit=fit,
        )
        if (tile.x | tile.y | tile.w | tile.h) & 1:
            raise ValueError(f"mosaic tile '{tile.slot}' extents must be even: {tile}")
        if tile.w <= 0 or tile.h <= 0:
            raise ValueError(f"mosaic tile '{tile.slot}' has empty size")
        if tile.x < 0 or tile.y < 0:
            raise ValueError(f"mosaic tile '{tile.slot}' origin is negative")
        if tile.x + tile.w > width or tile.y + tile.h > height:
            raise ValueError(f"mosaic tile '{tile.slot}' overflows {width}x{height}")
        tiles.append(tile)
    if not tiles:
        raise ValueError("mosaic.yaml has no tiles")
    return MosaicLayout(width=width, height=height, tiles=tuple(tiles))


def fitted_size(src_w: int, src_h: int, tile_w: int, tile_h: int) -> tuple[int, int]:
    """Largest even size that fits in the tile without stretching."""
    scale = min(tile_w / src_w, tile_h / src_h)
    out_w = max(2, int(src_w * scale) & ~1)
    out_h = max(2, int(src_h * scale) & ~1)
    return out_w, out_h


def cover_crop(rgb: np.ndarray, tile_w: int, tile_h: int) -> np.ndarray:
    """Center-crop `rgb` to the tile aspect. View when possible, no copy."""
    src_h, src_w = rgb.shape[:2]
    src_aspect = src_w / src_h
    tile_aspect = tile_w / tile_h
    if src_aspect > tile_aspect:
        crop_h = src_h
        crop_w = max(2, int(src_h * tile_aspect) & ~1)
        crop_w = min(crop_w, src_w & ~1)
    else:
        crop_w = src_w
        crop_h = max(2, int(src_w / tile_aspect) & ~1)
        crop_h = min(crop_h, src_h & ~1)
    cx = (src_w - crop_w) // 2
    cy = (src_h - crop_h) // 2
    return rgb[cy : cy + crop_h, cx : cx + crop_w]


class MosaicCompositor:
    """Reuse one RGB canvas. Resize only tiles whose source frame changed."""

    def __init__(self, layout: MosaicLayout):
        self.layout = layout
        self.canvas = np.zeros((layout.height, layout.width, 3), dtype=np.uint8)
        self._by_slot = {tile.slot: tile for tile in layout.tiles}
        self._scratch: dict[str, np.ndarray] = {}
        self._letterboxed: set[str] = set()

    def paste(self, slot: str, rgb: np.ndarray) -> None:
        """Fit `rgb` (H,W,3 uint8) into the slot tile. Does not copy `rgb`."""
        tile = self._by_slot[slot]
        src = rgb
        if tile.fit == "cover":
            src = cover_crop(rgb, tile.w, tile.h)
            out_w, out_h = tile.w, tile.h
            ox, oy = tile.x, tile.y
        elif tile.fit == "fill":
            out_w, out_h = tile.w, tile.h
            ox, oy = tile.x, tile.y
        else:
            src_h, src_w = rgb.shape[:2]
            out_w, out_h = fitted_size(src_w, src_h, tile.w, tile.h)
            ox = tile.x + (tile.w - out_w) // 2
            oy = tile.y + (tile.h - out_h) // 2
            needs_bar = out_w != tile.w or out_h != tile.h
            if needs_bar and slot not in self._letterboxed:
                self.canvas[tile.y : tile.y + tile.h, tile.x : tile.x + tile.w] = 0
                self._letterboxed.add(slot)
        dest = self.canvas[oy : oy + out_h, ox : ox + out_w]
        if src.shape[0] == out_h and src.shape[1] == out_w:
            dest[:] = src
            return
        # INTER_LINEAR for up and down: INTER_AREA on three 720p tiles misses 30 Hz.
        if dest.flags["C_CONTIGUOUS"]:
            cv2.resize(src, (out_w, out_h), dst=dest, interpolation=cv2.INTER_LINEAR)
            return
        scratch = self._scratch.get(slot)
        if scratch is None or scratch.shape[:2] != (out_h, out_w):
            scratch = np.empty((out_h, out_w, 3), dtype=np.uint8)
            self._scratch[slot] = scratch
        cv2.resize(src, (out_w, out_h), dst=scratch, interpolation=cv2.INTER_LINEAR)
        dest[:] = scratch
