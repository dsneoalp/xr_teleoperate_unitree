"""Latest-frame RGB reader for the sag-teleimager BGR ZMQ publisher.

Same path as SAG_Robot_Service/services/portal_robot/bgr_source.py: SUB to
zmq_port+1000 (raw BGR), not ImageClient's JPEG stream on zmq_port.
"""
from __future__ import annotations

import json
import struct
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np
import zmq

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

BGR_ZMQ_PORT_OFFSET = 1000


def fetch_teleimager_config(host: str, port: int, timeout_ms: int = 2000) -> Dict[str, Any]:
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(f"tcp://{host}:{port}")
        sock.send(b"GET_DATA")
        return json.loads(sock.recv().decode("utf-8"))
    finally:
        sock.close(linger=0)


def resolve_bgr_zmq_port(config: Dict[str, Any], camera_slot: str = "head_camera") -> int:
    slot = config.get(camera_slot)
    if not isinstance(slot, dict):
        raise RuntimeError(f"teleimager config missing {camera_slot}")
    if not slot.get("enable_bgr_zmq"):
        raise RuntimeError(
            f"teleimager {camera_slot} has enable_bgr_zmq=false "
            "(set TELEIMAGER_ENABLE_BGR_ZMQ=1 and rescan)"
        )
    zmq_port = int(slot.get("zmq_port") or 0)
    if zmq_port <= 0:
        raise RuntimeError(f"teleimager {camera_slot} has no zmq_port")
    return zmq_port + BGR_ZMQ_PORT_OFFSET


def parse_bgr_payload(payload: bytes) -> Tuple[int, int, Optional[int], bytes]:
    """Parse a teleimager BGR frame: 4-byte legacy or 12-byte with capture_ms."""
    if len(payload) < 4:
        raise ValueError("BGR ZMQ payload too short")
    height, width = struct.unpack("<HH", payload[:4])
    frame_bytes = height * width * 3
    if frame_bytes <= 0:
        raise ValueError("invalid BGR dimensions")
    if len(payload) == 4 + frame_bytes:
        return height, width, None, payload[4:]
    if len(payload) == 12 + frame_bytes:
        (capture_ms,) = struct.unpack("<Q", payload[4:12])
        return height, width, capture_ms, payload[12:]
    raise ValueError("BGR ZMQ payload length mismatch")


def _decimated(size: int, step: int) -> int:
    return (-(-size // step)) & ~1


def _fits(size: int, limit: int) -> bool:
    return limit <= 0 or size <= limit


def to_rgb(bgr: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    """BGR → contiguous RGB24, integer-factor scale, even width/height.

    Portal raises InvalidFrameDimensions on odd edges. Aspect is kept
    (1280x720 with max 640x480 → 640x360), not squashed per axis.
    """
    height, width = bgr.shape[:2]
    step = 1
    while not (
        _fits(_decimated(height, step), max_height)
        and _fits(_decimated(width, step), max_width)
    ):
        step += 1
    out = bgr[::step, ::step, ::-1]
    out = out[: out.shape[0] & ~1, : out.shape[1] & ~1]
    return np.ascontiguousarray(out)


class BgrSource:
    """Background SUB thread holding the most recent RGB frame."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        max_width: int,
        max_height: int,
    ) -> None:
        self._host = host
        self._port = port
        self._max_width = max_width
        self._max_height = max_height
        self._lock = threading.Lock()
        self._latest: Optional[Tuple[np.ndarray, Optional[int]]] = None
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="bgr-source", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def latest(self) -> Optional[Tuple[np.ndarray, Optional[int], int]]:
        """Newest frame as (rgb, capture_us, seq), or None before the first one."""
        with self._lock:
            if self._latest is None:
                return None
            rgb, capture_us = self._latest
            return rgb, capture_us, self._seq

    def _run(self) -> None:
        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self._host}:{self._port}")
        logger_mp.info(f"BGR SUB connected to tcp://{self._host}:{self._port}")
        try:
            while not self._stop.is_set():
                try:
                    payload = sock.recv()
                except zmq.Again:
                    continue
                try:
                    height, width, capture_ms, raw = parse_bgr_payload(payload)
                    bgr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                    rgb = to_rgb(bgr, self._max_width, self._max_height)
                except (ValueError, zmq.ZMQError) as exc:
                    logger_mp.warning(f"dropping malformed BGR frame: {exc}")
                    continue
                capture_us = capture_ms * 1000 if capture_ms is not None else None
                with self._lock:
                    self._latest = (rgb, capture_us)
                    self._seq += 1
        finally:
            sock.close(linger=0)
