"""Raw BGR ZMQ ingest for Portal RGB publish (one SUB per camera slot).

Subscribes to teleimager's BGR publisher (zmq_port + 1000) so the robot
process does not JPEG-decode the same stream video-ingress already
consumes as raw BGR. Color convert stays packed RGB24 for livekit-portal
FFI; no downscale.
"""
from __future__ import annotations

import json
import struct
import threading
from typing import Any, Dict, Optional, Tuple

import numpy as np

import logging_mp

logger_mp = logging_mp.getLogger(__name__)

BGR_ZMQ_PORT_OFFSET = 1000
_RGB_RING = 3


def fetch_teleimager_config(
    host: str, port: int = 60000, timeout_ms: int = 2000
) -> Optional[Dict[str, Any]]:
    """GET_DATA from teleimager. Returns None on timeout or parse error."""
    import zmq

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        sock.connect(f"tcp://{host}:{port}")
        sock.send(b"GET_DATA")
        return json.loads(sock.recv().decode("utf-8"))
    except Exception:
        return None
    finally:
        sock.close(linger=0)


def bgr_zmq_port(
    config: Dict[str, Any], camera_slot: str = "head_camera"
) -> Optional[int]:
    """Return BGR SUB port, or None if the slot has no raw-BGR publisher."""
    slot = config.get(camera_slot)
    if not isinstance(slot, dict):
        return None
    if not slot.get("enable_bgr_zmq"):
        return None
    zmq_port = int(slot.get("zmq_port") or 0)
    if zmq_port <= 0:
        return None
    return zmq_port + BGR_ZMQ_PORT_OFFSET


def parse_bgr_payload(payload: bytes) -> Tuple[int, int, Optional[int], memoryview]:
    """Parse teleimager BGR: `<HH>` height/width, optional `<Q>` capture_ms, packed BGR."""
    if len(payload) < 4:
        raise ValueError("BGR ZMQ payload too short")
    height, width = struct.unpack("<HH", payload[:4])
    frame_bytes = height * width * 3
    if frame_bytes <= 0:
        raise ValueError("invalid BGR dimensions")
    view = memoryview(payload)
    if len(payload) == 4 + frame_bytes:
        return height, width, None, view[4:]
    if len(payload) == 12 + frame_bytes:
        (capture_ms,) = struct.unpack("<Q", payload[4:12])
        return height, width, capture_ms, view[12:]
    raise ValueError("BGR ZMQ payload length mismatch")


def even_crop(frame: np.ndarray) -> np.ndarray:
    """Even W/H crop for Portal. Does not resize or copy when already even."""
    height, width = frame.shape[:2]
    height_even = height & ~1
    width_even = width & ~1
    if height_even <= 0 or width_even <= 0:
        raise ValueError(f"frame too small for even crop: {width}x{height}")
    if height_even == height and width_even == width:
        return frame
    return frame[:height_even, :width_even]


def bgr_to_rgb(bgr: np.ndarray, buf: Optional[np.ndarray]) -> np.ndarray:
    """Pack BGR into a reused C-contiguous RGB buffer. Does not mutate `bgr`."""
    import cv2

    if buf is None or buf.shape != bgr.shape or buf.dtype != np.uint8:
        buf = np.empty(bgr.shape, dtype=np.uint8)
    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB, dst=buf)
    return buf


class BgrCameraSource:
    """CONFLATE SUB thread holding the latest even-sized RGB frame (native WxH)."""

    def __init__(self, host: str, port: int, slot: str = "head_camera"):
        self._host = host
        self._port = port
        self._slot = slot
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._seq = 0
        self._locked_wh: Optional[Tuple[int, int]] = None
        self._size_mismatch_logged = False
        self._bufs: list[Optional[np.ndarray]] = [None] * _RGB_RING
        self._ring = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=f"bgr-{slot}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        logger_mp.info(f"BGR source '{self._slot}' closed.")

    def latest_rgb(self) -> Optional[Tuple[np.ndarray, int]]:
        """Newest RGB frame and seq, or None before the first accepted frame."""
        with self._lock:
            if self._latest is None:
                return None
            return self._latest, self._seq

    def _run(self) -> None:
        import zmq

        ctx = zmq.Context.instance()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.SUBSCRIBE, b"")
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt(zmq.RCVTIMEO, 500)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self._host}:{self._port}")
        logger_mp.info(
            f"BGR ZMQ '{self._slot}' {self._host}:{self._port} (JPEG skipped)"
        )
        try:
            while not self._stop.is_set():
                try:
                    payload = sock.recv()
                except zmq.Again:
                    continue
                try:
                    height, width, _capture_ms, raw = parse_bgr_payload(payload)
                    bgr = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
                    bgr = even_crop(bgr)
                    out_h, out_w = bgr.shape[:2]
                except (ValueError, zmq.ZMQError) as exc:
                    logger_mp.warning(f"dropping malformed BGR frame: {exc}")
                    continue
                if self._locked_wh is None:
                    self._locked_wh = (out_w, out_h)
                    logger_mp.info(
                        f"BGR '{self._slot}' locked {out_w}x{out_h} (encoder WxH pinned)"
                    )
                elif (out_w, out_h) != self._locked_wh:
                    if not self._size_mismatch_logged:
                        logger_mp.warning(
                            f"dropping BGR '{self._slot}' {out_w}x{out_h}; "
                            f"encoder pinned at {self._locked_wh[0]}x{self._locked_wh[1]}"
                        )
                        self._size_mismatch_logged = True
                    continue
                rgb = bgr_to_rgb(bgr, self._bufs[self._ring])
                self._bufs[self._ring] = rgb
                self._ring = (self._ring + 1) % _RGB_RING
                with self._lock:
                    self._latest = rgb
                    self._seq += 1
        finally:
            sock.close(linger=0)


HeadBgrSource = BgrCameraSource
