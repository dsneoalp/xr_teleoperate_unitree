"""G1 check: Portal RGB send_video_frame must open the Jetson HW encoder.

Does not start DDS / teleop_robot. Uses a separate LiveKit room so a running
teleop session is not joined. Run inside the robot container:

    python portal_hw_encode_check.py

Exit 0 only when /proc/self/fd points at nvhost-msenc after publishing RGB
frames AND the FFI logs 'Using Jetson MMAPI encoder for H264'. Device
existence alone is not HW encode (OpenH264 fallback).
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import uuid

import logging_mp

logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.portal_robot import (  # noqa: E402
    PortalRobotTransport,
    hw_encode_open_fds,
)

MMAPI_LOG = "Using Jetson MMAPI encoder for H264"


def _test_pattern(h: int, w: int, t: float):
    import numpy as np
    y = np.linspace(0, 255, h, dtype=np.uint8)[:, None]
    x = np.linspace(0, 255, w, dtype=np.uint8)[None, :]
    frame = np.zeros((h, w, 3), dtype=np.uint8)
    frame[:, :, 0] = x
    frame[:, :, 1] = y
    col = int((t * 80) % w)
    frame[:, max(0, col - 4):col + 4, 2] = 255
    return frame


class _FdTee:
    """Capture C-level writes to fd 1/2 (Rust FFI bypasses sys.stdout)."""

    def __init__(self):
        self._chunks: list[bytes] = []
        self._lock = threading.Lock()
        self._orig_out = os.dup(1)
        self._orig_err = os.dup(2)
        self._pipe_r, self._pipe_w = os.pipe()
        os.set_blocking(self._pipe_r, False)
        os.dup2(self._pipe_w, 1)
        os.dup2(self._pipe_w, 2)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self) -> None:
        while not self._stop.is_set():
            try:
                data = os.read(self._pipe_r, 8192)
            except BlockingIOError:
                time.sleep(0.01)
                continue
            if not data:
                time.sleep(0.01)
                continue
            with self._lock:
                self._chunks.append(data)
            try:
                os.write(self._orig_err, data)
            except OSError:
                pass

    def text(self) -> str:
        self._drain()
        with self._lock:
            return b"".join(self._chunks).decode("utf-8", "replace")

    def _drain(self) -> None:
        while True:
            try:
                data = os.read(self._pipe_r, 8192)
            except BlockingIOError:
                break
            if not data:
                break
            with self._lock:
                self._chunks.append(data)
            try:
                os.write(self._orig_err, data)
            except OSError:
                pass

    def close(self) -> None:
        try:
            os.fsync(self._pipe_w)
        except OSError:
            pass
        time.sleep(0.05)
        self._drain()
        self._stop.set()
        try:
            os.dup2(self._orig_out, 1)
            os.dup2(self._orig_err, 2)
        except OSError:
            pass
        for fd in (self._pipe_w, self._pipe_r, self._orig_out, self._orig_err):
            try:
                os.close(fd)
            except OSError:
                pass
        self._thread.join(timeout=1.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--portal-yaml", type=str, default=os.path.join(current_dir, "portal.yaml"))
    parser.add_argument("--portal-mapping", type=str, default=os.path.join(current_dir, "portal_mapping.yaml"))
    parser.add_argument("--env-file", type=str, default=os.path.join(current_dir, ".env"))
    parser.add_argument("--livekit-url", type=str, default=None)
    parser.add_argument("--livekit-room", type=str, default=None)
    parser.add_argument("--portal-identity", type=str, default="xr-hw-encode-check")
    parser.add_argument("--connect-timeout", type=float, default=45.0)
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv(args.env_file, override=False)
    if args.livekit_url:
        os.environ["LIVEKIT_URL"] = args.livekit_url
    if not os.environ.get("LIVEKIT_URL"):
        print("FAIL  LIVEKIT_URL missing; cannot publish frames", file=sys.stderr)
        return 2

    # Dedicated room: never join the live teleop room.
    if args.livekit_room:
        room = args.livekit_room
    else:
        base = os.environ.get("LIVEKIT_ROOM", "g1-portal")
        room = f"{base}-hwcheck-{uuid.uuid4().hex[:8]}"

    tee = _FdTee()
    portal = None
    try:
        portal = PortalRobotTransport(
            portal_yaml=args.portal_yaml,
            mapping_yaml=args.portal_mapping,
            env_file=args.env_file,
            identity=args.portal_identity,
            room=room,
            url=args.livekit_url,
        )
        portal.wait_until_connected(timeout=args.connect_timeout)
        track = portal.video_tracks[0] if portal.video_tracks else None
        if not track:
            print("FAIL  portal.yaml has no video tracks", file=sys.stderr)
            return 1

        interval = 1.0 / max(args.fps, 1.0)
        t0 = time.time()
        n = 0
        fds: list[str] = []
        while time.time() - t0 < args.duration:
            rgb = _test_pattern(480, 640, time.time())
            portal.send_video_frame(track, rgb, timestamp_us=int(time.time() * 1_000_000))
            n += 1
            fds = hw_encode_open_fds()
            if fds and MMAPI_LOG in tee.text():
                break
            time.sleep(interval)

        fds = fds or hw_encode_open_fds()
        log_blob = tee.text()
        mmapi_hit = MMAPI_LOG in log_blob

        if not fds:
            print(
                f"FAIL  published {n} RGB frames but no FD on nvhost-msenc "
                "(Portal likely fell back to OpenH264)",
                file=sys.stderr,
            )
            return 1
        if not mmapi_hit:
            print(
                f"FAIL  encoder FD {fds} after {n} frames but FFI log "
                f"{MMAPI_LOG!r} not seen (not proven MMAPI)",
                file=sys.stderr,
            )
            return 1

        print(f"OK  HW encoder FD {fds} after {n} RGB frames")
        print(f"OK  FFI log {MMAPI_LOG!r}")
        return 0
    finally:
        if portal is not None:
            portal.close()
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())
