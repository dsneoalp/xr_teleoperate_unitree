"""Encode capability port (HW vs software) and post-frame DoD evidence.

Device presence alone does not prove the FFI opened nvhost-msenc.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

EncodeBackend = Literal["hardware", "software", "unavailable"]
HwEncodeVerdict = Literal["ok", "openh264", "pending"]

_ENCODE_BACKEND_ENV = "SAG_ENCODE_BACKEND"
_CONFIRM_S_ENV = "SAG_HW_ENCODE_CONFIRM_S"
_DEFAULT_CONFIRM_S = 15.0

MMAPI_LOG = "Using Jetson MMAPI encoder for H264"
OPENH264_LOG = "[OpenH264]"


class EncodeUnavailableError(RuntimeError):
    """Fail-closed: HW encode required but not proven / OpenH264 fallback."""


@dataclass(frozen=True, slots=True)
class EncodeCapabilityReport:
    backend: EncodeBackend
    device_path: str | None
    detail: str


class EncodeCapability(Protocol):
    """Probe whether Portal RGB encode can use Jetson HW (msenc) or must fail-closed."""

    def probe(self) -> EncodeCapabilityReport:
        """Return current encode capability without starting encode."""
        ...


@dataclass(slots=True)
class JetsonMsencEncodeCapability:
    """Fail-closed HW encode probe for G1 / lab-jetson."""

    device_path: str = "/dev/nvhost-msenc"
    require_hw: bool = True

    def probe(self) -> EncodeCapabilityReport:
        forced = os.environ.get(_ENCODE_BACKEND_ENV, "").strip().lower()
        if forced == "software":
            if self.require_hw:
                return EncodeCapabilityReport(
                    backend="unavailable",
                    device_path=None,
                    detail="software encode forbidden when require_hw=true",
                )
            return EncodeCapabilityReport(
                backend="software",
                device_path=None,
                detail=f"{_ENCODE_BACKEND_ENV}=software",
            )
        path = Path(self.device_path)
        if path.exists():
            return EncodeCapabilityReport(
                backend="hardware",
                device_path=str(path),
                detail="nvhost-msenc present",
            )
        if self.require_hw:
            return EncodeCapabilityReport(
                backend="unavailable",
                device_path=str(path),
                detail="msenc missing; fail-closed",
            )
        return EncodeCapabilityReport(
            backend="software",
            device_path=None,
            detail="msenc missing; software allowed",
        )


def assert_encode_ready(capability: EncodeCapability) -> EncodeCapabilityReport:
    report = capability.probe()
    if report.backend == "unavailable":
        raise EncodeUnavailableError(f"encode_unavailable: {report.detail}")
    return report


def hw_encode_confirm_timeout_s() -> float:
    raw = os.environ.get(_CONFIRM_S_ENV, "").strip()
    if not raw:
        return _DEFAULT_CONFIRM_S
    try:
        return max(1.0, float(raw))
    except ValueError:
        return _DEFAULT_CONFIRM_S


def hw_encode_open_fds(fd_dir: str = "/proc/self/fd") -> list[str]:
    """Return symlink targets under fd_dir that point at the Jetson encoder device."""
    found: list[str] = []
    try:
        names = os.listdir(fd_dir)
    except OSError:
        return found
    for name in names:
        link = os.path.join(fd_dir, name)
        try:
            target = os.readlink(link)
        except OSError:
            continue
        if "nvhost-msenc" in target:
            found.append(target)
    return found


def evaluate_hw_encode_evidence(fds: list[str], log_text: str) -> HwEncodeVerdict:
    """DoD T5: MMAPI log + encoder FD. OpenH264 without MMAPI is fail-closed."""
    mmapi = MMAPI_LOG in log_text
    if mmapi and fds:
        return "ok"
    if OPENH264_LOG in log_text and not mmapi:
        return "openh264"
    return "pending"


def hw_encode_unavailable_message(verdict: HwEncodeVerdict, *, fds: list[str], n_frames: int) -> str:
    if verdict == "openh264":
        return (
            "encode_unavailable: FFI logged OpenH264 without "
            f"{MMAPI_LOG!r} (silent software fallback)"
        )
    return (
        f"encode_unavailable: no Jetson MMAPI after {n_frames} RGB frames "
        f"(fds={fds or []}; need {MMAPI_LOG!r} and FD on nvhost-msenc)"
    )


class FdTee:
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
