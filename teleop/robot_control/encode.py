"""Encode capability port (HW vs software).

Device presence only — does not prove the FFI opened nvhost-msenc.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

EncodeBackend = Literal["hardware", "software", "unavailable"]

_ENCODE_BACKEND_ENV = "SAG_ENCODE_BACKEND"


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
        raise RuntimeError(f"encode_unavailable: {report.detail}")
    return report
