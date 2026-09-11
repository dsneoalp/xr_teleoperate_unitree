"""Arm PD gain schedule: tanh(x)/x fade from 1 to 0."""
from __future__ import annotations

import math

ARM_STIFFNESS_FADE_S = 3.0
ARM_STIFFNESS_X_END = 3.0


def tanh_sinc(x: float) -> float:
    """lim x→0 tanh(x)/x = 1."""
    if abs(x) < 1e-12:
        return 1.0
    return math.tanh(x) / x


def arm_stiffness_scale(
    t: float,
    duration: float = ARM_STIFFNESS_FADE_S,
    x_end: float = ARM_STIFFNESS_X_END,
) -> float:
    """Scale in [0, 1]: 1 at t<=0, 0 at t>=duration.

    Shape is tanh(x)/x with x = x_end * t/duration, affine-mapped so the
    finite interval actually reaches 0 (raw tanh(3)/3 is still ~0.33).
    """
    if duration <= 0.0 or t >= duration:
        return 0.0
    if t <= 0.0:
        return 1.0
    x = x_end * (t / duration)
    g_end = tanh_sinc(x_end)
    return (tanh_sinc(x) - g_end) / (1.0 - g_end)
