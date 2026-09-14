"""Joint-command smoothing used by the robot control loop.

Pure time-domain filters. No Portal, SDK, or camera I/O.
"""
from __future__ import annotations

import math

import numpy as np


def interp_cmd(state, q_cmd, now, tau):
    """Linear segment q0→q1 over tau s. Restart on a new target; hold at s=1."""
    q_cmd = np.asarray(q_cmd, dtype=float)
    if tau <= 0.0:
        state['q'] = q_cmd.copy()
        return q_cmd
    if 'q' not in state:
        state['q'] = q_cmd.copy()
        state['q0'] = q_cmd.copy()
        state['q1'] = q_cmd.copy()
        state['t0'] = now
        return q_cmd.copy()
    prev = state.get('q1')
    if prev is None or prev.shape != q_cmd.shape or not np.allclose(q_cmd, prev):
        state['q0'] = np.asarray(state['q'], dtype=float).copy()
        state['q1'] = q_cmd.copy()
        state['t0'] = now
    s = min(1.0, (now - state['t0']) / tau)
    q = (1.0 - s) * state['q0'] + s * state['q1']
    state['q'] = q
    return q


def filter_cmd(state, q_cmd, dt, tau):
    """PT1 toward q_cmd. Continues catching up while the target is held."""
    q_cmd = np.asarray(q_cmd, dtype=float)
    if tau <= 0.0:
        state['q'] = q_cmd.copy()
        return q_cmd
    if 'q' not in state:
        state['q'] = q_cmd.copy()
        return q_cmd.copy()
    alpha = 1.0 - math.exp(-max(dt, 1e-6) / tau)
    q = state['q'] + alpha * (q_cmd - state['q'])
    state['q'] = q
    return q
