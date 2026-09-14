"""interp_cmd / filter_cmd: tau=0 pass-through, restart, PT1 step. No Portal."""
from __future__ import annotations

import math

import numpy as np

from teleop.utils.cmd_filter import filter_cmd, interp_cmd


def test_interp_tau_zero_copies_target():
    state = {}
    q = interp_cmd(state, [1.0, 2.0], now=0.0, tau=0.0)
    np.testing.assert_array_equal(q, [1.0, 2.0])
    np.testing.assert_array_equal(state["q"], [1.0, 2.0])


def test_interp_first_call_holds_target():
    state = {}
    q = interp_cmd(state, [4.0, 0.0], now=1.0, tau=0.5)
    np.testing.assert_array_equal(q, [4.0, 0.0])
    np.testing.assert_array_equal(state["q0"], [4.0, 0.0])
    np.testing.assert_array_equal(state["q1"], [4.0, 0.0])
    assert state["t0"] == 1.0


def test_interp_midpoint_and_hold():
    state = {}
    interp_cmd(state, [0.0], now=0.0, tau=1.0)
    start = interp_cmd(state, [10.0], now=0.0, tau=1.0)
    np.testing.assert_allclose(start, [0.0])
    mid = interp_cmd(state, [10.0], now=0.5, tau=1.0)
    np.testing.assert_allclose(mid, [5.0])
    end = interp_cmd(state, [10.0], now=2.0, tau=1.0)
    np.testing.assert_allclose(end, [10.0])


def test_interp_restarts_on_new_target():
    state = {}
    interp_cmd(state, [0.0], now=0.0, tau=1.0)
    interp_cmd(state, [10.0], now=0.0, tau=1.0)
    interp_cmd(state, [10.0], now=0.5, tau=1.0)
    np.testing.assert_allclose(state["q"], [5.0])
    q = interp_cmd(state, [0.0], now=0.5, tau=1.0)
    np.testing.assert_allclose(q, [5.0])
    np.testing.assert_allclose(state["q0"], [5.0])
    np.testing.assert_allclose(state["q1"], [0.0])


def test_filter_tau_zero_copies_target():
    state = {}
    q = filter_cmd(state, [3.0, 1.0], dt=0.1, tau=0.0)
    np.testing.assert_array_equal(q, [3.0, 1.0])


def test_filter_pt1_one_time_constant():
    state = {}
    filter_cmd(state, [0.0], dt=0.1, tau=1.0)
    q = filter_cmd(state, [1.0], dt=1.0, tau=1.0)
    np.testing.assert_allclose(q, [1.0 - math.exp(-1.0)])
