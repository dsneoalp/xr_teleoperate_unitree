"""CLI smoke for mock robot/operator scripts (no LiveKit connect)."""
from __future__ import annotations

import subprocess
import sys

from teleop.tests.paths import OPERATOR_MOCK, REPO_ROOT, ROBOT_MOCK, TELEOP_ROBOT


def _help(path: str) -> str:
    r = subprocess.run(
        [sys.executable, path, "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_robot_mock_help():
    out = _help(ROBOT_MOCK)
    assert "--duration" in out
    assert "--fps" in out


def test_operator_mock_help():
    out = _help(OPERATOR_MOCK)
    assert "--expect-echo" in out
    assert "--duration" in out


def test_teleop_robot_help():
    out = _help(TELEOP_ROBOT)
    assert "--frequency" in out
    assert "--no-img" in out
