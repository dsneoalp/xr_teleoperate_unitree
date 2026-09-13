"""Smoke: mapping pack/unpack, CLI help, LiveKit round-trip via both mocks.

    conda run -n tv python teleop/tests/portal_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import subprocess
import traceback

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

import numpy as np

from teleop.tests.paths import (
    ENV_FILE,
    OPERATOR_MOCK,
    PORTAL_MAPPING,
    PORTAL_YAML,
    REPO_ROOT,
    ROBOT_MOCK,
    TELEOP_ROBOT,
)

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

SMOKE_ROBOT_DURATION_S = 25.0
SMOKE_OPERATOR_DURATION_S = 20.0
SMOKE_ROBOT_STARTUP_S = 2.0


def test_mapping_roundtrip() -> None:
    from teleop.robot_control.portal_mapping import PortalMapping

    m = PortalMapping(PORTAL_MAPPING, PORTAL_YAML)
    assert m.arm_dof == 14, m.arm_dof
    assert m.hand_dof == 14, m.hand_dof
    arm = np.linspace(0.1, 1.4, 14)
    hand = np.linspace(-0.7, 0.6, 14)
    packed = m.pack_action(arm_q=arm, hand_q=hand, vx=0.11, vy=-0.22, vyaw=0.33, fsm_id=1)
    assert packed["fsm_id"] == 1
    assert packed["vx"] == 0.11
    assert packed["L_SHOULDER_PITCH"] == float(arm[0])
    assert packed["R_WRIST_YAW"] == float(arm[13])
    assert packed["left_thumb_mcp"] == float(hand[0])
    unpacked = m.unpack_action(packed)
    assert unpacked.fsm_id == 1
    assert abs(unpacked.vx - 0.11) < 1e-9
    assert abs(unpacked.vy + 0.22) < 1e-9
    assert abs(unpacked.vyaw - 0.33) < 1e-9
    np.testing.assert_allclose(unpacked.arm_q, arm)
    np.testing.assert_allclose(unpacked.hand_q, hand)

    motor = (np.arange(29, dtype=np.float64) + 1.0) * 0.01
    state = m.pack_state(motor_q=motor, hand_q=hand, fsm_id=1)
    assert abs(state["L_LEG_HIP_PITCH"] - 0.01) < 1e-9
    assert state["L_SHOULDER_PITCH"] == float(motor[15])
    assert state["fsm_id"] == 1
    np.testing.assert_allclose(m.unpack_arm_q(state), motor[15:29])
    np.testing.assert_allclose(m.unpack_hand_q(state), hand)

    state2 = m.pack_state(arm_q=arm, hand_q=hand, fsm_id=2)
    np.testing.assert_allclose(m.unpack_arm_q(state2), arm)
    print("OK  mapping pack/unpack")


def test_cli_help() -> None:
    for path in (TELEOP_ROBOT, ROBOT_MOCK, OPERATOR_MOCK):
        r = subprocess.run(
            [sys.executable, path, "--help"], cwd=REPO_ROOT,
            capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(
                f"{os.path.basename(path)} --help failed:\n{r.stderr}\n{r.stdout}")
    print("OK  CLI --help")


def test_portal_roundtrip() -> None:
    from teleop.robot_control.portal_operator import _load_dotenv
    _load_dotenv(ENV_FILE)
    url = os.environ.get("LIVEKIT_URL")
    if not url:
        raise SystemExit("LIVEKIT_URL missing; cannot run portal round-trip")

    room = os.environ.get("LIVEKIT_ROOM", "g1-portal") + "-smoke-" + uuid.uuid4().hex[:8]
    robot = subprocess.Popen(
        [sys.executable, ROBOT_MOCK,
         "--duration", str(SMOKE_ROBOT_DURATION_S),
         "--livekit-room", room,
         "--portal-identity", "smoke-robot"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        time.sleep(SMOKE_ROBOT_STARTUP_S)
        if robot.poll() is not None:
            out, _ = robot.communicate(timeout=2)
            raise RuntimeError(f"mock robot exited early:\n{out}")

        op = subprocess.run(
            [sys.executable, OPERATOR_MOCK,
             "--duration", str(SMOKE_OPERATOR_DURATION_S),
             "--expect-echo",
             "--livekit-room", room,
             "--portal-identity", "smoke-operator"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=SMOKE_OPERATOR_DURATION_S + 15.0,
        )
        log = (op.stdout or "") + (op.stderr or "")
        if op.returncode != 0:
            raise RuntimeError(
                f"mock operator failed (code={op.returncode}):\n{log}")
        if "echo+matched observation" not in log:
            raise RuntimeError(f"mock operator succeeded without match log:\n{log}")
        print("OK  mock robot ↔ mock operator (echo + matched observation)")
    finally:
        try:
            os.killpg(robot.pid, 9)
        except OSError:
            pass
        try:
            robot.communicate(timeout=3)
        except Exception:
            pass


if __name__ == "__main__":
    failed = 0
    for name, fn in (
        ("mapping", test_mapping_roundtrip),
        ("cli", test_cli_help),
        ("portal", test_portal_roundtrip),
    ):
        try:
            fn()
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
    if failed:
        raise SystemExit(1)
    print("ALL SMOKE TESTS PASSED")
