"""Smoke tests: mapping pack/unpack, then LiveKit Portal round-trip via mock robot.

    conda run -n tv python teleop/portal_smoke_test.py
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import subprocess
import traceback

import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

PORTAL_YAML = os.path.join(current_dir, "portal.yaml")
MAPPING_YAML = os.path.join(current_dir, "portal_mapping.yaml")
ENV_FILE = os.path.join(current_dir, ".env")


def test_mapping_roundtrip() -> None:
    from teleop.robot_control.portal_mapping import PortalMapping

    m = PortalMapping(MAPPING_YAML, PORTAL_YAML)
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


def test_portal_roundtrip() -> None:
    from dotenv import load_dotenv
    load_dotenv(ENV_FILE, override=False)
    url = os.environ.get("LIVEKIT_URL")
    if not url:
        raise SystemExit("LIVEKIT_URL missing; cannot run portal round-trip")

    room = os.environ.get("LIVEKIT_ROOM", "g1-portal") + "-smoke-" + uuid.uuid4().hex[:8]
    mock = subprocess.Popen(
        [sys.executable, os.path.join(current_dir, "portal_robot_mock.py"),
         "--duration", "25",
         "--livekit-room", room,
         "--portal-identity", "smoke-robot"],
        cwd=parent_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        from teleop.robot_control.portal_operator import PortalTeleopBridge

        time.sleep(2.0)
        if mock.poll() is not None:
            out, _ = mock.communicate(timeout=2)
            raise RuntimeError(f"mock exited early:\n{out}")

        bridge = PortalTeleopBridge(
            portal_yaml=PORTAL_YAML,
            mapping_yaml=MAPPING_YAML,
            env_file=ENV_FILE,
            identity="smoke-operator",
            room=room,
            ee=None,
        )
        try:
            bridge.wait_until_connected(timeout=20.0)
            target = np.linspace(0.05, 0.18, 14)
            hand = np.linspace(0.2, 0.33, 14)
            got = None
            for i in range(40):
                bridge.send_targets(target, hand_q=hand, vx=0.12, vy=-0.04, vyaw=0.08, fsm_id=1)
                time.sleep(0.15)
                q = bridge.get_current_dual_arm_q()
                if q is not None and np.allclose(q, target, atol=1e-3):
                    got = q
                    break
            if got is None:
                raise AssertionError(
                    f"operator never saw echoed arm q; last={bridge.get_current_dual_arm_q()}")
            print(f"OK  portal action→state echo (arm[0]={got[0]:.3f})")

            frame = None
            for _ in range(30):
                head = bridge.get_head_frame()
                if head is not None and head.bgr is not None:
                    frame = head.bgr
                    break
                time.sleep(0.1)
            if frame is None:
                print("WARN portal video frame not received (state echo still OK)")
            else:
                print(f"OK  portal video frame {frame.shape}")
        finally:
            bridge.close()
    finally:
        try:
            os.killpg(mock.pid, 9)
        except OSError:
            pass
        try:
            mock.communicate(timeout=3)
        except Exception:
            pass


def test_cli_help() -> None:
    for script in ("teleop_robot.py", "portal_robot_mock.py"):
        path = os.path.join(current_dir, script)
        r = subprocess.run([sys.executable, path, "--help"], cwd=parent_dir,
                           capture_output=True, text=True, timeout=30)
        if r.returncode != 0:
            raise RuntimeError(f"{script} --help failed:\n{r.stderr}\n{r.stdout}")
    print("OK  CLI --help")


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
