"""LiveKit Portal round-trip against portal_robot_mock. Skipped without .env.

    pytest tests/test_portal_smoke.py
    python tests/test_portal_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
import uuid
import unittest
import subprocess

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

_TELEOP = os.path.join(_repo_root, "teleop")
PORTAL_YAML = os.path.join(_TELEOP, "portal.yaml")
MAPPING_YAML = os.path.join(_TELEOP, "portal_mapping.yaml")
ENV_FILE = os.path.join(_TELEOP, ".env")
MOCK_SCRIPT = os.path.join(_TELEOP, "portal_robot_mock.py")


def _load_livekit_url() -> str | None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return os.environ.get("LIVEKIT_URL")
    if os.path.isfile(ENV_FILE):
        load_dotenv(ENV_FILE, override=False)
    return os.environ.get("LIVEKIT_URL")


class TestCliHelp(unittest.TestCase):
    def test_robot_and_mock_help(self):
        for script in ("teleop_robot.py", "portal_robot_mock.py"):
            path = os.path.join(_TELEOP, script)
            result = subprocess.run(
                [sys.executable, path, "--help"],
                cwd=_repo_root,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                result.returncode, 0,
                f"{script} --help failed:\n{result.stderr}\n{result.stdout}")


@unittest.skipUnless(_load_livekit_url(), "LIVEKIT_URL missing; skip Portal round-trip")
class TestPortalRoundtrip(unittest.TestCase):
    def test_action_state_echo_and_video(self):
        from teleop.robot_control.portal_operator import PortalTeleopBridge

        url = _load_livekit_url()
        self.assertTrue(url)
        room = os.environ.get("LIVEKIT_ROOM", "g1-portal") + "-smoke-" + uuid.uuid4().hex[:8]
        mock = subprocess.Popen(
            [sys.executable, MOCK_SCRIPT,
             "--duration", "25",
             "--livekit-room", room,
             "--portal-identity", "smoke-robot"],
            cwd=_repo_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            time.sleep(2.0)
            if mock.poll() is not None:
                out, _ = mock.communicate(timeout=2)
                self.fail(f"mock exited early:\n{out}")

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
                for _ in range(40):
                    bridge.send_targets(
                        target, hand_q=hand, vx=0.12, vy=-0.04, vyaw=0.08, fsm_id=1)
                    time.sleep(0.15)
                    q = bridge.get_reported_arm_q()
                    if q is not None and np.allclose(q, target, atol=1e-3):
                        got = q
                        break
                self.assertIsNotNone(
                    got, f"operator never saw echoed arm q; last={bridge.get_reported_arm_q()}")
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
                    self.assertEqual(len(frame.shape), 3)
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


if __name__ == "__main__":
    unittest.main()
