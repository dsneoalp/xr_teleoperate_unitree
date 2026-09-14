"""Unit tests for PortalMapping pack/unpack. No LiveKit session."""
from __future__ import annotations

import os
import sys
import unittest

import numpy as np

_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from teleop.robot_control.portal_mapping import PortalMapping

_TELEOP = os.path.join(_repo_root, "teleop")
PORTAL_YAML = os.path.join(_TELEOP, "portal.yaml")
MAPPING_YAML = os.path.join(_TELEOP, "portal_mapping.yaml")


class TestPortalMapping(unittest.TestCase):
    def setUp(self):
        self.m = PortalMapping(MAPPING_YAML, PORTAL_YAML)

    def test_dof(self):
        self.assertEqual(self.m.arm_dof, 14)
        self.assertEqual(self.m.hand_dof, 14)

    def test_action_roundtrip(self):
        arm = np.linspace(0.1, 1.4, 14)
        hand = np.linspace(-0.7, 0.6, 14)
        packed = self.m.pack_action(
            arm_q=arm, hand_q=hand, vx=0.11, vy=-0.22, vyaw=0.33, fsm_id=1)
        self.assertEqual(packed["fsm_id"], 1)
        self.assertEqual(packed["vx"], 0.11)
        self.assertEqual(packed["L_SHOULDER_PITCH"], float(arm[0]))
        self.assertEqual(packed["R_WRIST_YAW"], float(arm[13]))
        self.assertEqual(packed["left_thumb_mcp"], float(hand[0]))
        unpacked = self.m.unpack_action(packed, timestamp_us=10, in_reply_to_ts_us=9)
        self.assertEqual(unpacked.fsm_id, 1)
        self.assertEqual(unpacked.timestamp_us, 10)
        self.assertEqual(unpacked.in_reply_to_ts_us, 9)
        self.assertAlmostEqual(unpacked.vx, 0.11)
        self.assertAlmostEqual(unpacked.vy, -0.22)
        self.assertAlmostEqual(unpacked.vyaw, 0.33)
        np.testing.assert_allclose(unpacked.arm_q, arm)
        np.testing.assert_allclose(unpacked.hand_q, hand)

    def test_state_from_motor_q(self):
        hand = np.linspace(-0.7, 0.6, 14)
        motor = (np.arange(29, dtype=np.float64) + 1.0) * 0.01
        state = self.m.pack_state(motor_q=motor, hand_q=hand, fsm_id=1)
        self.assertAlmostEqual(state["L_LEG_HIP_PITCH"], 0.01)
        self.assertEqual(state["L_SHOULDER_PITCH"], float(motor[15]))
        self.assertEqual(state["fsm_id"], 1)
        np.testing.assert_allclose(self.m.unpack_arm_q(state), motor[15:29])
        np.testing.assert_allclose(self.m.unpack_hand_q(state), hand)

    def test_state_from_arm_q(self):
        arm = np.linspace(0.1, 1.4, 14)
        hand = np.linspace(-0.7, 0.6, 14)
        state = self.m.pack_state(arm_q=arm, hand_q=hand, fsm_id=2)
        np.testing.assert_allclose(self.m.unpack_arm_q(state), arm)


if __name__ == "__main__":
    unittest.main()
