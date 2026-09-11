"""Unit tests for the arm stiffness fade curve. No SDK / Portal."""
import os
import sys
import unittest

_teleop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_teleop_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from teleop.utils.arm_stiffness import (
    ARM_STIFFNESS_FADE_S,
    arm_stiffness_scale,
    tanh_sinc,
)


class TestArmStiffnessScale(unittest.TestCase):
    def test_tanh_sinc_at_zero(self):
        self.assertAlmostEqual(tanh_sinc(0.0), 1.0)

    def test_bounds(self):
        self.assertAlmostEqual(arm_stiffness_scale(0.0), 1.0)
        self.assertAlmostEqual(arm_stiffness_scale(-1.0), 1.0)
        self.assertAlmostEqual(arm_stiffness_scale(ARM_STIFFNESS_FADE_S), 0.0)
        self.assertAlmostEqual(arm_stiffness_scale(ARM_STIFFNESS_FADE_S + 1.0), 0.0)

    def test_monotonic_decrease(self):
        samples = [arm_stiffness_scale(i * 0.1) for i in range(31)]
        for a, b in zip(samples, samples[1:]):
            self.assertLessEqual(b, a + 1e-12)

    def test_midpoint_in_open_interval(self):
        mid = arm_stiffness_scale(ARM_STIFFNESS_FADE_S / 2.0)
        self.assertGreater(mid, 0.0)
        self.assertLess(mid, 1.0)


if __name__ == "__main__":
    unittest.main()
