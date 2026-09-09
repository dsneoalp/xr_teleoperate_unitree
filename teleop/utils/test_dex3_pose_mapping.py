"""Unit tests for dex3 custom pose mapping. No Tk, no Portal."""
import os
import sys
import tempfile
import unittest

import yaml

_teleop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_teleop_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from teleop.utils.dex3_pose_mapping import (
    Binding,
    PoseSpec,
    combo_to_display_list,
    dumps,
    hand_dict_to_q,
    load,
    match_binding,
    needs_gui,
    parse_combo,
    pressed_from_tele_data,
    q_to_hand_dict,
    ramp_towards,
    resolve_mapping_mode,
    save,
    target_q,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "testdata", "hand_pose_expected.yaml")


def _example_spec():
    default_q = [0.0] * 14
    left = {
        "left_thumb_mcp": 0.0,
        "left_thumb_aux": 0.9,
        "left_thumb_pip": 0.43,
        "left_index_mcp": 0.0,
        "left_index_pip": 0.0,
        "left_middle_mcp": 0.0,
        "left_middle_pip": 0.0,
    }
    q = hand_dict_to_q(left, None, default_q=default_q)
    return PoseSpec(
        duration_s=1.5,
        default_q=default_q,
        bindings=[Binding(combo=parse_combo(["L2", "X"]), q=q, has_left=True, has_right=False)],
    )


def _independent_l1_r1_spec():
    default_q = [0.0] * 14
    left_q = hand_dict_to_q({"left_thumb_aux": 0.5}, None, default_q=default_q)
    right_q = hand_dict_to_q(None, {"right_thumb_aux": -0.5}, default_q=default_q)
    return PoseSpec(
        default_q=default_q,
        bindings=[
            Binding(combo=parse_combo("L1"), q=left_q, has_left=True, has_right=False),
            Binding(combo=parse_combo("R1"), q=right_q, has_left=False, has_right=True),
        ],
    )


class TestCombo(unittest.TestCase):
    def test_parse_string_and_list(self):
        a = parse_combo("L2+X")
        b = parse_combo(["L2", "X"])
        c = parse_combo(["L_squeeze", "L_A"])
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_display_order(self):
        self.assertEqual(combo_to_display_list(parse_combo("X+L2")), ["L2", "X"])


class TestYamlDump(unittest.TestCase):
    def test_dump_matches_fixture(self):
        spec = _example_spec()
        dumped = yaml.safe_load(dumps(spec))
        with open(FIXTURE, encoding="utf-8") as f:
            expected = yaml.safe_load(f)
        self.assertEqual(dumped, expected)

    def test_fixture_has_expected_schema(self):
        with open(FIXTURE, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        self.assertEqual(raw["duration_s"], 1.5)
        self.assertEqual(list(raw["default"]["left_hand"]), [
            "left_thumb_mcp", "left_thumb_aux", "left_thumb_pip",
            "left_index_mcp", "left_index_pip",
            "left_middle_mcp", "left_middle_pip",
        ])
        self.assertEqual(list(raw["default"]["right_hand"]), [
            "right_thumb_mcp", "right_thumb_aux", "right_thumb_pip",
            "right_index_mcp", "right_index_pip",
            "right_middle_mcp", "right_middle_pip",
        ])
        self.assertEqual(raw["bindings"][0]["combo"], ["L2", "X"])
        self.assertNotIn("right_hand", raw["bindings"][0])
        self.assertEqual(raw["bindings"][0]["left_hand"]["left_thumb_aux"], 0.9)
        self.assertEqual(raw["bindings"][0]["left_hand"]["left_thumb_pip"], 0.43)

    def test_roundtrip_fixture(self):
        spec = load(FIXTURE)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.yaml")
            save(path, spec)
            with open(path, encoding="utf-8") as f:
                again = yaml.safe_load(f)
            with open(FIXTURE, encoding="utf-8") as f:
                original = yaml.safe_load(f)
            self.assertEqual(again, original)

    def test_load_fills_missing_side_from_default(self):
        spec = load(FIXTURE)
        q = spec.bindings[0].q
        self.assertEqual(len(q), 14)
        self.assertEqual(q[1], 0.9)
        self.assertEqual(q[2], 0.43)
        self.assertTrue(all(v == 0.0 for v in q[7:]))


class TestMatchAndRamp(unittest.TestCase):
    def setUp(self):
        self.spec = load(FIXTURE)
        self.combo_l2x = parse_combo("L2+X")
        self.combo_l2 = parse_combo("L2")
        l2_q = hand_dict_to_q(
            {"left_thumb_aux": 0.2}, None, default_q=self.spec.default_q)
        self.spec.bindings.append(
            Binding(combo=self.combo_l2, q=l2_q, has_left=True, has_right=False))

    def test_longest_combo_wins(self):
        pressed = parse_combo("L2+X")
        matched = match_binding(pressed, self.spec.bindings, side="left")
        self.assertEqual(matched.combo, self.combo_l2x)

    def test_no_press_is_default(self):
        q = target_q(frozenset(), self.spec)
        self.assertEqual(q, self.spec.default_q)

    def test_left_only_fixture_keeps_right_default(self):
        q = target_q(self.combo_l2x, self.spec)
        self.assertEqual(q[1], 0.9)
        self.assertEqual(q[2], 0.43)
        self.assertTrue(all(v == 0.0 for v in q[7:]))

    def test_independent_hands_both_pressed(self):
        spec = _independent_l1_r1_spec()
        q = target_q(parse_combo("L1+R1"), spec)
        self.assertEqual(q[1], 0.5)
        self.assertEqual(q[8], -0.5)

    def test_release_one_hand_keeps_the_other(self):
        spec = _independent_l1_r1_spec()
        q = target_q(parse_combo("R1"), spec)
        self.assertEqual(q[1], 0.0)
        self.assertTrue(all(v == 0.0 for v in q[:7]))
        self.assertEqual(q[8], -0.5)

    def test_release_ramps_to_default(self):
        bound = target_q(self.combo_l2x, self.spec)
        current = list(bound)
        step = 0.2
        for _ in range(20):
            current = ramp_towards(current, self.spec.default_q, step)
        self.assertEqual(current, self.spec.default_q)

    def test_ramp_clips_to_target(self):
        current = [0.0, 0.0]
        target = [0.05, 1.0]
        out = ramp_towards(current, target, 0.1)
        self.assertEqual(out[0], 0.05)
        self.assertEqual(out[1], 0.1)


class TestHandQ(unittest.TestCase):
    def test_length_and_order(self):
        left = {n: float(i + 1) for i, n in enumerate(
            ["left_thumb_mcp", "left_thumb_aux", "left_thumb_pip",
             "left_index_mcp", "left_index_pip", "left_middle_mcp", "left_middle_pip"])}
        right = {n: float(i + 10) for i, n in enumerate(
            ["right_thumb_mcp", "right_thumb_aux", "right_thumb_pip",
             "right_index_mcp", "right_index_pip", "right_middle_mcp", "right_middle_pip"])}
        q = hand_dict_to_q(left, right)
        self.assertEqual(len(q), 14)
        self.assertEqual(q[0], 1.0)
        self.assertEqual(q[3], 4.0)
        self.assertEqual(q[7], 10.0)
        back_l, back_r = q_to_hand_dict(q)
        self.assertEqual(list(back_l), list(left))
        self.assertEqual(back_l, left)
        self.assertEqual(back_r, right)


class TestResolve(unittest.TestCase):
    def test_needs_gui(self):
        self.assertFalse(needs_gui(FIXTURE))
        self.assertTrue(needs_gui("/tmp/does-not-exist-dex3-poses.yaml"))

    def test_cli_mode(self):
        self.assertEqual(resolve_mapping_mode(False, True), "xy")
        self.assertEqual(resolve_mapping_mode(False, False), "xy")
        self.assertEqual(resolve_mapping_mode(True, True), "load")
        self.assertEqual(resolve_mapping_mode(True, False), "gui")

    def test_pressed_from_tele_data(self):
        class Tele:
            left_ctrl_trigger = False
            left_ctrl_squeeze = True
            left_ctrl_aButton = True
            left_ctrl_bButton = False
            right_ctrl_trigger = False
            right_ctrl_squeeze = False
            right_ctrl_bButton = False

        self.assertEqual(pressed_from_tele_data(Tele()), parse_combo("L2+X"))


class TestGUIState(unittest.TestCase):
    def test_add_or_replace_binding(self):
        from teleop.utils.dex3_pose_gui_state import Dex3PoseGUIState
        state = Dex3PoseGUIState()
        q = [0.1] * 14
        state.set_q(q, notify=False)
        label = state.add_or_replace_binding(["L2", "X"], has_left=True, has_right=True)
        self.assertEqual(label, "L2+X (LR)")
        spec = state.snapshot_spec()
        self.assertEqual(len(spec.bindings), 1)
        self.assertEqual(spec.bindings[0].q, q)
        state.add_or_replace_binding(["L2", "X"], has_left=True, has_right=True)
        self.assertEqual(len(state.snapshot_spec().bindings), 1)

    def test_left_and_right_bindings_coexist(self):
        from teleop.utils.dex3_pose_gui_state import Dex3PoseGUIState
        state = Dex3PoseGUIState()
        state.set_q([0.2] * 14, notify=False)
        self.assertEqual(
            state.add_or_replace_binding(["L1"], has_left=True, has_right=False),
            "L1 (L)")
        state.set_q([0.3] * 14, notify=False)
        self.assertEqual(
            state.add_or_replace_binding(["R1"], has_left=False, has_right=True),
            "R1 (R)")
        spec = state.snapshot_spec()
        self.assertEqual(len(spec.bindings), 2)
        self.assertEqual(state.bindings_labels(), ["L1 (L)", "R1 (R)"])
        state.add_or_replace_binding(["L1"], has_left=True, has_right=False)
        self.assertEqual(len(state.snapshot_spec().bindings), 2)


if __name__ == "__main__":
    unittest.main()
