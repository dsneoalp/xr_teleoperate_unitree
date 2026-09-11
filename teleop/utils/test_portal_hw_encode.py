"""Offline tests for Portal Jetson HW-encode probe and compose contract.

No LiveKit, no DDS, no Docker. Run from repo root:

    python -m unittest teleop.utils.test_portal_hw_encode
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np
import yaml

_teleop_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_repo_root = os.path.dirname(_teleop_dir)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from teleop.robot_control import portal_robot as portal_robot_mod
from teleop.robot_control.encode import (
    JetsonMsencEncodeCapability,
    assert_encode_ready,
)
from teleop.robot_control.portal_robot import hw_encode_open_fds

COMPOSE_YML = os.path.join(_repo_root, "docker", "compose.yml")
PORTAL_YAML = os.path.join(_teleop_dir, "portal.yaml")
MAPPING_YAML = os.path.join(_teleop_dir, "portal_mapping.yaml")


class ProbeTests(unittest.TestCase):
    def test_require_hw_missing_device_unavailable(self):
        missing = os.path.join(tempfile.gettempdir(), "no-msenc")
        cap = JetsonMsencEncodeCapability(device_path=missing, require_hw=True)
        report = cap.probe()
        self.assertEqual(report.backend, "unavailable")
        with self.assertRaises(RuntimeError) as ctx:
            assert_encode_ready(cap)
        self.assertIn("encode_unavailable", str(ctx.exception))
        self.assertIn("msenc missing", str(ctx.exception))

    def test_missing_device_software_when_not_required(self):
        missing = os.path.join(tempfile.gettempdir(), "no-msenc")
        report = JetsonMsencEncodeCapability(
            device_path=missing, require_hw=False
        ).probe()
        self.assertEqual(report.backend, "software")
        self.assertIsNone(report.device_path)
        assert_encode_ready(
            JetsonMsencEncodeCapability(device_path=missing, require_hw=False)
        )

    def test_present_device_hardware(self):
        with tempfile.NamedTemporaryFile() as fake:
            path = fake.name
            report = assert_encode_ready(
                JetsonMsencEncodeCapability(device_path=path, require_hw=True)
            )
        self.assertEqual(report.backend, "hardware")
        self.assertEqual(report.device_path, path)

    def test_software_backend_forbidden_when_require_hw(self):
        with mock.patch.dict(os.environ, {"SAG_ENCODE_BACKEND": "software"}):
            cap = JetsonMsencEncodeCapability(require_hw=True)
            self.assertEqual(cap.probe().backend, "unavailable")
            with self.assertRaises(RuntimeError) as ctx:
                assert_encode_ready(cap)
            self.assertIn("encode_unavailable", str(ctx.exception))

    def test_software_backend_allowed_when_not_required(self):
        with mock.patch.dict(os.environ, {"SAG_ENCODE_BACKEND": "software"}):
            report = JetsonMsencEncodeCapability(require_hw=False).probe()
        self.assertEqual(report.backend, "software")
        self.assertIn("SAG_ENCODE_BACKEND=software", report.detail)

    def test_open_fds_scans_symlinks(self):
        with tempfile.TemporaryDirectory() as td:
            os.symlink("/dev/nvhost-msenc", os.path.join(td, "12"))
            os.symlink("/dev/null", os.path.join(td, "1"))
            os.symlink("/dev/v4l2-nvenc", os.path.join(td, "13"))
            found = hw_encode_open_fds(td)
        self.assertEqual(found, ["/dev/nvhost-msenc"])


def _fake_livekit_module(order: list[str]):
    mod = mock.MagicMock()

    class RobotConfig:
        @staticmethod
        def from_yaml_file(path, room):
            order.append("config")
            return object()

    class Robot:
        def __init__(self, cfg):
            order.append("robot")
            self._cfg = cfg

        def on_action(self, cb):
            pass

        def connect(self, *a, **k):
            raise AssertionError("connect must not run in this test")

    mod.RobotConfig = RobotConfig
    mod.Robot = Robot
    return mod


class InitOrderTests(unittest.TestCase):
    def test_init_probes_after_robot_before_connect(self):
        order: list[str] = []
        real_assert = portal_robot_mod.assert_encode_ready
        real_cap = portal_robot_mod.JetsonMsencEncodeCapability

        def wrapped_assert(capability):
            order.append("probe")
            return real_assert(capability)

        env = {
            "LIVEKIT_URL": "ws://127.0.0.1:7880",
            "LIVEKIT_API_KEY": "devkey",
            "LIVEKIT_API_SECRET": "secret",
            "LIVEKIT_ROOM": "g1-portal",
            "SAG_REQUIRE_HW_ENCODE": "1",
        }
        fake_portal = _fake_livekit_module(order)
        with tempfile.NamedTemporaryFile() as fake_dev:
            def cap_factory(*, require_hw=True, device_path=None):
                return real_cap(
                    device_path=fake_dev.name if device_path is None else device_path,
                    require_hw=require_hw,
                )

            with mock.patch.dict(os.environ, env):
                with mock.patch.object(portal_robot_mod, "assert_encode_ready", wrapped_assert):
                    with mock.patch.object(
                        portal_robot_mod, "JetsonMsencEncodeCapability", cap_factory
                    ):
                        with mock.patch.dict(sys.modules, {"livekit.portal": fake_portal}):
                            with mock.patch.object(portal_robot_mod, "_load_dotenv", lambda _p: None):
                                with mock.patch.object(threading.Thread, "start", lambda self: None):
                                    portal_robot_mod.PortalRobotTransport(
                                        portal_yaml=PORTAL_YAML,
                                        mapping_yaml=MAPPING_YAML,
                                        env_file=os.path.join(_teleop_dir, ".env"),
                                        identity="test-robot",
                                    )
        self.assertIn("probe", order)
        self.assertIn("config", order)
        self.assertIn("robot", order)
        self.assertLess(order.index("config"), order.index("probe"))
        self.assertLess(order.index("robot"), order.index("probe"))


class ComposeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(COMPOSE_YML, "r") as f:
            cls.data = yaml.safe_load(f)
        cls.services = cls.data["services"]

    def _env(self, service: dict) -> dict:
        env = service.get("environment") or {}
        if isinstance(env, list):
            out = {}
            for item in env:
                if isinstance(item, str) and "=" in item:
                    k, v = item.split("=", 1)
                    out[k] = v
            return out
        return dict(env)

    def test_robot_has_nvidia_msenc_fail_closed(self):
        robot = self.services["robot"]
        self.assertEqual(robot.get("runtime"), "nvidia")
        devices = [str(d) for d in (robot.get("devices") or [])]
        self.assertTrue(any("nvhost-msenc" in d for d in devices))
        self.assertFalse(any("v4l2-nvenc" in d for d in devices))
        self.assertNotEqual(robot.get("privileged"), True)
        env = self._env(robot)
        self.assertEqual(str(env.get("SAG_REQUIRE_HW_ENCODE")), "1")
        self.assertNotIn("PORTAL_REQUIRE_HW_ENCODE", env)
        self.assertIn("video", str(env.get("NVIDIA_DRIVER_CAPABILITIES", "")))
        self.assertEqual(str(env.get("NVIDIA_VISIBLE_DEVICES")), "all")
        self.assertEqual(str(env.get("LD_LIBRARY_PATH")), "/opt/tegra-libs")
        for extra in ("nvmap", "nvhost-ctrl", "nvhost-vic"):
            self.assertTrue(any(extra in d for d in devices), extra)
        groups = {str(g) for g in (robot.get("group_add") or [])}
        for g in ("44", "103", "994"):
            self.assertIn(g, groups)
        vols = [str(v) for v in (robot.get("volumes") or [])]
        self.assertTrue(any("nv_tegra_release" in v for v in vols))
        self.assertTrue(any("libnvtvmr.so" in v for v in vols))
        self.assertTrue(any("10_nvidia.json" in v or "nvidia.json" in v for v in vols))
        self.assertTrue(any("libEGL_nvidia" in v for v in vols))

    def test_mock_and_operator_not_fail_closed(self):
        for name in ("mock", "operator"):
            svc = self.services[name]
            env = self._env(svc)
            self.assertNotEqual(str(env.get("SAG_REQUIRE_HW_ENCODE", "")), "1", name)
            self.assertNotEqual(str(env.get("PORTAL_REQUIRE_HW_ENCODE", "")), "1", name)
            self.assertNotEqual(svc.get("runtime"), "nvidia", name)
            devices = [str(d) for d in (svc.get("devices") or [])]
            self.assertFalse(any("nvhost-msenc" in d for d in devices), name)


class RgbContractTests(unittest.TestCase):
    def test_video_loop_sends_rgb_not_h264(self):
        loop_path = os.path.join(_teleop_dir, "teleop_robot.py")
        with open(loop_path, "r") as f:
            src = f.read()
        self.assertIn("rgb = np.ascontiguousarray(head.bgr[:, :, ::-1])", src)
        self.assertIn("portal.send_video_frame(", src)
        self.assertIn("track, rgb", src)

        h, w = 8, 12
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[..., 0] = 10
        bgr[..., 1] = 20
        bgr[..., 2] = 30
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        self.assertEqual(rgb.shape, (h, w, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        np.testing.assert_array_equal(rgb[0, 0], [30, 20, 10])
        self.assertNotIn(b"\x00\x00\x00\x01", rgb.tobytes()[:32])

    def test_portal_yaml_h264_not_preencoded(self):
        with open(PORTAL_YAML, "r") as f:
            wire = yaml.safe_load(f)
        videos = wire.get("videos") or []
        self.assertTrue(videos)
        self.assertEqual(videos[0]["codec"], "h264")
        self.assertEqual(videos[0]["name"], "head_camera")


if __name__ == "__main__":
    unittest.main()
