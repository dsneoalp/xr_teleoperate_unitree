"""Offline tests for Portal Jetson HW-encode probe and compose contract.

No LiveKit, no DDS, no Docker. From repo root:

    pytest teleop/tests/robot_control/test_portal_hw_encode.py
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

from teleop.tests.paths import (
    BGR_SOURCE,
    PORTAL_MAPPING,
    PORTAL_ROBOT,
    PORTAL_YAML,
    REPO_ROOT,
    TELEOP_DIR,
    TELEOP_ROBOT,
)
from teleop.robot_control import portal_robot as portal_robot_mod
from teleop.robot_control.encode import (
    EncodeUnavailableError,
    JetsonMsencEncodeCapability,
    MMAPI_LOG,
    OPENH264_LOG,
    assert_encode_ready,
    evaluate_hw_encode_evidence,
    hw_encode_open_fds,
    hw_encode_unavailable_message,
)

COMPOSE_YML = os.path.join(REPO_ROOT, "docker", "compose.yml")
COMPOSE_G1_YML = os.path.join(REPO_ROOT, "docker", "compose.g1.yml")


class ProbeTests(unittest.TestCase):
    def test_require_hw_missing_device_unavailable(self):
        missing = os.path.join(tempfile.gettempdir(), "no-msenc")
        cap = JetsonMsencEncodeCapability(device_path=missing, require_hw=True)
        report = cap.probe()
        self.assertEqual(report.backend, "unavailable")
        with self.assertRaises(EncodeUnavailableError) as ctx:
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
            with self.assertRaises(EncodeUnavailableError) as ctx:
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

    def test_evidence_ok_needs_fd_and_mmapi_log(self):
        self.assertEqual(
            evaluate_hw_encode_evidence(["/dev/nvhost-msenc"], f"x {MMAPI_LOG} y"),
            "ok",
        )

    def test_evidence_pending_without_log_or_fd(self):
        self.assertEqual(evaluate_hw_encode_evidence([], MMAPI_LOG), "pending")
        self.assertEqual(
            evaluate_hw_encode_evidence(["/dev/nvhost-msenc"], "idle"),
            "pending",
        )

    def test_evidence_openh264_without_mmapi_is_fail(self):
        self.assertEqual(
            evaluate_hw_encode_evidence([], f"{OPENH264_LOG} software"),
            "openh264",
        )
        self.assertEqual(
            evaluate_hw_encode_evidence(
                ["/dev/nvhost-msenc"], f"{OPENH264_LOG} and {MMAPI_LOG}"
            ),
            "ok",
        )
        msg = hw_encode_unavailable_message("openh264", fds=[], n_frames=3)
        self.assertIn("encode_unavailable", msg)
        self.assertIn("OpenH264", msg)


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

        def on_operator_joined(self, cb):
            pass

        def on_operator_left(self, cb):
            pass

        def send_video_frame(self, *a, **k):
            order.append("send_video")

        def send_state(self, *a, **k):
            pass

        def disconnect(self):
            pass

        def close(self):
            pass

    mod.RobotConfig = RobotConfig
    mod.Robot = Robot
    return mod


def _livekit_env():
    return {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
        "LIVEKIT_ROOM": "g1-portal",
        "SAG_HW_ENCODE_CONFIRM_S": "1",
    }


class InitOrderTests(unittest.TestCase):
    def test_init_probes_after_robot_when_require_hw(self):
        order: list[str] = []
        real_assert = portal_robot_mod.assert_encode_ready
        real_cap = portal_robot_mod.JetsonMsencEncodeCapability

        def wrapped_assert(capability):
            order.append("probe")
            return real_assert(capability)

        fake_portal = _fake_livekit_module(order)
        with tempfile.NamedTemporaryFile() as fake_dev:
            def cap_factory(*, require_hw=True, device_path=None):
                return real_cap(
                    device_path=fake_dev.name if device_path is None else device_path,
                    require_hw=require_hw,
                )

            with mock.patch.dict(os.environ, _livekit_env()):
                with mock.patch.object(portal_robot_mod, "assert_encode_ready", wrapped_assert):
                    with mock.patch.object(
                        portal_robot_mod, "JetsonMsencEncodeCapability", cap_factory
                    ):
                        with mock.patch.dict(sys.modules, {"livekit.portal": fake_portal}):
                            with mock.patch.object(portal_robot_mod, "_load_dotenv", lambda _p: None):
                                with mock.patch.object(threading.Thread, "start", lambda self: None):
                                    portal_robot_mod.PortalRobotTransport(
                                        portal_yaml=PORTAL_YAML,
                                        mapping_yaml=PORTAL_MAPPING,
                                        env_file=os.path.join(TELEOP_DIR, ".env"),
                                        identity="test-robot",
                                        require_hw_encode=True,
                                    )
        self.assertIn("probe", order)
        self.assertIn("config", order)
        self.assertIn("robot", order)
        self.assertLess(order.index("config"), order.index("probe"))
        self.assertLess(order.index("robot"), order.index("probe"))

    def test_init_skips_probe_without_require_hw(self):
        order: list[str] = []

        def wrapped_assert(capability):
            order.append("probe")
            return portal_robot_mod.assert_encode_ready(capability)

        fake_portal = _fake_livekit_module(order)
        with mock.patch.dict(os.environ, _livekit_env()):
            with mock.patch.object(portal_robot_mod, "assert_encode_ready", wrapped_assert):
                with mock.patch.dict(sys.modules, {"livekit.portal": fake_portal}):
                    with mock.patch.object(portal_robot_mod, "_load_dotenv", lambda _p: None):
                        with mock.patch.object(threading.Thread, "start", lambda self: None):
                            portal_robot_mod.PortalRobotTransport(
                                portal_yaml=PORTAL_YAML,
                                mapping_yaml=PORTAL_MAPPING,
                                env_file=os.path.join(TELEOP_DIR, ".env"),
                                identity="test-robot",
                                require_hw_encode=False,
                            )
        self.assertNotIn("probe", order)
        self.assertIn("robot", order)


def _make_transport(order=None, require_hw_encode=True):
    order = order if order is not None else []
    fake_portal = _fake_livekit_module(order)
    fake_dev = tempfile.NamedTemporaryFile()
    real_cap = portal_robot_mod.JetsonMsencEncodeCapability

    def cap_factory(*, require_hw=True, device_path=None):
        return real_cap(
            device_path=fake_dev.name if device_path is None else device_path,
            require_hw=require_hw,
        )

    with mock.patch.dict(os.environ, _livekit_env()):
        with mock.patch.object(portal_robot_mod, "JetsonMsencEncodeCapability", cap_factory):
            with mock.patch.dict(sys.modules, {"livekit.portal": fake_portal}):
                with mock.patch.object(portal_robot_mod, "_load_dotenv", lambda _p: None):
                    with mock.patch.object(threading.Thread, "start", lambda self: None):
                        portal = portal_robot_mod.PortalRobotTransport(
                            portal_yaml=PORTAL_YAML,
                            mapping_yaml=PORTAL_MAPPING,
                            env_file=os.path.join(TELEOP_DIR, ".env"),
                            identity="test-robot",
                            require_hw_encode=require_hw_encode,
                        )
    portal._fake_dev = fake_dev
    return portal


class ConfirmOnSendTests(unittest.TestCase):
    def tearDown(self):
        fake = getattr(self, "_fake_dev", None)
        if fake is not None:
            fake.close()

    def test_send_video_raises_on_openh264(self):
        portal = _make_transport()
        self._fake_dev = portal._fake_dev
        tee = mock.Mock()
        tee.text.return_value = f"{OPENH264_LOG} fallback"
        portal._tee = tee
        with mock.patch.object(portal_robot_mod, "hw_encode_open_fds", return_value=[]):
            with self.assertRaises(EncodeUnavailableError) as ctx:
                portal.send_video_frame("head_camera", b"\x00" * 12, width=2, height=2)
        self.assertIn("OpenH264", str(ctx.exception))
        self.assertIsNotNone(portal.hw_encode_error)
        self.assertFalse(portal.hw_encode_confirmed)

    def test_send_video_confirms_on_fd_and_mmapi(self):
        portal = _make_transport()
        self._fake_dev = portal._fake_dev
        tee = mock.Mock()
        tee.text.return_value = f"ok {MMAPI_LOG}"
        portal._tee = tee
        with mock.patch.object(
            portal_robot_mod, "hw_encode_open_fds", return_value=["/dev/nvhost-msenc"]
        ):
            portal.send_video_frame("head_camera", b"\x00" * 12, width=2, height=2)
        self.assertTrue(portal.hw_encode_confirmed)
        self.assertIsNone(portal.hw_encode_error)
        tee.close.assert_called()

    def test_send_video_timeout_without_evidence(self):
        portal = _make_transport()
        self._fake_dev = portal._fake_dev
        portal._confirm_s = 0.0
        tee = mock.Mock()
        tee.text.return_value = "silence"
        portal._tee = tee
        with mock.patch.object(portal_robot_mod, "hw_encode_open_fds", return_value=[]):
            with self.assertRaises(EncodeUnavailableError) as ctx:
                portal.send_video_frame("head_camera", b"\x00" * 12, width=2, height=2)
        self.assertIn("no Jetson MMAPI", str(ctx.exception))

    def test_send_video_skips_dod_without_require_hw(self):
        portal = _make_transport(require_hw_encode=False)
        self._fake_dev = portal._fake_dev
        portal.send_video_frame("head_camera", b"\x00" * 12, width=2, height=2)
        self.assertFalse(portal.hw_encode_confirmed)
        self.assertIsNone(portal.hw_encode_error)


class ComposeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(COMPOSE_YML, "r") as f:
            cls.data = yaml.safe_load(f)
        with open(COMPOSE_G1_YML, "r") as f:
            cls.g1 = yaml.safe_load(f)
        cls.services = cls.data["services"]
        cls.g1_services = cls.g1["services"]

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

    def test_pc_robot_has_no_nvidia_and_no_hw_env(self):
        robot = self.services["robot"]
        self.assertNotEqual(robot.get("runtime"), "nvidia")
        env = self._env(robot)
        self.assertNotEqual(str(env.get("SAG_REQUIRE_HW_ENCODE", "")), "1")
        dockerfile = (robot.get("build") or {}).get("dockerfile", "")
        self.assertIn("Dockerfile.robot", dockerfile)
        self.assertNotIn("robot-g1", dockerfile)

    def test_robot_g1_has_nvidia_msenc_without_require_env(self):
        robot = self.g1_services["robot-g1"]
        self.assertEqual(robot.get("runtime"), "nvidia")
        devices = [str(d) for d in (robot.get("devices") or [])]
        self.assertTrue(any("nvhost-msenc" in d for d in devices))
        self.assertFalse(any("v4l2-nvenc" in d for d in devices))
        self.assertNotEqual(robot.get("privileged"), True)
        env = self._env(robot)
        self.assertNotEqual(str(env.get("SAG_REQUIRE_HW_ENCODE", "")), "1")
        self.assertIn("video", str(env.get("NVIDIA_DRIVER_CAPABILITIES", "")))
        self.assertEqual(str(env.get("NVIDIA_VISIBLE_DEVICES")), "all")
        self.assertEqual(str(env.get("LD_LIBRARY_PATH")), "/opt/tegra-libs")
        cmd = [str(c) for c in (robot.get("command") or [])]
        self.assertNotIn("--require-hw-encode", cmd)
        dockerfile = (robot.get("build") or {}).get("dockerfile", "")
        self.assertIn("Dockerfile.robot-g1", dockerfile)
        groups = {str(g) for g in (robot.get("group_add") or [])}
        for g in ("44", "103", "994"):
            self.assertIn(g, groups)
        vols = [str(v) for v in (robot.get("volumes") or [])]
        self.assertTrue(any("nv_tegra_release" in v for v in vols))
        self.assertTrue(any("libnvtvmr.so" in v for v in vols))

    def test_mock_and_operator_not_fail_closed(self):
        for name in ("mock", "operator"):
            svc = self.services[name]
            env = self._env(svc)
            self.assertNotEqual(str(env.get("SAG_REQUIRE_HW_ENCODE", "")), "1", name)
            self.assertNotEqual(svc.get("runtime"), "nvidia", name)
            dockerfile = (svc.get("build") or {}).get("dockerfile", "")
            self.assertIn(f"Dockerfile.{name}", dockerfile)


class RgbContractTests(unittest.TestCase):
    def test_video_loop_sends_rgb_not_h264(self):
        with open(TELEOP_ROBOT, "r") as f:
            robot_src = f.read()
        with open(PORTAL_ROBOT, "r") as f:
            portal_src = f.read()
        with open(BGR_SOURCE, "r") as f:
            ingest_src = f.read()
        self.assertIn("latest_rgb()", portal_src)
        self.assertIn("portal.send_video_frame(", portal_src)
        self.assertIn("timestamp_us=ts", portal_src)
        self.assertIn("EncodeUnavailableError", robot_src)
        self.assertIn("raise SystemExit(1)", robot_src)
        self.assertIn("--require-hw-encode", robot_src)
        self.assertIn("connect_frame_sources", robot_src)
        self.assertIn("video_publish_loop", robot_src)
        self.assertIn("BgrCameraSource", ingest_src)
        self.assertIn("JpegSlotSource", ingest_src)
        self.assertNotIn("ascontiguousarray(head.bgr", robot_src + portal_src + ingest_src)

        h, w = 8, 12
        bgr = np.zeros((h, w, 3), dtype=np.uint8)
        bgr[..., 0] = 10
        bgr[..., 1] = 20
        bgr[..., 2] = 30
        rgb = np.empty_like(bgr)
        rgb[:] = bgr[:, :, ::-1]
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
        self.assertNotIn("annexb", str(videos[0]).lower())

    def test_bgr_zmq_port_offset_and_disabled(self):
        from teleop.utils.bgr_source import bgr_zmq_port
        cfg = {
            "head_camera": {"zmq_port": 55555, "enable_bgr_zmq": True},
            "left_wrist_camera": {"zmq_port": 55556, "enable_bgr_zmq": True},
        }
        self.assertEqual(bgr_zmq_port(cfg), 56555)
        self.assertEqual(bgr_zmq_port(cfg, "left_wrist_camera"), 56556)
        self.assertIsNone(
            bgr_zmq_port({"head_camera": {"zmq_port": 55555, "enable_bgr_zmq": False}})
        )
        self.assertIsNone(bgr_zmq_port({"head_camera": {"zmq_port": 55555}}))
        self.assertIsNone(bgr_zmq_port({}))

    def test_parse_bgr_payload_legacy_and_capture_ms(self):
        import struct
        from teleop.utils.bgr_source import parse_bgr_payload

        frame = bytes(2 * 2 * 3)
        legacy = struct.pack("<HH", 2, 2) + frame
        h, w, ts, raw = parse_bgr_payload(legacy)
        self.assertEqual((h, w, ts), (2, 2, None))
        self.assertEqual(bytes(raw), frame)
        stamped = struct.pack("<HH", 2, 2) + struct.pack("<Q", 99) + frame
        h, w, ts, raw = parse_bgr_payload(stamped)
        self.assertEqual((h, w, ts), (2, 2, 99))
        with self.assertRaises(ValueError):
            parse_bgr_payload(b"\x00")
        with self.assertRaises(ValueError):
            parse_bgr_payload(struct.pack("<HH", 2, 2) + b"\x00")

    def test_even_crop(self):
        from teleop.utils import bgr_source as bgr_mod
        even = np.zeros((8, 10, 3), dtype=np.uint8)
        cropped = bgr_mod.even_crop(even)
        self.assertIs(cropped, even)
        odd = np.zeros((9, 11, 3), dtype=np.uint8)
        out = bgr_mod.even_crop(odd)
        self.assertEqual(out.shape[:2], (8, 10))
