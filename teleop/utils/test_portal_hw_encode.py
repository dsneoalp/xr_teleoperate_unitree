"""Offline tests for Portal Jetson HW-encode probe and compose contract.

No LiveKit, no DDS, no Docker. Run from repo root:

    python -m unittest teleop.utils.test_portal_hw_encode
"""
from __future__ import annotations

import os
import struct
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
    EncodeUnavailableError,
    JetsonMsencEncodeCapability,
    MMAPI_LOG,
    OPENH264_LOG,
    assert_encode_ready,
    evaluate_hw_encode_evidence,
    hw_encode_open_fds,
    hw_encode_unavailable_message,
)

COMPOSE_YML = os.path.join(_repo_root, "docker", "compose.yml")
PORTAL_YAML = os.path.join(_teleop_dir, "portal.yaml")
MAPPING_YAML = os.path.join(_teleop_dir, "portal_mapping.yaml")


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


def _make_transport(order=None):
    order = order if order is not None else []
    env = {
        "LIVEKIT_URL": "ws://127.0.0.1:7880",
        "LIVEKIT_API_KEY": "devkey",
        "LIVEKIT_API_SECRET": "secret",
        "LIVEKIT_ROOM": "g1-portal",
        "SAG_REQUIRE_HW_ENCODE": "1",
        "SAG_HW_ENCODE_CONFIRM_S": "1",
    }
    fake_portal = _fake_livekit_module(order)
    fake_dev = tempfile.NamedTemporaryFile()
    real_cap = portal_robot_mod.JetsonMsencEncodeCapability

    def cap_factory(*, require_hw=True, device_path=None):
        return real_cap(
            device_path=fake_dev.name if device_path is None else device_path,
            require_hw=require_hw,
        )

    with mock.patch.dict(os.environ, env):
        with mock.patch.object(portal_robot_mod, "JetsonMsencEncodeCapability", cap_factory):
            with mock.patch.dict(sys.modules, {"livekit.portal": fake_portal}):
                with mock.patch.object(portal_robot_mod, "_load_dotenv", lambda _p: None):
                    with mock.patch.object(threading.Thread, "start", lambda self: None):
                        portal = portal_robot_mod.PortalRobotTransport(
                            portal_yaml=PORTAL_YAML,
                            mapping_yaml=MAPPING_YAML,
                            env_file=os.path.join(_teleop_dir, ".env"),
                            identity="test-robot",
                        )
    portal._fake_dev = fake_dev  # keep file until test ends
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
        self.assertIn("graphics", str(env.get("NVIDIA_DRIVER_CAPABILITIES", "")))
        self.assertEqual(str(env.get("EGL_PLATFORM")), "device")
        self.assertEqual(str(env.get("NVIDIA_VISIBLE_DEVICES")), "all")
        self.assertEqual(str(env.get("LD_LIBRARY_PATH")), "/opt/tegra-libs")
        for extra in ("nvmap", "nvhost-ctrl", "nvhost-vic", "nvhost-gpu"):
            self.assertTrue(any(extra in d for d in devices), extra)
        groups = {str(g) for g in (robot.get("group_add") or [])}
        for g in ("44", "103", "994"):
            self.assertIn(g, groups)
        vols = [str(v) for v in (robot.get("volumes") or [])]
        self.assertTrue(any("nv_tegra_release" in v for v in vols))
        self.assertTrue(any("libnvtvmr.so" in v for v in vols))
        self.assertTrue(any("10_nvidia.json" in v or "nvidia.json" in v for v in vols))
        self.assertTrue(any("libEGL_nvidia" in v for v in vols))
        self.assertTrue(any("libnvidia-glcore" in v for v in vols))
        self.assertTrue(any("libnvidia-egl-gbm" in v for v in vols))
        self.assertEqual(
            str(env.get("__EGL_EXTERNAL_PLATFORM_CONFIG_DIRS")),
            "/usr/share/egl/egl_external_platform.d",
        )
        self.assertFalse(any("/dev/dri" in d for d in devices))

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
        bgr_path = os.path.join(_teleop_dir, "utils", "bgr_source.py")
        with open(loop_path, "r") as f:
            src = f.read()
        with open(bgr_path, "r") as f:
            src += f.read()
        self.assertIn("cv2.cvtColor", src)
        self.assertIn("COLOR_BGR2RGB", src)
        self.assertIn("bgr_to_rgb", src)
        self.assertIn("JPEG skipped", src)
        self.assertIn("_load_local_cam_config", src)
        self.assertIn("cam_config_server.yaml", src)
        self.assertIn("BgrCameraSource", src)
        self.assertIn("bgr_zmq_port", src)
        self.assertIn("_connect_frame_sources", src)
        self.assertIn("MosaicCompositor", src)
        self.assertIn("compositor.canvas", src)
        self.assertIn("mosaic.yaml", src)
        self.assertNotIn("ascontiguousarray(head.bgr", src)
        self.assertNotIn("cv2.resize", src)
        self.assertIn("portal.send_video_frame(", src)
        self.assertIn("send_ms", src)
        self.assertIn("loop_ms", src)
        self.assertIn("_log_cam_config", src)
        self.assertIn("EncodeUnavailableError", src)
        self.assertIn("raise SystemExit(1)", src)

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
        self.assertEqual(len(videos), 1)
        self.assertEqual(videos[0]["name"], "head_camera")
        self.assertEqual(videos[0]["codec"], "h264")
        self.assertGreaterEqual(int(videos[0]["max_bitrate_kbps"]), 4000)


class BgrSourceTests(unittest.TestCase):
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
        from teleop.utils.bgr_source import parse_bgr_payload

        height, width = 720, 1280
        pixels = bytes(range(256)) * ((height * width * 3 + 255) // 256)
        pixels = pixels[: height * width * 3]
        legacy = struct.pack("<HH", height, width) + pixels
        h, w, ts, raw = parse_bgr_payload(legacy)
        self.assertEqual((h, w, ts), (720, 1280, None))
        self.assertEqual(len(raw), height * width * 3)

        stamped = struct.pack("<HH", height, width) + struct.pack("<Q", 42) + pixels
        h, w, ts, raw = parse_bgr_payload(stamped)
        self.assertEqual((h, w, ts), (720, 1280, 42))
        self.assertEqual(len(raw), height * width * 3)

        with self.assertRaises(ValueError):
            parse_bgr_payload(b"\x00")
        with self.assertRaises(ValueError):
            parse_bgr_payload(struct.pack("<HH", 2, 2) + b"\x00")

    def test_even_crop_no_resize(self):
        from teleop.utils import bgr_source as bgr_mod

        even = np.zeros((720, 1280, 3), dtype=np.uint8)
        cropped = bgr_mod.even_crop(even)
        self.assertIs(cropped, even)
        self.assertEqual(cropped.shape, (720, 1280, 3))

        odd = np.zeros((479, 641, 3), dtype=np.uint8)
        out = bgr_mod.even_crop(odd)
        self.assertEqual(out.shape, (478, 640, 3))
        with open(bgr_mod.__file__, "r", encoding="utf-8") as src_file:
            src = src_file.read()
        self.assertNotIn("cv2.resize", src)
        self.assertNotIn("max_width", src)
        self.assertNotIn("INTER_AREA", src)


class MosaicTests(unittest.TestCase):
    def test_layout_720p_2x2_even(self):
        from teleop.utils.mosaic import load_mosaic_layout

        layout_path = os.path.join(_teleop_dir, "mosaic.yaml")
        layout = load_mosaic_layout(layout_path)
        self.assertEqual((layout.width, layout.height), (1280, 720))
        self.assertEqual(layout.slots, ("head_camera", "left_wrist_camera", "right_wrist_camera"))
        head, left, right = layout.tiles
        self.assertEqual((head.x, head.y, head.w, head.h), (0, 0, 640, 360))
        self.assertEqual(head.fit, "fill")
        self.assertEqual((left.x, left.y, left.w, left.h), (640, 0, 640, 360))
        self.assertEqual(left.fit, "fill")
        self.assertEqual((right.x, right.y, right.w, right.h), (0, 360, 640, 360))
        self.assertEqual(right.fit, "fill")

    def test_compose_places_tiles(self):
        from teleop.utils.mosaic import MosaicCompositor, load_mosaic_layout

        layout = load_mosaic_layout(os.path.join(_teleop_dir, "mosaic.yaml"))
        comp = MosaicCompositor(layout)
        head = np.zeros((720, 1280, 3), dtype=np.uint8)
        head[:] = (10, 20, 30)
        left = np.zeros((720, 1280, 3), dtype=np.uint8)
        left[:] = (40, 50, 60)
        right = np.zeros((720, 1280, 3), dtype=np.uint8)
        right[:] = (70, 80, 90)
        comp.paste("head_camera", head)
        comp.paste("left_wrist_camera", left)
        comp.paste("right_wrist_camera", right)
        canvas = comp.canvas
        self.assertEqual(canvas.shape, (720, 1280, 3))
        np.testing.assert_array_equal(canvas[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(canvas[180, 320], [10, 20, 30])
        np.testing.assert_array_equal(canvas[180, 960], [40, 50, 60])
        np.testing.assert_array_equal(canvas[540, 320], [70, 80, 90])
        np.testing.assert_array_equal(canvas[540, 960], [0, 0, 0])
        head[:] = (11, 22, 33)
        comp.paste("head_camera", head)
        np.testing.assert_array_equal(comp.canvas[0, 0], [11, 22, 33])
        np.testing.assert_array_equal(comp.canvas[180, 960], [40, 50, 60])
        np.testing.assert_array_equal(comp.canvas[540, 320], [70, 80, 90])


class LoopTimingTests(unittest.TestCase):
    def test_flush_includes_hz(self):
        from teleop.utils.loop_timing import LoopTiming

        logs: list[str] = []

        class _Log:
            def info(self, msg):
                logs.append(msg)

        timing = LoopTiming(_Log(), interval_s=0.0)
        timing.add("loop_ms", 5.0)
        timing.add("video_send_ms", 12.5)
        timing.count("video_new", 3)
        joined = " ".join(logs)
        self.assertIn("loop_ms", joined)
        self.assertIn("video_send_ms", joined)
        self.assertIn("hz=", joined)
        self.assertIn("p50=", joined)
        self.assertIn("video_new", joined)


if __name__ == "__main__":
    unittest.main()
