"""Teleimager ingest: BGR preferred, JPEG fallback, no Portal FFI."""
from __future__ import annotations

from unittest import mock

import numpy as np

from teleop.utils.bgr_source import JpegSlotSource, connect_frame_sources


class _Frame:
    def __init__(self, bgr):
        self.bgr = bgr


class _FakeSubscriber:
    def __init__(self, frame):
        self.frame = frame
        self.subscribes = []

    def subscribe(self, host, port, request_bgr=True):
        self.subscribes.append((host, port, request_bgr))
        return self.frame


class _FakeBgrSource:
    def __init__(self, host, port, slot="head_camera"):
        self.host = host
        self.port = port
        self.slot = slot
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def close(self):
        self.closed = True


class _FakeJpegSource:
    def __init__(self, host, cam_config, slot, request_bgr=True):
        self.host = host
        self.slot = slot
        self.cam_config = cam_config
        self.request_bgr = request_bgr

    def close(self):
        pass


def test_jpeg_latest_rgb_converts_bgr():
    bgr = np.zeros((4, 6, 3), dtype=np.uint8)
    bgr[..., 0] = 10
    bgr[..., 2] = 30
    mgr = _FakeSubscriber(_Frame(bgr))
    src = JpegSlotSource(
        "127.0.0.1",
        {"head_camera": {"enable_zmq": True, "zmq_port": 55555}},
        "head_camera",
        subscriber_manager=mgr,
    )
    rgb, seq = src.latest_rgb()
    assert seq == 1
    np.testing.assert_array_equal(rgb[0, 0], [30, 0, 10])
    assert mgr.subscribes[-1] == ("127.0.0.1", 55555, True)


def test_jpeg_latest_rgb_none_without_frame():
    mgr = _FakeSubscriber(None)
    src = JpegSlotSource(
        "127.0.0.1",
        {"head_camera": {"enable_zmq": True, "zmq_port": 1}},
        "head_camera",
        subscriber_manager=mgr,
    )
    assert src.latest_rgb() is None


def test_connect_prefers_bgr_zmq():
    cfg = {
        "head_camera": {
            "zmq_port": 55555,
            "enable_bgr_zmq": True,
            "enable_zmq": True,
        },
    }
    with mock.patch(
        "teleop.utils.bgr_source.fetch_teleimager_config", return_value=cfg
    ), mock.patch(
        "teleop.utils.bgr_source.BgrCameraSource", _FakeBgrSource
    ):
        sources = connect_frame_sources("10.0.0.1", ["head_camera"], retries=1, retry_s=0)
    assert set(sources) == {"head_camera"}
    assert sources["head_camera"].port == 56555
    assert sources["head_camera"].started is True


def test_connect_jpeg_fallback_without_bgr():
    cfg = {"head_camera": {"zmq_port": 55555, "enable_zmq": True}}
    with mock.patch(
        "teleop.utils.bgr_source.fetch_teleimager_config", return_value=cfg
    ), mock.patch(
        "teleop.utils.bgr_source.JpegSlotSource", _FakeJpegSource
    ):
        sources = connect_frame_sources("10.0.0.1", ["head_camera"], retries=1, retry_s=0)
    assert set(sources) == {"head_camera"}
    assert sources["head_camera"].slot == "head_camera"


def test_connect_empty_when_no_config():
    with mock.patch(
        "teleop.utils.bgr_source.fetch_teleimager_config", return_value=None
    ), mock.patch(
        "teleop.utils.bgr_source.load_local_cam_config", return_value=None
    ):
        sources = connect_frame_sources("10.0.0.1", ["head_camera"], retries=1, retry_s=0)
    assert sources == {}
