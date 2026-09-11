"""Thin LiveKit Portal Robot transport.

Not an arm_ctrl stand-in. The robot control loop owns G1_29_ArmController /
Dex3 / loco and calls this only to receive actions and publish state.
"""
from __future__ import annotations

import os
import sys
import time
import asyncio
import threading

import yaml

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
parent2_dir = os.path.dirname(parent_dir)
if parent2_dir not in sys.path:
    sys.path.append(parent2_dir)

from teleop.robot_control.encode import (
    EncodeUnavailableError,
    FdTee,
    JetsonMsencEncodeCapability,
    MMAPI_LOG,
    assert_encode_ready,
    evaluate_hw_encode_evidence,
    hw_encode_confirm_timeout_s,
    hw_encode_open_fds,
    hw_encode_unavailable_message,
)
from teleop.robot_control.portal_mapping import PortalMapping
from teleop.robot_control.portal_operator import mint_portal_token, _load_dotenv


class PortalRobotTransport:
    """connect / on_action / send_state around livekit.portal.Robot."""

    def __init__(self,
                 portal_yaml: str,
                 env_file: str,
                 identity: str = "xr-robot",
                 room: str | None = None,
                 url: str | None = None,
                 mapping_yaml: str | None = None):
        from livekit.portal import Robot, RobotConfig

        _load_dotenv(env_file)
        self._url = url or os.environ.get("LIVEKIT_URL")
        self._room = room or os.environ.get("LIVEKIT_ROOM", "g1-portal")
        api_key = os.environ.get("LIVEKIT_API_KEY")
        api_secret = os.environ.get("LIVEKIT_API_SECRET")
        if not all([self._url, api_key, api_secret]):
            raise RuntimeError(
                "LIVEKIT_URL / LIVEKIT_API_KEY / LIVEKIT_API_SECRET missing "
                f"(looked in env and {env_file})"
            )

        mapping_yaml = mapping_yaml or os.path.join(parent_dir, "portal_mapping.yaml")
        self.mapping = PortalMapping(mapping_yaml, portal_yaml)
        self._identity = identity

        with open(portal_yaml, "r") as f:
            wire = yaml.safe_load(f) or {}
        self.video_tracks = [v["name"] for v in (wire.get("videos") or [])]

        cfg = RobotConfig.from_yaml_file(portal_yaml, self._room)
        self._robot = Robot(cfg)
        self._action_cb = None
        self._robot.on_action(self._on_action)

        self._require_hw = os.environ.get("SAG_REQUIRE_HW_ENCODE", "0") == "1"
        if self.video_tracks or self._require_hw:
            report = assert_encode_ready(
                JetsonMsencEncodeCapability(require_hw=self._require_hw)
            )
            logger_mp.info(
                f"[portal-robot] encode_capability backend={report.backend} "
                f"detail={report.detail}"
            )

        self._hw_lock = threading.Lock()
        self._hw_confirmed = False
        self._hw_deadline: float | None = None
        self._hw_frames = 0
        self._hw_error: BaseException | None = None
        self._tee: FdTee | None = None
        self._confirm_s = hw_encode_confirm_timeout_s()

        self._stop_evt = threading.Event()
        self._connected_evt = threading.Event()
        self._connect_error = None
        self._loop = None
        self._thread = threading.Thread(target=self._portal_loop, daemon=True)
        self._thread.start()
        logger_mp.info(f"[portal-robot] connecting '{identity}' to room '{self._room}' at {self._url} ...")

    @property
    def hw_encode_error(self) -> BaseException | None:
        return self._hw_error

    @property
    def hw_encode_confirmed(self) -> bool:
        return self._hw_confirmed

    def on_unpacked_action(self, callback) -> None:
        """callback(UnpackedAction) on the portal thread."""
        self._action_cb = callback

    def _on_action(self, action) -> None:
        raw = getattr(action, "raw_values", None) or getattr(action, "values", None) or {}
        unpacked = self.mapping.unpack_action(raw)
        cb = self._action_cb
        if cb is not None:
            try:
                cb(unpacked)
            except Exception as exc:
                logger_mp.warning(f"[portal-robot] action callback failed: {exc}")

    def send_state(self, motor_q=None, arm_q=None, hand_q=None, fsm_id=0, timestamp_us=None) -> None:
        values = self.mapping.pack_state(motor_q=motor_q, arm_q=arm_q, hand_q=hand_q, fsm_id=fsm_id)
        ts = timestamp_us if timestamp_us is not None else int(time.time() * 1_000_000)
        try:
            self._robot.send_state(values, timestamp_us=ts)
        except Exception as exc:
            logger_mp.warning(f"[portal-robot] send_state failed: {exc}")

    def send_video_frame(self, track: str, frame, width=None, height=None, timestamp_us=None) -> None:
        if self._hw_error is not None:
            raise self._hw_error
        try:
            self._robot.send_video_frame(track, frame, width=width, height=height,
                                         timestamp_us=timestamp_us)
        except EncodeUnavailableError:
            raise
        except Exception as exc:
            logger_mp.warning(f"[portal-robot] send_video_frame '{track}' failed: {exc}")
            return
        self._confirm_hw_encode()

    def _confirm_hw_encode(self) -> None:
        if not self._require_hw or not self.video_tracks:
            return
        with self._hw_lock:
            if self._hw_confirmed or self._hw_error is not None:
                return
            self._hw_frames += 1
            if self._hw_deadline is None:
                self._hw_deadline = time.monotonic() + self._confirm_s
            fds = hw_encode_open_fds()
            log_text = self._tee.text() if self._tee is not None else ""
            verdict = evaluate_hw_encode_evidence(fds, log_text)
            if verdict == "ok":
                self._hw_confirmed = True
                logger_mp.info(
                    f"[portal-robot] HW encode confirmed after {self._hw_frames} frames "
                    f"fds={fds} log={MMAPI_LOG!r}"
                )
                self._close_tee()
                return
            if verdict == "openh264":
                err = EncodeUnavailableError(
                    hw_encode_unavailable_message(
                        verdict, fds=fds, n_frames=self._hw_frames
                    )
                )
                self._hw_error = err
                self._close_tee()
                raise err
            if time.monotonic() >= self._hw_deadline:
                err = EncodeUnavailableError(
                    hw_encode_unavailable_message(
                        "pending", fds=fds, n_frames=self._hw_frames
                    )
                )
                self._hw_error = err
                self._close_tee()
                raise err

    def _close_tee(self) -> None:
        tee = self._tee
        self._tee = None
        if tee is not None:
            try:
                tee.close()
            except Exception:
                logger_mp.warning("[portal-robot] FFI log tee close failed")

    def _portal_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:
            self._connect_error = exc
            logger_mp.error(f"[portal-robot] loop terminated: {exc}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _async_main(self):
        if self._require_hw and self.video_tracks:
            self._tee = FdTee()
        token = mint_portal_token(
            os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"],
            self._identity, self._room)
        await self._robot.connect(self._url, token)
        self._connected_evt.set()
        logger_mp.info(f"[portal-robot] connected as '{self._robot.local_identity()}'")
        while not self._stop_evt.is_set():
            await asyncio.sleep(0.05)
        logger_mp.info("[portal-robot] disconnecting ...")
        try:
            await self._robot.disconnect()
        finally:
            self._robot.close()

    def wait_until_connected(self, timeout: float = 15.0) -> None:
        if not self._connected_evt.wait(timeout):
            msg = self._connect_error or "timeout"
            raise RuntimeError(f"[portal-robot] failed to connect: {msg}")

    def close(self) -> None:
        logger_mp.info("[portal-robot] closing ...")
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._close_tee()
        logger_mp.info("[portal-robot] closed.")
