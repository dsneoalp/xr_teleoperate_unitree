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

        self._stop_evt = threading.Event()
        self._connected_evt = threading.Event()
        self._connect_error = None
        self._loop = None
        self._thread = threading.Thread(target=self._portal_loop, daemon=True)
        self._thread.start()
        logger_mp.info(f"[portal-robot] connecting '{identity}' to room '{self._room}' at {self._url} ...")

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
        try:
            self._robot.send_video_frame(track, frame, width=width, height=height,
                                         timestamp_us=timestamp_us)
        except Exception as exc:
            logger_mp.warning(f"[portal-robot] send_video_frame '{track}' failed: {exc}")

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
        logger_mp.info("[portal-robot] closed.")
