"""LiveKit Portal based robot interface for xr_teleoperate.

Replaces the Unitree DDS arm/hand controllers when teleop_hand_and_arm.py
runs with --portal:

  * IK arm joint targets and dex3 hand retargeting targets are published
    as Portal actions (operator role) at the teleop control rate.
  * Robot state (j0..j13 = dual arm q) is received via on_observation and
    fed back into the IK as warm start; if no robot state arrives the
    bridge dead-reckons with the last sent targets.
  * Video frames (declared in portal.yaml, currently head_camera) are
    received via the same observation stream and exposed through an
    ImageClient-compatible API (get_head_frame() -> .bgr) so that
    render_to_xr() and recording keep working unchanged.

No unitree_sdk2py import happens anywhere in this module.
"""
from __future__ import annotations

import os
import sys
import time
import asyncio
import threading

import numpy as np
import yaml

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))          # .../teleop/robot_control
parent_dir = os.path.dirname(current_dir)                          # .../teleop
parent2_dir = os.path.dirname(parent_dir)                          # repo root
if parent2_dir not in sys.path:
    sys.path.append(parent2_dir)

# Number of f32 joint fields per part in the portal action schema.
PORTAL_ARM_DOF = 7    # per arm (G1_29 / H1_2 style)
PORTAL_HAND_DOF = 7   # per hand (dex3-1)
PORTAL_STATE_ARM_FIELDS = 14  # j0..j13 map to left (0..6) / right (7..13) arm


class _Frame:
    """Minimal stand-in for teleimager's frame object (only .bgr is used)."""

    __slots__ = ("bgr", "timestamp_us")

    def __init__(self, bgr, timestamp_us=0):
        self.bgr = bgr
        self.timestamp_us = timestamp_us


def _load_dotenv(env_file: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        raise SystemExit("python-dotenv is required for portal mode: pip install python-dotenv")
    if env_file and os.path.isfile(env_file):
        load_dotenv(env_file, override=False)


def mint_portal_token(api_key: str, api_secret: str, identity: str, room: str,
                      ttl_hours: int = 6,
                      min_playout_delay_ms: int = 0,
                      max_playout_delay_ms: int = 1) -> str:
    """Mint a LiveKit JWT for a portal participant (Robot or Operator).

    Mirrors the upstream examples' mint_token: the portal roles self-set the
    `lk.portal.role` attribute on connect, which requires
    can_update_own_metadata; the RoomConfiguration with tight playout-delay
    bounds (0..1 ms) minimizes video latency for teleop.
    """
    import datetime
    from livekit import api
    from livekit.protocol.room import RoomConfiguration

    grants = api.VideoGrants(
        room_join=True,
        room=room,
        can_publish=True,
        can_subscribe=True,
        can_update_own_metadata=True,
    )
    room_config = RoomConfiguration(
        name=room,
        min_playout_delay=min_playout_delay_ms,
        max_playout_delay=max_playout_delay_ms,
    )
    token = (
        api.AccessToken(api_key, api_secret)
        .with_identity(identity)
        .with_grants(grants)
        .with_room_config(room_config)
        .with_ttl(datetime.timedelta(hours=ttl_hours))
    )
    return token.to_jwt()


async def claim_active_operator(op, identity: str | None = None,
                                attempts: int = 5, delay: float = 1.0) -> None:
    """Claim control with retries. Right after connect the Robot peer's
    role attribute may not have propagated yet, causing a transient
    PortalError.NoPeer on the first RPC."""
    ident = identity or op.local_identity()
    for attempt in range(1, attempts + 1):
        try:
            await op.set_active_operator(ident)
            return
        except Exception as exc:
            if attempt == attempts:
                raise
            logger_mp.debug(f"[portal] set_active_operator retry {attempt}/{attempts}: {exc}")
            await asyncio.sleep(delay)


class PortalTeleopBridge:
    """Drop-in replacement for the Unitree arm/hand controllers that
    transports actions, state and video over LiveKit Portal.

    Exposes the subset of the G1_29_ArmController / ImageClient interfaces
    used by teleop_hand_and_arm.py:

      Arm controller side:
        ctrl_dual_arm(q_target, tauff_target)
        ctrl_dual_arm_go_home()
        get_current_dual_arm_q() / get_current_dual_arm_dq()
        set_arm_velocity_limit(limit)

      Image client side:
        get_cam_config()
        get_head_frame() / get_left_wrist_frame() / get_right_wrist_frame()
        close()

    Dex3 hand retargeting runs in its own thread (same math as
    Dex3_1_Controller.control_process, minus the DDS publishing) and the
    resulting joint targets are appended to every outgoing action.
    """

    def __init__(self,
                 portal_yaml: str,
                 env_file: str,
                 identity: str = "xr-teleop",
                 room: str | None = None,
                 url: str | None = None,
                 ee: str | None = None,
                 hand_fps: float = 100.0,
                 # dex3 shared-memory wiring (same objects the DDS controller used)
                 left_hand_array_in=None,
                 right_hand_array_in=None,
                 dual_hand_data_lock=None,
                 dual_hand_state_array_out=None,
                 dual_hand_action_array_out=None,
                 xr_motion_data_ready_in=None,
                 cam_config_path: str | None = None,
                 state_timeout: float = 0.5):
        from livekit.portal import Operator, OperatorConfig

        self._portal_yaml = portal_yaml
        self._env_file = env_file
        self._identity = identity
        self._ee = ee
        self._state_timeout = state_timeout

        # --- configuration ----------------------------------------------
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

        with open(portal_yaml, "r") as f:
            self._wire = yaml.safe_load(f)
        self._declared_videos = [v["name"] for v in (self._wire.get("videos") or [])]

        self._cam_config_path = cam_config_path or os.path.join(
            parent_dir, "utils", "portal_cam_config.yaml")

        cfg = OperatorConfig.from_yaml_file(portal_yaml, self._room)
        self._op = Operator(cfg)
        self._op.on_observation(self._on_observation)
        self._op.on_drop(lambda drops: logger_mp.debug(f"[portal] dropped states: {len(drops)}"))

        # --- caches guarded by locks ------------------------------------
        self._obs_lock = threading.Lock()
        self._state_q = None            # np.array(14) from robot state
        self._state_dq = np.zeros(PORTAL_STATE_ARM_FIELDS)
        self._state_ts_wall = 0.0       # time.time() when last state arrived
        self._prev_state_q = None
        self._prev_state_ts_us = None
        self._obs_ts_us = None          # timestamp of last observation
        self._frames = {}               # track name -> _Frame(bgr)

        self._arm_lock = threading.Lock()
        self._q_target = np.zeros(2 * PORTAL_ARM_DOF)
        self._last_sent_q = np.zeros(2 * PORTAL_ARM_DOF)

        self._hand_lock = threading.Lock()
        self._hand_q = np.zeros(2 * PORTAL_HAND_DOF)
        self._fsm_id = 0

        # --- threads -----------------------------------------------------
        self._stop_evt = threading.Event()
        self._connected_evt = threading.Event()
        self._connect_error = None
        self._loop = None

        self._hand_thread = None
        if self._ee == "dex3" and left_hand_array_in is not None and right_hand_array_in is not None:
            self._left_hand_array_in = left_hand_array_in
            self._right_hand_array_in = right_hand_array_in
            self._dual_hand_data_lock = dual_hand_data_lock
            self._dual_hand_state_array_out = dual_hand_state_array_out
            self._dual_hand_action_array_out = dual_hand_action_array_out
            self._xr_motion_data_ready_in = xr_motion_data_ready_in
            self._hand_fps = hand_fps
            self._hand_thread = threading.Thread(target=self._hand_retarget_loop, daemon=True)
            self._hand_thread.start()

        self._portal_thread = threading.Thread(target=self._portal_loop, daemon=True)
        self._portal_thread.start()

        logger_mp.info(f"[portal] connecting operator '{identity}' to room '{self._room}' at {self._url} ...")

    # ------------------------------------------------------------------
    # portal asyncio plumbing
    # ------------------------------------------------------------------
    def _portal_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:  # connection failures etc.
            self._connect_error = exc
            logger_mp.error(f"[portal] operator loop terminated: {exc}")
        finally:
            try:
                self._loop.close()
            except Exception:
                pass

    async def _async_main(self):
        token = mint_portal_token(
            os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"],
            self._identity, self._room)
        await self._op.connect(self._url, token)
        await claim_active_operator(self._op)
        self._connected_evt.set()
        logger_mp.info(f"[portal] connected as '{self._op.local_identity()}'; "
                       f"active operator claimed. videos={self._declared_videos}")
        while not self._stop_evt.is_set():
            await asyncio.sleep(0.05)
        logger_mp.info("[portal] disconnecting operator ...")
        try:
            await self._op.disconnect()
        finally:
            self._op.close()

    def wait_until_connected(self, timeout: float = 15.0) -> None:
        if not self._connected_evt.wait(timeout):
            msg = self._connect_error or "timeout"
            raise RuntimeError(f"[portal] operator failed to connect: {msg}")

    # ------------------------------------------------------------------
    # inbound data
    # ------------------------------------------------------------------
    def _on_observation(self, obs) -> None:
        ts_us = getattr(obs, "timestamp_us", None)
        wall = time.time()
        q_new = None
        state = getattr(obs, "state", None) or {}
        vals = []
        for k in range(PORTAL_STATE_ARM_FIELDS):
            v = state.get(f"j{k}")
            if v is None:
                vals = None
                break
            vals.append(float(v))
        if vals is not None:
            q_new = np.array(vals, dtype=np.float64)

        frames = {}
        for name, frame in (getattr(obs, "frames", None) or {}).items():
            try:
                data = frame.data
                w, h = frame.width, frame.height
                if data is None or not w or not h:
                    continue
                rgb = np.frombuffer(bytes(data), dtype=np.uint8).reshape(int(h), int(w), 3)
                frames[name] = _Frame(np.ascontiguousarray(rgb[:, :, ::-1]),  # RGB -> BGR
                                      getattr(frame, "timestamp_us", 0))
            except Exception as exc:
                logger_mp.warning(f"[portal] failed to decode frame '{name}': {exc}")

        with self._obs_lock:
            self._obs_ts_us = ts_us
            if q_new is not None:
                if (self._prev_state_q is not None and ts_us is not None
                        and self._prev_state_ts_us is not None
                        and ts_us > self._prev_state_ts_us):
                    dt = (ts_us - self._prev_state_ts_us) / 1e6
                    if dt > 1e-4:
                        self._state_dq = (q_new - self._prev_state_q) / dt
                self._prev_state_q = q_new
                self._prev_state_ts_us = ts_us
                self._state_q = q_new
                self._state_ts_wall = wall
            self._frames.update(frames)

    # ------------------------------------------------------------------
    # outbound actions (arm controller API)
    # ------------------------------------------------------------------
    def ctrl_dual_arm(self, q_target, tauff_target) -> None:
        """Queue one action: arm targets from IK + latest hand targets."""
        q = np.asarray(q_target, dtype=np.float64).copy()
        with self._arm_lock:
            self._q_target[:] = q
        if self._loop is not None and self._connected_evt.is_set():
            try:
                self._loop.call_soon_threadsafe(self._send_action_now, q)
            except RuntimeError:
                pass  # loop already closed during shutdown

    def _send_action_now(self, q14: np.ndarray) -> None:
        with self._hand_lock:
            hand_q = self._hand_q.copy()
            fsm_id = self._fsm_id
        with self._obs_lock:
            in_reply_to = self._obs_ts_us
        values = {"fsm_id": int(fsm_id)}
        for i in range(PORTAL_ARM_DOF):
            values[f"left_arm_q{i}"] = float(q14[i])
            values[f"right_arm_q{i}"] = float(q14[PORTAL_ARM_DOF + i])
        for i in range(PORTAL_HAND_DOF):
            values[f"left_hand_q{i}"] = float(hand_q[i])
            values[f"right_hand_q{i}"] = float(hand_q[PORTAL_HAND_DOF + i])
        try:
            self._op.send_action(values,
                                 timestamp_us=int(time.time() * 1_000_000),
                                 in_reply_to_ts_us=in_reply_to)
        except Exception as exc:
            logger_mp.warning(f"[portal] send_action failed: {exc}")
            return
        with self._arm_lock:
            self._last_sent_q[:] = q14

    def set_fsm_id(self, fsm_id: int) -> None:
        with self._hand_lock:
            self._fsm_id = int(fsm_id)

    def set_arm_velocity_limit(self, velocity_limit: float = 30.0) -> None:
        # Velocity clipping is the robot gateway's responsibility in portal
        # mode; accepted for API compatibility only.
        logger_mp.debug(f"[portal] set_arm_velocity_limit ignored ({velocity_limit})")

    # ------------------------------------------------------------------
    # state feedback for the IK
    # ------------------------------------------------------------------
    def get_current_dual_arm_q(self) -> np.ndarray:
        with self._obs_lock:
            fresh = (time.time() - self._state_ts_wall) < self._state_timeout
            if fresh and self._state_q is not None:
                return self._state_q.copy()
        with self._arm_lock:
            return self._last_sent_q.copy()

    def get_current_dual_arm_dq(self) -> np.ndarray:
        with self._obs_lock:
            fresh = (time.time() - self._state_ts_wall) < self._state_timeout
            if fresh:
                return self._state_dq.copy()
        return np.zeros(PORTAL_STATE_ARM_FIELDS)

    def ctrl_dual_arm_go_home(self) -> None:
        logger_mp.info("[portal] ctrl_dual_arm_go_home ...")
        self.ctrl_dual_arm(np.zeros(2 * PORTAL_ARM_DOF), np.zeros(2 * PORTAL_ARM_DOF))
        for _ in range(100):
            if np.all(np.abs(self.get_current_dual_arm_q()) < 0.05):
                logger_mp.info("[portal] both arms reached home position (reported).")
                return
            time.sleep(0.05)
        logger_mp.warning("[portal] go_home timed out waiting for state feedback.")

    # ------------------------------------------------------------------
    # dex3 hand retargeting (replaces Dex3_1_Controller.control_process)
    # ------------------------------------------------------------------
    def set_xr_motion_data_ready(self, value) -> None:
        """Attach the xr_motion_data_ready shared Value (created later in
        teleop_hand_and_arm.py) so the retargeting thread can gate on it."""
        self._xr_motion_data_ready_in = value

    def _hand_retarget_loop(self) -> None:
        from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

        logger_mp.info("[portal] starting local dex3 hand retargeting ...")
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)

        left_q_target = np.zeros(PORTAL_HAND_DOF)
        right_q_target = np.zeros(PORTAL_HAND_DOF)

        while not self._stop_evt.is_set():
            start_time = time.time()
            try:
                with self._left_hand_array_in.get_lock():
                    left_hand_data = np.array(self._left_hand_array_in[:]).reshape(25, 3).copy()
                with self._right_hand_array_in.get_lock():
                    right_hand_data = np.array(self._right_hand_array_in[:]).reshape(25, 3).copy()

                if self._xr_motion_data_ready_in is not None:
                    with self._xr_motion_data_ready_in.get_lock():
                        xr_ready = self._xr_motion_data_ready_in.value
                else:
                    xr_ready = True

                if xr_ready:
                    ref_left = left_hand_data[hand_retargeting.left_indices[1, :]] - \
                        left_hand_data[hand_retargeting.left_indices[0, :]]
                    ref_right = right_hand_data[hand_retargeting.right_indices[1, :]] - \
                        right_hand_data[hand_retargeting.right_indices[0, :]]
                    left_q_target = hand_retargeting.left_retargeting.retarget(ref_left)[
                        hand_retargeting.left_dex_retargeting_to_hardware]
                    right_q_target = hand_retargeting.right_retargeting.retarget(ref_right)[
                        hand_retargeting.right_dex_retargeting_to_hardware]

                action_data = np.concatenate((left_q_target, right_q_target))
                with self._hand_lock:
                    self._hand_q[:] = action_data
                if self._dual_hand_state_array_out is not None and self._dual_hand_action_array_out is not None:
                    # No real hand state over portal (state schema has no hand
                    # fields): report the commanded targets as state so that
                    # recording keeps a plausible trajectory.
                    with self._dual_hand_data_lock:
                        self._dual_hand_state_array_out[:] = action_data
                        self._dual_hand_action_array_out[:] = action_data
            except Exception as exc:
                logger_mp.warning(f"[portal] hand retargeting error: {exc}")

            sleep_time = max(0.0, (1.0 / self._hand_fps) - (time.time() - start_time))
            time.sleep(sleep_time)
        logger_mp.info("[portal] hand retargeting stopped.")

    # ------------------------------------------------------------------
    # ImageClient-compatible video API
    # ------------------------------------------------------------------
    def get_cam_config(self) -> dict:
        """Load the static camera config and gate it on the portal.yaml
        video tracks. Portal carries the video, so teleimager-WebRTC is
        always disabled in portal mode (render_to_xr is used instead)."""
        with open(self._cam_config_path, "r") as f:
            cam_config = yaml.safe_load(f)
        for cam, track in (("head_camera", "head_camera"),
                           ("left_wrist_camera", "left_wrist_camera"),
                           ("right_wrist_camera", "right_wrist_camera")):
            if cam not in cam_config:
                continue
            active = cam_config[cam].get("enable_zmq", False) and track in self._declared_videos
            cam_config[cam]["enable_zmq"] = active
            cam_config[cam]["enable_webrtc"] = False
        return cam_config

    def _get_frame(self, track: str):
        with self._obs_lock:
            frame = self._frames.get(track)
            return frame if frame is not None else _Frame(None)

    def get_head_frame(self):
        return self._get_frame("head_camera")

    def get_left_wrist_frame(self):
        return self._get_frame("left_wrist_camera")

    def get_right_wrist_frame(self):
        return self._get_frame("right_wrist_camera")

    # ------------------------------------------------------------------
    def close(self) -> None:
        logger_mp.info("[portal] closing bridge ...")
        self._stop_evt.set()
        if self._hand_thread is not None:
            self._hand_thread.join(timeout=3.0)
        if self._portal_thread is not None:
            self._portal_thread.join(timeout=5.0)
        logger_mp.info("[portal] bridge closed.")
