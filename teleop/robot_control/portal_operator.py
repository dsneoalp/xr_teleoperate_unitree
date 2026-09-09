"""LiveKit Portal operator bridge for xr_teleoperate.

Used by teleop_operator.py (not as arm_ctrl):

  * IK arm targets, dex3 retargeting, and loco (vx/vy/vyaw) are published
    as Portal actions at the teleop control rate.
  * Control state is received via on_state (no video wait). IK warm-starts
    from last sent targets so delayed WAN state does not oscillate the
    solver. Missing state still dead-reckons with those targets.
  * Video: on_video_frame copies raw RGB; TeleVuer / get_head_frame()
    decodes on the caller thread. Extra tracks map to left/right wrist
    in declaration order.
  * Recordings consume on_observation (state + matched frames) and join
    actions via in_reply_to_ts_us.

No unitree_sdk2py import happens anywhere in this module.
Joint names come from portal_mapping.yaml, not hardcoded tuples.
"""
from __future__ import annotations

import copy
import os
import sys
import time
import asyncio
import threading
from collections import deque
from dataclasses import dataclass
from queue import Empty, Full, Queue

import numpy as np
import yaml

import logging_mp
logger_mp = logging_mp.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))          # .../teleop/robot_control
parent_dir = os.path.dirname(current_dir)                          # .../teleop
parent2_dir = os.path.dirname(parent_dir)                          # repo root
if parent2_dir not in sys.path:
    sys.path.append(parent2_dir)

from teleop.robot_control.portal_mapping import PortalMapping

# ImageClient / TeleVuer slot names. Portal tracks bind by yaml order.
_TELEVUER_SLOTS = ("head_camera", "left_wrist_camera", "right_wrist_camera")
_ACTION_RING = 64
_RECORD_QUEUE = 8


class _Frame:
    """Minimal stand-in for teleimager's frame object (only .bgr is used)."""

    __slots__ = ("bgr", "timestamp_us")

    def __init__(self, bgr, timestamp_us=0):
        self.bgr = bgr
        self.timestamp_us = timestamp_us


@dataclass
class _RawFrame:
    rgb: bytes
    width: int
    height: int
    timestamp_us: int


@dataclass
class _SentAction:
    timestamp_us: int
    in_reply_to_ts_us: int | None
    arm_q: np.ndarray
    hand_q: np.ndarray
    vx: float
    vy: float
    vyaw: float
    fsm_id: int


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
    """Mint a LiveKit JWT for a portal participant (Robot or Operator)."""
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
    """Operator-side LiveKit transport for teleop_operator.py.

    Call as `teleop_bridge` (not `arm_ctrl`). Sends actions via
    `send_targets`; exposes ImageClient-style video getters.
    """

    def __init__(self,
                 portal_yaml: str,
                 env_file: str,
                 identity: str = "xr-teleop",
                 room: str | None = None,
                 url: str | None = None,
                 ee: str | None = None,
                 hand_fps: float = 100.0,
                 left_hand_array_in=None,
                 right_hand_array_in=None,
                 dual_hand_data_lock=None,
                 dual_hand_state_array_out=None,
                 dual_hand_action_array_out=None,
                 xr_motion_data_ready_in=None,
                 cam_config_path: str | None = None,
                 mapping_yaml: str | None = None,
                 state_timeout: float = 0.5):
        from livekit.portal import Operator, OperatorConfig

        self._portal_yaml = portal_yaml
        self._env_file = env_file
        self._identity = identity
        self._ee = ee
        self._state_timeout = state_timeout

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
        self._map = PortalMapping(mapping_yaml, portal_yaml)
        logger_mp.info(f"[portal] mapping {mapping_yaml}: "
                       f"arm_dof={self._map.arm_dof} hand_dof={self._map.hand_dof}")

        with open(portal_yaml, "r") as f:
            self._wire = yaml.safe_load(f)
        self._declared_videos = [v["name"] for v in (self._wire.get("videos") or [])]
        self._xr_track = self._declared_videos[0] if self._declared_videos else None

        self._cam_config_path = cam_config_path or os.path.join(
            parent_dir, "utils", "portal_cam_config.yaml")

        with open(self._cam_config_path, "r") as f:
            self._cam_config_raw = yaml.safe_load(f) or {}
        self._expected_hw = {}
        for cam, cfg_cam in self._cam_config_raw.items():
            if not isinstance(cfg_cam, dict):
                continue
            shape = cfg_cam.get("image_shape")
            if isinstance(shape, (list, tuple)) and len(shape) >= 2:
                self._expected_hw[cam] = (int(shape[0]), int(shape[1]))

        cfg = OperatorConfig.from_yaml_file(portal_yaml, self._room)
        self._op = Operator(cfg)
        self._op.on_state(self._on_state)
        self._op.on_observation(self._on_observation)
        self._op.on_drop(self._on_drop)
        for track in self._declared_videos:
            self._op.on_video_frame(track, self._on_video_frame)
        self._frames_logged = set()

        arm_dof = self._map.arm_dof
        hand_dof = self._map.hand_dof

        self._obs_lock = threading.Lock()
        self._state_q = None
        self._state_dq = np.zeros(arm_dof)
        self._state_ts_wall = 0.0
        self._prev_state_q = None
        self._prev_state_ts_us = None
        self._obs_ts_us = None
        self._raw_frames: dict[str, _RawFrame] = {}
        self._decoded_bgr: dict[str, _Frame] = {}
        self._decoded_src: dict[str, int] = {}
        self._pending_action_wall = None
        self._rtt_cb = None
        self._drop_cb = None
        self._drop_count = 0
        self._drop_log_t = 0.0

        self._arm_lock = threading.Lock()
        self._q_target = np.zeros(arm_dof)
        self._last_sent_q = np.zeros(arm_dof)
        self._last_sent_dq = np.zeros(arm_dof)
        self._last_send_wall = 0.0

        self._hand_lock = threading.Lock()
        self._hand_q = np.zeros(hand_dof)
        self._fsm_id = 0
        self._vx = 0.0
        self._vy = 0.0
        self._vyaw = 0.0

        self._action_lock = threading.Lock()
        self._action_ring: deque[_SentAction] = deque(maxlen=_ACTION_RING)

        self._recording = threading.Event()
        self._record_cb = None
        self._record_binocular = False
        self._record_loco = False
        self._record_q: Queue = Queue(maxsize=_RECORD_QUEUE)

        self._stop_evt = threading.Event()
        self._connected_evt = threading.Event()
        self._connect_error = None
        self._loop = None

        self._hand_thread = None
        self._left_hand_array_in = left_hand_array_in
        self._right_hand_array_in = right_hand_array_in
        self._dual_hand_data_lock = dual_hand_data_lock
        self._dual_hand_state_array_out = dual_hand_state_array_out
        self._dual_hand_action_array_out = dual_hand_action_array_out
        self._xr_motion_data_ready_in = xr_motion_data_ready_in
        if self._ee == "dex3" and left_hand_array_in is not None and right_hand_array_in is not None:
            self._hand_fps = hand_fps
            self._hand_thread = threading.Thread(target=self._hand_retarget_loop, daemon=True)
            self._hand_thread.start()

        self._record_thread = threading.Thread(target=self._record_loop, daemon=True)
        self._record_thread.start()

        self._portal_thread = threading.Thread(target=self._portal_loop, daemon=True)
        self._portal_thread.start()

        logger_mp.info(f"[portal] connecting operator '{identity}' to room '{self._room}' at {self._url} ...")

    def _portal_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_main())
        except Exception as exc:
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
                       f"active operator claimed. videos={self._declared_videos} "
                       f"televuer={self._xr_track!r}")
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

    def _state_dict(self, msg) -> dict:
        raw = getattr(msg, "raw_state", None)
        if not raw:
            raw = getattr(msg, "state", None) or {}
        return raw

    def _on_state(self, state) -> None:
        ts_us = getattr(state, "timestamp_us", None)
        wall = time.time()
        raw = self._state_dict(state)
        q_new = self._map.unpack_arm_q(raw)
        hand_new = self._map.unpack_hand_q(raw)

        rtt_ms = None
        with self._obs_lock:
            self._obs_ts_us = ts_us
            if q_new is not None:
                self._prev_state_q = q_new
                self._prev_state_ts_us = ts_us
                self._state_q = q_new
                self._state_ts_wall = wall
            sent = self._pending_action_wall
            if sent is not None:
                rtt_ms = (wall - sent) * 1000.0
                self._pending_action_wall = None
        if rtt_ms is not None and self._rtt_cb is not None:
            try:
                self._rtt_cb(rtt_ms)
            except Exception as exc:
                logger_mp.debug(f"[portal] rtt callback failed: {exc}")

        if (hand_new is not None
                and self._dual_hand_state_array_out is not None
                and self._dual_hand_data_lock is not None):
            with self._dual_hand_data_lock:
                n = min(len(self._dual_hand_state_array_out), hand_new.size)
                self._dual_hand_state_array_out[:n] = hand_new[:n]

    def _on_drop(self, dropped) -> None:
        n = len(dropped) if dropped else 0
        if n <= 0:
            return
        self._drop_count += n
        if self._drop_cb is not None:
            try:
                self._drop_cb(n)
            except Exception as exc:
                logger_mp.debug(f"[portal] drop callback failed: {exc}")
        now = time.monotonic()
        if now - self._drop_log_t >= 1.0:
            logger_mp.info(f"[portal] dropped states n={n} total={self._drop_count}")
            self._drop_log_t = now

    def _on_observation(self, obs) -> None:
        if not self._recording.is_set() or self._record_cb is None:
            return
        ts_us = getattr(obs, "timestamp_us", None)
        raw = self._state_dict(obs)
        frames = getattr(obs, "frames", None) or {}
        copied = {}
        for name, frame in frames.items():
            raw_f = self._copy_raw_frame(frame)
            if raw_f is not None:
                copied[name] = raw_f
        sample = {
            "ts_us": ts_us,
            "raw_state": dict(raw) if raw else {},
            "frames": copied,
            "action": self._match_action(ts_us),
        }
        try:
            self._record_q.put_nowait(sample)
        except Full:
            try:
                self._record_q.get_nowait()
                self._record_q.task_done()
            except (Empty, ValueError):
                pass
            try:
                self._record_q.put_nowait(sample)
            except Full:
                logger_mp.warning("[portal] record queue full, dropping observation")

    def _copy_raw_frame(self, frame) -> _RawFrame | None:
        try:
            data = getattr(frame, "data", None)
            w, h = int(getattr(frame, "width", 0) or 0), int(getattr(frame, "height", 0) or 0)
            if data is None or not w or not h:
                return None
            return _RawFrame(
                rgb=bytes(data),
                width=w,
                height=h,
                timestamp_us=int(getattr(frame, "timestamp_us", 0) or 0),
            )
        except Exception as exc:
            logger_mp.warning(f"[portal] failed to copy video frame: {exc}")
            return None

    def _on_video_frame(self, track: str, frame) -> None:
        raw = self._copy_raw_frame(frame)
        if raw is None:
            return
        with self._obs_lock:
            self._raw_frames[track] = raw
        if track not in self._frames_logged:
            self._frames_logged.add(track)
            slot = self._slot_for_track(track)
            dest = "TeleVuer" if track == self._xr_track else (slot or "record")
            logger_mp.info(f"[portal] video '{track}' {raw.width}x{raw.height} → {dest}")

    def _slot_for_track(self, track: str) -> str | None:
        try:
            idx = self._declared_videos.index(track)
        except ValueError:
            return None
        if idx >= len(_TELEVUER_SLOTS):
            return None
        return _TELEVUER_SLOTS[idx]

    def _bgr_from_raw(self, raw: _RawFrame, slot: str | None) -> np.ndarray:
        rgb = np.frombuffer(raw.rgb, dtype=np.uint8).reshape(raw.height, raw.width, 3)
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        if slot and slot in self._expected_hw:
            eh, ew = self._expected_hw[slot]
            if (raw.height, raw.width) != (eh, ew):
                import cv2
                bgr = cv2.resize(bgr, (ew, eh), interpolation=cv2.INTER_LINEAR)
        return bgr

    def _match_action(self, state_ts_us) -> _SentAction | None:
        with self._action_lock:
            if not self._action_ring:
                return None
            if state_ts_us is not None:
                for item in reversed(self._action_ring):
                    if item.in_reply_to_ts_us is not None and item.in_reply_to_ts_us == state_ts_us:
                        return item
            return min(
                self._action_ring,
                key=lambda a: abs((a.timestamp_us or 0) - (state_ts_us or 0)),
            )

    def _record_loop(self) -> None:
        while not self._stop_evt.is_set():
            try:
                sample = self._record_q.get(timeout=0.2)
            except Empty:
                continue
            try:
                item = self._build_record_item(sample)
                cb = self._record_cb
                if item is not None and cb is not None:
                    cb(item)
            except Exception as exc:
                logger_mp.warning(f"[portal] record sample failed: {exc}")
            finally:
                try:
                    self._record_q.task_done()
                except ValueError:
                    pass

    def _build_record_item(self, sample: dict) -> dict | None:
        raw_state = sample.get("raw_state") or {}
        arm_q = self._map.unpack_arm_q(raw_state)
        hand_q = self._map.unpack_hand_q(raw_state)
        if arm_q is None:
            arm_q = np.zeros(self._map.arm_dof)
        if hand_q is None:
            hand_q = np.zeros(self._map.hand_dof)
        n_left_arm = len(self._map.left_arm)
        n_left_hand = len(self._map.left_hand)
        left_arm_state = arm_q[:n_left_arm]
        right_arm_state = arm_q[n_left_arm:]
        left_ee_state = hand_q[:n_left_hand].tolist()
        right_ee_state = hand_q[n_left_hand:].tolist()

        action = sample.get("action")
        if action is not None:
            left_arm_action = action.arm_q[:n_left_arm]
            right_arm_action = action.arm_q[n_left_arm:]
            left_hand_action = action.hand_q[:n_left_hand].tolist()
            right_hand_action = action.hand_q[n_left_hand:].tolist()
            body_action = [action.vx, action.vy, action.vyaw] if self._record_loco else []
        else:
            left_arm_action = np.zeros(n_left_arm)
            right_arm_action = np.zeros(len(self._map.right_arm))
            left_hand_action = [0.0] * n_left_hand
            right_hand_action = [0.0] * len(self._map.right_hand)
            body_action = []

        colors = self._colors_from_raw(sample.get("frames") or {})
        return {
            "colors": colors,
            "depths": {},
            "states": {
                "left_arm": {"qpos": left_arm_state.tolist(), "qvel": [], "torque": []},
                "right_arm": {"qpos": right_arm_state.tolist(), "qvel": [], "torque": []},
                "left_ee": {"qpos": left_ee_state, "qvel": [], "torque": []},
                "right_ee": {"qpos": right_ee_state, "qvel": [], "torque": []},
                "body": {"qpos": []},
            },
            "actions": {
                "left_arm": {"qpos": left_arm_action.tolist(), "qvel": [], "torque": []},
                "right_arm": {"qpos": right_arm_action.tolist(), "qvel": [], "torque": []},
                "left_ee": {"qpos": left_hand_action, "qvel": [], "torque": []},
                "right_ee": {"qpos": right_hand_action, "qvel": [], "torque": []},
                "body": {"qpos": body_action},
            },
        }

    def _colors_from_raw(self, frames: dict[str, _RawFrame]) -> dict:
        colors = {}
        binocular = self._record_binocular
        for i, track in enumerate(self._declared_videos):
            raw = frames.get(track)
            if raw is None:
                continue
            slot = self._slot_for_track(track)
            bgr = self._bgr_from_raw(raw, slot)
            if i == 0:
                if binocular and bgr.shape[1] >= 2:
                    w = bgr.shape[1] // 2
                    colors["color_0"] = bgr[:, :w]
                    colors["color_1"] = bgr[:, w:]
                else:
                    colors["color_0"] = bgr
            elif i == 1:
                colors["color_2" if binocular else "color_1"] = bgr
            elif i == 2:
                colors["color_3" if binocular else "color_2"] = bgr
        return colors

    def configure_recording(self, *, binocular: bool, record_loco: bool, on_sample) -> None:
        self._record_binocular = bool(binocular)
        self._record_loco = bool(record_loco)
        self._record_cb = on_sample

    def start_recording(self) -> None:
        self._recording.set()

    def stop_recording(self, timeout: float = 2.0) -> None:
        self._recording.clear()
        deadline = time.time() + timeout
        while not self._record_q.empty() and time.time() < deadline:
            time.sleep(0.02)

    def send_targets(self, arm_q, hand_q=None, vx=0.0, vy=0.0, vyaw=0.0, fsm_id=None) -> None:
        """Publish one action: arm (+ optional hand) targets and loco."""
        q = np.asarray(arm_q, dtype=np.float64).copy()
        with self._arm_lock:
            self._q_target[:] = q
        with self._hand_lock:
            if hand_q is not None:
                self._hand_q[:] = np.asarray(hand_q, dtype=np.float64).reshape(-1)
            self._vx = float(vx)
            self._vy = float(vy)
            self._vyaw = float(vyaw)
            if fsm_id is not None:
                self._fsm_id = int(fsm_id)
        if self._loop is not None and self._connected_evt.is_set():
            try:
                self._loop.call_soon_threadsafe(self._send_action_now, q)
            except RuntimeError:
                pass

    def _send_action_now(self, q_arm: np.ndarray) -> None:
        with self._hand_lock:
            hand_q = self._hand_q.copy()
            fsm_id = self._fsm_id
            vx, vy, vyaw = self._vx, self._vy, self._vyaw
        with self._obs_lock:
            in_reply_to = self._obs_ts_us
        values = self._map.pack_action(
            arm_q=q_arm, hand_q=hand_q, vx=vx, vy=vy, vyaw=vyaw, fsm_id=fsm_id)
        ts_us = int(time.time() * 1_000_000)
        try:
            self._op.send_action(values,
                                 timestamp_us=ts_us,
                                 in_reply_to_ts_us=in_reply_to)
        except Exception as exc:
            logger_mp.warning(f"[portal] send_action failed: {exc}")
            return
        with self._action_lock:
            self._action_ring.append(_SentAction(
                timestamp_us=ts_us,
                in_reply_to_ts_us=in_reply_to,
                arm_q=np.asarray(q_arm, dtype=np.float64).copy(),
                hand_q=hand_q,
                vx=vx, vy=vy, vyaw=vyaw, fsm_id=fsm_id,
            ))
        with self._obs_lock:
            self._pending_action_wall = time.time()
        now = time.time()
        with self._arm_lock:
            if self._last_send_wall > 0:
                dt = now - self._last_send_wall
                if dt > 1e-4:
                    self._last_sent_dq[:] = (q_arm - self._last_sent_q) / dt
            self._last_sent_q[:] = q_arm
            self._last_send_wall = now

    def on_rtt(self, callback) -> None:
        """callback(rtt_ms) on the first state after a successful send_action."""
        self._rtt_cb = callback

    def on_drops(self, callback) -> None:
        """callback(n) when Portal drops unmatched observation states."""
        self._drop_cb = callback

    def state_age_ms(self) -> float | None:
        with self._obs_lock:
            ts = self._state_ts_wall
        if not ts:
            return None
        return (time.time() - ts) * 1000.0

    def set_fsm_id(self, fsm_id: int) -> None:
        with self._hand_lock:
            self._fsm_id = int(fsm_id)

    def get_current_dual_arm_q(self) -> np.ndarray:
        """IK warm start: last commanded q, not delayed WAN robot state."""
        with self._arm_lock:
            return self._last_sent_q.copy()

    def get_current_dual_arm_dq(self) -> np.ndarray:
        with self._arm_lock:
            return self._last_sent_dq.copy()

    def get_reported_arm_q(self) -> np.ndarray | None:
        with self._obs_lock:
            if self._state_q is None:
                return None
            return self._state_q.copy()

    def send_go_home(self) -> None:
        logger_mp.info("[portal] send_go_home ...")
        zeros = np.zeros(self._map.arm_dof)
        self.send_targets(zeros, hand_q=np.zeros(self._map.hand_dof),
                          vx=0.0, vy=0.0, vyaw=0.0, fsm_id=2)
        for _ in range(100):
            reported = self.get_reported_arm_q()
            if reported is not None and np.all(np.abs(reported) < 0.05):
                logger_mp.info("[portal] both arms reached home position (reported).")
                return
            time.sleep(0.05)
        logger_mp.warning("[portal] go_home timed out waiting for state feedback.")

    def set_xr_motion_data_ready(self, value) -> None:
        self._xr_motion_data_ready_in = value

    def _hand_retarget_loop(self) -> None:
        from teleop.robot_control.hand_retargeting import HandRetargeting, HandType

        logger_mp.info("[portal] starting local dex3 hand retargeting ...")
        hand_retargeting = HandRetargeting(HandType.UNITREE_DEX3)

        left_q_target = np.zeros(len(self._map.left_hand))
        right_q_target = np.zeros(len(self._map.right_hand))

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
                if self._dual_hand_action_array_out is not None:
                    with self._dual_hand_data_lock:
                        self._dual_hand_action_array_out[:] = action_data
            except Exception as exc:
                logger_mp.warning(f"[portal] hand retargeting error: {exc}")

            sleep_time = max(0.0, (1.0 / self._hand_fps) - (time.time() - start_time))
            time.sleep(sleep_time)
        logger_mp.info("[portal] hand retargeting stopped.")

    def get_cam_config(self) -> dict:
        cam_config = copy.deepcopy(self._cam_config_raw)
        for slot in _TELEVUER_SLOTS:
            if slot not in cam_config:
                continue
            cam_config[slot]["enable_webrtc"] = False
            cam_config[slot]["enable_zmq"] = False
        for i, track in enumerate(self._declared_videos):
            if i >= len(_TELEVUER_SLOTS):
                logger_mp.warning(f"[portal] ignoring extra video track '{track}'")
                continue
            slot = _TELEVUER_SLOTS[i]
            if slot not in cam_config:
                cam_config[slot] = {
                    "enable_zmq": True,
                    "enable_webrtc": False,
                    "binocular": False,
                    "image_shape": [480, 640],
                    "fps": 30,
                    "webrtc_port": 60001 + i,
                }
            else:
                cam_config[slot]["enable_zmq"] = True
                cam_config[slot]["enable_webrtc"] = False
            logger_mp.info(f"[portal] yaml video[{i}] '{track}' → {slot}")
        return cam_config

    def _get_frame_at(self, index: int):
        if index >= len(self._declared_videos):
            return _Frame(None)
        track = self._declared_videos[index]
        with self._obs_lock:
            raw = self._raw_frames.get(track)
            if raw is None:
                return _Frame(None)
            src_id = id(raw)
            cached = self._decoded_bgr.get(track)
            if cached is not None and self._decoded_src.get(track) == src_id:
                return cached
        slot = self._slot_for_track(track)
        try:
            wrapped = _Frame(self._bgr_from_raw(raw, slot), raw.timestamp_us)
        except Exception as exc:
            logger_mp.warning(f"[portal] failed to decode frame '{track}': {exc}")
            return _Frame(None)
        with self._obs_lock:
            if self._raw_frames.get(track) is raw:
                self._decoded_bgr[track] = wrapped
                self._decoded_src[track] = src_id
        return wrapped

    def get_head_frame(self):
        return self._get_frame_at(0)

    def get_left_wrist_frame(self):
        return self._get_frame_at(1)

    def get_right_wrist_frame(self):
        return self._get_frame_at(2)

    def close(self) -> None:
        logger_mp.info("[portal] closing bridge ...")
        self._recording.clear()
        self._stop_evt.set()
        if self._hand_thread is not None:
            self._hand_thread.join(timeout=3.0)
        if self._record_thread is not None:
            self._record_thread.join(timeout=2.0)
        if self._portal_thread is not None:
            self._portal_thread.join(timeout=5.0)
        logger_mp.info("[portal] bridge closed.")
