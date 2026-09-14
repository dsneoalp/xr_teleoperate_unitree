"""LiveKit Portal operator bridge for xr_teleoperate.

Used by teleop_operator.py (not as arm_ctrl):

  * IK arm targets, dex3 retargeting, and loco (vx/vy/vyaw) are published
    as Portal actions at the teleop control rate.
  * Robot state is received via on_observation. IK warm-starts from last
    sent targets so delayed WAN state does not oscillate the solver.
    Missing state still dead-reckons with those targets.
  * Video: every track listed under portal.yaml `videos:` is subscribed.
    TeleVuer / get_head_frame() is filled from unmatched on_video_frame
    (lowest display latency) and from matched obs.frames, so XR still
    updates when tick-synced frames never take the unmatched path.
    Recording uses framed obs paired with send_action via in_reply_to_ts_us
    (in_reply_to is published only after the observation is buffered).

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
from teleop.robot_control.obs_record_buffer import (
    RecordingObservation,
    RecordingPair,
    RecordingObsBuffer,
    record_buffer_maxlen,
)
from teleop.robot_control.tick_slot import now_us
from teleop.utils.loop_timing import LoopTiming

# ImageClient / TeleVuer slot names. Portal tracks bind by yaml order.
_TELEVUER_SLOTS = ("head_camera", "left_wrist_camera", "right_wrist_camera")


class _Frame:
    """Minimal stand-in for teleimager's frame object (only .bgr is used)."""

    __slots__ = ("bgr", "timestamp_us")

    def __init__(self, bgr, timestamp_us=0):
        self.bgr = bgr
        self.timestamp_us = timestamp_us


def _apply_env_file(env_file: str) -> None:
    """Load KEY=VAL lines into os.environ without overriding existing keys."""
    if not env_file or not os.path.isfile(env_file):
        return
    with open(env_file, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            if not key:
                continue
            val = val.strip().strip("'").strip('"')
            os.environ.setdefault(key, val)


def _load_dotenv(env_file: str) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        _apply_env_file(env_file)
        return
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
        self._op.on_observation(self._on_observation)
        self._op.on_drop(self._on_drop)
        for track in self._declared_videos:
            self._op.on_video_frame(track, self._on_video_frame)
        self._frames_logged = set()
        self._match_timing = LoopTiming(logger_mp, prefix="[timing-match]")

        arm_dof = self._map.arm_dof
        hand_dof = self._map.hand_dof

        self._obs_lock = threading.Lock()
        self._state_q = None
        self._state_dq = np.zeros(arm_dof)
        self._state_ts_wall = 0.0
        self._prev_state_q = None
        self._prev_state_ts_us = None
        self._obs_ts_us = None
        self._frames = {}
        self._pending_action_wall = None
        self._rtt_cb = None
        self._recording_enabled = False
        slack = int(self._wire.get("slack") or 5)
        self._rec_buf = RecordingObsBuffer(maxlen=record_buffer_maxlen(slack))
        self._record_pair_cb = None
        self._last_obs_had_frames = False

        self._arm_lock = threading.Lock()
        self._last_sent_q = np.zeros(arm_dof)
        self._last_sent_dq = np.zeros(arm_dof)
        self._last_send_wall = 0.0

        self._hand_lock = threading.Lock()
        self._hand_q = np.zeros(hand_dof)
        self._fsm_id = 0

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

    def _count_match(self, name: str, n: int = 1) -> None:
        timing = getattr(self, "_match_timing", None)
        if timing is not None and n:
            timing.count(name, n)

    @staticmethod
    def _drop_n(drops) -> int:
        if drops is None:
            return 0
        if isinstance(drops, int):
            return max(0, drops)
        try:
            return len(drops)
        except TypeError:
            return 1

    def _on_drop(self, drops) -> None:
        n = self._drop_n(drops)
        self._count_match("drop", n)
        if n:
            logger_mp.info(f"[portal] dropped states: {n}")

    def _on_observation(self, obs) -> None:
        ts_us = getattr(obs, "timestamp_us", None)
        wall = time.time()
        raw = getattr(obs, "raw_state", None)
        if not raw:
            raw = getattr(obs, "state", None) or {}
        q_new = self._map.unpack_arm_q(raw)
        hand_new = self._map.unpack_hand_q(raw)
        raw_frames = getattr(obs, "frames", None) or {}
        had_frames = bool(raw_frames)
        self._count_match("obs")
        if had_frames:
            self._count_match("obs_framed")

        rtt_ms = None
        with self._obs_lock:
            self._last_obs_had_frames = had_frames
            recording = self._recording_enabled
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

        stored_all = {}
        for track, frame in raw_frames.items():
            stored_all.update(self._decode_video_frame(track, frame))
        self._merge_display_frames(stored_all)
        if recording and ts_us is not None:
            rec_frames = {}
            for name, wrapped in stored_all.items():
                if wrapped.bgr is not None:
                    rec_frames[name] = np.ascontiguousarray(wrapped.bgr.copy())
            if rec_frames:
                self._rec_buf.push(RecordingObservation(
                    timestamp_us=int(ts_us),
                    arm_q=None if q_new is None else q_new.copy(),
                    hand_q=None if hand_new is None else hand_new.copy(),
                    frames=rec_frames,
                ))
        with self._obs_lock:
            self._obs_ts_us = ts_us

    def _on_video_frame(self, track: str, frame) -> None:
        if track == getattr(self, "_xr_track", None):
            self._count_match("unmatched_video")
        self._merge_display_frames(self._decode_video_frame(track, frame))

    def _merge_display_frames(self, stored: dict) -> None:
        """Write decoded frames into the TeleVuer cache.

        Unmatched on_video_frame stays the low-latency path. Matched
        obs.frames fill the same cache so get_head_frame() still works
        when Portal never emits unmatched video. A newer timestamp is
        never replaced by an older one.
        """
        if not stored:
            return
        with self._obs_lock:
            for track, wrapped in stored.items():
                existing = self._frames.get(track)
                if (existing is None
                        or int(wrapped.timestamp_us) >= int(existing.timestamp_us)):
                    self._frames[track] = wrapped

    def _slot_for_track(self, track: str) -> str | None:
        try:
            idx = self._declared_videos.index(track)
        except ValueError:
            return None
        if idx >= len(_TELEVUER_SLOTS):
            return None
        return _TELEVUER_SLOTS[idx]

    def _decode_video_frame(self, track: str, frame, *, log: bool = True) -> dict:
        try:
            data = frame.data
            w, h = int(frame.width), int(frame.height)
            if data is None or not w or not h:
                return {}
            rgb = np.frombuffer(bytes(data), dtype=np.uint8).reshape(h, w, 3)
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            slot = self._slot_for_track(track)
            if slot and slot in self._expected_hw:
                eh, ew = self._expected_hw[slot]
                if (h, w) != (eh, ew):
                    import cv2
                    bgr = cv2.resize(bgr, (ew, eh), interpolation=cv2.INTER_LINEAR)
            wrapped = _Frame(bgr, getattr(frame, "timestamp_us", 0) or 0)
            if log and track not in self._frames_logged:
                self._frames_logged.add(track)
                dest = "TeleVuer" if track == self._xr_track else (slot or "record")
                logger_mp.info(f"[portal] video '{track}' {w}x{h} → {dest}")
            return {track: wrapped}
        except Exception as exc:
            logger_mp.warning(f"[portal] failed to decode frame '{track}': {exc}")
            return {}

    def send_targets(self, arm_q, hand_q=None, vx=0.0, vy=0.0, vyaw=0.0, fsm_id=None) -> None:
        """Publish an action from the caller thread (sync, fire-and-forget)."""
        if not self._connected_evt.is_set():
            return
        q_arm = np.asarray(arm_q, dtype=np.float64).reshape(-1).copy()
        with self._hand_lock:
            if hand_q is not None:
                q_hand = np.asarray(hand_q, dtype=np.float64).reshape(-1).copy()
            else:
                q_hand = self._hand_q.copy()
            if fsm_id is None:
                fsm = self._fsm_id
            else:
                fsm = int(fsm_id)
                self._fsm_id = fsm
        self._send_action_now(q_arm, q_hand, float(vx), float(vy), float(vyaw), fsm)

    def _send_action_now(self, q_arm, q_hand, vx, vy, vyaw, fsm_id) -> None:
        with self._obs_lock:
            in_reply_to = self._obs_ts_us
        action_ts = now_us()
        values = self._map.pack_action(
            arm_q=q_arm, hand_q=q_hand, vx=vx, vy=vy, vyaw=vyaw, fsm_id=fsm_id)
        try:
            self._op.send_action(values,
                                 timestamp_us=action_ts,
                                 in_reply_to_ts_us=in_reply_to)
        except Exception as exc:
            logger_mp.warning(f"[portal] send_action failed: {exc}")
            return
        with self._obs_lock:
            self._pending_action_wall = time.time()
            recording = self._recording_enabled
            record_cb = self._record_pair_cb
        now = time.time()
        with self._arm_lock:
            if self._last_send_wall > 0:
                dt = now - self._last_send_wall
                if dt > 1e-4:
                    self._last_sent_dq[:] = (q_arm - self._last_sent_q) / dt
            self._last_sent_q[:] = q_arm
            self._last_send_wall = now
        if recording:
            pair = self.take_recording_pair(
                in_reply_to, action_ts, q_arm, q_hand, vx, vy, vyaw, fsm_id)
            if pair is None:
                logger_mp.debug(
                    f"[portal] no recording obs for in_reply_to={in_reply_to}")
            elif record_cb is not None:
                try:
                    record_cb(pair)
                except Exception as exc:
                    logger_mp.warning(f"[portal] record pair callback failed: {exc}")

    def on_rtt(self, callback) -> None:
        """callback(rtt_ms) on the first observation after a successful send_action."""
        self._rtt_cb = callback

    def on_record_pair(self, callback) -> None:
        """callback(RecordingPair) after a successful send_action while recording."""
        self._record_pair_cb = callback

    def set_recording_enabled(self, enabled: bool) -> None:
        """Decode obs.frames into the recording buffer only while True."""
        with self._obs_lock:
            self._recording_enabled = bool(enabled)
        if not enabled:
            self._rec_buf.clear()
        logger_mp.info(f"[portal] recording buffer {'on' if enabled else 'off'}")

    def take_recording_pair(
            self, in_reply_to_ts_us, action_ts, arm_action, hand_action,
            vx, vy, vyaw, fsm_id) -> RecordingPair | None:
        """Pop the observation that this action replied to. None if missing."""
        obs = self._rec_buf.take(in_reply_to_ts_us)
        if obs is None or not obs.frames:
            return None
        return RecordingPair(
            obs=obs,
            action_timestamp_us=int(action_ts),
            arm_action=np.asarray(arm_action, dtype=np.float64).copy(),
            hand_action=np.asarray(hand_action, dtype=np.float64).copy(),
            vx=float(vx),
            vy=float(vy),
            vyaw=float(vyaw),
            fsm_id=int(fsm_id),
        )

    @property
    def xr_track(self) -> str | None:
        return self._xr_track

    def get_last_obs_ts_us(self) -> int | None:
        with self._obs_lock:
            return self._obs_ts_us

    def last_obs_had_frames(self) -> bool:
        with self._obs_lock:
            return self._last_obs_had_frames

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
            frame = self._frames.get(track)
            return frame if frame is not None else _Frame(None)

    def get_head_frame(self):
        return self._get_frame_at(0)

    def get_left_wrist_frame(self):
        return self._get_frame_at(1)

    def get_right_wrist_frame(self):
        return self._get_frame_at(2)

    def close(self) -> None:
        logger_mp.info("[portal] closing bridge ...")
        self._stop_evt.set()
        if self._hand_thread is not None:
            self._hand_thread.join(timeout=3.0)
        if self._portal_thread is not None:
            self._portal_thread.join(timeout=5.0)
        logger_mp.info("[portal] bridge closed.")
