"""Thin LiveKit Portal Robot transport.

Not an arm_ctrl stand-in. The robot control loop owns G1_29_ArmController /
Dex3 / loco and calls this only to receive actions and publish state.

``video_publish_loop`` stamps RGB from a camera source onto control-loop
ticks via ``LatestTickSlot`` and ``send_video_frame``.

HW encode DoD runs only when ``require_hw_encode=True`` (CLI
``--require-hw-encode``). Default is software / sim, matching control_fix.
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
from teleop.robot_control.tick_slot import LatestTickSlot, now_us
from teleop.utils.loop_timing import LoopTiming

VIDEO_WAIT_S = 0.05
VIDEO_SKIP_LOG_EVERY = 30


def video_publish_loop(
    frame_source,
    portal,
    track,
    stop_evt,
    tick_slot: LatestTickSlot,
    timing: LoopTiming | None = None,
):
    """Wait for a control-loop tick, grab latest RGB, stamp it once.

    Pacing comes from the control loop via the slot, not a second 1/fps sleep.
    Duplicate source seq is skipped; the consumed timestamp is not reused.
    """
    logged = False
    skip_count = 0
    last_seq = -1
    timing = timing or LoopTiming(logger_mp, prefix="[timing-video]")
    last_encode_t = None
    while not stop_evt.is_set():
        ts = tick_slot.wait_take(timeout=VIDEO_WAIT_S)
        if ts is None:
            continue
        try:
            grab_t0 = time.perf_counter()
            item = frame_source.latest_rgb()
            grab_ms = (time.perf_counter() - grab_t0) * 1000.0
            if item is None:
                skip_count += 1
                if skip_count == 1 or skip_count % VIDEO_SKIP_LOG_EVERY == 0:
                    logger_mp.debug(
                        f"video skip: no camera frame for tick {ts} "
                        f"track={track} (skips={skip_count})"
                    )
                continue
            rgb_buf, seq = item
            if seq == last_seq:
                timing.count(f"{track}_dup")
                continue
            if last_seq >= 0:
                skipped = seq - last_seq - 1
                if skipped > 0:
                    timing.count(f"{track}_src_skipped", skipped)
            last_seq = seq
            timing.count(f"{track}_new")
            encode_t0 = time.perf_counter()
            if last_encode_t is not None:
                timing.add("video_gap_ms", (encode_t0 - last_encode_t) * 1000.0)
            last_encode_t = encode_t0
            portal.send_video_frame(track, rgb_buf, timestamp_us=ts)
            timing.add("grab_ms", grab_ms)
            timing.add("encode_ms", (time.perf_counter() - encode_t0) * 1000.0)
            skip_count = 0
            if not logged:
                h, w = rgb_buf.shape[:2]
                logger_mp.info(f"publishing '{track}' {w}x{h} (video thread)")
                logged = True
        except EncodeUnavailableError as exc:
            logger_mp.error(f"video thread HW encode DoD failed: {exc}")
            stop_evt.set()
            return
        except Exception as exc:
            logger_mp.warning(f"video thread: {exc}")


def video_publish_loop_maybe_profile(
    frame_source, portal, track, stop_evt, tick_slot, timing
):
    """Optional cProfile around the video thread when SAG_PROFILE_VIDEO=1."""
    if os.environ.get("SAG_PROFILE_VIDEO", "0") != "1":
        video_publish_loop(frame_source, portal, track, stop_evt, tick_slot, timing)
        return
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        video_publish_loop(frame_source, portal, track, stop_evt, tick_slot, timing)
    finally:
        profiler.disable()
        dump_path = os.environ.get("SAG_PROFILE_VIDEO_PATH", "/tmp/teleop_video.cprof")
        profiler.dump_stats(dump_path)
        logger_mp.info(f"video thread cProfile dumped to {dump_path}")


class PortalRobotTransport:
    """connect / on_action / send_state around livekit.portal.Robot."""

    def __init__(self,
                 portal_yaml: str,
                 env_file: str,
                 identity: str = "xr-robot",
                 room: str | None = None,
                 url: str | None = None,
                 mapping_yaml: str | None = None,
                 require_hw_encode: bool = False):
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
        self._operators = set()
        self._operators_lock = threading.Lock()
        self._robot.on_operator_joined(self._on_operator_joined)
        self._robot.on_operator_left(self._on_operator_left)

        self._require_hw = bool(require_hw_encode)
        self._hw_lock = threading.Lock()
        self._hw_confirmed = False
        self._hw_deadline: float | None = None
        self._hw_frames = 0
        self._hw_error: BaseException | None = None
        self._tee: FdTee | None = None
        self._confirm_s = hw_encode_confirm_timeout_s()

        if self._require_hw:
            report = assert_encode_ready(
                JetsonMsencEncodeCapability(require_hw=True)
            )
            logger_mp.info(
                f"[portal-robot] encode_capability backend={report.backend} "
                f"detail={report.detail} require_hw_encode=True"
            )

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

    def has_operator(self) -> bool:
        """True if at least one Portal operator is currently in the room."""
        with self._operators_lock:
            return bool(self._operators)

    def _on_operator_joined(self, identity: str) -> None:
        with self._operators_lock:
            self._operators.add(identity)
            n = len(self._operators)
        logger_mp.info(f"[portal-robot] operator joined: {identity} (n={n})")

    def _on_operator_left(self, identity: str) -> None:
        with self._operators_lock:
            self._operators.discard(identity)
            n = len(self._operators)
        logger_mp.info(f"[portal-robot] operator left: {identity} (n={n})")

    def _seed_operators(self) -> None:
        try:
            current = list(self._robot.operators() or [])
        except Exception as exc:
            logger_mp.warning(f"[portal-robot] operators() seed failed: {exc}")
            return
        with self._operators_lock:
            self._operators.update(current)
            n = len(self._operators)
        if current:
            logger_mp.info(f"[portal-robot] seeded operators: {current} (n={n})")

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
        ts = timestamp_us if timestamp_us is not None else now_us()
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
        self._seed_operators()
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
