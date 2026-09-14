"""Portal robot control loop: apply joint targets via SDK, stream state.

No TeleVuer / IK. Pair with teleop_operator.py in the same LiveKit room.

    python teleop/teleop_robot.py --ee dex3 --arm G1_29
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --motion
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --sim
"""
import time
import math
import argparse
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os
import sys
import numpy as np
import yaml

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.encode import EncodeUnavailableError
from teleop.robot_control.portal_robot import PortalRobotTransport
from teleop.robot_control.portal_mapping import UnpackedAction
from teleop.utils.bgr_source import (
    BgrCameraSource,
    bgr_to_rgb,
    bgr_zmq_port,
    even_crop,
    fetch_teleimager_config,
)
from teleop.utils.loop_timing import LoopTiming
from teleop.utils.mosaic import MosaicCompositor, load_mosaic_layout

FSM_IDLE = 0
FSM_TELEOP = 1
FSM_HOME = 2
FSM_HAND_SETUP = 3
ACTION_TIMEOUT = 0.2
IMAGE_CLIENT_RETRIES = 50
IMAGE_CLIENT_RETRY_S = 0.1
MOSAIC_YAML = os.path.join(current_dir, "mosaic.yaml")


def interp_cmd(state, q_cmd, now, tau):
    """Linear segment q0→q1 over tau s. Restart on a new target; hold at s=1."""
    q_cmd = np.asarray(q_cmd, dtype=float)
    if tau <= 0.0:
        state['q'] = q_cmd.copy()
        return q_cmd
    if 'q' not in state:
        state['q'] = q_cmd.copy()
        state['q0'] = q_cmd.copy()
        state['q1'] = q_cmd.copy()
        state['t0'] = now
        return q_cmd.copy()
    prev = state.get('q1')
    if prev is None or prev.shape != q_cmd.shape or not np.allclose(q_cmd, prev):
        state['q0'] = np.asarray(state['q'], dtype=float).copy()
        state['q1'] = q_cmd.copy()
        state['t0'] = now
    s = min(1.0, (now - state['t0']) / tau)
    q = (1.0 - s) * state['q0'] + s * state['q1']
    state['q'] = q
    return q


def filter_cmd(state, q_cmd, dt, tau):
    """PT1 toward q_cmd. Continues catching up while the target is held."""
    q_cmd = np.asarray(q_cmd, dtype=float)
    if tau <= 0.0:
        state['q'] = q_cmd.copy()
        return q_cmd
    if 'q' not in state:
        state['q'] = q_cmd.copy()
        return q_cmd.copy()
    alpha = 1.0 - math.exp(-max(dt, 1e-6) / tau)
    q = state['q'] + alpha * (q_cmd - state['q'])
    state['q'] = q
    return q


def _local_cam_config_paths():
    """Bind-mounted teleop YAMLs. site-packages ZMQ_Requester looks two dirs above
    image_client.py and misses these when GET_DATA on :60000 times out."""
    return (
        os.path.join(current_dir, "teleimager", "cam_config_client.yaml"),
        os.path.join(current_dir, "teleimager", "cam_config_server.yaml"),
        os.path.join(current_dir, "utils", "portal_cam_config.yaml"),
    )


def _load_local_cam_config():
    for path in _local_cam_config_paths():
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
        except Exception as exc:
            logger_mp.warning(f"Failed to load local {path}: {exc}")
            continue
        if cfg:
            logger_mp.info(f"Loaded camera config from local {path}")
            return cfg
    return None


def _log_cam_config(cfg):
    """Log teleimager slots (shape is [H, W])."""
    logger_mp.info(f"teleimager cam_config:\n{yaml.safe_dump(cfg, sort_keys=False).rstrip()}")


class _JpegSlotSource:
    """JPEG fallback for one teleimager slot when raw BGR is unavailable."""

    def __init__(self, host: str, cam_config, slot: str, request_bgr: bool = True):
        from teleimager.image_client import ZMQ_SubscriberManager

        self._host = host
        self._slot = slot
        self._request_bgr = request_bgr
        self._cam_config = cam_config
        self._rgb_buf = None
        self._seq = 0
        self._subscriber_manager = ZMQ_SubscriberManager.get_instance()
        cam = self._cam_config.get(slot) or {}
        if not cam.get("enable_zmq"):
            logger_mp.warning(
                f"[Image Client] {slot} ZMQ is not enabled; track will not publish"
            )
            return
        port = cam["zmq_port"]
        self._subscriber_manager.subscribe(host, port, request_bgr=request_bgr)
        logger_mp.info(f"JPEG ZMQ '{slot}' {host}:{port} (BGR unavailable)")

    def latest_rgb(self):
        cam = self._cam_config[self._slot]
        frame = self._subscriber_manager.subscribe(
            self._host, cam["zmq_port"], request_bgr=self._request_bgr
        )
        if frame is None or getattr(frame, "bgr", None) is None:
            return None
        rgb = bgr_to_rgb(even_crop(frame.bgr), self._rgb_buf)
        self._rgb_buf = rgb
        self._seq += 1
        return rgb, self._seq

    def close(self):
        logger_mp.info(f"JPEG source '{self._slot}' closed.")


def _connect_frame_sources(host: str, slots):
    """One ingest per mosaic slot. BGR ZMQ preferred, JPEG fallback."""
    last_error = None
    for attempt in range(1, IMAGE_CLIENT_RETRIES + 1):
        try:
            cfg = fetch_teleimager_config(host)
            if cfg is not None:
                logger_mp.info(f"Received camera config from server {host}:60000")
                _log_cam_config(cfg)
            else:
                cfg = _load_local_cam_config()
                if cfg is not None:
                    _log_cam_config(cfg)
            if cfg is None:
                raise RuntimeError("Failed to get camera configuration.")
            sources = {}
            try:
                for slot in slots:
                    port = bgr_zmq_port(cfg, slot)
                    if port is not None:
                        source = BgrCameraSource(host, port, slot=slot)
                        source.start()
                        sources[slot] = source
                        logger_mp.info(
                            f"slot '{slot}' BGR {host}:{port} (attempt {attempt})"
                        )
                        continue
                    cam = cfg.get(slot) if isinstance(cfg.get(slot), dict) else None
                    if cam and cam.get("enable_zmq"):
                        sources[slot] = _JpegSlotSource(
                            host=host, cam_config=cfg, slot=slot, request_bgr=True
                        )
                        logger_mp.info(
                            f"slot '{slot}' JPEG fallback (attempt {attempt})"
                        )
                        continue
                    logger_mp.warning(
                        f"mosaic slot '{slot}' has no teleimager BGR/JPEG source; skip"
                    )
                if sources:
                    return sources
                raise RuntimeError("No video sources matched mosaic.yaml slots.")
            except Exception:
                for source in sources.values():
                    try:
                        source.close()
                    except Exception:
                        pass
                raise
        except Exception as exc:
            last_error = exc
            time.sleep(IMAGE_CLIENT_RETRY_S)
    logger_mp.warning(
        f"ImageClient failed after {IMAGE_CLIENT_RETRIES} tries: {last_error}"
    )
    return {}


def _video_publish_loop(sources, compositor, portal, track, stop_evt, fps: float, timing):
    """Compose dirty tiles onto the mosaic canvas and send one RGB frame."""
    logged = False
    last_seq = {slot: -1 for slot in compositor.layout.slots}
    interval = 1.0 / max(fps, 1.0)
    while not stop_evt.is_set():
        t0 = time.perf_counter()
        try:
            any_new = False
            mosaic_t0 = time.perf_counter()
            for slot, source in sources.items():
                item = source.latest_rgb()
                if item is None:
                    continue
                rgb_buf, seq = item
                prev = last_seq.get(slot, -1)
                if seq == prev:
                    continue
                if prev >= 0:
                    skipped = seq - prev - 1
                    if skipped > 0:
                        timing.count(f"{slot}_src_skipped", skipped)
                last_seq[slot] = seq
                compositor.paste(slot, rgb_buf)
                any_new = True
                timing.count(f"{slot}_new")
            timing.add("mosaic_ms", (time.perf_counter() - mosaic_t0) * 1000.0)
            if any_new:
                send_t0 = time.perf_counter()
                portal.send_video_frame(
                    track, compositor.canvas, timestamp_us=int(time.time() * 1_000_000)
                )
                timing.add("send_ms", (time.perf_counter() - send_t0) * 1000.0)
                if not logged:
                    h, w = compositor.canvas.shape[:2]
                    logger_mp.info(
                        f"publishing '{track}' {w}x{h} mosaic (video thread)"
                    )
                    logged = True
            else:
                timing.count("mosaic_dup")
        except EncodeUnavailableError as exc:
            logger_mp.error(f"video thread HW encode DoD failed: {exc}")
            stop_evt.set()
            return
        except Exception as exc:
            logger_mp.warning(f"video thread: {exc}")
        sleep = interval - (time.perf_counter() - t0)
        if sleep > 0:
            stop_evt.wait(sleep)


def _video_publish_loop_maybe_profile(
    sources, compositor, portal, track, stop_evt, fps: float, timing
):
    """Optional cProfile around the video thread when SAG_PROFILE_VIDEO=1."""
    if os.environ.get("SAG_PROFILE_VIDEO", "0") != "1":
        _video_publish_loop(
            sources, compositor, portal, track, stop_evt, fps, timing
        )
        return
    import cProfile

    profiler = cProfile.Profile()
    profiler.enable()
    try:
        _video_publish_loop(
            sources, compositor, portal, track, stop_evt, fps, timing
        )
    finally:
        profiler.disable()
        dump_path = os.environ.get("SAG_PROFILE_VIDEO_PATH", "/tmp/teleop_video.cprof")
        profiler.dump_stats(dump_path)
        logger_mp.info(f"video thread cProfile dumped to {dump_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=float, default=30.0, help='state publish rate')
    parser.add_argument('--arm', type=str, choices=['G1_29'], default='G1_29')
    parser.add_argument('--ee', type=str, choices=['dex3'], default=None)
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--network-interface', type=str, default=None)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--no-img', action='store_true',
                        help='do not subscribe to teleimager / publish video')
    parser.add_argument('--img-server-ip', type=str, default='127.0.0.1',
                        help='teleimager ZMQ host (config port 60000)')
    parser.add_argument('--portal-yaml', type=str, default=os.path.join(current_dir, 'portal.yaml'))
    parser.add_argument('--portal-mapping', type=str, default=os.path.join(current_dir, 'portal_mapping.yaml'))
    parser.add_argument('--env-file', type=str, default=os.path.join(current_dir, '.env'))
    parser.add_argument('--livekit-url', type=str, default=None)
    parser.add_argument('--livekit-room', type=str, default=None)
    parser.add_argument('--portal-identity', type=str, default='xr-robot')
    parser.add_argument('--cmd-tau', type=float, default=0.15,
                        help='command smoothing time (s); 0 = off. Switch interp/filter in the TELEOP loop.')
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(1 if args.sim else 0, networkInterface=args.network_interface)

    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper

    loco_wrapper = None
    motion_switcher = None
    if args.motion:
        loco_wrapper = LocoClientWrapper(robot_type="G1")
    else:
        motion_switcher = MotionSwitcher()
        status, result = motion_switcher.Enter_Debug_Mode()
        logger_mp.info(f"Enter debug mode: {'Success' if status == 0 else 'Failed'}")

    arm_ctrl = G1_29_ArmController(motion_mode=args.motion, simulation_mode=args.sim)

    # DDS apply only. XR→Dex3 retargeting runs on the operator; actions arrive as hand_q.
    hand_ctrl = None
    if args.ee == "dex3":
        from teleop.robot_control.robot_hand_unitree import Dex3_1_Controller
        hand_ctrl = Dex3_1_Controller(apply_targets=True, simulation_mode=args.sim)

    portal = PortalRobotTransport(
        portal_yaml=args.portal_yaml,
        mapping_yaml=args.portal_mapping,
        env_file=args.env_file,
        identity=args.portal_identity,
        room=args.livekit_room,
        url=args.livekit_url)
    portal.wait_until_connected()

    lock = threading.Lock()
    latest = {'action': None, 'wall': 0.0}
    applied_fsm = FSM_IDLE
    timing = LoopTiming(logger_mp)

    mosaic_layout = load_mosaic_layout(MOSAIC_YAML)
    compositor = MosaicCompositor(mosaic_layout)
    img_sources = {} if args.no_img else _connect_frame_sources(
        args.img_server_ip, mosaic_layout.slots
    )
    video_stop = threading.Event()
    video_threads = []
    track = portal.video_tracks[0] if portal.video_tracks else None
    if img_sources and track:
        thread = threading.Thread(
            target=_video_publish_loop_maybe_profile,
            args=(
                img_sources, compositor, portal, track, video_stop,
                args.frequency, timing,
            ),
            name="video-mosaic",
            daemon=True)
        thread.start()
        video_threads.append(thread)
        logger_mp.info(
            f"mosaic {mosaic_layout.width}x{mosaic_layout.height} "
            f"slots={list(img_sources)} track='{track}'"
        )
    last_action_wall = {'t': 0.0}
    arm_cmd = {}
    hand_cmd = {}
    last_tick = 0.0

    def on_action(action: UnpackedAction):
        now = time.time()
        with lock:
            prev = last_action_wall['t']
            latest['action'] = action
            latest['wall'] = now
            last_action_wall['t'] = now
        if prev:
            timing.add("action_gap_ms", (now - prev) * 1000.0)

    portal.on_unpacked_action(on_action)
    logger_mp.info("robot loop running; waiting for operator actions")

    try:
        while True:
            start = time.time()
            dt = (start - last_tick) if last_tick else (1.0 / args.frequency)
            last_tick = start
            with lock:
                action = latest['action']
                age = start - latest['wall'] if latest['wall'] else 1e9

            fsm = action.fsm_id if action is not None else FSM_IDLE

            if fsm == FSM_HOME:
                if applied_fsm != FSM_HOME:
                    arm_ctrl.ctrl_dual_arm_go_home()
                    applied_fsm = FSM_HOME
                arm_cmd.clear()
                hand_cmd.clear()
            elif fsm == FSM_HAND_SETUP and action:
                if applied_fsm != FSM_HAND_SETUP:
                    arm_cmd.clear()
                    hand_cmd.clear()
                    if hand_ctrl is not None:
                        hand_cmd['q'] = np.asarray(hand_ctrl.get_current_dual_hand_q(), dtype=float).copy()
                if hand_ctrl is not None and action.hand_q.size:
                    half = action.hand_q.size // 2
                    hand_q = interp_cmd(hand_cmd, action.hand_q, start, args.cmd_tau)
                    hand_ctrl.ctrl_dual_hand(hand_q[:half], hand_q[half:])
                applied_fsm = FSM_HAND_SETUP
            elif fsm == FSM_TELEOP and action:
                if applied_fsm != FSM_TELEOP:
                    arm_ctrl.speed_gradual_max()
                    arm_cmd.clear()
                    arm_cmd['q'] = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float).copy()
                    hand_cmd.clear()
                    if hand_ctrl is not None:
                        hand_cmd['q'] = np.asarray(hand_ctrl.get_current_dual_hand_q(), dtype=float).copy()
                tauff = np.zeros_like(action.arm_q)
                # pick one (arm + hand must match):
                arm_q = interp_cmd(arm_cmd, action.arm_q, start, args.cmd_tau)
                # arm_q = filter_cmd(arm_cmd, action.arm_q, dt, args.cmd_tau)
                arm_ctrl.ctrl_dual_arm(arm_q, tauff)
                if hand_ctrl is not None and action.hand_q.size:
                    half = action.hand_q.size // 2
                    hand_q = interp_cmd(hand_cmd, action.hand_q, start, args.cmd_tau)
                    # hand_q = filter_cmd(hand_cmd, action.hand_q, dt, args.cmd_tau)
                    hand_ctrl.ctrl_dual_hand(hand_q[:half], hand_q[half:])
                if loco_wrapper is not None:
                    loco_wrapper.Move(action.vx, action.vy, action.vyaw)
                applied_fsm = FSM_TELEOP
            else:
                applied_fsm = FSM_IDLE
                arm_cmd.clear()
                hand_cmd.clear()

            motor_q = arm_ctrl.get_current_motor_q()
            hand_q = hand_ctrl.get_current_dual_hand_q() if hand_ctrl is not None else None
            portal.send_state(motor_q=motor_q, hand_q=hand_q, fsm_id=fsm)

            hw_err = portal.hw_encode_error
            if hw_err is not None:
                raise hw_err

            timing.add("loop_ms", (time.time() - start) * 1000.0)
            sleep = max(0.0, (1.0 / args.frequency) - (time.time() - start))
            time.sleep(sleep)
    except EncodeUnavailableError as exc:
        logger_mp.error(f"HW encode DoD failed: {exc}")
        raise SystemExit(1) from exc
    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting ...")
    finally:
        try:
            if motion_switcher is not None:
                motion_switcher.Exit_Debug_Mode()
                logger_mp.info(f"Exit debug / SelectMode('ai'): status={status} result={result}")
                _, mode = motion_switcher.msc.CheckMode()
                logger_mp.info(f"CheckMode after exit: {mode}")
                
        except Exception as e:
            logger_mp.error(f"Exit debug mode failed: {e}")
        try:
            portal.close()
        except Exception as e:
            logger_mp.error(f"portal close failed: {e}")
        video_stop.set()
        for thread in video_threads:
            thread.join(timeout=2.0)
        for source in img_sources.values():
            try:
                source.close()
            except Exception as e:
                logger_mp.error(f"video source close failed: {e}")
        logger_mp.info("robot exited.")
