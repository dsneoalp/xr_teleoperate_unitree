"""Portal robot control loop: apply joint targets via SDK, stream state.

No TeleVuer / IK. Pair with teleop_operator.py in the same LiveKit room.

Control loop is the clock: one tick_ts for send_state and every video track.
Camera ingest lives in ``teleop.utils.bgr_source``; video publish in
``portal_robot.video_publish_loop``. Command smoothing is ``cmd_filter``.

    python teleop/teleop_robot.py --ee dex3 --arm G1_29
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --motion
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --sim
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --require-hw-encode
"""
import time
import argparse
import threading
import logging_mp
try:
    logging_mp.basicConfig(level=logging_mp.INFO)
except RuntimeError:
    pass
logger_mp = logging_mp.getLogger(__name__)

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.encode import EncodeUnavailableError
from teleop.robot_control.portal_robot import (
    PortalRobotTransport,
    video_publish_loop,
    video_publish_loop_maybe_profile,
)
from teleop.robot_control.portal_mapping import UnpackedAction
from teleop.robot_control.tick_slot import LatestTickSlot, now_us
from teleop.utils.bgr_source import connect_frame_sources
from teleop.utils.cmd_filter import filter_cmd, interp_cmd
from teleop.utils.loop_timing import LoopTiming
from teleop.utils.arm_stiffness import ARM_STIFFNESS_FADE_S

FSM_IDLE = 0
FSM_TELEOP = 1
FSM_HOME = 2
FSM_HAND_SETUP = 3
ACTION_TIMEOUT = 0.2
VIDEO_OVERWRITE_LOG_EVERY = 30


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--frequency', type=float, default=30.0,
        help='control-loop and send_state rate (Hz). Video stamps the same '
             'tick_ts once (Portal unified sampling); DDS uses this rate.')
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
    parser.add_argument('--cmd-tau', type=float, default=0.3,
                        help='command smoothing time (s); 0 = off. Switch interp/filter in the TELEOP loop.')
    parser.add_argument(
        '--require-hw-encode', action='store_true',
        help='Fail-closed Jetson MMAPI encode (nvhost-msenc FD + FFI log). '
             'Off by default so PC/sim/mock keep software encode.')
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
        url=args.livekit_url,
        require_hw_encode=args.require_hw_encode)
    portal.wait_until_connected()

    lock = threading.Lock()
    latest = {'action': None, 'wall': 0.0}
    applied_fsm = FSM_IDLE
    timing = LoopTiming(logger_mp)

    img_sources = {} if args.no_img else connect_frame_sources(
        args.img_server_ip, portal.video_tracks
    )
    video_stop = threading.Event()
    video_threads = []
    tick_slots: list[LatestTickSlot] = []
    for i, (track, source) in enumerate(img_sources.items()):
        slot = LatestTickSlot()
        tick_slots.append(slot)
        target = (
            video_publish_loop_maybe_profile if i == 0 else video_publish_loop
        )
        thread = threading.Thread(
            target=target,
            args=(source, portal, track, video_stop, slot, timing),
            name=f"video-{track}",
            daemon=True)
        thread.start()
        video_threads.append(thread)
    if video_threads:
        logger_mp.info(f"video publish threads started: {list(img_sources)}")

    last_action_wall = {'t': 0.0}
    arm_cmd = {}
    hand_cmd = {}
    last_tick = 0.0
    loco_stale = False
    operator_present = False
    had_operator = False
    video_overwrites = 0

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
            present = portal.has_operator()
            if present and not operator_present:
                logger_mp.info("operator in session; restore arm stiffness")
                arm_ctrl.restore_arm_stiffness()
                had_operator = True
            elif (not present) and operator_present and had_operator:
                logger_mp.info("no operator in session; fading arm stiffness")
                arm_ctrl.fade_arm_stiffness()
                applied_fsm = FSM_IDLE
                arm_cmd.clear()
                hand_cmd.clear()
            operator_present = present

            if present and fsm == FSM_HOME:
                if applied_fsm != FSM_HOME:
                    arm_ctrl.ctrl_dual_arm_go_home()
                    applied_fsm = FSM_HOME
                arm_cmd.clear()
                hand_cmd.clear()
            elif present and fsm == FSM_HAND_SETUP and action:
                if applied_fsm != FSM_HAND_SETUP:
                    arm_cmd.clear()
                    hand_cmd.clear()
                    if hand_ctrl is not None:
                        hand_cmd['q'] = np.asarray(hand_ctrl.get_current_dual_hand_q(), dtype=float).copy()
                if hand_ctrl is not None and action.hand_q.size:
                    half = action.hand_q.size // 2
                    hand_q = action.hand_q
                    hand_ctrl.ctrl_dual_hand(hand_q[:half], hand_q[half:])
                applied_fsm = FSM_HAND_SETUP
            elif present and fsm == FSM_TELEOP and action:
                if applied_fsm != FSM_TELEOP:
                    arm_ctrl.speed_gradual_max()
                    arm_cmd.clear()
                    arm_cmd['q'] = np.asarray(arm_ctrl.get_current_dual_arm_q(), dtype=float).copy()
                    hand_cmd.clear()
                    if hand_ctrl is not None:
                        hand_cmd['q'] = np.asarray(hand_ctrl.get_current_dual_hand_q(), dtype=float).copy()
                tauff = np.zeros_like(action.arm_q)
                arm_q = filter_cmd(arm_cmd, action.arm_q, dt, args.cmd_tau)
                arm_ctrl.ctrl_dual_arm(arm_q, tauff)
                if hand_ctrl is not None and action.hand_q.size:
                    half = action.hand_q.size // 2
                    hand_q = interp_cmd(hand_cmd, action.hand_q, start, args.cmd_tau)
                    hand_ctrl.ctrl_dual_hand(hand_q[:half], hand_q[half:])
                applied_fsm = FSM_TELEOP
            elif present:
                applied_fsm = FSM_IDLE
                arm_cmd.clear()
                hand_cmd.clear()

            if loco_wrapper is not None:
                vx, vy, vyaw = 0.0, 0.0, 0.0
                if present and fsm == FSM_TELEOP and action is not None and age <= ACTION_TIMEOUT:
                    vx, vy, vyaw = action.vx, action.vy, action.vyaw
                    if loco_stale:
                        logger_mp.info(f"action fresh (age={age:.3f}s), loco linear resume")
                        loco_stale = False
                elif present and fsm == FSM_TELEOP and action is not None:
                    vyaw = action.vyaw
                    if not loco_stale:
                        logger_mp.info(f"action stale (age={age:.3f}s), loco linear stop")
                        loco_stale = True
                loco_wrapper.Move(vx, vy, vyaw)

            motor_q = arm_ctrl.get_current_motor_q()
            hand_q = hand_ctrl.get_current_dual_hand_q() if hand_ctrl is not None else None
            tick_ts = now_us()
            portal.send_state(motor_q=motor_q, hand_q=hand_q, fsm_id=fsm,
                              timestamp_us=tick_ts)
            replaced_any = False
            for slot in tick_slots:
                if not slot.offer(tick_ts):
                    replaced_any = True
            if replaced_any:
                video_overwrites += 1
                if video_overwrites == 1 or video_overwrites % VIDEO_OVERWRITE_LOG_EVERY == 0:
                    logger_mp.info(
                        f"video still encoding, offered newer tick "
                        f"(overwrites={video_overwrites})")
            elif video_overwrites:
                logger_mp.info(
                    f"video caught up after {video_overwrites} overwritten ticks")
                video_overwrites = 0

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
            logger_mp.info("robot shutdown; fading arm stiffness")
            arm_ctrl.fade_arm_stiffness()
            arm_ctrl.wait_arm_stiffness_fade(timeout=ARM_STIFFNESS_FADE_S + 0.5)
        except Exception as e:
            logger_mp.error(f"arm stiffness fade on shutdown failed: {e}")
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
