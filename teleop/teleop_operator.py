"""Portal operator control loop: TeleVuer, IK, hand retargeting, send_targets.

No Unitree SDK. Pair with teleop_robot.py in the same LiveKit room.

    python teleop/teleop_operator.py --ee dex3 --arm G1_29 --input-mode hand
    python teleop/teleop_operator.py --ee dex3 --arm G1_29 --input-mode controller
    python teleop/teleop_operator.py --ee dex3 --arm G1_29 --input-mode controller --motion
    python teleop/teleop_operator.py --ee dex3 --arm G1_29 --input-mode controller --custom_mapping --hand-pose-yaml teleop/my_poses.yaml
"""
import time
import argparse
from multiprocessing import Value, Array, Lock
import threading
import numpy as np
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from televuer import TeleVuerWrapper
from teleop.robot_control.robot_arm_ik import G1_29_ArmIK
from teleop.utils.episode_writer import EpisodeWriter
from teleop.utils.ipc import IPC_Server
from sshkeyboard import listen_keyboard, stop_listening
from teleop.robot_control.portal_operator import PortalTeleopBridge
from teleop.utils.loop_timing import LoopTiming
from teleop.utils.dex3_pose_mapping import (
    load,
    pressed_from_tele_data,
    ramp_step,
    ramp_towards,
    resolve_mapping_mode,
    target_q,
)

LOCO_SCALE = 0.3
FSM_IDLE = 0
FSM_TELEOP = 1
FSM_HOME = 2
FSM_HAND_SETUP = 3

# Dex3 hardware-order open/close poses (thumb only; index/middle stay open).
# Left: thumb0, thumb1, thumb2, middle0, middle1, index0, index1
# Right: thumb0, thumb1, thumb2, index0, index1, middle0, middle1
DEX3_OPEN_Q_LEFT = np.zeros(7)
DEX3_CLOSED_Q_LEFT = np.array([0.25, 0.78, 1.48, 0.0, 0.0, 0.0, 0.0])
DEX3_OPEN_Q_RIGHT = np.zeros(7)
DEX3_CLOSED_Q_RIGHT = np.array([-0.25, -0.78, -1.48, 0.0, 0.0, 0.0, 0.0])


def ramp_dex3_oc(s, close_pressed, open_pressed, ramp):
    """Update open/close scalar in [0, 1]. X (close) wins if both are held."""
    if close_pressed:
        return min(1.0, s + ramp)
    if open_pressed:
        return max(0.0, s - ramp)
    return s


def dex3_oc_hand_q(s):
    s = float(np.clip(s, 0.0, 1.0))
    left = (1.0 - s) * DEX3_OPEN_Q_LEFT + s * DEX3_CLOSED_Q_LEFT
    right = (1.0 - s) * DEX3_OPEN_Q_RIGHT + s * DEX3_CLOSED_Q_RIGHT
    return np.concatenate((left, right))


START = False
STOP = False
READY = False
RECORD_RUNNING = False
RECORD_TOGGLE = False
MAPPING_GUI_OPEN = False


def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
        if MAPPING_GUI_OPEN:
            logger_mp.warning("Accept the pose GUI first, then press [r].")
            return
        START = True
    elif key == 'q':
        START = False
        STOP = True
    elif key == 's' and START is True:
        RECORD_TOGGLE = True
    else:
        logger_mp.warning(f"[on_press] {key} was pressed, but no action is defined for this key.")


def get_state() -> dict:
    global START, STOP, RECORD_RUNNING, READY
    return {
        "START": START,
        "STOP": STOP,
        "READY": READY,
        "RECORD_RUNNING": RECORD_RUNNING,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=float, default=30.0)
    parser.add_argument('--input-mode', type=str, choices=['hand', 'controller'], default='hand')
    parser.add_argument('--display-mode', type=str, choices=['immersive', 'ego', 'pass-through'], default='immersive')
    parser.add_argument('--arm', type=str, choices=['G1_29'], default='G1_29')
    parser.add_argument('--ee', type=str, choices=['dex3'], default=None)
    parser.add_argument('--motion', action='store_true',
                        help='Read vx/vy/vyaw from controller thumbsticks (robot must also use --motion)')
    parser.add_argument('--dex3-oc-duration', type=float, default=1.5,
                        help='Seconds for a full Dex3 open↔close ramp via left X/Y (controller mode only)')
    parser.add_argument('--custom_mapping', action='store_true',
                        help='Bind Dex3 poses to controller combos (requires --ee dex3 --input-mode controller)')
    parser.add_argument('--hand-pose-yaml', type=str,
                        default=os.path.join(current_dir, 'hand_pose_session.yaml'),
                        help='YAML path for --custom_mapping (load if present, else GUI writes it)')
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--ipc', action='store_true')
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--task-dir', type=str, default='./utils/data/')
    parser.add_argument('--task-name', type=str, default='pick cube')
    parser.add_argument('--task-goal', type=str, default='pick up cube.')
    parser.add_argument('--task-desc', type=str, default='task description')
    parser.add_argument('--task-steps', type=str, default='step1: do this; step2: do that;')
    parser.add_argument('--portal-yaml', type=str, default=os.path.join(current_dir, 'portal.yaml'))
    parser.add_argument('--portal-mapping', type=str, default=os.path.join(current_dir, 'portal_mapping.yaml'))
    parser.add_argument('--env-file', type=str, default=os.path.join(current_dir, '.env'))
    parser.add_argument('--livekit-url', type=str, default=None)
    parser.add_argument('--livekit-room', type=str, default=None)
    parser.add_argument('--portal-identity', type=str, default='xr-teleop')
    parser.add_argument('--cam-config', type=str, default=os.path.join(current_dir, 'utils', 'portal_cam_config.yaml'))
    args = parser.parse_args()

    if args.ee == "dex3" and args.input_mode == "controller" and args.dex3_oc_duration <= 0:
        parser.error("--dex3-oc-duration must be > 0.")
    if args.custom_mapping:
        if args.ee != "dex3" or args.input_mode != "controller":
            parser.error("--custom_mapping requires --ee dex3 --input-mode controller")
    mapping_mode = resolve_mapping_mode(args.custom_mapping, os.path.isfile(args.hand_pose_yaml))
    if mapping_mode == "gui" and args.headless:
        parser.error("--custom_mapping --headless requires an existing --hand-pose-yaml file")

    custom_spec = None

    try:
        if args.ipc:
            ipc_server = IPC_Server(on_press=on_press, get_state=get_state)
            ipc_server.start()
        else:
            listen_keyboard_thread = threading.Thread(
                target=listen_keyboard,
                kwargs={"on_press": on_press, "until": None, "sequential": False},
                daemon=True)
            listen_keyboard_thread.start()

        left_hand_pos_array = None
        right_hand_pos_array = None
        dual_hand_data_lock = None
        dual_hand_state_array = None
        dual_hand_action_array = None
        if args.ee == "dex3":
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock=False)
            dual_hand_action_array = Array('d', 14, lock=False)
            if args.input_mode == "hand":
                left_hand_pos_array = Array('d', 75, lock=True)
                right_hand_pos_array = Array('d', 75, lock=True)

        teleop_bridge = PortalTeleopBridge(
            portal_yaml=args.portal_yaml,
            mapping_yaml=args.portal_mapping,
            env_file=args.env_file,
            identity=args.portal_identity,
            room=args.livekit_room,
            url=args.livekit_url,
            ee=args.ee,
            left_hand_array_in=left_hand_pos_array,
            right_hand_array_in=right_hand_pos_array,
            dual_hand_data_lock=dual_hand_data_lock,
            dual_hand_state_array_out=dual_hand_state_array,
            dual_hand_action_array_out=dual_hand_action_array,
            cam_config_path=args.cam_config)
        camera_config = teleop_bridge.get_cam_config()
        teleop_bridge.wait_until_connected()

        xr_need_local_img = not (
            args.display_mode == 'pass-through' or camera_config['head_camera']['enable_webrtc'])

        tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=args.input_mode == "hand",
            binocular=camera_config['head_camera']['binocular'],
            img_shape=camera_config['head_camera']['image_shape'],
            display_mode=args.display_mode,
            zmq=camera_config['head_camera']['enable_zmq'],
            webrtc=camera_config['head_camera']['enable_webrtc'],
            webrtc_url=f"https://127.0.0.1:{camera_config['head_camera']['webrtc_port']}/offer",
            arm_reference_mode="head_yaw")

        xr_motion_data_ready = Value('b', False, lock=True)
        teleop_bridge.set_xr_motion_data_ready(xr_motion_data_ready)

        arm_ik = G1_29_ArmIK()

        if args.record:
            recorder = EpisodeWriter(
                task_dir=os.path.join(args.task_dir, args.task_name),
                task_goal=args.task_goal,
                task_desc=args.task_desc,
                task_steps=args.task_steps,
                frequency=args.frequency,
                rerun_log=not args.headless)

        if mapping_mode == "load":
            custom_spec = load(args.hand_pose_yaml)
            logger_mp.info(
                f"Loaded Dex3 pose mapping from {args.hand_pose_yaml} "
                f"({len(custom_spec.bindings)} bindings).")
        elif mapping_mode == "gui":
            from teleop.utils.dex3_pose_gui import run_dex3_pose_gui
            from teleop.utils.dex3_pose_gui_state import Dex3PoseGUIState
            from teleop.utils.dex3_pose_mapping import PoseSpec

            logger_mp.info(
                "Dex3 pose GUI: bind combos, Accept to write YAML, then press [r].")
            try:
                import tkinter  # noqa: F401
            except ImportError:
                logger_mp.error(
                    "tkinter is required for --custom_mapping when the YAML file is missing. "
                    "Install tk (conda-forge) or python3-tk.")
                STOP = True
            if not STOP:
                gui_state = Dex3PoseGUIState(PoseSpec(duration_s=args.dex3_oc_duration))
                gui_stop = threading.Event()
                gui_thread = threading.Thread(
                    target=run_dex3_pose_gui,
                    kwargs={
                        "state": gui_state,
                        "output_path": args.hand_pose_yaml,
                        "stop_event": gui_stop,
                    },
                    daemon=True)
                MAPPING_GUI_OPEN = True
                gui_thread.start()
                hold_arm = np.zeros(14)
                dt = 1.0 / args.frequency
                while (not gui_state.accepted.is_set()
                       and not gui_state.cancelled.is_set()
                       and not STOP):
                    tick = time.time()
                    if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                        head_img = teleop_bridge.get_head_frame()
                        if head_img.bgr is not None:
                            tv_wrapper.render_to_xr(head_img.bgr)
                    teleop_bridge.send_targets(
                        hold_arm,
                        hand_q=gui_state.get_q(),
                        vx=0.0, vy=0.0, vyaw=0.0,
                        fsm_id=FSM_HAND_SETUP)
                    time.sleep(max(0.0, dt - (time.time() - tick)))
                gui_stop.set()
                MAPPING_GUI_OPEN = False
                if STOP or not gui_state.accepted.is_set():
                    logger_mp.info("Pose GUI cancelled; exiting.")
                    STOP = True
                else:
                    custom_spec = load(args.hand_pose_yaml)
                    START = False
                    logger_mp.info(
                        f"Wrote Dex3 pose mapping to {args.hand_pose_yaml} "
                        f"({len(custom_spec.bindings)} bindings).")

        if not STOP:
            logger_mp.info("----------------------------------------------------------------")
            logger_mp.info("Press [r] to start syncing the robot with your movements.")
            if custom_spec is not None:
                logger_mp.info(
                    "Dex3 custom mapping: longest matching combo ramps to its pose; "
                    f"release ramps to default (duration={args.dex3_oc_duration}s). Right A=e-stop.")
            elif args.ee == "dex3" and args.input_mode == "controller":
                logger_mp.info("Dex3 X/Y: left X=close, left Y=open, right A=e-stop "
                               f"(duration={args.dex3_oc_duration}s).")
            if args.motion:
                logger_mp.info("Motion: thumbsticks send vx/vy/vyaw (robot must also use --motion).")
            if args.record:
                logger_mp.info("Press [s] to START or SAVE recording (toggle cycle).")
            logger_mp.info("Press [q] to stop and exit the program.")
            READY = True
            teleop_bridge.set_fsm_id(FSM_IDLE)
            teleop_bridge.send_go_home()
            while not START and not STOP:
                time.sleep(0.033)
                if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                    head_img = teleop_bridge.get_head_frame()
                    if head_img.bgr is not None:
                        tv_wrapper.render_to_xr(head_img.bgr)

        if not STOP:
            logger_mp.info("start Tracking")
            teleop_bridge.set_fsm_id(FSM_TELEOP)

        timing = LoopTiming(logger_mp)
        teleop_bridge.on_rtt(lambda rtt_ms: timing.add("rtt_ms", rtt_ms))
        last_send_t = None

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        dex3_oc_s = 0.0
        dex3_oc_ramp = 0.0
        custom_current_q = None
        custom_ramp = 0.0
        if custom_spec is not None:
            custom_current_q = list(custom_spec.default_q)
            custom_ramp = ramp_step(args.dex3_oc_duration, args.frequency)
        elif args.ee == "dex3" and args.input_mode == "controller":
            dex3_oc_ramp = 1.0 / (args.dex3_oc_duration * args.frequency)

        while not STOP:
            start_time = time.time()
            if camera_config['head_camera']['enable_zmq']:
                if args.record or xr_need_local_img:
                    head_img = teleop_bridge.get_head_frame()
                if xr_need_local_img and head_img is not None and head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            if camera_config['left_wrist_camera']['enable_zmq'] and args.record:
                left_wrist_img = teleop_bridge.get_left_wrist_frame()
            if camera_config['right_wrist_camera']['enable_zmq'] and args.record:
                right_wrist_img = teleop_bridge.get_right_wrist_frame()

            if args.record and RECORD_TOGGLE:
                RECORD_TOGGLE = False
                if not RECORD_RUNNING:
                    if recorder.create_episode():
                        RECORD_RUNNING = True
                    else:
                        logger_mp.error("Failed to create episode. Recording not started.")
                else:
                    RECORD_RUNNING = False
                    recorder.save_episode()

            tele_data = tv_wrapper.get_tele_data()
            controller_hand_q = None
            if args.ee == "dex3" and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            elif args.ee == "dex3" and args.input_mode == "controller":
                if custom_spec is not None:
                    pressed = pressed_from_tele_data(tele_data)
                    tgt = target_q(pressed, custom_spec)
                    custom_current_q = ramp_towards(custom_current_q, tgt, custom_ramp)
                    controller_hand_q = np.asarray(custom_current_q, dtype=float)
                else:
                    dex3_oc_s = ramp_dex3_oc(
                        dex3_oc_s,
                        tele_data.left_ctrl_aButton,
                        tele_data.left_ctrl_bButton,
                        dex3_oc_ramp)
                    controller_hand_q = dex3_oc_hand_q(dex3_oc_s)
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready

            vx = vy = vyaw = 0.0
            if args.input_mode == "controller":
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                if args.motion:
                    vx = -tele_data.left_ctrl_thumbstickValue[1] * LOCO_SCALE
                    vy = -tele_data.left_ctrl_thumbstickValue[0] * LOCO_SCALE
                    vyaw = -tele_data.right_ctrl_thumbstickValue[0] * LOCO_SCALE

            current_lr_arm_q = teleop_bridge.get_current_dual_arm_q()
            current_lr_arm_dq = teleop_bridge.get_current_dual_arm_dq()
            ik_t0 = time.perf_counter()
            sol_q, sol_tauff = arm_ik.solve_ik(
                tele_data.left_wrist_pose, tele_data.right_wrist_pose,
                current_lr_arm_q, current_lr_arm_dq)
            timing.add("ik_ms", (time.perf_counter() - ik_t0) * 1000.0)
            del sol_tauff
            fsm = FSM_TELEOP if START else FSM_IDLE
            send_now = time.perf_counter()
            if last_send_t is not None:
                timing.add("send_gap_ms", (send_now - last_send_t) * 1000.0)
            last_send_t = send_now
            age_ms = teleop_bridge.state_age_ms()
            if age_ms is not None:
                timing.add("state_age_ms", age_ms)
            teleop_bridge.send_targets(sol_q, hand_q=controller_hand_q, vx=vx, vy=vy, vyaw=vyaw, fsm_id=fsm)

            if args.record:
                READY = recorder.is_ready()
                left_ee_state = []
                right_ee_state = []
                left_hand_action = []
                right_hand_action = []
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                elif args.ee == "dex3" and args.input_mode == "controller":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                    if controller_hand_q is not None:
                        left_hand_action = controller_hand_q[:7].tolist()
                        right_hand_action = controller_hand_q[7:].tolist()
                body_action = [vx, vy, vyaw] if args.input_mode == "controller" and args.motion else []
                reported = teleop_bridge.get_reported_arm_q()
                rec_q = reported if reported is not None else current_lr_arm_q
                half = len(rec_q) // 2
                left_arm_state, right_arm_state = rec_q[:half], rec_q[half:]
                left_arm_action, right_arm_action = sol_q[:half], sol_q[half:]
                if RECORD_RUNNING:
                    colors = {}
                    depths = {}
                    if camera_config['head_camera']['binocular']:
                        if head_img is not None and head_img.bgr is not None:
                            w = camera_config['head_camera']['image_shape'][1] // 2
                            colors["color_0"] = head_img.bgr[:, :w]
                            colors["color_1"] = head_img.bgr[:, w:]
                    elif head_img is not None and head_img.bgr is not None:
                        colors["color_0"] = head_img.bgr
                    if left_wrist_img is not None and left_wrist_img.bgr is not None:
                        colors["color_2" if camera_config['head_camera']['binocular'] else "color_1"] = left_wrist_img.bgr
                    if right_wrist_img is not None and right_wrist_img.bgr is not None:
                        colors["color_3" if camera_config['head_camera']['binocular'] else "color_2"] = right_wrist_img.bgr
                    states = {
                        "left_arm": {"qpos": left_arm_state.tolist(), "qvel": [], "torque": []},
                        "right_arm": {"qpos": right_arm_state.tolist(), "qvel": [], "torque": []},
                        "left_ee": {"qpos": left_ee_state, "qvel": [], "torque": []},
                        "right_ee": {"qpos": right_ee_state, "qvel": [], "torque": []},
                        "body": {"qpos": []},
                    }
                    actions = {
                        "left_arm": {"qpos": left_arm_action.tolist(), "qvel": [], "torque": []},
                        "right_arm": {"qpos": right_arm_action.tolist(), "qvel": [], "torque": []},
                        "left_ee": {"qpos": left_hand_action, "qvel": [], "torque": []},
                        "right_ee": {"qpos": right_hand_action, "qvel": [], "torque": []},
                        "body": {"qpos": body_action},
                    }
                    recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

            timing.add("loop_ms", (time.time() - start_time) * 1000.0)
            sleep_time = max(0, (1 / args.frequency) - (time.time() - start_time))
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting program...")
    except Exception:
        import traceback
        logger_mp.error(traceback.format_exc())
    finally:
        try:
            teleop_bridge.send_go_home()
        except Exception as e:
            logger_mp.error(f"Failed to send_go_home: {e}")
        try:
            if args.ipc:
                ipc_server.stop()
            else:
                stop_listening()
                listen_keyboard_thread.join()
        except Exception as e:
            logger_mp.error(f"Failed to stop keyboard listener or ipc server: {e}")
        try:
            teleop_bridge.close()
        except Exception as e:
            logger_mp.error(f"Failed to close teleop_bridge: {e}")
        try:
            tv_wrapper.close()
        except Exception as e:
            logger_mp.error(f"Failed to close televuer wrapper: {e}")
        try:
            if args.record:
                recorder.close()
        except Exception as e:
            logger_mp.error(f"Failed to close recorder: {e}")
        logger_mp.info("Finally, exiting program.")
        raise SystemExit(0)
