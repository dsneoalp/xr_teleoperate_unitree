"""Portal operator control loop: TeleVuer, IK, hand retargeting, send_targets.

No Unitree SDK. Pair with teleop_robot.py in the same LiveKit room.

    python teleop/teleop_operator.py --ee dex3 --arm G1_29 --input-mode hand
    python teleop/teleop_operator.py --arm G1_29 --input-mode controller
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

LOCO_SCALE = 0.3
FSM_IDLE = 0
FSM_TELEOP = 1
FSM_HOME = 2


def _mat_is_valid(mat) -> bool:
    det = np.linalg.det(mat)
    return bool(np.isfinite(det) and not np.isclose(det, 0.0, atol=1e-6))


def _xyz(mat) -> np.ndarray:
    return np.asarray(mat)[:3, 3]


def _bar_col(bgr) -> int:
    if bgr is None or getattr(bgr, "size", 0) == 0:
        return -1
    return int(np.argmax(bgr[:, :, 0].mean(axis=0)))


def _guess_lan_ip() -> str:
    import socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "<host-ip>"

START = False
STOP = False
READY = False
RECORD_RUNNING = False
RECORD_TOGGLE = False


def on_press(key):
    global STOP, START, RECORD_TOGGLE
    if key == 'r':
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

    if args.ee == "dex3" and args.input_mode == "controller":
        parser.error("--ee dex3 does not support controller input mode.")

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
            left_hand_pos_array = Array('d', 75, lock=True)
            right_hand_pos_array = Array('d', 75, lock=True)
            dual_hand_data_lock = Lock()
            dual_hand_state_array = Array('d', 14, lock=False)
            dual_hand_action_array = Array('d', 14, lock=False)

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

        logger_mp.info("----------------------------------------------------------------")
        lan_ip = _guess_lan_ip()
        logger_mp.info(
            f"XR headset: open https://{lan_ip}:8012/?ws=wss://{lan_ip}:8012 "
            "then click Virtual Reality and allow hand tracking. "
            "Native Quest/Pico hands are NOT TeleVuer events.")
        logger_mp.info("Until cam_move>0, wrist poses stay CONST (no HAND_MOVE/CAMERA_MOVE).")
        logger_mp.info("Press [r] to start syncing the robot with your movements.")
        if args.record:
            logger_mp.info("Press [s] to START or SAVE recording (toggle cycle).")
        logger_mp.info("Press [q] to stop and exit the program.")
        READY = True
        teleop_bridge.set_fsm_id(FSM_IDLE)
        wait_dbg = {"last_log": 0.0, "video_cb": 0, "hand_move": 0, "cam_move": 0}
        while not START and not STOP:
            time.sleep(0.033)
            if camera_config['head_camera']['enable_zmq'] and xr_need_local_img:
                head_img = teleop_bridge.get_head_frame()
                if head_img.bgr is not None:
                    tv_wrapper.render_to_xr(head_img.bgr)
            now = time.time()
            if now - wait_dbg["last_log"] >= 1.0:
                dt = now - wait_dbg["last_log"] if wait_dbg["last_log"] else 1.0
                video = teleop_bridge.get_video_debug()
                hand_n = tv_wrapper.tvuer.hand_move_count
                cam_n = tv_wrapper.tvuer.cam_move_count
                idle_head = teleop_bridge.get_head_frame()
                bar = _bar_col(idle_head.bgr)
                hand_rate = int((hand_n - wait_dbg['hand_move']) / dt)
                cam_rate = int((cam_n - wait_dbg['cam_move']) / dt)
                xr_hint = "" if cam_rate > 0 else " NO_XR_SESSION"
                logger_mp.info(
                    f"[op idle 1Hz] hand_move={hand_rate}/s "
                    f"cam_move={cam_rate}/s "
                    f"video_cb_fps={int((video['video_cb_count'] - wait_dbg['video_cb']) / dt)} "
                    f"bar_col={bar} (press r to start tracking){xr_hint}"
                )
                wait_dbg["last_log"] = now
                wait_dbg["video_cb"] = video["video_cb_count"]
                wait_dbg["hand_move"] = hand_n
                wait_dbg["cam_move"] = cam_n

        logger_mp.info("start Tracking")
        teleop_bridge.set_fsm_id(FSM_TELEOP)

        head_img = None
        left_wrist_img = None
        right_wrist_img = None
        dbg = {
            "last_log": 0.0,
            "hand_move": 0,
            "cam_move": 0,
            "video_cb": 0,
            "obs_frame": 0,
        }

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
            if args.ee == "dex3" and args.input_mode == "hand":
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            with xr_motion_data_ready.get_lock():
                xr_motion_data_ready.value = tele_data.motion_data_ready

            vx = vy = vyaw = 0.0
            if args.input_mode == "controller":
                if tele_data.right_ctrl_aButton:
                    START = False
                    STOP = True
                vx = -tele_data.left_ctrl_thumbstickValue[1] * LOCO_SCALE
                vy = -tele_data.left_ctrl_thumbstickValue[0] * LOCO_SCALE
                vyaw = -tele_data.right_ctrl_thumbstickValue[0] * LOCO_SCALE

            current_lr_arm_q = teleop_bridge.get_current_dual_arm_q()
            current_lr_arm_dq = teleop_bridge.get_current_dual_arm_dq()
            sol_q, sol_tauff = arm_ik.solve_ik(
                tele_data.left_wrist_pose, tele_data.right_wrist_pose,
                current_lr_arm_q, current_lr_arm_dq)
            del sol_tauff
            fsm = FSM_TELEOP if START else FSM_IDLE
            teleop_bridge.send_targets(sol_q, vx=vx, vy=vy, vyaw=vyaw, fsm_id=fsm)

            now = time.time()
            if now - dbg["last_log"] >= 1.0:
                dt = now - dbg["last_log"] if dbg["last_log"] else 1.0
                raw_l = tv_wrapper.tvuer.left_arm_pose
                raw_r = tv_wrapper.tvuer.right_arm_pose
                wrist_l = _xyz(tele_data.left_wrist_pose)
                hand_l = tele_data.left_hand_pos
                hand_norm = float(np.linalg.norm(hand_l)) if hand_l is not None else 0.0
                video = teleop_bridge.get_video_debug()
                state = teleop_bridge.get_state_debug()
                hand_n = tv_wrapper.tvuer.hand_move_count
                cam_n = tv_wrapper.tvuer.cam_move_count
                dbg_head = head_img if (head_img is not None and head_img.bgr is not None) \
                    else teleop_bridge.get_head_frame()
                bar = _bar_col(dbg_head.bgr) if dbg_head is not None else -1
                if dbg_head is not None and getattr(dbg_head, "timestamp_us", 0):
                    ts_age_ms = (now * 1_000_000 - dbg_head.timestamp_us) / 1000.0
                elif video["last_wall"]:
                    ts_age_ms = (now - video["last_wall"]) * 1000.0
                else:
                    ts_age_ms = -1.0
                obs_age = state["obs_age_ms"]
                obs_age_s = "inf" if not np.isfinite(obs_age) else f"{obs_age:.0f}"
                raw_l_xyz = _xyz(raw_l)
                logger_mp.info(
                    f"[op 1Hz] xr_ready={int(bool(tele_data.motion_data_ready))} "
                    f"hand_move={int((hand_n - dbg['hand_move']) / dt)}/s "
                    f"cam_move={int((cam_n - dbg['cam_move']) / dt)}/s "
                    f"rawL_det={np.linalg.det(raw_l):.3f} "
                    f"rawL_xyz={raw_l_xyz[0]:.2f},{raw_l_xyz[1]:.2f},{raw_l_xyz[2]:.2f} "
                    f"validL={int(_mat_is_valid(raw_l))} validR={int(_mat_is_valid(raw_r))} "
                    f"wristL={wrist_l[0]:.2f},{wrist_l[1]:.2f},{wrist_l[2]:.2f} "
                    f"handL_norm={hand_norm:.3f} "
                    f"ik L_PITCH={float(sol_q[0]):.4f} R_PITCH={float(sol_q[7]):.4f} "
                    f"warm L_PITCH={float(current_lr_arm_q[0]):.4f} "
                    f"q_src={state['q_src']} obs_age_ms={obs_age_s} "
                    f"video_cb_fps={int((video['video_cb_count'] - dbg['video_cb']) / dt)} "
                    f"obs_frame_fps={int((video['obs_frame_count'] - dbg['obs_frame']) / dt)} "
                    f"bar_col={bar} ts_age_ms={ts_age_ms:.0f} "
                    f"loop_ms={(now - start_time) * 1000.0:.1f}"
                )
                dbg["last_log"] = now
                dbg["hand_move"] = hand_n
                dbg["cam_move"] = cam_n
                dbg["video_cb"] = video["video_cb_count"]
                dbg["obs_frame"] = video["obs_frame_count"]

            if args.record:
                READY = recorder.is_ready()
                if args.ee == "dex3" and args.input_mode == "hand":
                    with dual_hand_data_lock:
                        left_ee_state = dual_hand_state_array[:7]
                        right_ee_state = dual_hand_state_array[-7:]
                        left_hand_action = dual_hand_action_array[:7]
                        right_hand_action = dual_hand_action_array[-7:]
                else:
                    left_ee_state = []
                    right_ee_state = []
                    left_hand_action = []
                    right_hand_action = []
                    current_body_action = [vx, vy, vyaw] if args.input_mode == "controller" else []
                half = len(current_lr_arm_q) // 2
                left_arm_state, right_arm_state = current_lr_arm_q[:half], current_lr_arm_q[half:]
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
                        "body": {"qpos": [vx, vy, vyaw] if args.input_mode == "controller" else []},
                    }
                    recorder.add_item(colors=colors, depths=depths, states=states, actions=actions)

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
