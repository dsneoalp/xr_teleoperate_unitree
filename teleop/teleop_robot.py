"""Portal robot control loop: apply joint targets via SDK, stream state.

No TeleVuer / IK. Pair with teleop_operator.py in the same LiveKit room.

    python teleop/teleop_robot.py --ee dex3 --arm G1_29
    python teleop/teleop_robot.py --ee dex3 --arm G1_29 --motion
"""
import time
import argparse
import threading
import logging_mp
logging_mp.basicConfig(level=logging_mp.INFO)
logger_mp = logging_mp.getLogger(__name__)

import os
import sys
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from teleop.robot_control.portal_robot import PortalRobotTransport
from teleop.robot_control.portal_mapping import UnpackedAction

FSM_IDLE = 0
FSM_TELEOP = 1
FSM_HOME = 2
ACTION_TIMEOUT = 0.2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frequency', type=float, default=30.0, help='state publish rate')
    parser.add_argument('--arm', type=str, choices=['G1_29'], default='G1_29')
    parser.add_argument('--ee', type=str, choices=['dex3'], default=None)
    parser.add_argument('--motion', action='store_true')
    parser.add_argument('--network-interface', type=str, default=None)
    parser.add_argument('--sim', action='store_true')
    parser.add_argument('--portal-yaml', type=str, default=os.path.join(current_dir, 'portal.yaml'))
    parser.add_argument('--portal-mapping', type=str, default=os.path.join(current_dir, 'portal_mapping.yaml'))
    parser.add_argument('--env-file', type=str, default=os.path.join(current_dir, '.env'))
    parser.add_argument('--livekit-url', type=str, default=None)
    parser.add_argument('--livekit-room', type=str, default=None)
    parser.add_argument('--portal-identity', type=str, default='xr-robot')
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ChannelFactoryInitialize(1 if args.sim else 0, networkInterface=args.network_interface)

    from teleop.robot_control.robot_arm import G1_29_ArmController
    from teleop.utils.motion_switcher import MotionSwitcher, LocoClientWrapper

    loco_wrapper = None
    if args.motion:
        loco_wrapper = LocoClientWrapper()
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
        url=args.livekit_url)
    portal.wait_until_connected()

    lock = threading.Lock()
    latest = {'action': None, 'wall': 0.0}
    applied_fsm = FSM_IDLE

    def on_action(action: UnpackedAction):
        with lock:
            latest['action'] = action
            latest['wall'] = time.time()

    portal.on_unpacked_action(on_action)
    logger_mp.info("robot loop running; waiting for operator actions")

    try:
        while True:
            start = time.time()
            with lock:
                action = latest['action']
                age = start - latest['wall'] if latest['wall'] else 1e9

            fsm = action.fsm_id if action is not None else FSM_IDLE
            fresh = action is not None and age < ACTION_TIMEOUT

            if fsm == FSM_HOME:
                if applied_fsm != FSM_HOME:
                    arm_ctrl.ctrl_dual_arm_go_home()
                    applied_fsm = FSM_HOME
            elif fsm == FSM_TELEOP and fresh:
                tauff = np.zeros_like(action.arm_q)
                arm_ctrl.ctrl_dual_arm(action.arm_q, tauff)
                if hand_ctrl is not None and action.hand_q.size:
                    half = action.hand_q.size // 2
                    hand_ctrl.ctrl_dual_hand(action.hand_q[:half], action.hand_q[half:])
                if loco_wrapper is not None:
                    loco_wrapper.Move(action.vx, action.vy, action.vyaw)
                applied_fsm = FSM_TELEOP
            else:
                applied_fsm = FSM_IDLE

            motor_q = arm_ctrl.get_current_motor_q()
            hand_q = hand_ctrl.get_current_dual_hand_q() if hand_ctrl is not None else None
            portal.send_state(motor_q=motor_q, hand_q=hand_q, fsm_id=fsm)

            sleep = max(0.0, (1.0 / args.frequency) - (time.time() - start))
            time.sleep(sleep)
    except KeyboardInterrupt:
        logger_mp.info("KeyboardInterrupt, exiting ...")
    finally:
        try:
            arm_ctrl.ctrl_dual_arm_go_home()
        except Exception as e:
            logger_mp.error(f"go_home failed: {e}")
        try:
            portal.close()
        except Exception as e:
            logger_mp.error(f"portal close failed: {e}")
        logger_mp.info("robot exited.")
