from numpy import isin
from traitlets import observe_compat
import zmq
import torch
import scipy.spatial.transform as st
import numpy as np
from src.rdt.polymetis_robot_utils.interfaces.diffik import (
    DiffIKWrapper,
    GripperInterface,
)
from rdt.spacemouse.spacemouse_shared_memory import Spacemouse
from rdt.teleop.utils import scale_scripted_action
from rdt.polymetis_robot_utils.interfaces.reqrep import (
    Client,
    GripperAction,
    Reset,
    Action,
)
import time
from rdt.teleop.teleop_sm import ActionContainer, precise_wait
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.polymetis_robot_utils.interfaces.controller import (
    MultiSpacemouseControl,
    Chirality,
)


class D:
    frequency = 20
    max_pos_speed = 1
    max_rot_speed = 1.5
    sm_dpos_scalar = 1.8
    sm_drot_scalar = 4.0
    keyboard = KeyboardInterface()
    gripper_open = False
    record_latency_when_grasping = 15


def to_pose_mat(pose_):
    pose_mat = np.eye(4)
    pose_mat[:-1, -1] = pose_[:3]
    pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_[3:]).as_matrix()
    return pose_mat


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-p", "--port_vis", type=int, default=6000)
    # parser.add_argument("--frequency", type=int, default=10)  # 30
    # parser.add_argument("--command_latency", type=float, default=0.01)
    # parser.add_argument("--deadzone", type=float, default=0.05)
    # parser.add_argument("--max-pos-speed", type=float, default=2)
    # parser.add_argument("--max-rot-speed", type=float, default=3)
    # parser.add_argument("--resize-images", action="store_true")
    # parser.add_argument("--use_lcm", action="store_true")
    # parser.add_argument("--save_dir", required=True)
    # parser.add_argument("--task", type=str, required=True)
    # parser.add_argument("--n-demos", type=int, default=1)
    # args = parser.parse_args()

    platform_client = Client()
    from multiprocessing.managers import SharedMemoryManager

    msc = MultiSpacemouseControl([Chirality.RIGHT])

    platform_client.reset()

    time.sleep(1)
    while True:
        time.sleep(0.01)
        obs = platform_client.get_obs()
        action = msc.get_action(Chirality.RIGHT, obs)

        platform_client.act(
            Action(
                next_pose_mat=action.next_pose_mat,
                gripper_action=GripperAction.OPEN,
                action_taken=action.action_taken,
            )
        )
