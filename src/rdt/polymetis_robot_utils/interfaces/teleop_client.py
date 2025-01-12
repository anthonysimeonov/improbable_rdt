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


def get_action(sm, target_pose):
    # get teleop command
    sm_state = sm.get_motion_state_transformed()

    # scale pos command
    dpos = sm_state[:3] * (D.max_pos_speed / D.frequency) * D.sm_dpos_scalar

    # convert and scale rot command
    drot_xyz = sm_state[3:]
    drot_rotvec = st.Rotation.from_euler("xyz", drot_xyz).as_rotvec()
    drot_rotvec *= (D.max_rot_speed / D.frequency) * D.sm_drot_scalar
    drot = st.Rotation.from_rotvec(drot_rotvec)

    # get keyboard actions/flags
    keyboard_action, collect_enum = D.keyboard.get_action()

    # check if action is taken
    if np.allclose(dpos, 0.0) and np.allclose(drot_xyz, 0.0):
        action_taken = False
    else:
        action_taken = True

    # manage grasping
    # D.steps_since_grasp += 1
    # if D.steps_since_grasp < D.record_latency_when_grasping:
    #     action_taken = True

    # D.last_grip_step += 1
    # is_gripper_open = self.gripper_open
    # if sm.is_button_pressed(0) or sm.is_button_pressed(1) and self.last_grip_step > 10:
    #     toggle_gripper = True

    #     self.gripper_open = not self.gripper_open
    #     self.last_grip_step = 0
    #     self.grasp_flag = -1 * self.grasp_flag
    #     self.steps_since_grasp = 0
    # else:
    #     toggle_gripper = False
    toggle_gripper = False

    # Make a delta action of xyz + quat_xyzw that we can scale before sending to robot
    delta_action = np.concatenate([dpos, drot.as_quat(), np.array([False])])

    # overwrite action from keyboard action (for screwing)
    kb_taken = False
    if not (np.allclose(keyboard_action[3:6], 0.0)):
        delta_action[3:7] = keyboard_action[3:7]
        kb_taken = True
        action_taken = True

    pos_bounds_m = 0.025 * 2
    ori_bounds_deg = 20

    delta_action = (
        scale_scripted_action(
            torch.from_numpy(delta_action).unsqueeze(0),
            pos_bounds_m=pos_bounds_m,
            ori_bounds_deg=ori_bounds_deg,
        )
        .squeeze()
        .numpy()
    )

    # write out action
    new_target_pose = target_pose.copy()
    dpos, drot = delta_action[:3], st.Rotation.from_quat(delta_action[3:7])
    new_target_pose[:3] += dpos
    # new_target_pose[3:] = (
    #     drot * st.Rotation.from_rotvec(self.target_pose[3:])
    # ).as_rotvec()
    if kb_taken:
        # right multiply (more intuitive for screwing)
        new_target_pose[3:] = (
            st.Rotation.from_rotvec(target_pose[3:]) * drot
        ).as_rotvec()
    else:
        # left multiply (more intuitive for spacemouse)
        new_target_pose[3:] = (
            drot * st.Rotation.from_rotvec(target_pose[3:])
        ).as_rotvec()
    new_target_pose_mat = to_pose_mat(new_target_pose)
    current_pose_mat = to_pose_mat(target_pose)

    # TODO: consolidate to action
    action_struct = ActionContainer(
        current_pose_mat=current_pose_mat,
        next_pose_mat=new_target_pose_mat,
        grasp_flag=False,
        action_taken=action_taken,
        collect_enum=collect_enum,
        is_gripper_open=False,
        toggle_gripper=toggle_gripper,
    )
    return action_struct


if __name__ == "__main__":

    c = Client()
    from multiprocessing.managers import SharedMemoryManager

    smm = SharedMemoryManager()
    smm.start()

    sm = Spacemouse(smm)
    sm.start()

    c.reset()

    time.sleep(1)
    while True:
        time.sleep(0.01)
        obs = c.get_obs()
        action = get_action(sm, obs.xyz_rotvec)

        c.act(
            Action(
                next_pose_mat=action.next_pose_mat,
                gripper_action=GripperAction.OPEN,
                action_taken=action.action_taken,
            )
        )
