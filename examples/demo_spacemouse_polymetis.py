# to run this, must also run pip install params-proto

import os
import sys
import time
from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager

import torch
import numpy as np
import scipy.spatial.transform as st
import meshcat

from rdt.spacemouse.spacemouse_shared_memory import Spacemouse

from polymetis import GripperInterface, RobotInterface
# from rdt.polymetis_robot_utils.plan_exec_util import PlanningHelper
# from rdt.polymetis_robot_utils.traj_util import PolymetisTrajectoryUtil
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-p', '--port_vis', type=int, default=6000)  # 30
parser.add_argument('--frequency', type=int, default=10)  # 30
parser.add_argument('--command_latency', type=float, default=0.01)
parser.add_argument('--deadzone', type=float, default=0.05)
parser.add_argument('--max_pos_speed', type=float, default=0.3)
parser.add_argument('--max_rot_speed', type=float, default=0.7)

args = parser.parse_args()

poly_util = PolymetisHelper()


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


def main():
    frequency = args.frequency
    dt = 1 / frequency
    command_latency = args.command_latency

    class SimpleGripper:
        def __init__(self, gripper):
            self.gripper = gripper
            self.gripper_speed, self.gripper_force = 0.5, 10.0

        def gripper_close(self, speed=None, force=None):
            if speed is None:
                speed = self.gripper_speed
            if force is None:
                force = self.gripper_force

            self.gripper.goto(self.gripper_close_pos, speed, force)

        def gripper_grasp(self, speed=None, force=None):
            if speed is None:
                speed = self.gripper_speed
            if force is None:
                force = self.gripper_force

            self.gripper.grasp(speed, force)

    franka_ip = "173.16.0.1" 
    # robot = RobotInterface(ip_address=franka_ip)
    robot = DiffIKWrapper(ip_address=franka_ip)
    # robot.set_pos_rot_scalars(pos=np.array([1.5]*3))
    robot.set_pos_rot_scalars(rot=np.array([1.5, 1.5, 4.0]))

    # Kq_new = torch.Tensor([40., 30., 50., 25., 35., 25., 10.])
    # Kqd_new = torch.Tensor([4., 6., 5., 5., 3., 2., 1.])
    pd_ratio = torch.Tensor([10.,  5., 10.,  5., 11.67, 12.5, 10.])
    Kq_new = torch.Tensor([320., 240., 350., 200., 200., 260.,  70.])  # pretty good with 30hz, more jerky with 10hz
    Kq_new = torch.Tensor([350., 250., 350., 210., 220., 260.,  70.])  
    Kqd_new = Kq_new / pd_ratio
    # Kq_new = torch.Tensor([400.0, 400.0, 400.0, 400.0, 250.0, 150.0, 50.0])
    # Kqd_new = torch.Tensor([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])

    # Kx_new = torch.Tensor([750., 750., 750.,  15.,  15.,  15.]) 
    # Kxd_new = torch.Tensor([37., 37., 37.,  2.,  2.,  2.])

    robot.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new, adaptive=True)
    # robot.start_resolved_rate_control()
    gripper = GripperInterface(ip_address=franka_ip)
    simple_gripper = SimpleGripper(gripper)
    init_joint_positions = robot.home_pose.numpy()
    # init_joint_positions = neutral_joint_positions
    gripper_open = True

    zmq_url=f'tcp://127.0.0.1:{args.port_vis}'
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis['scene'].delete()

    translation, quat_xyzw = robot.get_ee_pose()
    # pose_mat = poly_util.polypose2mat(robot.get_ee_pose())
    pose_mat = poly_util.polypose2mat((translation, quat_xyzw))
    rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
    target_pose = np.array([*translation.numpy(), *rotvec])

    def polypose2target(poly_pose):
        translation, quat_xyzw = poly_pose[0], poly_pose[1]
        rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
        target_pose = np.array([*translation.numpy(), *rotvec])
        return target_pose

    def to_pose_mat(pose_):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = pose_[:3]
        pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_[3:]).as_matrix() 
        return pose_mat

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone) as sm:
            t_start = time.monotonic()
            iter_idx = 0
            stop = False

            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                # t_command_target = t_cycle_end + dt
                precise_wait(t_sample)

                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (args.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (args.max_rot_speed / frequency)
                drot = st.Rotation.from_euler("xyz", drot_xyz)
                
                if False:
                    if sm.is_button_pressed(0) or sm.is_button_pressed(1):
                        gripper.gripper_close() if gripper_open else gripper.gripper_open()
                        gripper_open = not gripper_open

                new_target_pose = target_pose.copy()
                new_target_pose[:3] += dpos
                new_target_pose[3:] = (
                    drot * st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()
                new_target_pose_mat = to_pose_mat(new_target_pose)

                # directly send the delta commands to our diffik controller
                des_ee_vel = torch.Tensor([*dpos, *drot_xyz])

                # robot.update_desired_ee_velocities(des_ee_vel)
                robot.update_desired_ee_pose(new_target_pose_mat, dt=dt) #, scalar=0.5)
                # target_pose = new_target_pose
                target_pose = polypose2target(robot.get_ee_pose())
                
                # Draw the current target pose (in meshcat)
                mc_util.meshcat_frame_show(mc_vis, f'scene/target_pose', new_target_pose_mat)
                mc_util.meshcat_frame_show(mc_vis, f'scene/current_pose', poly_util.polypose2mat(robot.get_ee_pose()))
                # with suppress_stdout():
                    # remove_handles(pose_handles)
                    # pose_handles = draw_pose(to_pb_pose(target_pose))

                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == "__main__":
    main()
