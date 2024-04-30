import os, os.path as osp
from pathlib import Path
import sys
import pickle
import time
import threading
from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager

import torch
import numpy as np
import scipy.spatial.transform as st
import meshcat

from rdt.spacemouse.spacemouse_shared_memory import Spacemouse

from polymetis import GripperInterface, RobotInterface

from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.common.demo_util import CollectEnum
from datetime import datetime


import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port_vis", type=int, default=6000)
parser.add_argument("--frequency", type=int, default=10)  # 30
parser.add_argument("--command_latency", type=float, default=0.01)
parser.add_argument("--deadzone", type=float, default=0.05)
parser.add_argument("--max_pos_speed", type=float, default=0.3)
parser.add_argument("--max_rot_speed", type=float, default=0.7)
parser.add_argument("--save_dir", required=True)
parser.add_argument("--use_lcm", action="store_true")

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

    franka_ip = "173.16.0.1"
    # robot = RobotInterface(ip_address=franka_ip)
    robot = DiffIKWrapper(ip_address=franka_ip)
    # robot.set_pos_rot_scalars(pos=np.array([1.5]*3))
    # robot.set_pos_rot_scalars(rot=np.array([1.5, 1.5, 4.0]))

    sm_dpos_scalar = np.array([1.5] * 3)
    sm_drot_scalar = np.array([1.5] * 3)
    # sm_drot_scalar = np.array([1.5, 1.5, 4.0])

    # Kq_new = torch.Tensor([40., 30., 50., 25., 35., 25., 10.])
    # Kqd_new = torch.Tensor([4., 6., 5., 5., 3., 2., 1.])
    pd_ratio = torch.Tensor([10.0, 5.0, 10.0, 5.0, 11.67, 12.5, 10.0])
    Kq_new = torch.Tensor([150.0, 120.0, 160.0, 100.0, 110.0, 100.0, 40.0])

    # Kq_new = torch.Tensor([320., 240., 350., 200., 200., 260.,  70.])  # pretty good with 30hz, more jerky with 10hz
    # Kq_new = torch.Tensor([350., 250., 350., 210., 220., 260.,  70.])
    # Kqd_new = Kq_new / pd_ratio

    # Kq_new = torch.Tensor([400.0, 400.0, 400.0, 400.0, 250.0, 150.0, 50.0])
    # Kqd_new = torch.Tensor([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])

    Kqd_new = torch.Tensor([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 8.0])

    # Kx_new = torch.Tensor([750., 750., 750.,  15.,  15.,  15.])
    # Kxd_new = torch.Tensor([37., 37., 37.,  2.,  2.,  2.])

    # robot.start_joint_impedance()
    robot.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new, adaptive=True)
    # robot.start_resolved_rate_control()

    gripper = GripperInterface(ip_address=franka_ip)
    gripper.goto(0.08, 0.05, 0.1, blocking=False)
    init_joint_positions = robot.home_pose.numpy()
    # init_joint_positions = neutral_joint_positions
    gripper_open = True
    grasp_flag = -1

    zmq_url = f"tcp://127.0.0.1:{args.port_vis}"
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis["scene"].delete()

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

    # Setup camera streams (via either LCM or pyrealsense)

    if args.use_lcm:
        from rdt.image.factory import get_realsense_rgbd_subscribers
        import lcm

        lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")
        rs_cfg = get_default_multi_realsense_cfg()
        img_subscribers = get_realsense_rgbd_subscribers(lc, rs_cfg)

        def lc_th(lc):
            while True:
                lc.handle_timeout(1)
                time.sleep(0.001)

        lc_thread = threading.Thread(target=lc_th, args=(lc,))
        lc_thread.daemon = True
        lc_thread.start()

        def get_rgbd_lcm(subs):
            rgbd_list = []
            for name, img_sub, info_sub in subs:
                rgb_image, depth_image = img_sub.get_rgb_and_depth(block=True)
                if rgb_image is None or depth_image is None:
                    return

                img_dict = dict(rgb=rgb_image, depth=depth_image)
                rgbd_list.append(img_dict)

            return rgbd_list

    else:
        from rdt.image.factory import enable_realsense_devices
        import pyrealsense2 as rs

        rs_cfg = get_default_multi_realsense_cfg()
        resolution_width = rs_cfg.WIDTH  # pixels
        resolution_height = rs_cfg.HEIGHT  # pixels
        frame_rate = rs_cfg.FRAME_RATE  # fps

        ctx = rs.context()  # Create librealsense context for managing devices
        serials = rs_cfg.SERIAL_NUMBERS

        print(f"Enabling devices with serial numbers: {serials}")
        pipelines = enable_realsense_devices(
            serials, ctx, resolution_width, resolution_height, frame_rate
        )

        def get_rgbd_rs(pipelines):
            rgbd_list = []

            align_to = rs.stream.color
            align = rs.align(align_to)

            for device, pipe in pipelines:
                try:
                    # Get frameset of color and depth
                    frames = pipe.wait_for_frames(2000)  # 100
                except RuntimeError as e:
                    print(f"Couldn't get frame for device: {device}")
                    # continue
                    raise

                # Align the depth frame to color frame
                aligned_frames = align.process(frames)

                # Get aligned frames
                aligned_depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()

                # Validate that both frames are valid
                if not aligned_depth_frame or not color_frame:
                    continue

                depth_image = np.asanyarray(aligned_depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                img_dict = dict(rgb=color_image, depth=depth_image)
                rgbd_list.append(img_dict)

            return rgbd_list

    # Setup data saving
    demo_save_dir = Path(args.save_dir) / str(datetime.now())
    demo_save_dir.mkdir(exist_ok=True, parents=True)

    episode_dict = {}
    episode_dict["robot_state"] = []
    episode_dict["image_front"] = []
    episode_dict["image_wrist"] = []
    episode_dict["actions"] = []

    timestep = 0

    # Setup other interfaces
    keyboard = KeyboardInterface()

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone) as sm:
            t_start = time.monotonic()
            iter_idx = 0
            stop = False

            last_grip_step = 0
            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                # t_command_target = t_cycle_end + dt
                precise_wait(t_sample)

                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (args.max_pos_speed / frequency)
                # drot_xyz = sm_state[3:] * (args.max_rot_speed / frequency)
                # drot = st.Rotation.from_euler("xyz", drot_xyz)

                drot_xyz = (
                    sm_state[3:] * (args.max_rot_speed / frequency) * sm_dpos_scalar
                )
                drot_rotvec = st.Rotation.from_euler("xyz", drot_xyz).as_rotvec()
                drot_rotvec *= sm_drot_scalar
                drot = st.Rotation.from_rotvec(drot_rotvec)

                keyboard_action, collect_enum = keyboard.get_action()
                if collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                    break

                # get observations
                rgbd_list = (
                    get_rgbd_lcm(img_subscribers)
                    if args.use_lcm
                    else get_rgbd_rs(pipelines)
                )
                current_ee_pose = torch.cat(robot.get_ee_pose(), dim=-1)
                current_joint_positions = robot.get_joint_positions()
                robot_state_dict = dict(
                    ee_pose=current_ee_pose.numpy(),
                    joint_positions=current_joint_positions.numpy(),
                )

                # if False:
                last_grip_step += 1
                if (
                    sm.is_button_pressed(0)
                    or sm.is_button_pressed(1)
                    and last_grip_step > 10
                ):
                    # gripper.gripper_close() if gripper_open else gripper.gripper_open()
                    (
                        gripper.grasp(0.07, 70, blocking=False)
                        if gripper_open
                        else gripper.goto(0.08, 0.05, 0.1, blocking=False)
                    )  # goto for opening
                    gripper_open = not gripper_open
                    last_grip_step = 0
                    grasp_flag = -1 * grasp_flag

                if not (np.allclose(keyboard_action[3:6], 0.0)):
                    drot = st.Rotation.from_quat(keyboard_action[3:7])

                new_target_pose = target_pose.copy()
                new_target_pose[:3] += dpos
                new_target_pose[3:] = (
                    drot * st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()
                new_target_pose_mat = to_pose_mat(new_target_pose)

                # directly send the delta commands to our diffik controller
                des_ee_vel = torch.Tensor([*dpos, *drot_xyz])

                # robot.update_desired_ee_velocities(des_ee_vel)
                robot.update_desired_ee_pose(
                    new_target_pose_mat, dt=dt
                )  # , scalar=0.5)

                # log the data
                action = np.zeros(8)
                action[:3] = new_target_pose_mat[:-1, -1]
                action[3:7] = st.Rotation.from_matrix(
                    new_target_pose_mat[:-1, :-1]
                ).as_quat()
                action[-1] = grasp_flag
                episode_dict["robot_state"].append(robot_state_dict)
                episode_dict["actions"].append(action)

                # TODO - ensure we are taking care of the right order of cameras here... 
                img_keys = ["image_wrist", "image_front"]
                for img_key, img in zip(img_keys, rgbd_list):
                    episode_dict[img_key].append(img) 

                # target_pose = new_target_pose
                target_pose = polypose2target(robot.get_ee_pose())

                # Draw the current target pose (in meshcat)
                mc_util.meshcat_frame_show(
                    mc_vis, f"scene/target_pose", new_target_pose_mat
                )
                mc_util.meshcat_frame_show(
                    mc_vis,
                    f"scene/current_pose",
                    poly_util.polypose2mat(robot.get_ee_pose()),
                )
                # with suppress_stdout():
                # remove_handles(pose_handles)
                # pose_handles = draw_pose(to_pb_pose(target_pose))

                precise_wait(t_cycle_end)
                iter_idx += 1

    if collect_enum == CollectEnum.SUCCESS:
        # save the data
        obs_action_pkl_fname = demo_save_dir / "episode_data.pkl"
        with open(obs_action_pkl_fname, "wb") as f:
            pickle.dump(episode_dict, f)


if __name__ == "__main__":
    main()
