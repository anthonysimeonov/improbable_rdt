from pathlib import Path
import pickle
import time
from multiprocessing.managers import SharedMemoryManager
import torch
import numpy as np
from datetime import datetime
import meshcat
import scipy.spatial.transform as st
import pyrealsense2 as rs

from polymetis import GripperInterface

from rdt.spacemouse.spacemouse_shared_memory import Spacemouse

from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.interfaces.diffik import DiffIKWrapper
from rdt.common import mc_util
from rdt.common.keyboard_interface import KeyboardInterface
from rdt.common.demo_util import CollectEnum
from rdt.image.factory import enable_single_realsense
from rdt.teleop.utils import scale_scripted_action
from rdt.robot.transforms import convert_tip2wrist, convert_wrist2tip

# from src.real.serials import CAMERA1_SERIAL, CAMERA2_SERIAL
CAMERA1_SERIAL = "242622071805"
CAMERA2_SERIAL = "242522072326"

from ipdb import set_trace as bp

import argparse


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


def polypose2target(poly_pose):
    """
    Converts a pose given as a tuple of translation and quaternion to a target pose.

    Args:
        poly_pose (tuple): A tuple containing the translation and quaternion of the polygonal pose.

    Returns:
        numpy.ndarray: The target pose, represented as a numpy array with the translation and rotation vector.

    """
    translation, quat_xyzw = poly_pose[0], poly_pose[1]
    rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
    target_pose = np.array([*translation.numpy(), *rotvec])
    return target_pose


def to_pose_mat(pose_):
    pose_mat = np.eye(4)
    pose_mat[:-1, -1] = pose_[:3]
    pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_[3:]).as_matrix()
    return pose_mat


def wrist_target_to_tip(wrist_target_pose_rv):
    wrist_target_pose_mat = to_pose_mat(wrist_target_pose_rv)
    tip_target_pose_mat = convert_wrist2tip(wrist_target_pose_mat)
    tip_target_pos = tip_target_pose_mat[:-1, -1]
    tip_target_rv = st.Rotation.from_matrix(tip_target_pose_mat[:-1, :-1]).as_rotvec()
    tip_target_pose_rv = np.array([*tip_target_pos, *tip_target_rv])
    return tip_target_pose_rv


def execute_gripper_action(
    gripper: GripperInterface, toggle_gripper: bool, gripper_open: bool
):
    if not toggle_gripper:
        return
    if gripper_open:
        gripper.grasp(0.07, 70, blocking=False)
    else:
        gripper.goto(0.065, 0.05, 0.1, blocking=False)


class ActionContainer:
    def __init__(
        self,
        current_pose_mat: np.ndarray,
        next_pose_mat: np.ndarray,
        grasp_flag: int,
        action_taken: bool,
        collect_enum: CollectEnum,
        is_gripper_open: bool,
        toggle_gripper: bool,
    ):
        self.current_pose_mat = current_pose_mat
        self.next_pose_mat = next_pose_mat
        self.grasp_flag = grasp_flag
        self.action_taken = action_taken
        self.collect_enum = collect_enum
        self.is_gripper_open = is_gripper_open
        self.toggle_gripper = toggle_gripper


# Setup observation and action helpers
class ObsActHelper:
    def __init__(
        self,
        sm: Spacemouse,
        keyboard: KeyboardInterface,
        robot: DiffIKWrapper,
        gripper: GripperInterface,
        image1_pipeline: rs.pipeline,
        image2_pipeline: rs.pipeline,
    ):

        self.sm = sm

        self.keyboard = keyboard

        self.robot = robot
        self.gripper = gripper

        self.image1_pipeline = image1_pipeline
        self.image2_pipeline = image2_pipeline

        self._setup()

    def set_constants(
        self,
        max_pos_speed: float,
        max_rot_speed: float,
        sm_dpos_scalar: float,
        sm_drot_scalar: float,
        frequency: float,
    ):
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.sm_dpos_scalar = sm_dpos_scalar
        self.sm_drot_scalar = sm_drot_scalar
        self.frequency = frequency

    def _setup(self):
        self.grasp_flag = -1
        self.gripper_open = True
        self.last_grip_step = 0
        self.steps_since_grasp = 0
        self.record_latency_when_grasping = 15

    @staticmethod
    def to_isaac_dpose_from_abs(current_pose_mat, goal_pose_mat, grasp_flag, rm=True):
        """
        Convert from absolute current and desired pose to delta pose

        Args:
            rm (bool): 'rm' stands for 'right multiplication' - If True, assume commands send as right multiply (local rotations)
        """
        if rm:
            delta_rot_mat = (
                np.linalg.inv(current_pose_mat[:-1, :-1]) @ goal_pose_mat[:-1, :-1]
            )
        else:
            delta_rot_mat = goal_pose_mat[:-1:-1] @ np.linalg.inv(
                current_pose_mat[:-1, :-1]
            )

        target_translation = goal_pose_mat[:-1, -1] - current_pose_mat[:-1, -1]
        target_quat_xyzw = st.Rotation.from_matrix(delta_rot_mat).as_quat()

        target_dpose = np.concatenate(
            (target_translation, target_quat_xyzw, np.array([grasp_flag])), axis=-1
        )

        return target_dpose

    @staticmethod
    def to_pose_mat(pose_):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = pose_[:3]
        pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_[3:]).as_matrix()
        return pose_mat

    def set_target_pose(self, target_pose: np.ndarray):
        # [x, y, z, dx, dy, dz] (rotvec!)
        self.target_pose = target_pose

    def get_action(self):
        # get teleop command
        sm_state = self.sm.get_motion_state_transformed()

        # scale pos command
        dpos = (
            sm_state[:3] * (self.max_pos_speed / self.frequency) * self.sm_dpos_scalar
        )

        # convert and scale rot command
        drot_xyz = sm_state[3:] * (self.max_rot_speed / self.frequency)
        drot_rotvec = st.Rotation.from_euler("xyz", drot_xyz).as_rotvec()
        drot_rotvec *= self.sm_drot_scalar
        drot = st.Rotation.from_rotvec(drot_rotvec)

        # get keyboard actions/flags
        keyboard_action, collect_enum = self.keyboard.get_action()

        # check if action is taken
        if np.allclose(dpos, 0.0) and np.allclose(drot_xyz, 0.0):
            action_taken = False
        else:
            action_taken = True

        # manage grasping
        self.steps_since_grasp += 1
        if self.steps_since_grasp < self.record_latency_when_grasping:
            action_taken = True

        self.last_grip_step += 1
        is_gripper_open = self.gripper_open
        if (
            self.sm.is_button_pressed(0)
            or self.sm.is_button_pressed(1)
            and self.last_grip_step > 10
        ):
            toggle_gripper = True

            self.gripper_open = not self.gripper_open
            self.last_grip_step = 0
            self.grasp_flag = -1 * self.grasp_flag
            self.steps_since_grasp = 0
        else:
            toggle_gripper = False

        # Make a delta action of xyz + quat_xyzw that we can scale before sending to robot
        delta_action = np.concatenate(
            [dpos, drot.as_quat(), np.array([self.grasp_flag])]
        )

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
        new_target_pose = self.target_pose.copy()
        dpos, drot = delta_action[:3], st.Rotation.from_quat(delta_action[3:7])
        new_target_pose[:3] += dpos
        # new_target_pose[3:] = (
        #     drot * st.Rotation.from_rotvec(self.target_pose[3:])
        # ).as_rotvec()
        if kb_taken:
            # right multiply (more intuitive for screwing)
            new_target_pose[3:] = (
                st.Rotation.from_rotvec(self.target_pose[3:]) * drot
            ).as_rotvec()
        else:
            # left multiply (more intuitive for spacemouse)
            new_target_pose[3:] = (
                drot * st.Rotation.from_rotvec(self.target_pose[3:])
            ).as_rotvec()
        new_target_pose_mat = self.to_pose_mat(new_target_pose)
        current_pose_mat = self.to_pose_mat(self.target_pose)

        action_struct = ActionContainer(
            current_pose_mat=current_pose_mat,
            next_pose_mat=new_target_pose_mat,
            grasp_flag=self.grasp_flag,
            action_taken=action_taken,
            collect_enum=collect_enum,
            is_gripper_open=is_gripper_open,
            toggle_gripper=toggle_gripper,
        )
        return action_struct

    def get_rgbd_rs(self, pipe) -> dict:
        align_to = rs.stream.color
        align = rs.align(align_to)

        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)  # 100
            # frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
            print(f"Runtime error: {e}")
            print(
                f"Couldn't get frame for device: {pipe.get_active_profile().get_device()}"
            )
            # continue
            raise

        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # the .copy() here is super important!
        img_dict = dict(rgb=color_image.copy(), depth=depth_image.copy())

        return img_dict

    def get_observation(self):
        # get the rgb images
        image1_rgbd = self.get_rgbd_rs(self.image1_pipeline)
        image2_rgbd = self.get_rgbd_rs(self.image2_pipeline)

        # get the robot state
        current_ee_wrist_pose_mat = poly_util.polypose2mat(self.robot.get_ee_pose())
        # convert to tip
        current_ee_tip_pose_mat = convert_wrist2tip(current_ee_wrist_pose_mat)
        current_ee_tip_pose = poly_util.mat2polypose(current_ee_tip_pose_mat)
        current_ee_pose = torch.cat(current_ee_tip_pose, dim=-1)

        current_joint_positions = self.robot.get_joint_positions()
        jacobian = self.robot.robot_model.compute_jacobian(current_joint_positions)
        ee_spatial_velocity = jacobian @ self.robot.get_joint_velocities()

        robot_state_dict = dict(
            ee_pos=current_ee_pose[:3].numpy(),
            ee_quat=current_ee_pose[3:7].numpy(),
            ee_pos_vel=ee_spatial_velocity[:3].numpy(),
            ee_ori_vel=ee_spatial_velocity[3:6].numpy(),
            gripper_width=self.gripper.get_state().width,
            joint_positions=current_joint_positions.numpy(),
        )

        # pack these
        obs = dict(
            color_image1=image1_rgbd["rgb"],
            color_image2=image2_rgbd["rgb"],
            depth_image1=image1_rgbd["depth"],
            depth_image2=image2_rgbd["depth"],
            robot_state=robot_state_dict,
        )

        return obs

    def posemat2action(self, target_pose_mat: np.ndarray, grasp_flag: int):
        action = np.zeros(8)
        action[:3] = target_pose_mat[:-1, -1]
        action[3:7] = st.Rotation.from_matrix(target_pose_mat[:-1, :-1]).as_quat()
        action[-1] = grasp_flag

        return action


def main():

    # === One-time setup ===
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port_vis", type=int, default=6000)
    parser.add_argument("--frequency", type=int, default=10)  # 30
    parser.add_argument("--command_latency", type=float, default=0.01)
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("--max_pos_speed", type=float, default=0.3)
    parser.add_argument("--max_rot_speed", type=float, default=0.7)
    parser.add_argument("--use_lcm", action="store_true")
    parser.add_argument("--save_dir", required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--n-demos", type=int, default=1)

    args = parser.parse_args()

    # some main args
    frequency = args.frequency
    dt = 1 / frequency
    command_latency = args.command_latency

    n_successes = 0

    # setup robot
    franka_ip = "173.16.0.1"
    robot_home = torch.Tensor(
        # [-0.0931, 0.0382, 0.1488, -2.3811, -0.0090, 2.4947, 0.1204]
        [-0.253, -0.198, 0.026, -2.388, 0.327, 2.407, 1.473]
    )
    Kq = torch.Tensor([150.0, 120.0, 160.0, 100.0, 110.0, 100.0, 40.0])
    Kqd = torch.Tensor([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 8.0])

    robot = DiffIKWrapper(
        ip_address=franka_ip,
        robot_home=robot_home,
        Kq=Kq,
        Kqd=Kqd,
    )
    gripper = GripperInterface(ip_address=franka_ip)

    # manual home
    gripper.goto(0.065, 0.05, 0.1, blocking=False)
    robot.reset()

    sm_dpos_scalar = np.array([1.8] * 3)
    sm_drot_scalar = np.array([4.0] * 3)

    # setup visuals
    zmq_url = f"tcp://127.0.0.1:{args.port_vis}"
    # mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    # mc_vis["scene"].delete()

    # Setup camera streams
    rs_cfg = get_default_multi_realsense_cfg()
    resolution_width = rs_cfg.WIDTH  # pixels
    resolution_height = rs_cfg.HEIGHT  # pixels
    frame_rate = rs_cfg.FRAME_RATE  # fps

    ctx = rs.context()  # Create librealsense context for managing devices

    print(
        f"Enabling devices with serial numbers: {CAMERA1_SERIAL} (front) and {CAMERA2_SERIAL} (wrist)"
    )
    image1_pipeline = enable_single_realsense(
        CAMERA1_SERIAL, ctx, resolution_width, resolution_height, frame_rate
    )
    time.sleep(1.0)
    image2_pipeline = enable_single_realsense(
        CAMERA2_SERIAL, ctx, resolution_width, resolution_height, frame_rate
    )
    time.sleep(1.0)

    # Setup data saving
    demo_save_dir = Path(args.save_dir)
    demo_save_dir.mkdir(exist_ok=True, parents=True)

    # Setup control interfaces
    keyboard = KeyboardInterface()
    shm_manager = SharedMemoryManager()
    shm_manager.start()
    sm = Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone)
    sm.start()

    # === Main loop ===
    while n_successes < args.n_demos:
        pkl_path = demo_save_dir / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"

        episode_data = {}
        episode_data["observations"] = []
        episode_data["actions"] = []
        episode_data["joint_targets"] = []
        episode_data["task"] = args.task

        # assume all real world demos that we actually save are success
        episode_data["success"] = True
        episode_data["args"] = args.__dict__

        # initial metadata dict
        metadata = dict(
            sm_dpos_scalar=sm_dpos_scalar,
            sm_drot_scalar=sm_drot_scalar,
            Kq=Kq.cpu().numpy(),
            Kqd=Kqd.cpu().numpy(),
        )
        episode_data["metadata"] = metadata

        translation, quat_xyzw = robot.get_ee_pose()
        rotvec = st.Rotation.from_quat(quat_xyzw.numpy()).as_rotvec()
        target_pose = np.array([*translation.numpy(), *rotvec])
        tip_target_pose = wrist_target_to_tip(target_pose)

        t_start = time.monotonic()
        iter_idx = 0
        stop = False

        obs_act_helper = ObsActHelper(
            sm=sm,
            keyboard=keyboard,
            robot=robot,
            gripper=gripper,
            image1_pipeline=image1_pipeline,
            image2_pipeline=image2_pipeline,
        )

        # obs_act_helper.set_target_pose(target_pose)
        obs_act_helper.set_target_pose(tip_target_pose)
        obs_act_helper.set_constants(
            max_pos_speed=args.max_pos_speed,
            max_rot_speed=args.max_rot_speed,
            sm_dpos_scalar=sm_dpos_scalar,
            sm_drot_scalar=sm_drot_scalar,
            frequency=args.frequency,
        )

        robot.reset(randomize=True)

        global_start_time = time.time()
        print(f"Start collecting!")
        while not stop:
            # calculate timing
            t_cycle_end = t_start + (iter_idx + 1) * dt
            t_sample = t_cycle_end - command_latency
            # t_command_target = t_cycle_end + dt
            precise_wait(t_sample)

            # get robot state/image observation
            observation = obs_act_helper.get_observation()

            # get and unpack action
            action_struct = obs_act_helper.get_action()
            action_current_pose_mat = action_struct.current_pose_mat
            action_next_pose_mat = action_struct.next_pose_mat
            grasp_flag = action_struct.grasp_flag
            action_taken = action_struct.action_taken
            collect_enum = action_struct.collect_enum
            is_gripper_open = action_struct.is_gripper_open
            toggle_gripper = action_struct.toggle_gripper

            if collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
                break

            # send command to the robot
            # robot.update_desired_ee_pose(action_next_pose_mat, dt=dt)
            joint_position_targets = robot.update_desired_ee_pose(
                convert_tip2wrist(action_next_pose_mat), dt=dt
            )
            execute_gripper_action(gripper, toggle_gripper, is_gripper_open)

            # log the data
            if action_taken:
                # convert to delta actions (where action quat is a right mult.)
                action = obs_act_helper.to_isaac_dpose_from_abs(
                    current_pose_mat=action_current_pose_mat,
                    goal_pose_mat=action_next_pose_mat,
                    grasp_flag=grasp_flag,
                    rm=True,
                )
                episode_data["actions"].append(action)
                episode_data["joint_targets"].append(joint_position_targets)
                episode_data["observations"].append(observation)

            target_pose = polypose2target(robot.get_ee_pose())
            tip_target_pose = wrist_target_to_tip(target_pose)
            # obs_act_helper.set_target_pose(target_pose)
            obs_act_helper.set_target_pose(tip_target_pose)

            # # Draw the current and target pose (in meshcat)
            # mc_util.meshcat_frame_show(
            #     mc_vis,
            #     f"scene/target_pose_wrist",
            #     convert_tip2wrist(action_next_pose_mat),
            # )
            # mc_util.meshcat_frame_show(
            #     mc_vis, f"scene/target_pose_tip", action_next_pose_mat
            # )
            # mc_util.meshcat_frame_show(
            #     mc_vis,
            #     f"scene/current_pose",
            #     poly_util.polypose2mat(robot.get_ee_pose()),
            # )

            precise_wait(t_cycle_end)
            iter_idx += 1

        global_total_time = time.time() - global_start_time
        print(f"Time elapsed: {global_total_time}")
        if collect_enum == CollectEnum.SUCCESS:
            # save the data
            with open(pkl_path, "wb") as f:
                pickle.dump(episode_data, f)

            n_successes += 1

    # Clean up resources
    sm.stop()
    shm_manager.shutdown()


if __name__ == "__main__":
    main()
