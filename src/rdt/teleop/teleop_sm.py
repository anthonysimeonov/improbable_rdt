from pathlib import Path
import pickle
import random
import time

# from multiprocessing.managers import SharedMemoryManager
from typing import List
import cv2
from src.rdt.polymetis_robot_utils.interfaces.reqrep import GripperAction, RobotState
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
from rdt.polymetis_robot_utils.interfaces.reqrep import Action, RobotState, Reset, Ack
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
        gripper.grasp(0.1, 0.0001, 0.01, blocking=False)
    else:
        gripper.goto(0.08, 0.2, 0.1, blocking=False)


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
class Platform:
    def __init__(
        self,
        sm: Spacemouse,
        keyboard: KeyboardInterface,
        robot: DiffIKWrapper,
        gripper: GripperInterface,
        image_pipelines: List[rs.pipeline],
        resize_images: bool = False,
        include_depth: bool = False,
        resize_size: tuple = (256, 256),
        show_images: bool = False,
    ):
        self.sm = sm
        self.keyboard = keyboard
        self.robot = robot
        self.gripper = gripper
        self.image_pipelines = image_pipelines
        self.resize_images = resize_images
        self.include_depth = include_depth
        self.image_height, self.image_width = resize_size
        self.show_images = show_images

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

    def get_observation(self) -> RobotState:
        obs = dict()

        # get the robot state
        current_ee_wrist_pose_mat = poly_util.polypose2mat(self.robot.get_ee_pose())

        # convert to tip
        current_ee_tip_pose_mat = convert_wrist2tip(current_ee_wrist_pose_mat)
        current_ee_tip_pose = poly_util.mat2polypose(current_ee_tip_pose_mat)
        current_ee_pose = torch.cat(current_ee_tip_pose, dim=-1)

        current_joint_positions = self.robot.get_joint_positions()
        jacobian = self.robot.robot_model.compute_jacobian(current_joint_positions)
        ee_spatial_velocity = jacobian @ self.robot.get_joint_velocities()

        robot_state = RobotState()
        robot_state.ee_pos = current_ee_pose[:3].numpy()
        robot_state.ee_quat = current_ee_pose[3:7].numpy()
        robot_state.ee_lin_vel = ee_spatial_velocity[:3].numpy()
        robot_state.ee_ang_vel = ee_spatial_velocity[3:6].numpy()
        robot_state.gripper_qpos_scalar = np.array([self.gripper.get_state().width])
        robot_state.qpos = current_joint_positions.numpy()

        return robot_state


def main():

    # === One-time setup ===
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--port_vis", type=int, default=6000)
    parser.add_argument("--frequency", type=int, default=10)  # 30
    parser.add_argument("--command_latency", type=float, default=0.01)
    parser.add_argument("--deadzone", type=float, default=0.05)
    parser.add_argument("--max-pos-speed", type=float, default=2)
    parser.add_argument("--max-rot-speed", type=float, default=3)
    parser.add_argument("--resize-images", action="store_true")
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
        [
            -3.2031e-01,
            -1.1461e-01,
            1.4063e-01,
            -2.4171e00,
            -5.8475e-02,
            2.4469e00,
            -8.3714e-01,
        ]
    )  # New sim home
    Kq = torch.Tensor([150.0, 120.0, 160.0, 100.0, 110.0, 100.0, 40.0])
    Kqd = torch.Tensor([20.0, 20.0, 20.0, 20.0, 12.0, 12.0, 8.0])

    robot = DiffIKWrapper(
        ip_address=franka_ip,
        robot_home=robot_home,
        Kq=Kq,
        Kqd=Kqd,
        chirality="left",
    )
    gripper = GripperInterface(ip_address=franka_ip)

    # manual home
    gripper.goto(0.08, 0.05, 0.1, blocking=False)  # 0.065

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

    camera_serials = [
        "242622071805",  # Global camera 1
        "317422075533",  # Wrist camera
        "242522072326",  # Global camera 2
    ]

    print(f"Camera serials: {camera_serials}")

    ctx = rs.context()  # Create librealsense context for managing devices

    image_pipelines = []

    for serial in camera_serials:
        pipeline = enable_single_realsense(
            serial, ctx, resolution_width, resolution_height, frame_rate
        )
        image_pipelines.append(pipeline)
        time.sleep(1.0)

    # Setup data saving
    demo_save_dir = Path(args.save_dir)
    demo_save_dir.mkdir(exist_ok=True, parents=True)

    # Setup control interfaces
    keyboard = KeyboardInterface()
    # shm_manager = SharedMemoryManager()
    # shm_manager.start()
    # sm = Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone)
    # sm.start()

    import zmq

    # TODO: remove harcoded port
    port = 5555
    ctx = zmq.Context()
    socket = ctx.socket(zmq.REP)

    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.SNDHWM, 1)
    socket.setsockopt(zmq.IMMEDIATE, 1)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVTIMEO, -1)
    socket.setsockopt(zmq.SNDTIMEO, -1)
    socket.bind(f"tcp://*:{port}")

    # === Main loop ===
    while n_successes < args.n_demos:
        pkl_path = demo_save_dir / f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}.pkl"
        robot.reset(randomize=True)

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

        obs_act_helper = Platform(
            sm=None,
            keyboard=keyboard,
            robot=robot,
            gripper=gripper,
            image_pipelines=image_pipelines,
            resize_images=args.resize_images,
            include_depth=True,
            resize_size=(252, 448),
            show_images=False,
        )

        obs_act_helper.set_target_pose(tip_target_pose)
        obs_act_helper.set_constants(
            max_pos_speed=args.max_pos_speed,
            max_rot_speed=args.max_rot_speed,
            sm_dpos_scalar=sm_dpos_scalar,
            sm_drot_scalar=sm_drot_scalar,
            frequency=args.frequency,
        )

        global_start_time = time.time()
        print(f"Start collecting!")
        while not stop:
            # calculate timing
            # t_cycle_end = t_start + (iter_idx + 1) * dt
            # t_sample = t_cycle_end - command_latency
            # t_command_target = t_cycle_end + dt
            # precise_wait(t_sample)

            # get robot state/image observation
            # observation = obs_act_helper.get_observation()

            # get and unpack action
            # action_struct = obs_act_helper.get_action()
            # action_current_pose_mat = action_struct.current_pose_mat
            # action_next_pose_mat = action_struct.next_pose_mat
            # grasp_flag = action_struct.grasp_flag
            # action_taken = action_struct.action_taken
            # collect_enum = action_struct.collect_enum
            # is_gripper_open = action_struct.is_gripper_open
            # toggle_gripper = action_struct.toggle_gripper

            # if collect_enum in [CollectEnum.SUCCESS, CollectEnum.FAIL]:
            #     break

            data = socket.recv_pyobj()
            if isinstance(data, RobotState):
                # obs = RobotState.from_matrices(
                #     ee_pos=oh_obs["robot_state"]["ee_pos"],
                #     ee_quat=oh_obs["robot_state"]["ee_quat"],
                #     qvel=np.array([-99999] * 7),
                #     qpos=oh_obs["robot_state"]["joint_positions"],
                #     gripper_qpos_scalar=np.array(
                #         [oh_obs["robot_state"]["gripper_width"]]
                #     ),
                # )
                response = obs_act_helper.get_observation()
                response.qvel = np.array([-99999] * 7)

            elif isinstance(data, Action):
                # send command to the robot
                joint_position_targets = robot.update_desired_ee_pose(
                    convert_tip2wrist(data.next_pose_mat[:16].reshape(4, 4)), dt=dt
                )

                # # execute_gripper_action(
                # #     gripper,
                # #     (data.gripper_action == GripperAction.OPEN)
                # #     != (data.gripper_qpos_scalar > 0.04),
                # #     data.gripper_qpos_scalar > 0.04,
                # # )

                target_pose = polypose2target(robot.get_ee_pose())
                tip_target_pose = wrist_target_to_tip(target_pose)
                obs_act_helper.set_target_pose(tip_target_pose)

                response = Ack()
            elif isinstance(data, Reset):
                robot.reset(randomize=True)
                response = Ack()

            socket.send_pyobj(response)

            # log the data
            # if action_taken:
            #     # convert to delta actions (where action quat is a right mult.)
            #     action = obs_act_helper.to_isaac_dpose_from_abs(
            #         current_pose_mat=action_current_pose_mat,
            #         goal_pose_mat=action_next_pose_mat,
            #         grasp_flag=grasp_flag,
            #         rm=True,
            #     )
            #     episode_data["actions"].append(action)
            #     episode_data["joint_targets"].append(joint_position_targets)
            #     episode_data["observations"].append(observation)

            # for key in observation.keys():
            #     if not key.startswith("color_image"):
            #         continue
            #     cv2.imshow(
            #         key,
            #         cv2.cvtColor(observation[key], cv2.COLOR_BGR2RGB),
            #     )

            # cv2.waitKey(1)

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

            # precise_wait(t_cycle_end)
            iter_idx += 1

            # print(
            #     f"Iteration {iter_idx} complete, time elapsed: {time.time() - global_start_time}, frequency: {iter_idx / (time.time() - global_start_time)}"
            # )

        global_total_time = time.time() - global_start_time
        print(f"Time elapsed: {global_total_time}")
        # if collect_enum == CollectEnum.SUCCESS:
        #     # save the data
        #     with open(pkl_path, "wb") as f:
        #         pickle.dump(episode_data, f)

        #     n_successes += 1

    # Clean up resources
    # sm.stop()
    # shm_manager.shutdown()


if __name__ == "__main__":
    main()
