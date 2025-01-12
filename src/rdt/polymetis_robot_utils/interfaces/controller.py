import threading
import pyspacemouse
import numpy as np
import numpy as np
import time
import torch
import scipy.spatial.transform as st
from typing import Iterable, Dict, Optional, List
from abc import abstractmethod
from enum import Enum
from easyhid import Enumeration
from mujoco_ar import MujocoARConnector
from bifranka.util.teleop_util import scale_scripted_action
from bifranka.util.robot_config import RobotConfig
from bifranka.interfaces.keyboard_interface import CollectEnum
from dataclasses import dataclass


class RobotState:
    qpos: np.ndarray
    qvel: np.ndarray
    ee_pos: np.ndarray
    ee_quat: np.ndarray
    ee_lin_vel: Optional[np.ndarray]
    ee_ang_vel: Optional[np.ndarray]
    gripper_qpos_scalar: np.ndarray

    @property
    def xyz_rotvec(self):
        return np.concatenate(
            [self.ee_pos, st.Rotation.from_quat(self.ee_quat).as_rotvec()]
        )


class Chirality(Enum):
    LEFT = "left"
    RIGHT = "right"


class GripperAction(Enum):
    OPEN = "open"
    CLOSE = "close"


@dataclass
class Action:
    next_pose_mat: np.ndarray
    gripper_action: GripperAction
    action_taken: bool


class TeleopControllerBase:
    @abstractmethod
    def get_actions(self) -> Dict[Chirality, Action]:
        pass

    @abstractmethod
    def get_action(self, chirality: Chirality) -> Action:
        pass

    @staticmethod
    def to_pose_mat(pose_):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = pose_[:3]
        pose_mat[:-1, :-1] = st.Rotation.from_rotvec(pose_[3:]).as_matrix()
        return pose_mat

    @property
    def terminal(self) -> List[CollectEnum]:
        return CollectEnum.terminal()


class MultiSpacemouseControl(TeleopControllerBase):
    def __init__(
        self,
        chiralities: Iterable[Chirality],
        operating_frequency: float = 20,
        read_frequency: float = 200,
        sm_dpos_scalar: np.array = np.array([1.8] * 3),
        sm_drot_scalar: np.array = np.array([4.0] * 3),
        max_pos_speed: float = 3,
        max_rot_speed: float = 5,
    ):
        super().__init__()
        self.sm_dpos_scalar = sm_dpos_scalar
        self.sm_drot_scalar = sm_drot_scalar
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.read_frequency = read_frequency
        self.operating_frequency = operating_frequency

        hid_paths = [
            dev.path
            for dev in Enumeration().find()
            if "3Dconnexion" in dev.product_string and dev.usage == 58
        ]

        assert (
            len(hid_paths) == 1
        ), f"Check connections; found {len(hid_paths)} spacemouse devices, expected 2"
        print("[Done] Successfully registered 2/2 spacemice")

        # TODO: Fix
        self._chiralities = chiralities
        self._spacemice = {
            ch: pyspacemouse.open(path=hid_paths[i])
            for i, ch in enumerate(self._chiralities)
        }

        if any(sm is None for sm in self._spacemice.values()):
            raise ValueError("Failed to open one or more spacemice")

        self._state_lock = threading.Lock()
        self._threads = [
            threading.Thread(target=self._read_spacemouse, args=(ch, read_frequency))
            for ch in self._chiralities
        ]

        # Initialize the memory before starting the threads
        self.latest_state = {
            ch: {
                "action": np.zeros(6),
                "timestamp": time.time(),
                "gripper": [False, False],
            }
            for ch in self._chiralities
        }

        for t in self._threads:
            t.daemon = True
            t.start()

        self.last_valid_gripper_states = {
            ch: GripperAction.OPEN for ch in self._chiralities
        }
        print("[Done] Successfully registered keyboard interface")

    @property
    def chirality(self):
        return self._chiralities

    def _read_spacemouse(self, chirality: Chirality, read_frequency: float):
        """
        Read spacemouse data and update the latest state.
        """
        sm_ref = self._spacemice[chirality]
        while True:
            state = sm_ref.read()
            with self._state_lock:
                # Spacemouse axis matched with robot base frame
                self.latest_state[chirality]["action"] = np.array(
                    [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw]
                )
                self.latest_state[chirality]["gripper"] = [
                    bool(state.buttons[0]),
                    bool(state.buttons[-1]),
                ]
                self.latest_state[chirality]["timestamp"] = time.time()
            time.sleep(1 / read_frequency)

    def get_actions(
        self, platform_state: Dict[Chirality, RobotState]
    ) -> Dict[Chirality, Action]:
        """Returns the latest action and button state of the SpaceMouse."""
        return {ch: self.get_action(ch, platform_state[ch]) for ch in self._chiralities}

    def get_action(self, chirality: Chirality, robot_state: RobotState) -> Action:
        """Returns the latest action and button state of the SpaceMouse."""
        with self._state_lock:
            latest_state = self.latest_state[chirality].copy()

        dpos = (
            latest_state["action"][:3]
            * (self.max_pos_speed / self.operating_frequency)
            * self.sm_dpos_scalar
        )

        drot_xyz = latest_state["action"][3:]
        drot_rotvec = st.Rotation.from_euler("xyz", drot_xyz).as_rotvec()
        drot_rotvec *= (
            self.max_rot_speed / self.operating_frequency
        ) * self.sm_drot_scalar
        drot = st.Rotation.from_rotvec(drot_rotvec)

        action_taken = not np.allclose(dpos, 0.0) and not np.allclose(drot_xyz, 0.0)
        # Determine gripper action (if there is any to be taken)
        gripper_action = None
        if latest_state["gripper"][0]:
            gripper_action = GripperAction.OPEN
        elif latest_state["gripper"][1]:
            gripper_action = GripperAction.CLOSE

        if gripper_action is not None:
            self.last_valid_gripper_states[chirality] = gripper_action
        else:
            gripper_action = self.last_valid_gripper_states[chirality]

        delta_action = torch.from_numpy(
            np.concatenate([dpos, drot.as_quat(), np.array([-1])])
        ).unsqueeze(0)
        delta_action = (
            scale_scripted_action(
                delta_action, pos_bounds_m=0.025 * 2, ori_bounds_deg=20
            )
            .squeeze()
            .numpy()
        )[:-1]

        ee_pos = robot_state.ee_pos
        ee_quat = robot_state.ee_quat
        ee_pos += delta_action[:3]
        ee_rotvec = (
            st.Rotation.from_quat(delta_action[3:]) * st.Rotation.from_quat(ee_quat)
        ).as_rotvec()

        return Action(
            next_pose_mat=self.to_pose_mat(np.concatenate([ee_pos, ee_rotvec])),
            gripper_action=gripper_action,
            action_taken=action_taken,
        )


class AVPControl(TeleopControllerBase):
    def __init__(
        self,
        chiralities: Iterable[Chirality],
        avp_ip: str,  # IP address of the AVP server
    ):
        super().__init__()
        import avp_stream as avp
        import tqdm

        self._chiralities = chiralities

        self.avp = avp.VisionProStreamer(ip=avp_ip)
        self.avp.start_streaming()

        print("=== Put your hands on the table! ===")
        # Progress bar for 5 seconds
        pbar = tqdm.tqdm(total=50)
        pbar.set_description(
            "We need to calibrate where you are in global frame. Put your hands on the table."
        )
        for _ in tqdm.tqdm(range(50)):
            time.sleep(0.01)
            pbar.update(1)

        self.table_height, self.left_x, self.right_x, self.left_y, self.right_y = (
            self._get_calibration()
        )
        print("[Done] Successfully calibrated the table height and the hand positions")
        print(
            f"Table Z: {self.table_height} | Left Pos: {self.left_x} | Right Pos: {self.right_x}"
        )

        rot_z90 = st.Rotation.from_euler("z", -np.pi / 2).as_matrix()
        self.rot_z90 = np.eye(4)
        self.rot_z90[:3, :3] = rot_z90
        # expand 0 dims
        self.rot_z90 = np.expand_dims(self.rot_z90, axis=0)

        print(self.rot_z90.shape)

        print("[Done] Successfully registered keyboard interface")

    def get_latest_ee_gripper(self):

        latest = self.avp.get_latest()
        latest["right_wrist"][:, 2, 3] -= self.table_height
        latest["right_wrist"][:, 0, 3] -= self.right_x
        # latest["right_wrist"][:, 1, 3] -= self.right_y
        latest["right_wrist"][:, 1, 3] += 0.3
        latest["right_wrist"] = self.rot_z90 @ latest["right_wrist"]

        latest["left_wrist"][:, 2, 3] -= self.table_height
        latest["left_wrist"][:, 0, 3] -= self.left_x
        # latest["left_wrist"][:, 1, 3] -= self.left_y
        latest["left_wrist"][:, 1, 3] += 0.3
        latest["left_wrist"] = self.rot_z90 @ latest["left_wrist"]

        if latest is not None:

            ee_frames = self._construct_ee_matrix(latest)
            gripper = self._get_gripper(latest)

            return ee_frames, gripper

        return None, None

    def _get_gripper(self, latest):
        """
        True if the gripper is opened, False if the gripper is closed.
        Determined by the distance between the thumb and the index finger.
        """

        right_fingers = latest["right_wrist"] @ latest["right_fingers"]
        left_fingers = latest["left_wrist"] @ latest["left_fingers"]

        right_f1 = right_fingers[4]
        right_f3 = right_fingers[9]

        left_f1 = left_fingers[4]
        left_f3 = left_fingers[9]

        right_distance = np.linalg.norm(right_f1 - right_f3)
        left_distance = np.linalg.norm(left_f1 - left_f3)

        return {
            Chirality.LEFT: right_distance > 0.04,
            Chirality.RIGHT: left_distance > 0.04,
        }

    def _construct_ee_matrix(self, latest):

        right_fingers = latest["right_wrist"] @ latest["right_fingers"]
        left_fingers = latest["left_wrist"] @ latest["left_fingers"]

        # RIGHT EE
        right_f1 = right_fingers[4][:3, -1]  # thumb tip
        right_f2 = right_fingers[3][:3, -1]
        right_f3 = right_fingers[9][:3, -1]  # index tip
        right_f4 = right_fingers[8][:3, -1]

        right_yf_axis = right_f4 - right_f1  # thumb tip to index tip
        right_z_axis = (
            -(right_f2 + right_f4) / 2 + (right_f1 + right_f3) / 2
        )  # middle finger to thumb and index tip
        right_z_axis = right_z_axis / np.linalg.norm(right_z_axis)  # normalize
        right_x_axis = np.cross(right_yf_axis, right_z_axis)
        right_y_axis = np.cross(right_z_axis, right_x_axis)

        # construct the rotation matrix
        right_ee = np.eye(4)
        right_ee[:3, 0] = right_x_axis / np.linalg.norm(right_x_axis)
        right_ee[:3, 1] = right_y_axis / np.linalg.norm(right_y_axis)
        right_ee[:3, 2] = right_z_axis / np.linalg.norm(right_z_axis)
        right_ee[:3, -1] = (right_f1 + right_f3) / 2

        right_ee[:3, :3] /= np.linalg.det(right_ee[:3, :3])

        # LEFT EE

        left_f1 = left_fingers[4][:3, -1]  # thumb tip
        left_f2 = left_fingers[3][:3, -1]
        left_f3 = left_fingers[9][:3, -1]  # index tip
        left_f4 = left_fingers[8][:3, -1]

        left_yf_axis = left_f4 - left_f2  # thumb tip to index tip
        left_yf_axis = left_yf_axis / np.linalg.norm(left_yf_axis)  # normalize
        left_z_axis = (left_f2 + left_f4) / 2 - (
            left_f1 + left_f3
        ) / 2  # middle finger to thumb and index tip
        left_z_axis = left_z_axis / np.linalg.norm(left_z_axis)  # normalize
        left_x_axis = np.cross(left_yf_axis, left_z_axis)
        left_y_axis = np.cross(left_z_axis, left_x_axis)

        # construct the rotation matrix
        left_ee = np.eye(4)
        left_ee[:3, 0] = left_x_axis
        left_ee[:3, 1] = left_y_axis
        left_ee[:3, 2] = left_z_axis
        left_ee[:3, 3] = (left_f1 + left_f3) / 2

        return {Chirality.LEFT: right_ee, Chirality.RIGHT: left_ee}

    def _get_calibration(self):

        latest = self.avp.get_latest()

        right_wrist = latest["right_wrist"]
        left_wrist = latest["left_wrist"]

        init_z = (right_wrist[0, 2, 3] + left_wrist[0, 2, 3]) / 2
        init_x_left = left_wrist[0, 0, 3]
        init_x_right = right_wrist[0, 0, 3]

        init_y_left = left_wrist[0, 1, 3]
        init_y_right = right_wrist[0, 1, 3]

        return init_z, init_x_left, init_x_right, init_y_left, init_y_right

    def get_actions(
        self, robot_state: Dict[Chirality, RobotState]
    ) -> Dict[Chirality, Action]:
        """Returns the latest action and button state of the SpaceMouse."""
        return {ch: self.get_action(ch, robot_state) for ch in self._chiralities}

    def get_action(self, chirality: Chirality, robot_state: RobotState) -> Action:

        ee_frames, gripper = self.get_latest_ee_gripper()

        # Determine gripper action (if there is any to be taken)

        gripper_action = None
        if gripper[chirality]:
            # Open gripper
            gripper_action = GripperAction.OPEN
        else:
            # Close gripper
            gripper_action = GripperAction.CLOSE

        ee_frame = ee_frames[chirality]

        if chirality == Chirality.RIGHT:
            ee_pos = robot_state[chirality].ee_pos
            ee_rotvec = st.Rotation.from_quat(
                robot_state[chirality].ee_quat
            ).as_rotvec()
        else:
            ee_pos = ee_frame[:3, -1]
            ee_rotvec = st.Rotation.from_matrix(ee_frame[:3, :3]).as_rotvec()

        print(ee_pos.shape, ee_rotvec.shape)

        action_taken = True
        return ActionContainer(
            next_pose_mat=self.to_pose_mat(np.concatenate([ee_pos, ee_rotvec])),
            gripper_action=gripper_action,
            action_taken=action_taken,
        )


class Iphone:
    def __init__(
        self,
        port: int,
        chirality: Chirality,
        pos_scale: float = 1.5,
        rot_scale: float = 1.5,
    ):
        self._connector = MujocoARConnector(port=port)
        self._connector.start()

        self._port = port
        self._chirality = chirality
        self._last_pose_mat = None

        # Scaling factors
        self._rot_scale = rot_scale
        self._pos_scale = pos_scale

    @property
    def chirality(self):
        return self._chirality

    def vibrate(self, duration: float = 1, intensity: int = 5, sharpness: int = 3):
        self._connector.vibrate(
            duration=duration, intensity=intensity, sharpness=sharpness
        )

    def get_action(
        self,
        home_state,
    ) -> Action:

        # Translate the iphone in the world frame by the position specified by the iphone AR controller
        latest_data = self._connector.get_latest_data()
        next_pose_mat = np.eye(4)
        next_pose_mat[:3, -1] = (
            home_state[0].numpy() + latest_data["position"] * self._pos_scale
        )

        # Rotate around the world frame by the rotation specified by the iPhone
        next_pose_mat[:3, :3] = (
            st.Rotation.from_rotvec(
                st.Rotation.from_matrix(latest_data["rotation"]).as_rotvec()
                * self._rot_scale
            )
            * st.Rotation.from_quat(home_state[1].numpy())
        ).as_matrix()

        action = Action(
            next_pose_mat=next_pose_mat,
            gripper_action=(
                GripperAction.OPEN if latest_data["toggle"] else GripperAction.CLOSE
            ),
            action_taken=self._last_pose_mat is None
            or not np.allclose(next_pose_mat, self._last_pose_mat),
        )
        self._last_pose_mat = next_pose_mat
        return action


class IPhoneARControl:
    def __init__(
        self,
        left_port: Optional[int] = 8887,
        right_port: Optional[int] = 8888,
    ):
        super().__init__()
        self._controllers = [
            Iphone(port=port, chirality=chirality)
            for port, chirality in [
                [left_port, Chirality.LEFT],
                [right_port, Chirality.RIGHT],
            ]
            if port is not None
        ]

        print("Awaiting iPhone AR controller connection completion...")
        start_time = time.time()
        while (
            not all(
                controller._connector.get_latest_data()["button"] is not None
                for controller in self._controllers
            )
            and time.time() < start_time + 10
        ):
            time.sleep(0.1)

        if not all(
            controller._connector.get_latest_data()["button"] is not None
            for controller in self._controllers
        ):
            raise ValueError("Failed to connect to all iPhone AR controllers.")
        print("[Done] Successfully connected to all iPhone AR controllers.")

    def get_actions(
        self, robot_state: Dict[Chirality, RobotState]
    ) -> Dict[Chirality, Action]:
        return {
            controller.chirality: controller.get_action(
                self._home_states[controller.chirality]
            )
            for controller in self._controllers
        }

    def get_action(self, chirality: Chirality) -> Action:
        raise NotImplementedError("Not implemented yet")

    def set_hs(self, home_states: Dict[Chirality, np.ndarray]):
        self._home_states = home_states

    @property
    def terminal(self) -> List[CollectEnum]:
        return [*CollectEnum.terminal(), CollectEnum.CLUTCH]

    def vibrate(self):
        for controller in self._controllers:
            controller.vibrate()
