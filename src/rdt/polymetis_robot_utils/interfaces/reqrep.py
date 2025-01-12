import abc
from re import A

from flask.cli import F
import zmq
import struct
import numpy as np
import scipy.spatial.transform as st
from array import array
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
from enum import IntEnum
from dataclasses import field, dataclass


# #
# # Data-like
# #
class GripperAction(IntEnum):
    OPEN = 0
    CLOSE = 1
    NOOP = 2


# class Message(IntEnum):
#     RobotState = 0
#     Action = 1
#     Reset = 2
#     Ack = 3


# ##
# ## Buffer views
# ##
# @dataclass
# class ActionView:
#     """Semantic view of the action for debug purposes"""

#     next_pose_mat: np.ndarray
#     gripper_action: GripperAction
#     action_taken: bool


# @dataclass
# class RobotStateView:
#     """Semantic view of the robot state for debug purposes"""

#     qpos: np.ndarray
#     qvel: np.ndarray
#     ee_pos: np.ndarray
#     ee_quat: np.ndarray
#     ee_lin_vel: Optional[np.ndarray]
#     ee_ang_vel: Optional[np.ndarray]
#     gripper_qpos_scalar: np.ndarray


# ###
# ### Message buffers
# ###
# class Buffer(ABC):
#     def __init__(self, bsize: int):
#         super().__init__()
#         self._buffer = array("f", [-1] * bsize)
#         self._struct = struct.Struct(f"{bsize}f")

#     def to_bytes(self) -> bytes:
#         return self._struct.pack(*self._buffer)

#     def from_array(self, data: array):
#         self._buffer[:] = data

#     @abstractmethod
#     def view(self):
#         pass


# class Action(Buffer):
#     BUFFER_LAYOUT = {
#         "__msg__": (0, 1),
#         "pose_mat": (1, 33),  # flattened
#         "gripper_action": (33, 36),  # enums as floats
#         "action_taken": (36, 37),  # bools as floats
#     }
#     BUFFER_SIZE = 37

#     def __init__(self):
#         """Initialize with pre-allocated buffer. This should be called once for each client."""
#         super().__init__(self.BUFFER_SIZE)
#         self._setup_views()

#     def _setup_views(self):
#         """Create numpy views into the buffer"""
#         buf_array = np.frombuffer(self._buffer, dtype=np.float32)
#         pose_slice = slice(*self.BUFFER_LAYOUT["pose_mat"])
#         self.next_pose_mat = buf_array[pose_slice]
#         self.gripper_ind = self.BUFFER_LAYOUT["gripper_action"]
#         self._action_taken_idx = self.BUFFER_LAYOUT["action_taken"][0]
#         self._buffer[self.BUFFER_LAYOUT["__msg__"][0]] = Message.Action.value

#     @property
#     def gripper_action(self) -> GripperAction:
#         """Get gripper action enum"""
#         return [GripperAction(int(self._buffer[x])) for x in self.gripper_ind]

#     @gripper_action.setter
#     def gripper_action(self, value: Tuple[GripperAction]):
#         """Set gripper action enum"""
#         for i, v in enumerate(value):
#             self._buffer[self.gripper_ind[i]] = float(v)

#     @property
#     def action_taken(self) -> bool:
#         """Get action taken flag"""
#         return bool(self._buffer[self._action_taken_idx])

#     @action_taken.setter
#     def action_taken(self, value: bool):
#         """Set action taken flag"""
#         self._buffer[self._action_taken_idx] = float(value)

#     def view(self) -> ActionView:
#         """Get a semantic view of the action for debugging"""
#         return ActionView(
#             next_pose_mat=self.next_pose_mat.copy(),
#             gripper_action=self.gripper_action,
#             action_taken=self.action_taken,
#         )

#     @classmethod
#     def from_matrices(
#         cls, pose_mat: np.ndarray, gripper: Tuple[GripperAction], taken: bool
#     ) -> "Action":
#         """Create action from numpy matrix and values"""
#         action = cls()
#         np.copyto(action.next_pose_mat[: pose_mat.size], pose_mat.flatten())
#         action.gripper_action = gripper
#         action.action_taken = taken
#         return action


# class RobotState(Buffer):
#     BUFFER_LAYOUT = {
#         "__msg__": (0, 1),
#         "qpos": (1, 15),  # joint positions
#         "qvel": (15, 29),  # joint velocities
#         "ee_pos": (29, 35),  # end effector position
#         "ee_quat": (35, 43),  # end effector orientation as quaternion
#         "ee_lin_vel": (43, 49),  # end effector linear velocity
#         "ee_ang_vel": (49, 55),  # end effector angular velocity
#         "gripper_qpos_scalar": (55, 57),  # scalar for gripper joint position
#     }
#     BUFFER_SIZE = 57

#     def __init__(self):
#         """Initialize with pre-allocated buffer. This should be called once."""
#         super().__init__(self.BUFFER_SIZE)
#         self._setup_views()

#     def _setup_views(self):
#         """Create numpy views into the buffer"""
#         buf_array = np.frombuffer(self._buffer, dtype=np.float32)
#         for field in [
#             "qpos",
#             "qvel",
#             "ee_pos",
#             "ee_quat",
#             "ee_lin_vel",
#             "ee_ang_vel",
#             "gripper_qpos_scalar",
#         ]:
#             setattr(self, field, buf_array[slice(*self.BUFFER_LAYOUT[field])])

#         self._buffer[0] = Message.RobotState.value

#     @property
#     def robot_xyz_rotvec(self) -> Tuple[np.ndarray, np.ndarray]:
#         """Get robot state as (xyz, quat_xyzw)"""
#         return np.stack(
#             [
#                 np.concatenate(
#                     [
#                         self.ee_pos[:3],
#                         st.Rotation.from_quat(self.ee_quat[:4]).as_rotvec(),
#                     ]
#                 ),
#                 np.concatenate(
#                     [
#                         self.ee_pos[3:],
#                         st.Rotation.from_quat(self.ee_quat[4:]).as_rotvec(),
#                     ]
#                 ),
#             ]
#         )

#     @classmethod
#     def from_matrices(
#         cls,
#         qpos: np.ndarray,
#         qvel: np.ndarray,
#         ee_pos: np.ndarray,
#         ee_quat: np.ndarray,
#         gripper_qpos_scalar: np.ndarray,
#         ee_lin_vel: Optional[np.ndarray] = None,
#         ee_ang_vel: Optional[np.ndarray] = None,
#     ) -> "RobotState":
#         """Create robot state from numpy matrices"""
#         state = cls()
#         np.copyto(state.qpos[: qpos.size], qpos)
#         np.copyto(state.qvel[: qvel.size], qvel)
#         np.copyto(state.ee_pos[: ee_pos.size], ee_pos)
#         np.copyto(state.ee_quat[: ee_quat.size], ee_quat)
#         np.copyto(
#             state.gripper_qpos_scalar[: gripper_qpos_scalar.size], gripper_qpos_scalar
#         )
#         if ee_lin_vel is not None:
#             np.copyto(state.ee_lin_vel[: ee_lin_vel.size], ee_lin_vel)
#         if ee_ang_vel is not None:
#             np.copyto(state.ee_ang_vel[: ee_ang_vel.size], ee_ang_vel)

#         return state

#     def view(self) -> RobotStateView:
#         """Get a semantic view of the robot state for debugging"""
#         return RobotStateView(
#             qpos=self.qpos.copy(),
#             qvel=self.qvel.copy(),
#             ee_pos=self.ee_pos.copy(),
#             ee_quat=self.ee_quat.copy(),
#             ee_lin_vel=self.ee_lin_vel.copy() if self.ee_lin_vel is not None else None,
#             ee_ang_vel=self.ee_ang_vel.copy() if self.ee_ang_vel is not None else None,
#             gripper_qpos_scalar=self.gripper_qpos_scalar,
#         )


# class Reset(Buffer):
#     BUFFER_SIZE = 1

#     def __init__(self):
#         super().__init__(self.BUFFER_SIZE)
#         self._buffer[0] = Message.Reset.value

#     def view(self):
#         return "__reset__"


# class Ack(Buffer):
#     BUFFER_SIZE = 1

#     def __init__(self):
#         super().__init__(self.BUFFER_SIZE)
#         self.BUFFER_SIZE[0] = Message.Ack.value

#     def view(self):
#         return "__ack__"


# def unpack_bytes(data: bytes) -> Union[RobotState, Action, Reset]:
#     """Unpack bytes into a message buffer"""
#     arr = array("f", data)
#     msg_type = Message(int(arr[0]))
#     d: Buffer
#     if msg_type == Message.RobotState:
#         d = RobotState
#     elif msg_type == Message.Action:
#         d = Action
#     elif msg_type == Message.Reset:
#         d = Reset
#     else:
#         raise ValueError(f"Unknown message type: {msg_type}")
#     d = d()
#     d.from_array(arr)

#     return d


# class Client:
#     def __init__(self, addr: str = "localhost", port: int = 5555):
#         self.ctx = zmq.Context()
#         self.socket = self.ctx.socket(zmq.REQ)
#         self.socket.setsockopt(zmq.RCVHWM, 1)
#         self.socket.setsockopt(zmq.SNDHWM, 1)
#         self.socket.setsockopt(zmq.IMMEDIATE, 1)
#         self.socket.setsockopt(zmq.LINGER, 0)
#         # self.socket.setsockopt(zmq.REQ_RELAXED, 1)
#         self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
#         self.socket.setsockopt(zmq.RECONNECT_IVL, 100)
#         self.socket.setsockopt(zmq.RCVTIMEO, -1)
#         self.socket.setsockopt(zmq.SNDTIMEO, -1)
#         self.socket.connect(f"tcp://{addr}:{port}")
#         print("connect")

#     def get_obs(self) -> RobotState:
#         r = RobotState()
#         self.socket.send(r.to_bytes())
#         return unpack_bytes(self.socket.recv())

#     def act(self, action: Action):
#         self.socket.send(action.to_bytes())
#         self.socket.recv()

#     def reset(self):
#         r = Reset()
#         self.socket.send(r.to_bytes())
#         return True if self.socket.recv() == b"ACK" else False

#     def __del__(self):
#         self.socket.close()

from typing import Dict, Type


@dataclass
class Action:
    next_pose_mat: np.ndarray
    gripper_action: GripperAction
    action_taken: bool


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


class Reset:
    pass


class Ack:
    pass


class Nack:
    pass


class Client:
    def __init__(self, addr: str = "localhost", port: int = 5555):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.REQ)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.IMMEDIATE, 1)
        self.socket.setsockopt(zmq.LINGER, 0)
        # self.socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.socket.setsockopt(zmq.RECONNECT_IVL, 100)
        self.socket.setsockopt(zmq.RCVTIMEO, -1)
        self.socket.setsockopt(zmq.SNDTIMEO, -1)
        self.socket.connect(f"tcp://{addr}:{port}")

    def get_obs(self) -> RobotState:
        r = RobotState()
        self.socket.send_pyobj(r)
        return self.socket.recv_pyobj()

    def act(self, action: Action):
        self.socket.send_pyobj(action)
        self.socket.recv_pyobj()

    def reset(self):
        r = Reset()
        self.socket.send_pyobj(r)
        self.socket.recv_pyobj()

    def __del__(self):
        self.socket.close()
