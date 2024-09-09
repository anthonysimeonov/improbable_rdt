from scipy.spatial.transform import Rotation as R

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torchcontrol as toco

from polymetis import GripperInterface, RobotInterface
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from collections import namedtuple

import grpc

poly_util = PolymetisHelper()


# === Resolved Rates Controller===
class ResolvedRateControl(toco.PolicyModule):
    """Resolved Rates Control --> End-Effector Control (dx, dy, dz, droll, dpitch, dyaw) via Joint Velocity Control"""

    def __init__(
        self,
        Kp: torch.Tensor,
        robot_model: torch.nn.Module,
        ignore_gravity: bool = True,
    ) -> None:
        """
        Initializes a Resolved Rates controller with the given P gains and robot model.

        :param Kp: P gains in joint space (7-DoF)
        :param robot_model: A robot model from torchcontrol.models
        :param ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize Modules --> Inverse Dynamics is necessary as it needs to be compensated for in output torques...
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )

        # Create LinearFeedback (P) Controller...
        self.p = toco.modules.feedback.LinearFeedback(Kp)

        # Reference End-Effector Velocity (dx, dy, dz, droll, dpitch, dyaw)
        self.ee_velocity_desired = torch.nn.Parameter(torch.zeros(6))

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute joint torques given desired EE velocity.

        :param state_dict: A dictionary containing robot states (joint positions, velocities, etc.)
        :return Dictionary containing joint torques.
        """
        # State Extraction
        joint_pos_current, joint_vel_current = (
            state_dict["joint_positions"],
            state_dict["joint_velocities"],
        )

        # Compute Target Joint Velocity via Resolved Rate Control...
        #   =>> Resolved Rate: joint_vel_desired = J.pinv() @ ee_vel_desired
        #                      >> Numerically stable --> torch.linalg.lstsq(J, ee_vel_desired).solution
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        joint_vel_desired = torch.linalg.lstsq(
            jacobian, self.ee_velocity_desired
        ).solution

        # Control Logic --> Compute P Torque (feedback) & Inverse Dynamics Torque (feedforward)
        torque_feedback = self.p(joint_vel_current, joint_vel_desired)
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}


class DiffIKWrapper(RobotInterface):
    """
    wrapper that addds the resolved rate controller + changes quaternion order to xyzw
    """

    def __init__(self, robot_home, Kq, Kqd, *args, **kwargs):

        super().__init__(*args, **kwargs)
        # self.pos_scalar = 1.0
        # self.rot_scalar = 2.0
        self.pos_scalar = np.array([1.0] * 3)  # x, y, z
        self.rot_scalar = np.array([1.0] * 3)  # r, p, y
        self.robot_home = robot_home
        self.Kq = Kq
        self.Kqd = Kqd

        # self.pos_scalar = np.array([0.5] * 3)  # x, y, z
        # self.rot_scalar = np.array([0.5] * 3)  # r, p, y

    def set_pos_rot_scalars(self, pos=None, rot=None):
        if pos is not None:
            self.pos_scalar = pos
        if rot is not None:
            self.rot_scalar = rot

    def get_ee_pose_with_rot_as_xyzw(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Polymetis defaults to returning a Tuple of (position, orientation), where orientation is a quaternion in
        *scalar-first* format (w, x, y, z). However, `scipy` and other libraries expect *scalar-last* (x, y, z, w);
        we take care of that here!

        :return Tuple of (3D position, 4D orientation as a quaternion w/ scalar-last)
        """
        pos, quat = super().get_ee_pose()
        return pos, torch.roll(quat, -1)

    def start_resolved_rate_control(
        self, Kq: Optional[List[float]] = None
    ) -> List[Any]:
        """
        Start Resolved-Rate Control (P control on Joint Velocity), as a non-blocking controller.

        The desired EE velocities can be updated using `update_desired_ee_velocities` (6-DoF)!
        """
        torch_policy = ResolvedRateControl(
            Kp=self.Kqd_default if Kq is None else Kq,
            robot_model=self.robot_model,
            ignore_gravity=self.use_grav_comp,
        )
        return self.send_torch_policy(torch_policy=torch_policy, blocking=False)

    def update_desired_ee_velocities(self, ee_velocities: torch.Tensor):
        """
        Update the desired end-effector velocities (6-DoF x, y, z, roll, pitch, yaw).

        Requires starting a resolved-rate controller via `start_resolved_rate_control` beforehand.
        """
        try:
            update_idx = self.update_current_policy(
                {"ee_velocity_desired": ee_velocities}
            )
        except grpc.RpcError as e:
            print(
                "Unable to update desired end-effector velocities. Use `start_resolved_rate_control` to start a "
                "resolved-rate controller."
            )
            raise e
        return update_idx

    def compute_ee_vel_desired(self, desired_ee_pose_mat, dt_pos=1.0, dt_rot=1.0):
        current_ee_pose_mat = poly_util.polypose2mat(self.get_ee_pose())

        # position
        ee_dpos = (desired_ee_pose_mat[:-1, -1] - current_ee_pose_mat[:-1, -1]) / dt_pos
        ee_dpos = ee_dpos * self.pos_scalar

        # rotation
        ee_rot_mat_error = desired_ee_pose_mat[:-1, :-1] @ np.linalg.inv(
            current_ee_pose_mat[:-1, :-1]
        )
        ee_drot = (R.from_matrix(ee_rot_mat_error).as_rotvec()) / dt_rot
        ee_drot = ee_drot * self.rot_scalar

        # combine into spatial velocity
        ee_dpose = torch.Tensor([*ee_dpos, *ee_drot]).float()
        return ee_dpose

    def update_desired_ee_pose(
        self, ee_pose_mat: torch.Tensor, dt=0.1, scalar=1.0
    ) -> torch.Tensor:
        """
        Update the desired end-effector pose (position and orientation) using a resolved-rate controller.

        :param ee_pose_mat: Desired end-effector pose as a 4x4 transformation matrix
        :param dt: Time step for velocity computation
        :param scalar: Scalar multiplier for the desired joint velocities
        :return: Tuple of (current joint positions, desired joint positions)
        """
        joint_pos_current = self.get_joint_positions()
        jacobian = self.robot_model.compute_jacobian(joint_pos_current)
        ee_velocity_desired = self.compute_ee_vel_desired(
            ee_pose_mat, dt_pos=dt, dt_rot=dt
        )

        joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired).solution
        joint_pos_desired = joint_pos_current + joint_vel_desired * dt * scalar

        self.update_desired_joint_positions(joint_pos_desired)

        return joint_pos_desired

    def reset(self, randomize=False):

        robot_home = self.robot_home

        if randomize:
            home_noise = (2 * torch.rand(7) - 1) * np.deg2rad(10)
            robot_home = robot_home + home_noise

        self.move_to_joint_positions(robot_home)
        self.start_joint_impedance(Kq=self.Kq, Kqd=self.Kqd, adaptive=True)
