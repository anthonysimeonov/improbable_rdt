from typing import Tuple, List, Dict, Optional
import time
import grpc

import numpy as np
from scipy.spatial.transform import Rotation as R
import torch

import torchcontrol as toco
from torchcontrol.utils.tensor_utils import to_tensor
from polymetis_pb2 import RobotState


class PolymetisTrajectoryUtil:
    def __init__(self, robot, start_tolerance_deg=5.0):
        self.robot = robot
        self.time_to_go_default = self.robot.time_to_go_default

        # self.Kq_default = self.robot.Kq_default
        # self.Kqd_default = self.robot.Kqd_default
        # self.Kq_default = torch.Tensor([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
        # self.Kqd_default = torch.Tensor([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])
        self.Kq_default = torch.Tensor([500.0, 500.0, 500.0, 500.0, 250.0, 150.0, 50.0])
        self.Kqd_default = torch.Tensor([40.0, 40.0, 40.0, 40.0, 25.0, 25.0, 15.0])
        self.Kx_default = self.robot.Kx_default
        self.Kxd_default = self.robot.Kxd_default

        self.robot_model = self.robot.robot_model
        self.use_grav_comp = self.robot.use_grav_comp

        # self.joint_vel_limits = self.robot_model.get_joint_velocity_limits() * 0.85
        self.joint_vel_limits = torch.Tensor([2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51]) * 0.75

        self.start_tolerance = np.deg2rad(start_tolerance_deg)

        self.diffik_lookahead = 10
        self.diffik_sleep_lag = 0.9

    @staticmethod
    def polypose2mat(polypose):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = polypose[0].numpy()
        pose_mat[:-1, :-1] = R.from_quat(polypose[1].numpy()).as_matrix()
        return pose_mat 

    @staticmethod 
    def mat2polypose(pose_mat):
        trans = torch.from_numpy(pose_mat[:-1, -1])
        quat = torch.from_numpy(R.from_matrix(pose_mat[:-1, :-1]).as_quat())
        return trans, quat
    
    def set_diffik_lookahead(self, lookahead):
        self.diffik_lookahead = lookahead
    
    def _adaptive_time_to_go(self, joint_displacement: torch.Tensor):
        """Compute adaptive time_to_go
        Computes the corresponding time_to_go such that the mean velocity is equal to one-eighth
        of the joint velocity limit:
        time_to_go = max_i(joint_displacement[i] / (joint_velocity_limit[i] / 8))
        (Note 1: The magic number 8 is deemed reasonable from hardware tests on a Franka Emika.)
        (Note 2: In a min-jerk trajectory, maximum velocity is equal to 1.875 * mean velocity.)
        The resulting time_to_go is also clipped to a minimum value of the default time_to_go.
        """
        joint_vel_limits = self.robot_model.get_joint_velocity_limits()
        joint_pos_diff = torch.abs(joint_displacement)
        time_to_go = torch.max(joint_pos_diff / joint_vel_limits * 8.0)
        return max(time_to_go, self.time_to_go_default)

    def _adaptive_time_to_go_full_path(self, joint_angle_seq: torch.Tensor):
        """Compute adaptive time_to_go
        Computes the corresponding time_to_go such that the mean velocity is equal to one-eighth
        of the joint velocity limit:
        time_to_go = max_i(joint_displacement[i] / (joint_velocity_limit[i] / 8))
        (Note 1: The magic number 8 is deemed reasonable from hardware tests on a Franka Emika.)
        (Note 2: In a min-jerk trajectory, maximum velocity is equal to 1.875 * mean velocity.)
        The resulting time_to_go is also clipped to a minimum value of the default time_to_go.
        """

        disp_list = []
        for i in range(joint_angle_seq.shape[0] - 1):
            disp = torch.abs(joint_angle_seq[i+1] - joint_angle_seq[i])
            disp_list.append(disp)
        
        waypoint_disps = torch.stack(disp_list, 0).reshape(-1, joint_angle_seq.shape[1])
        
        disp_total = torch.sum(waypoint_disps, dim=0)

        ttg_list = []
        for i in range(joint_angle_seq.shape[0] - 1):
            ttg = self._adaptive_time_to_go(
                joint_angle_seq[i+1] - joint_angle_seq[i]
            )
            ttg_list.append(ttg)
        
        # ttg = sum(ttg_list) / joint_angle_seq.shape[0]
        ttg = self._adaptive_time_to_go(disp_total)

        # print(f'Time to go: {ttg}')
        return ttg

    def _min_jerk_spaces(
        self, N: int, T: float
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generates a 1-dim minimum jerk trajectory from 0 to 1 in N steps & T seconds.
        Assumes zero velocity & acceleration at start & goal.
        The resulting trajectories can be scaled for different start & goals.
        Args:
            N: Length of resulting trajectory in steps
            T: Duration of resulting trajectory in seconds
        Returns:
            p_traj: Position trajectory of shape (N,)
            pd_traj: Velocity trajectory of shape (N,)
            pdd_traj: Acceleration trajectory of shape (N,)
        """
        assert N > 1, "Number of planning steps must be larger than 1."

        t_traj = torch.linspace(0, 1, N)
        p_traj = 10 * t_traj**3 - 15 * t_traj**4 + 6 * t_traj**5
        pd_traj = (30 * t_traj**2 - 60 * t_traj**3 + 30 * t_traj**4) / T
        pdd_traj = (60 * t_traj - 180 * t_traj**2 + 120 * t_traj**3) / (T**2)

        return p_traj, pd_traj, pdd_traj

    def _compute_num_steps(self, time_to_go: float, hz: float):
        return int(time_to_go * hz)
    
    def _foh(self, values, des_n_values):
        from IPython import embed; embed()
        # assume even spacing
        in_n_values = values.shape[0]
        n_values_per_step = des_n_values / in_n_values

        out_values = []
        for i in range(values.shape[0] - 1):
            vali = values[i]
            valip = values[i+1]

            interp = torch.linspace(vali.item(), valip.item(), n_values_per_step)
            # interp = torch.linspace(vali, valip)
            out_values.append(interp)
        out_values = torch.cat(out_values)
        return out_values
    
    def generate_path_waypoints(self, position_path: torch.Tensor, 
                                time_to_go: float = None, time_scaling: float = 1.0,
                                debug: bool = False) -> List[Dict]:
        """
        Primitive joint space minimum jerk trajectory planner.
        Assumes zero velocity & acceleration at start & goal.
        Args:
            start: Start joint position of shape (N,)
            goal: Goal joint position of shape (N,)
            time_to_go: Trajectory duration in seconds
            hz: Frequency of output trajectory
        Returns:
            waypoints: List of waypoints
        """
        if time_to_go is None:
            time_to_go = self._adaptive_time_to_go_full_path(position_path) / time_scaling
        hz = self.robot.hz

        steps = self._compute_num_steps(time_to_go, hz)
        dt = 1.0 / hz

        p_traj, pd_traj, pdd_traj = self._min_jerk_spaces(steps, time_to_go)

        # get sequence of displacements
        disp_list = []
        disp_total_list = []
        for i in range(position_path.shape[0] - 1):
            # disp = position_path[i+1] - position_path[i]
            disp = torch.abs(position_path[i+1] - position_path[i])
            disp_list.append(disp)

            disp_total_list.append(torch.mean(disp))
            # disp_total_list.append(torch.sum(disp))
        
        waypoint_disps = torch.stack(disp_list, 0).reshape(-1, position_path.shape[1])
        
        # get total displacement, and displacement fractions
        disp_total = torch.sum(waypoint_disps, dim=0)
        disp_fracs = waypoint_disps / disp_total.reshape(1, position_path.shape[1]).repeat((waypoint_disps.shape[0], 1))
        disp_fracs_single = disp_fracs[:, torch.argmax(disp_total)]
        # disp_fracs_single = disp_fracs[:, 0]
        disp_fracs_cumul = torch.cumsum(disp_fracs_single, 0)

        waypoint_total_disps = torch.stack(disp_total_list, 0).reshape(-1, 1)
        disp_total_total = torch.sum(waypoint_total_disps, dim=0).reshape(-1, 1)
        disp_total_fracs = waypoint_total_disps / disp_total_total
        disp_total_fracs_cumul = torch.cumsum(disp_total_fracs, 0).reshape(-1)

        disp_fracs_cumul = disp_total_fracs_cumul

        q_traj_list = []
        qd_traj_list = []
        qdd_traj_list = []
        ind_list = []
        traj_ind_start = 0
        for i in range(waypoint_disps.shape[0]):
            start = position_path[i]
            goal = position_path[i+1]
            D = goal - start

            if i == (waypoint_disps.shape[0] - 1):
                traj_ind_end = p_traj.shape[0]
            else:
                traj_ind_end = torch.argmin(torch.linalg.norm((disp_fracs_cumul[i] - p_traj).reshape(1, -1), dim=0))

            if traj_ind_start > traj_ind_end:
                print(f'here in generate waypoints with start > end')
                from IPython import embed; embed()
                raise RuntimeError('Start index cannot be greater than end index')
                return

            if traj_ind_start == traj_ind_end:
                traj_ind_start = traj_ind_end
                continue
            # print(f'Start: {traj_ind_start}, End: {traj_ind_end}, Total: {p_traj.shape[0]}')
            pi = p_traj[traj_ind_start:traj_ind_end, None]
            pdi = pd_traj[traj_ind_start:traj_ind_end, None]
            pddi = pdd_traj[traj_ind_start:traj_ind_end, None]

            try:
                scale = 1 / (pi.max() - pi.min())
            except RuntimeError as e:
                print(f'here in generate waypoints with runtime error')
                print(f'Error: {e}')
                from IPython import embed; embed()
                raise RuntimeError(e)
                return 

            if not pi.max() == pi.min(): # ow scale is inf
                qd_traj = D[None, :] * pdi * scale
                qdd_traj = D[None, :] * pddi * scale
            else:
                qd_traj = D[None, :] * pdi
                qdd_traj = D[None, :] * pddi

            deltas = qd_traj * dt 
            q_traj = start + torch.cumsum(deltas, dim=0)

            q_traj_list.append(q_traj)
            qd_traj_list.append(qd_traj)
            qdd_traj_list.append(qdd_traj)

            ind_list.append((traj_ind_start, traj_ind_end))
            traj_ind_start = traj_ind_end

        # if True:
        if False:
            fig = plt.figure() 
            for i in range(len(q_traj_list)):
                # plt.plot(np.arange(q_traj_list[i].shape[0]), q_traj_list[i][:, 3])
                plt.plot(np.arange(ind_list[i][0], ind_list[i][1]), q_traj_list[i][:, 3])
            
            plt.show()
        
        q_traj = torch.cat(q_traj_list)
        qd_traj = torch.cat(qd_traj_list)
        qdd_traj = torch.cat(qdd_traj_list)

        waypoints = [
            {
                "time_from_start": i * dt,
                "position": q_traj[i, :],
                "velocity": qd_traj[i, :],
                "acceleration": qdd_traj[i, :],
            }
            for i in range(steps)
        ]

        if False:
        # if True:
            import matplotlib.pyplot as plt
            import numpy as np
            pos = torch.stack([val['position'] for val in waypoints], dim=0)
            vel = torch.stack([val['velocity'] for val in waypoints], dim=0)

            fig, axs = plt.subplots(1, 2)
            plt_idx = 0
            # plt.plot(np.arange(pos.shape[0]), pos[:, 3])
            # plt.plot(np.arange(pos.shape[0]), vel[:, 3])
            axs[0].plot(np.arange(pos.shape[0]), pos[:, plt_idx])
            axs[1].plot(np.arange(pos.shape[0]), vel[:, plt_idx])
            plt.show()

            from IPython import embed; embed()
        
        if debug:
            from IPython import embed; embed()


        return waypoints
    
    # def move_to_joint_positions(
    def execute_position_path(
        self,
        position_path: torch.Tensor,
        time_to_go: float = None,
        delta: bool = False,
        Kq: torch.Tensor = None,
        Kqd: torch.Tensor = None,
        time_scaling: float = 1.0,
        **kwargs,
    ) -> List[RobotState]:

        waypoints = self.generate_path_waypoints(position_path, time_to_go=time_to_go, time_scaling=time_scaling)

        current_jpos = self.robot.get_joint_positions()
        first_jpos = waypoints[0]['position']
        delta = torch.abs(current_jpos - first_jpos)
        if (delta > self.start_tolerance).any():
            print(f'Start state: {current_jpos} too far from beginning of path: {first_jpos}')
            print(f'Please ensure start state matches beginning of trajectory')
            print(f'Exiting')
            return -1

        # Create & execute policy
        torch_policy = toco.policies.JointTrajectoryExecutor(
            joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
            joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
            Kq=self.Kq_default if Kq is None else Kq,
            Kqd=self.Kqd_default if Kqd is None else Kqd,
            Kx=self.Kx_default,
            Kxd=self.Kxd_default,
            robot_model=self.robot_model,
            ignore_gravity=self.use_grav_comp,
        )

        # torch_policy = toco.policies.JointTrajectoryExecutor(
        #     joint_pos_trajectory=[waypoint["position"] for waypoint in waypoints],
        #     joint_vel_trajectory=[waypoint["velocity"] for waypoint in waypoints],
        #     # Kp=self.Kq_default if Kq is None else Kq,
        #     # Kd=self.Kqd_default if Kqd is None else Kqd,
        #     robot_model=self.robot_model,
        #     ignore_gravity=self.use_grav_comp,
        # )

        return self.robot.send_torch_policy(torch_policy=torch_policy, **kwargs)
    
    def diffik_traj(self, ee_pose_des_traj, total_time=5.0, precompute=True, execute=False, 
                    start_ee_pose_mat=None, start_joint_pos=None):

        if start_ee_pose_mat is None:
            # get current ee pose
            current_ee_pose = self.robot.get_ee_pose()
            current_ee_pose_mat = self.polypose2mat(current_ee_pose)
        else:
            current_ee_pose_mat = start_ee_pose_mat

        # using desired ee pose, compute desired ee velocity
        ee_pose_mat_traj = np.asarray([current_ee_pose_mat] + ee_pose_des_traj).reshape(-1, 4, 4)
        ee_pos_traj = ee_pose_mat_traj[:, :-1, -1].reshape(-1, 3)
        dt = total_time / ee_pos_traj.shape[0]

        # get orientations represented as axis angles, and separate into angles and unit-length axes
        ee_rot_traj = R.from_matrix(ee_pose_mat_traj[:, :-1, :-1])

        # get translation velocities with finite difference
        ee_trans_vel_traj = (ee_pos_traj[1:] - ee_pos_traj[:-1]) / dt
        ee_trans_vel_traj = np.concatenate((ee_trans_vel_traj, np.array([[0.0, 0.0, 0.0]])), axis=0)

        ee_rot_traj_inv = ee_rot_traj.inv()
        ee_delta_rot_traj = (ee_rot_traj[1:] * ee_rot_traj_inv[:-1])
        ee_delta_axis_angle_traj = ee_delta_rot_traj.as_rotvec()
        ee_rot_vel_traj = ee_delta_axis_angle_traj / dt
        ee_rot_vel_traj = np.concatenate((ee_rot_vel_traj, np.array([[0.0, 0.0, 0.0]])), axis=0)

        # combine into trajectory of spatial velocities
        ee_des_spatial_vel_traj = np.concatenate((ee_trans_vel_traj, ee_rot_vel_traj), axis=1)
        ee_velocity_desired = torch.from_numpy(ee_des_spatial_vel_traj).float()

        if start_joint_pos is None:
            # get current configuration and compute target joint velocity
            current_joint_pos = self.robot.get_joint_positions()
            current_joint_vel = self.robot.get_joint_velocities()
            print(f'Joint velocity is {current_joint_vel}')
        else:
            print(f'Starting from joint angles: {start_joint_pos}')
            current_joint_pos = torch.Tensor(start_joint_pos)

        if precompute:
            # in a loop, compute target joint velocity, integrate to get next joint position, and repeat 
            joint_pos_traj = []
            joint_vel_traj = []
            for t in range(ee_velocity_desired.shape[0]):
                jacobian = self.robot_model.compute_jacobian(current_joint_pos) 
                joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired[t]).solution
                joint_pos_desired = current_joint_pos + joint_vel_desired*dt

                joint_pos_traj.append(joint_pos_desired)
                joint_vel_traj.append(joint_vel_desired.detach().cpu().numpy())

                current_joint_pos = joint_pos_desired.clone()

            joint_pos_desired = torch.stack(joint_pos_traj, dim=0)
            joint_vel_desired = np.stack(joint_vel_traj)
            max_idx = np.argmax(np.max(joint_vel_desired, axis=1))
            print(f'Max velocity is {joint_vel_desired[max_idx]}')
            if np.max(np.absolute(joint_vel_desired[max_idx])) > 2.7:
                print(f'Max velocity is over our limit (2.7), returning None')
                return

            if execute:
                self.execute_position_path(joint_pos_desired)

            return joint_pos_desired
        else:
            if not execute:
                print(f'[Trajectory Util DiffIK] "Execute" must be True when running with precompute=False')
                print(f'[Trajectory Util DiffIK] Exiting')
                return 
            
            # print('here before executing')
            # from IPython import embed; embed()

            # print(f'\n\n\n')
            # print(f'[Trajectory Util DiffIK] Executing...')
            # print(f'\n\n\n')

            # in a loop, compute target joint velocity, integrate to get next joint position, and repeat 
            vel_lp_alpha = 0.1
            vel_ramp_down_alpha = 0.9
            ramp_down_coef = 1.0

            pdt_ = self._min_jerk_spaces(ee_pose_mat_traj.shape[0], total_time)[1]
            pdt = pdt_ / pdt_.max()

            joint_pos_traj = []
            # for t in range(ee_pose_mat_traj.shape[0]):
            for t_idx in range(ee_pose_mat_traj.shape[0]):
                
                t = t_idx + self.diffik_lookahead
                if t >= (ee_pose_mat_traj.shape[0] - 1):
                    t = ee_pose_mat_traj.shape[0] - 1

                # compute velocity needed to get to next desired pose, from current pose
                current_ee_pose = self.robot.get_ee_pose()
                current_ee_pose_mat = self.polypose2mat(current_ee_pose)

                # get current
                current_ee_pos = current_ee_pose_mat[:-1, -1].reshape(1, 3)
                current_ee_ori_mat = current_ee_pose_mat[:-1, :-1]
                current_ee_rot = R.from_matrix(current_ee_ori_mat)

                # get desired
                ee_pos_des = ee_pos_traj[t].reshape(1, 3)
                ee_rot_des = ee_rot_traj[t]

                # compute desired rot as delta_rot, in form of axis angle
                delta_rot = (ee_rot_des * current_ee_rot.inv())
                delta_axis_angle = delta_rot.as_rotvec().reshape(1, 3)

                # stack into desired spatial vel
                trans_vel_des = (ee_pos_des - current_ee_pos) / dt
                rot_vel_des = (delta_axis_angle) / dt
                ee_velocity_desired = torch.from_numpy(
                    np.concatenate((trans_vel_des, rot_vel_des), axis=1).squeeze()).float()

                # solve J.pinv() @ ee_vel_des
                jacobian = self.robot_model.compute_jacobian(current_joint_pos) 
                joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired).solution
                joint_pos_desired = current_joint_pos + joint_vel_desired*dt

                # joint_vel_filtered = joint_vel_desired
                # joint_vel_filtered = vel_lp_alpha * joint_vel_desired + (1 - vel_lp_alpha) * current_joint_vel

                # if t > (ee_pose_mat_traj.shape[0] * 0.9):
                #     ramp_down_coef = ramp_down_coef * vel_ramp_down_alpha 
                #     joint_vel_filtered = ramp_down_coef * joint_vel_filtered
                
                # joint_vel_filtered = pdt[t] * joint_vel_desired
                # joint_vel_filtered = torch.clip(joint_vel_filtered, min=-1.0*self.joint_vel_limits, max=self.joint_vel_limits)
                # print(f'Joint velocity filtered: {joint_vel_filtered}')

                # joint_pos_desired = current_joint_pos + joint_vel_filtered*dt

                # send joint angle command
                self.robot.update_desired_joint_positions(joint_pos_desired)

                # try:
                #     self.robot.update_current_policy(
                #         {
                #             "joint_pos_desired": joint_pos_desired, 
                #             "joint_vel_desired": joint_vel_desired
                #         }
                #     )
                # except grpc.RpcError as e:
                #     print(
                #         "Unable to update desired joint positions. Use 'start_joint_impedance' to start a joint impedance controller."
                #     )
                #     raise e

                # time.sleep(dt)
                # time.sleep(dt * self.diffik_sleep_lag)
                current_joint_pos = self.robot.get_joint_positions()
                current_joint_vel = self.robot.get_joint_velocities()

            return None

    def diffik_step(self, ee_pose_des):
        # get current ee pose
        current_ee_pose = self.robot.get_ee_pose()
        current_ee_pose_mat = self.polypose2mat(current_ee_pose)

        # using desired ee pose, compute desired ee velocity
        ee_pose_mat_traj = np.asarray([current_ee_pose_mat] + ee_pose_des_traj).reshape(-1, 4, 4)
        ee_pos_traj = ee_pose_mat_traj[:, :-1, -1].reshape(-1, 3)
        dt = len(ee_pos_traj) / total_time

        # get translation velocities with finite difference
        ee_trans_vel_traj = (ee_pos_traj[1:] - ee_pos_traj[:-1]) / dt
        ee_trans_vel_traj = np.concatenate((ee_trans_vel_traj, np.array([[0.0, 0.0, 0.0]])), axis=0)

        # get orientations represented as axis angles, and separate into angles and unit-length axes
        ee_axis_angle_traj = R.from_matrix(ee_pose_mat_traj[:, :-1, :-1]).as_rotvec().reshape(-1, 3)
        ee_aa_theta_traj = np.linalg.norm(ee_axis_angle_traj, axis=-1)
        ee_aa_axis_traj = ee_axis_angle_traj / ee_aa_theta_traj.reshape(-1, 1)

        # get angular velocities with finite difference
        ee_aa_thetadot_traj = (ee_aa_theta_traj[1:] - ee_aa_theta_traj[:-1]) / dt
        ee_aa_thetadot_traj = np.concatenate((ee_aa_thetadot_traj, np.array([0.0])))
        ee_rot_vel_traj = ee_aa_axis_traj * ee_aa_thetadot_traj.reshape(-1, 1)

        # combine into trajectory of spatial velocities
        ee_des_spatial_vel_traj = np.concatenate((ee_trans_vel_traj, ee_rot_vel_traj), axis=1)
        ee_velocity_desired = torch.from_numpy(ee_des_spatial_vel_traj).float()

        # get current configuration and compute target joint velocity
        current_joint_pos = self.robot.get_joint_positions()

        jacobian = self.robot_model.compute_jacobian(current_joint_pos) 
        joint_vel_desired = torch.linalg.lstsq(jacobian, ee_velocity_desired[t]).solution
        joint_pos_desired = current_joint_pos + joint_vel_desired*dt

        self.robot.update_desired_joint_positions(joint_pos_desired)

    def start_joint_impedance_with_velocity(self, Kq=None, Kqd=None, **kwargs):
        """Starts joint position control mode.
        Runs an non-blocking joint impedance controller.
        The desired joint positions can be updated using `update_desired_joint_positions`
        """ 
        torch_policy = JointImpedanceControlWithVelocity(
                joint_pos_current=self.robot.get_joint_positions(),
                Kp=self.Kq_default if Kq is None else Kq,
                Kd=self.Kqd_default if Kqd is None else Kqd,
                robot_model=self.robot_model,
                ignore_gravity=self.use_grav_comp,
            )

        return self.robot.send_torch_policy(torch_policy=torch_policy, blocking=False)


class JointImpedanceControlWithVelocity(toco.PolicyModule):
    """
    Impedance control in joint space.
    """

    def __init__(
        self,
        joint_pos_current,
        Kp,
        Kd,
        robot_model: torch.nn.Module,
        ignore_gravity=True,
    ):
        """
        Args:
            joint_pos_current: Current joint positions
            Kp: P gains in joint space
            Kd: D gains in joint space
            robot_model: A robot model from torchcontrol.models
            ignore_gravity: `True` if the robot is already gravity compensated, `False` otherwise
        """
        super().__init__()

        # Initialize modules
        self.robot_model = robot_model
        self.invdyn = toco.modules.feedforward.InverseDynamics(
            self.robot_model, ignore_gravity=ignore_gravity
        )
        self.joint_pd = toco.modules.feedback.JointSpacePD(Kp, Kd)

        # Reference pose
        self.joint_pos_desired = torch.nn.Parameter(to_tensor(joint_pos_current))
        self.joint_vel_desired = torch.nn.Parameter(torch.zeros_like(self.joint_pos_desired))
        # self.joint_vel_desired = torch.zeros_like(self.joint_pos_desired)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            state_dict: A dictionary containing robot states
        Returns:
            A dictionary containing the controller output
        """
        # State extraction
        joint_pos_current = state_dict["joint_positions"]
        joint_vel_current = state_dict["joint_velocities"]

        # Control logic
        torque_feedback = self.joint_pd(
            joint_pos_current,
            joint_vel_current,
            self.joint_pos_desired,
            self.joint_vel_desired,
        )
        torque_feedforward = self.invdyn(
            joint_pos_current, joint_vel_current, torch.zeros_like(joint_pos_current)
        )  # coriolis
        torque_out = torque_feedback + torque_feedforward

        return {"joint_torques": torque_out}