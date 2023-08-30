import os, os.path as osp
import copy
import time
import trimesh
import numpy as np
import torch
import pybullet as p
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from rdt.common import util
from rdt.common.franka_ik import PbPlUtils
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from airobot import log_debug, log_warn, log_info

poly_util = PolymetisHelper()
pb_pl = PbPlUtils()


class PlanningHelper(object):
    def __init__(self, mc_vis, robot, gripper, 
                 ik_helper, traj_helper, tmp_obstacle_dir,
                 gripper_type='panda'):
        self.mc_vis = mc_vis
        self.robot = robot
        self.gripper = gripper
        self.ik_helper = ik_helper
        self.traj_helper = traj_helper

        self.tmp_obstacle_dir = tmp_obstacle_dir

        # self.loop_delay = 0.1
        self.loop_delay = 0.025
        self.cart_loop_delay = 0.01
        self.occnet_check_thresh = 0.5
        self.waypoint_jnts = np.array([-0.1329, -0.0262, -0.0448, -1.60,  0.0632,  1.9965, -0.8882])
        self.waypoint_pose = poly_util.polypose2mat(self.robot.robot_model.forward_kinematics(torch.from_numpy(self.waypoint_jnts)))
        # self.waypoint_jnts = self.robot.home_pose.numpy()
        self.max_planning_time = 5.0
        self.planning_alg = 'rrt_star'
        # self.planning_alg = 'birrt'

        self.attach_ik_obj_id = None 

        self.gripper_close_pos = 0.0
        self.gripper_open_pos = 0.078 if gripper_type == 'panda' else 0.08
        self.gripper_speed, self.gripper_force = 0.5, 10.0

        self._setup()

    def set_loop_delay(self, delay):
        self.loop_delay = delay

    def set_max_planning_time(self, max_time):
        self.max_planning_time = max_time

    def set_planning_alg(self, alg):
        self.planning_alg = alg
    
    def set_occnet_thresh(self, thresh=0.5):
        self.occnet_check_thresh = thresh
    
    def set_gripper_speed(self, speed):
        self.gripper_speed = speed
    
    def set_gripper_force(self, force):
        self.gripper_force = force

    def set_gripper_open_pos(self, pos):
        self.gripper_open_pos = pos
    
    def set_gripper_close_pos(self, pos):
        self.gripper_close_pos = pos
    
    def gripper_open(self, speed=None, force=None):
        if speed is None:
            speed = self.gripper_speed
        if force is None:
            force = self.gripper_force

        self.gripper.goto(self.gripper_open_pos, speed, force)

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
    
    def _setup(self):
        self.robot.start_joint_impedance()
    
    def execute_pb_loop(self, jnt_list):
        if jnt_list is None:
            print(f'Cannot execute in pybullet, jnt_list is None')
            return
        for jidx, jnt in enumerate(jnt_list):
            self.ik_helper.set_jpos(jnt)
            time.sleep(self.loop_delay)
        
    def check_states_close(self, state1, state2, tol=np.deg2rad(1)):
        dist = np.abs(state1 - state2)

        close = (dist < tol).all()

        if not close:
            print(f'States: {state1} and {state2} not close!')
            for i in range(dist.shape[0]):
                print(f'Joint angle: {i}, distance: {dist[i]}')
        return close

    def check_pose_mats_close(self, pose_mat1, pose_mat2, pos_tol=0.025, rot_tol=0.4, check_pos=True, check_rot=True):
        pos_dist = np.abs(pose_mat1[:-1, -1] - pose_mat2[:-1, -1])
        rot_mult = R.from_matrix(pose_mat1[:-1, :-1]) * R.from_matrix(pose_mat2[:-1, :-1]).inv()
        rot_dist = np.linalg.norm((np.eye(3) - (rot_mult.as_matrix())), ord='fro')
        # rot_dist = 1 - np.linalg.norm(rot_mult.as_matrix(), ord='fro')

        log_debug(f'[Check pose mats close] Position distance: {pos_dist}')
        log_debug(f'[Check pose mats close] Rotation distance: {rot_dist}')

        pos_close = (pos_dist < pos_tol).all()
        rot_close = rot_dist < rot_tol

        close = pos_close and rot_close

        if check_pos and (not pos_close):
            print(f'Position: {pose_mat1[:-1, -1]} and {pose_mat2[:-1, -1]} not close!')
        if check_rot and (not rot_close):
            print(f'Rotation: {pose_mat1[:-1, :-1]} and {pose_mat2[:-1, :-1]} not close!')
        
        if check_pos and check_rot:
            return close
        elif check_pos:
            return pos_close
        elif check_rot:
            return rot_close
        else:
            return True
    
    def execute_loop_jnt_impedance(self, jnt_list):
        current_jnts = self.robot.get_joint_positions().numpy()
        if not self.check_states_close(np.asarray(jnt_list[0]), current_jnts):
            print(f'Current joints: {current_jnts} too far from start of loop: {np.asarray(jnt_list[0])}')
            print(f'Exiting')
            return

        joint_pos_arr = torch.Tensor(jnt_list)
        for idx in range(joint_pos_arr.shape[0]):
            new_jp = joint_pos_arr[idx]
            self.robot.update_desired_joint_positions(new_jp)
            time.sleep(self.loop_delay)

    def execute_loop_cart_impedance(self, ee_pose_mat_list, total_time=5.0, check_start=True):
        if not check_start:
            log_warn(f'[Execute loop cartesian impedance] !! NOT CHECKING DISTANCE FROM START POSE !!')
        else:
            current_pose = self.robot.get_ee_pose()
            current_pose_mat = poly_util.polypose2mat(current_pose)
            if not self.check_pose_mats_close(current_pose_mat, ee_pose_mat_list[0]):
                print(f'Current pose: {current_pose_mat} too far from start of loop: {ee_pose_mat_list[0]}')
                print(f'Exiting')
                return

        cart_loop_delay = total_time * 1.0 / len(ee_pose_mat_list)
        ee_pose_list_list = [util.pose_stamped2list(util.pose_from_matrix(val)) for val in ee_pose_mat_list]
        ee_pose_arr = torch.Tensor(ee_pose_list_list)
        for idx in range(ee_pose_arr.shape[0]):
            new_ee_pose = ee_pose_arr[idx]
            # robot.update_desired_ee_pose(new_ee_pose[:3], new_ee_pose[3:])
            self.robot.update_desired_ee_pose(new_ee_pose[:3], new_ee_pose[3:])
            time.sleep(cart_loop_delay)

    def execute_loop(self, jnt_list, time_to_go=None, time_scaling=1.0):
        if jnt_list is None:
            print(f'Trajectory is None')
            return

        joint_pos_arr = torch.Tensor(jnt_list)
        self.traj_helper.execute_position_path(joint_pos_arr, time_to_go=time_to_go, time_scaling=time_scaling)
    
    def get_diffik_traj(self, pose_mat_des, from_current=True, pose_mat_start=None, N=500, show_frames=True, show_frame_name='interp_poses'):
        assert from_current or (pose_mat_start is not None), 'Cannot have from_current=False and pose_mat_start=None!'

        if from_current:
            # current -> place place
            current_pose = self.robot.get_ee_pose()
            current_pose_mat = poly_util.polypose2mat(current_pose)

            if show_frames:
                util.meshcat_frame_show(self.mc_vis, 'scene/poses/current', current_pose_mat)

            pose_mat_start = current_pose_mat

        interp_pose_mats = self.interpolate_pose(pose_mat_start, pose_mat_des, N)
        if show_frames:
            for i in range(len(interp_pose_mats)):
                if i % (len(interp_pose_mats) * 0.05) == 0:
                    util.meshcat_frame_show(self.mc_vis, f'scene/poses/{show_frame_name}/{i}', interp_pose_mats[i])
        
        return interp_pose_mats

    def feasible_diffik_joint_traj(self, pose_mat_des=None, from_current=True, pose_mat_start=None, start_joint_pos=None, 
                                   N=500, show_frames=True, show_frame_name='interp_poses', pose_mat_list=None, coll_pcd=None, coll_pcd_thresh=0.5, 
                                   total_time=2.5, check_first=True, show_pb=False, return_mat_list=False):

        assert (pose_mat_des is not None) or (pose_mat_list is not None), 'Must either provide "pose_mat_des" or "pose_mat_list"!'
        if (pose_mat_des is not None) and (pose_mat_list is None):
            pose_mat_list = self.get_diffik_traj(
                pose_mat_des, 
                from_current=from_current, 
                pose_mat_start=pose_mat_start, 
                N=N,
                show_frames=show_frames,
                show_frame_name=show_frame_name)

        valid_ik = True 

        joint_traj = self.traj_helper.diffik_traj(
            pose_mat_list, precompute=True, execute=False, total_time=total_time, 
            start_ee_pose_mat=pose_mat_start, start_joint_pos=start_joint_pos)
        if joint_traj is None:
            print(f'DiffIK not valid, singularity reached!')
            return [], False
        joint_traj_list = [val.numpy().tolist() for val in joint_traj]

        if len(joint_traj_list) > 50:
            joint_traj_list = joint_traj_list[::int(len(joint_traj_list) * 1.0 / 50)]

        if show_pb:
            self.execute_pb_loop(joint_traj_list)

        if check_first:
            for jnt_val in joint_traj_list:
                self.ik_helper.set_jpos(jnt_val)
                # in_collision = self.ik_helper.check_collision(pcd=coll_pcd, thresh=coll_pcd_thresh)[0]
                in_collision, in_coll_type = self.ik_helper.check_collision()
                valid_ik = valid_ik and (not in_collision)
                if not valid_ik:
                    print(f'DiffIK not valid!')
                    print(f'In Collision: {in_collision}, {in_coll_type}')
                    break

        if return_mat_list:
            return joint_traj_list, valid_ik, pose_mat_list
        return joint_traj_list, valid_ik
        
    def feasible_diffik_traj(self, pose_mat_des=None, from_current=True, pose_mat_start=None, N=500, show_frames=True, show_frame_name='interp_poses', 
                             pose_mat_list=None, total_time=2.5, check_first=True, show_pb=False, return_mat_list=False):

        assert (pose_mat_des is not None) or (pose_mat_list is not None), 'Must either provide "pose_mat_des" or "pose_mat_list"!'
        if (pose_mat_des is not None) and (pose_mat_list is None):
            pose_mat_list = self.get_diffik_traj(
                pose_mat_des, 
                from_current=from_current, 
                pose_mat_start=pose_mat_start, 
                N=N,
                show_frames=show_frames,
                show_frame_name=show_frame_name)

        valid_ik = True 
        if check_first:
            joint_traj = self.traj_helper.diffik_traj(pose_mat_list, precompute=True, execute=False, total_time=total_time)
            joint_traj_list = [val.numpy().tolist() for val in joint_traj]

            if show_pb:
                self.execute_pb_loop(joint_traj_list)
            for jnt_val in joint_traj_list:
                self.ik_helper.set_jpos(jnt_val)
                in_collision = self.ik_helper.check_collision()[0]
                valid_ik = valid_ik and (not in_collision)
                if not valid_ik:
                    break
        
        if valid_ik:
            self.traj_helper.diffik_traj(pose_mat_list, precompute=False, execute=True, total_time=total_time)
        else:
            print(f'DiffIK not feasible')

        if return_mat_list:
            return valid_ik, pose_mat_list        
        return valid_ik

    def get_waypoints_loop(self, jnt_list, time_to_go=None, time_scaling=1.0):        
        joint_pos_arr = torch.Tensor(jnt_list)
        waypoints = self.traj_helper.generate_path_waypoints(
            joint_pos_arr, 
            time_to_go=time_to_go, 
            time_scaling=time_scaling) 
        return waypoints

    def get_offset_poses(self, grasp_pose_mat, grasp_offset_dist=0.075):
        grasp_offset_mat = np.eye(4); grasp_offset_mat[2, -1] = -1.0*grasp_offset_dist
        offset_grasp_pose_mat = np.matmul(grasp_pose_mat, grasp_offset_mat)

        above_grasp_pose_mat = grasp_pose_mat.copy(); above_grasp_pose_mat[2, -1] += 0.15

        return offset_grasp_pose_mat, above_grasp_pose_mat
    
    def remove_all_attachments(self):
        self.ik_helper.clear_attachment_bodies()

    def remove_all_obstacles(self):
        self.ik_helper.clear_collision_bodies()
    
    def attach_obj_pcd(self, obj_pcd, grasp_pose_mat_world, 
                       obj_bb=True, name='attached_obj.obj'):
        if not obj_bb:
            raise NotImplementedError
        
        if not name.endswith('.obj'):
            name = name + '.obj'
        
        obj_bb = trimesh.PointCloud(obj_pcd).bounding_box_oriented.to_mesh()
        obj_bb_fname = osp.join(self.tmp_obstacle_dir, name)
        obj_bb.export(obj_bb_fname)

        obj_pose_ee = util.convert_reference_frame(
            pose_source=util.unit_pose(),
            pose_frame_target=util.pose_from_matrix(grasp_pose_mat_world),
            pose_frame_source=util.unit_pose()
        )
        obj_pose_ee_mat = util.matrix_from_pose(obj_pose_ee)

        obj_bb_pos = [0]*3
        obj_bb_ori = [0, 0, 0, 1]
        self.attach_ik_obj_id = pb_pl.load_pybullet(
            obj_bb_fname, 
            base_pos=obj_bb_pos,
            base_ori=obj_bb_ori,
            scale=1.0)
        p.resetBasePositionAndOrientation(
            self.attach_ik_obj_id, 
            obj_bb_pos, 
            obj_bb_ori, 
            physicsClientId=self.ik_helper.pb_client)

        self.ik_helper.add_attachment_bodies(
            parent_body=self.ik_helper.robot, 
            parent_link=self.ik_helper.tool_link, 
            grasp_pose_mat=obj_pose_ee_mat, 
            bodies={'target_obj': self.attach_ik_obj_id})
        
        return 

    def attach_obj(self, obj_mesh, grasp_pose_mat_world, name='attached_obj.obj'):
        
        if not name.endswith('.obj'):
            name = name + '.obj'
        
        obj_mesh_fname = osp.join(self.tmp_obstacle_dir, name)
        obj_mesh.export(obj_mesh_fname)

        obj_pose_ee = util.convert_reference_frame(
            pose_source=util.unit_pose(),
            pose_frame_target=util.pose_from_matrix(grasp_pose_mat_world),
            pose_frame_source=util.unit_pose()
        )
        obj_pose_ee_mat = util.matrix_from_pose(obj_pose_ee)

        obj_bb_pos = [0]*3
        obj_bb_ori = [0, 0, 0, 1]
        self.attach_ik_obj_id = pb_pl.load_pybullet(
            obj_mesh_fname, 
            base_pos=obj_bb_pos,
            base_ori=obj_bb_ori,
            scale=1.0)
        p.resetBasePositionAndOrientation(
            self.attach_ik_obj_id, 
            obj_bb_pos, 
            obj_bb_ori, 
            physicsClientId=self.ik_helper.pb_client)

        self.ik_helper.add_attachment_bodies(
            parent_body=self.ik_helper.robot, 
            parent_link=self.ik_helper.tool_link, 
            grasp_pose_mat=obj_pose_ee_mat, 
            bodies={'target_obj': self.attach_ik_obj_id})
        
        return 
    
    def interpolate_joints(self, plan, n_pts=None, des_step_dist=np.deg2rad(2.5)): #7.5)):
        """
        Densely interpolate a plan that was obtained from the motion planner
        """
        if plan is None:
            return None

        plan_np = np.asarray(plan) 

        try:
            if plan_np.shape[0] > 0:
                pass
        except IndexError as e:
            print(f'[Interpolate Joints] Exception: {e}')

        # print("here in interp")
        # ; 
        # if len(plan_np) > 0:
        if plan_np.shape[0] > 0:
            if n_pts is None:
                # rough heuristic of making sure every joint doesn't take a step larger than 0.1 radians per waypoint

                max_step_dist = 0.0
                # for i in range(len(plan) - 1):
                for i in range(plan_np.shape[0] - 1):
                    step_ = plan_np[i]
                    step_next = plan_np[i+1]

                    dists = np.abs(step_ - step_next)

                    max_dist = np.max(dists)

                    if max_dist > max_step_dist:
                        max_step_dist = max_dist

                n_pts_per_step = np.ceil(max_step_dist / des_step_dist)
                # n_pts = int(len(plan) * n_pts_per_step)
                n_pts = int(plan_np.shape[0] * n_pts_per_step)

                interp_info_str = f'Got max dist: {max_step_dist}. '
                interp_info_str += f'Going to make sure each step is interpolated to {n_pts_per_step} points, '
                interp_info_str += f'giving total of {n_pts} points'
                print(f'{interp_info_str}')
            else:
                # n_pts_per_step = int(np.ceil(n_pts / len(plan) * 1.0))
                n_pts_per_step = int(np.ceil(n_pts / plan_np.shape[0] * 1.0))
            
            n_pts_per_step = int(n_pts_per_step)
            print('n_pts_per_step', n_pts_per_step)
            # new_plan = np.asarray(plan_list[0]) 
            new_plan_np = plan_np[0]
            # for i in range(len(plan) - 1):
            for i in range(plan_np.shape[0] - 1):
                step_ = plan_np[i]
                step_next = plan_np[i+1]
                
                print('step_', step_)
                print('step_next', step_next)
                interp = np.linspace(step_, step_next, n_pts_per_step)
                # new_plan.extend(interp.tolist())
                new_plan_np = np.vstack((new_plan_np, interp))
            
            print(f'New plan shape: ({new_plan_np.shape[0]}, {new_plan_np.shape[1]})')
            new_plan = []
            for i in range(new_plan_np.shape[0]):
                new_plan.append(new_plan_np[i].tolist())

            return new_plan
        else:
            return []
    
    def interpolate_plan_full(self, plan_dict):
        out_plan_dict = {}
        for k, v in plan_dict.items():
            # out_plan_dict[k] = self.interpolate_joints(np.asarray(v))
            out_plan_dict[k] = self.interpolate_joints(v)
        
        return out_plan_dict
    
    def interpolate_pose(self, pose_mat_initial, pose_mat_final, N):
        """
        Function to interpolate between two poses using a combination of
        linear position interpolation and quaternion spherical-linear
        interpolation (SLERP)

        Args:
            pose_initial (PoseStamped): Initial pose
            pose_final (PoseStamped): Final pose
            N (int): Number of intermediate points.

        Returns:
            list: List of poses that interpolates between initial and final pose.
                Each element is PoseStamped. 
        """
        trans_initial = pose_mat_initial[:-1, -1]
        quat_initial = R.from_matrix(pose_mat_initial[:-1, :-1]).as_quat()

        trans_final = pose_mat_final[:-1, -1]
        quat_final = R.from_matrix(pose_mat_final[:-1, :-1]).as_quat()

        trans_interp_total = [np.linspace(trans_initial[0], trans_final[0], num=N),
                            np.linspace(trans_initial[1], trans_final[1], num=N),
                            np.linspace(trans_initial[2], trans_final[2], num=N)]
        
        key_rots = R.from_quat([quat_initial, quat_final])
        slerp = Slerp(np.arange(2), key_rots)
        interp_rots = slerp(np.linspace(0, 1, N))
        quat_interp_total = interp_rots.as_quat()    

        pose_mat_interp = []
        for counter in range(N):
            pose_tmp = [
                trans_interp_total[0][counter],
                trans_interp_total[1][counter],
                trans_interp_total[2][counter],
                quat_interp_total[counter][0], 
                quat_interp_total[counter][1],
                quat_interp_total[counter][2],
                quat_interp_total[counter][3],
            ]
            pose_mat_interp.append(util.matrix_from_list(pose_tmp))
        return pose_mat_interp
    
    def go_home_plan(self):
        current_jnts = self.robot.get_joint_positions().numpy()
        current_to_home = self.ik_helper.plan_joint_motion(
            current_jnts, 
            self.robot.home_pose.numpy(),
            alg=self.planning_alg)
        
        if current_to_home is not None:
            while True:
                self.execute_pb_loop(current_to_home)
                input_value3 = input('\nWould you like to execute? "y" for Yes, else for No\n')
                if input_value3 == 'y':
                    self.execute_loop(current_to_home)
                    return
                elif input_value3 == 'n':
                    print('Exiting...')
                    return
                else:
                    pass
                continue
        else:
            print(f'Could not plan path from current to home. Exiting')
            return 
    
    def plan_home(self, from_current=True, start_position=None, show_pb=True, execute=False):
        home_plan = self.plan_joint_target(
            self.robot.home_pose.numpy(),
            from_current=from_current,
            start_position=start_position,
            show_pb=show_pb,
            execute=execute
            )
        return home_plan
    
    def plan_joint_target(self, joint_position_desired, from_current=True, 
                        start_position=None, show_pb=True, execute=False):
        assert from_current or (start_position is not None), 'Cannot have "from_current" False and "start_position" None!'
        if from_current:
            start_position = self.robot.get_joint_positions().numpy()

        joint_traj = self.ik_helper.plan_joint_motion(
            start_position, 
            joint_position_desired,
            alg=self.planning_alg)

        if joint_traj is not None:
            if show_pb:
                self.execute_pb_loop(joint_traj)
            
            if execute:
                self.execute_loop(joint_traj)
            
            return joint_traj
        else:
            print(f'[Plan Joint Target] Path planning failed')
            return None
        
    def plan_pose_target(self, ee_pose_mat_desired, from_current=True,
                         start_position=None, show_pb=True, execute=False):
        
        ik_joints = self.ik_helper.get_feasible_ik(
            util.pose_stamped2list(util.pose_from_matrix(ee_pose_mat_desired)), 
            target_link=False)
        
        if ik_joints is not None:
            return self.plan_joint_target(ik_joints, from_current=from_current, start_position=start_position, show_pb=show_pb, execute=execute)
        else:
            print(f'[Plan Pose Target] IK sampling failed')
            return None
