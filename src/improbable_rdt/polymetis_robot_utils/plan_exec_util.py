import os, os.path as osp
import copy
import time
import trimesh
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from llm_robot.utils import util
from llm_robot.utils.real.polymetis_util import PolymetisHelper
from airobot import log_debug, log_warn, log_info

poly_util = PolymetisHelper()


class PlanningHelper:
    def __init__(self, mc_vis, robot, gripper, 
                 ik_helper, traj_helper, tmp_obstacle_dir,
                 gripper_type='panda'):
        self.mc_vis = mc_vis
        self.robot = robot
        self.gripper = gripper
        self.ik_helper = ik_helper
        self.traj_helper = traj_helper

        self.tmp_obstacle_dir = tmp_obstacle_dir

        self.cached_plan = {}
        self.cached_plan['grasp_to_offset'] = None
        self.cached_plan['grasp_to_grasp'] = None
        self.cached_plan['grasp_to_above'] = None
        self.cached_plan['waypoint'] = None
        self.cached_plan['place_to_offset'] = None
        self.cached_plan['place_to_place'] = None
        self.cached_plan['place_to_offset2'] = None
        self.cached_plan['home'] = None

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

    def get_feasible_grasp(self, grasp_pose_mat_list, relative_pose_mat, place_offset_vec=np.zeros(3), place_offset_dist=0.1,
                           use_offset=False, pcd=None, thresh=0.5, return_all=True):

        feas_hand_list = []
        for i, grasp_pose_mat in enumerate(grasp_pose_mat_list):
            grasp_jnts_feas = self.ik_helper.get_feasible_ik(
                util.pose_stamped2list(util.pose_from_matrix(grasp_pose_mat)),
                target_link=False,
                pcd=pcd,
                thresh=thresh,
                hand_only=True)

            place_pose_mat = np.matmul(relative_pose_mat, grasp_pose_mat)
            if use_offset:
                place_pose_mat[:-1, -1] += place_offset_vec * place_offset_dist
            place_jnts_feas = None 
            if grasp_jnts_feas is not None:
                place_jnts_feas = self.ik_helper.get_feasible_ik(
                    util.pose_stamped2list(util.pose_from_matrix(place_pose_mat)),
                    target_link=False,
                    pcd=pcd,
                    thresh=thresh,
                    hand_only=True)
            
            print(f'Grasp joints: {grasp_jnts_feas}')
            print(f'Place joints: {place_jnts_feas}')
            print(f'\n\n\n')

            if (grasp_jnts_feas is not None) and (place_jnts_feas is not None):
                feas_hand_list.append(True)
            else:
                feas_hand_list.append(False)

        out_grasp_pose_mat = None
        feas_grasp_jnts = []
        feas_list = []

        for i, grasp_pose_mat in enumerate(grasp_pose_mat_list):
            if not feas_hand_list[i]:
                feas_list.append(False)
                continue
            grasp_jnts_feas = self.ik_helper.get_feasible_ik(
                util.pose_stamped2list(util.pose_from_matrix(grasp_pose_mat)),
                target_link=False,
                pcd=pcd,
                thresh=thresh)

            place_pose_mat = np.matmul(relative_pose_mat, grasp_pose_mat)
            if use_offset:
                place_pose_mat[:-1, -1] += place_offset_vec * place_offset_dist
            place_jnts_feas = None 
            if grasp_jnts_feas is not None:
                place_jnts_feas = self.ik_helper.get_feasible_ik(
                    util.pose_stamped2list(util.pose_from_matrix(place_pose_mat)),
                    target_link=False,
                    pcd=pcd,
                    thresh=thresh)
            
            print(f'Grasp joints: {grasp_jnts_feas}')
            print(f'Place joints: {place_jnts_feas}')
            print(f'\n\n\n')

            if (grasp_jnts_feas is not None) and (place_jnts_feas is not None):
                feas_grasp_jnts.append(grasp_jnts_feas)
                feas_list.append(True)
            else:
                feas_list.append(False)

        feas_list_where = np.where(feas_list)[0]
        # sample a random one

        if len(feas_list_where) > 0:
            idx2 = np.random.randint(len(feas_list_where))
            idx = feas_list_where[idx2]

            if return_all: 
                return grasp_pose_mat_list[idx], feas_list
            return grasp_pose_mat_list[idx]
        if return_all:
            return None, feas_list
        return None
    
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

    def plan_full_path_with_grasp(self, 
            grasp_pose_mat, place_pose_mat, place_offset_pose_mat, 
            grasp_offset_dist=0.075, plan_pcd=None, obj_pcd=None,
            dense_plan=False, thresh=0.5, attach_obj=False,
            pb_execute=True, execute=False, use_cached=False,
            try_diffik_first=True, try_diffik_first_place_offset=True,
            *args, **kwargs):

        have_cached = sum([val is not None for val in list(self.cached_plan.values())]) == len(self.cached_plan)
        if use_cached and have_cached:
            if pb_execute:
                # current to grasp offset
                self.execute_pb_loop(self.cached_plan['grasp_to_offset'])
                self.execute_pb_loop(self.cached_plan['grasp_to_grasp'])
                self.execute_pb_loop(self.cached_plan['grasp_to_above'])
                self.execute_pb_loop(self.cached_plan['waypoint'])
                self.execute_pb_loop(self.cached_plan['place_to_offset'])
                self.execute_pb_loop(self.cached_plan['place_to_place'])
                self.execute_pb_loop(self.cached_plan['place_to_offset2'])
                self.execute_pb_loop(self.cached_plan['home'])

            if execute:
                input_value = input('Press Enter, if you want to execute, or "n" to exit\n')

                if input_value == 'n':
                    print(f'Exiting')
                    return

                current_jnts = self.robot.get_joint_positions().numpy()
                if not self.check_states_close(self.cached_plan['grasp_to_offset'][0], current_jnts):
                    print(f'Starting state different from current state')
                    print(f'Would you like to plan a path from current to start? If no, will exit')
                    input_value2 = input('\n"y" or "yp" for Yes (plan), "ye" for Yes (execute), else for No\n')

                    if input_value2 in ['y', 'yp']:
                        current_to_start = self.ik_helper.plan_joint_motion(
                            current_jnts, 
                            self.cached_plan['grasp_to_offset'][0],
                            pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                            alg=self.planning_alg)
                        
                        if current_to_start is not None:
                            input_value3 = input('\nWould you like to execute? "y" for Yes, else for No\n')
                            self.execute_pb_loop(current_to_start)
                            if input_value3 == 'y':
                                self.execute_loop(current_to_start)
                        else:
                            print(f'Could not plan path from current to start. Exiting')
                            return 
                    elif input_value2 == 'ye':
                        self.robot.move_to_joint_positions(torch.Tensor(self.cached_plan['grasp_to_offset'][0]))
                    else:
                        print(f'Exiting')
                        return

                self.gripper_open()
                self.execute_loop(self.cached_plan['grasp_to_offset'])
                self.execute_loop(self.cached_plan['grasp_to_grasp'])
                self.gripper_close()
                self.execute_loop(self.cached_plan['grasp_to_above'])
                self.execute_loop(self.cached_plan['waypoint'])
                self.execute_loop(self.cached_plan['place_to_offset'])
                self.execute_loop(self.cached_plan['place_to_place'])
                self.gripper_open()
                self.execute_loop(self.cached_plan['place_to_offset2'])
                self.execute_loop(self.cached_plan['home'])
                self.gripper_open()
            else:
                pass
            return

        # get offset from grasp pose
        offset_grasp_pose_mat, above_grasp_pose_mat = self.get_offset_poses(grasp_pose_mat, grasp_offset_dist=grasp_offset_dist)

        pl_mc = 'scene/planning_full'
        util.meshcat_frame_show(self.mc_vis, f'{pl_mc}/grasp_offset_frame', offset_grasp_pose_mat)
        util.meshcat_frame_show(self.mc_vis, f'{pl_mc}/grasp_frame', grasp_pose_mat)
        util.meshcat_frame_show(self.mc_vis, f'{pl_mc}/grasp_above_frame', above_grasp_pose_mat)
        util.meshcat_frame_show(self.mc_vis, f'{pl_mc}/place_offset_frame', place_offset_pose_mat)
        util.meshcat_frame_show(self.mc_vis, f'{pl_mc}/place_frame', place_pose_mat)

        plan_dict = {}
        plan_dict['grasp_to_offset'] = None
        plan_dict['grasp_to_grasp'] = None
        plan_dict['grasp_to_above'] = None
        plan_dict['waypoint'] = None
        plan_dict['place_to_offset'] = None
        plan_dict['place_to_place'] = None
        plan_dict['place_to_offset2'] = None
        plan_dict['home'] = None

        current_jnts = self.robot.get_joint_positions().numpy()

        self.remove_all_attachments()

        offset_pcd = None
        if obj_pcd is not None:
            offset_pcd = obj_pcd.copy()
        if plan_pcd is not None and offset_pcd is not None:
            offset_pcd = np.concatenate([offset_pcd, plan_pcd], axis=0)

        jnt_waypoint_dict = {}
        jnt_waypoint_dict['grasp_offset'] = self.ik_helper.get_feasible_ik(
            util.pose_stamped2list(util.pose_from_matrix(offset_grasp_pose_mat)), 
            target_link=False,
            pcd=offset_pcd,
            thresh=thresh)
        grasp_to_offset_jnt_list = self.ik_helper.plan_joint_motion(
                current_jnts, 
                jnt_waypoint_dict['grasp_offset'], 
                max_time=self.max_planning_time,
                pcd=offset_pcd, occnet_thresh=self.occnet_check_thresh,
                alg=self.planning_alg)

        grasp_to_grasp_jnt_list = None
        if grasp_to_offset_jnt_list is not None:
            if try_diffik_first:
                grasp_to_grasp_jnt_list, valid_diffik = self.feasible_diffik_joint_traj(
                    pose_mat_des=grasp_pose_mat,
                    pose_mat_start=offset_grasp_pose_mat,
                    start_joint_pos=grasp_to_offset_jnt_list[-1],
                    coll_pcd=plan_pcd,
                    coll_pcd_thresh=self.occnet_check_thresh,
                    show_frame_name='grasp_to_grasp',
                    from_current=False
                )

                if valid_diffik:
                    print(f'DiffIK feasible for grasp_to_grasp')
                    jnt_waypoint_dict['grasp'] = grasp_to_grasp_jnt_list[-1]
                else:
                    print(f'DiffIK NOT feasible for grasp_to_grasp')
                    jnt_waypoint_dict['grasp'] = self.ik_helper.get_feasible_ik(
                        util.pose_stamped2list(util.pose_from_matrix(grasp_pose_mat)), 
                        target_link=False,
                        pcd=plan_pcd,
                        thresh=thresh)
                    grasp_to_grasp_jnt_list = self.ik_helper.plan_joint_motion(
                            grasp_to_offset_jnt_list[-1],
                            jnt_waypoint_dict['grasp'], 
                            max_time=self.max_planning_time,
                            pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                            alg=self.planning_alg)
            else:
                jnt_waypoint_dict['grasp'] = self.ik_helper.get_feasible_ik(
                    util.pose_stamped2list(util.pose_from_matrix(grasp_pose_mat)), 
                    target_link=False,
                    pcd=plan_pcd,
                    thresh=thresh)
                grasp_to_grasp_jnt_list = self.ik_helper.plan_joint_motion(
                        grasp_to_offset_jnt_list[-1],
                        jnt_waypoint_dict['grasp'], 
                        max_time=self.max_planning_time,
                        pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                        alg=self.planning_alg)

        grasp_to_above_jnt_list = None
        if grasp_to_grasp_jnt_list is not None:
            if try_diffik_first:
                grasp_to_above_jnt_list, valid_diffik = self.feasible_diffik_joint_traj(
                    pose_mat_des=above_grasp_pose_mat,
                    pose_mat_start=grasp_pose_mat,
                    start_joint_pos=grasp_to_grasp_jnt_list[-1],
                    coll_pcd=plan_pcd,
                    coll_pcd_thresh=self.occnet_check_thresh,
                    show_frame_name='grasp_to_above',
                    from_current=False
                )

                if valid_diffik:
                    print(f'DiffIK feasible for grasp_to_above')
                    jnt_waypoint_dict['grasp_above'] = grasp_to_above_jnt_list[-1]
                else:
                    print(f'DiffIK NOT feasible for grasp_to_above')
                    jnt_waypoint_dict['grasp_above'] = self.ik_helper.get_feasible_ik(
                        util.pose_stamped2list(util.pose_from_matrix(above_grasp_pose_mat)), 
                        target_link=False,
                        pcd=plan_pcd,
                        thresh=thresh)
                    grasp_to_above_jnt_list = self.ik_helper.plan_joint_motion(
                            grasp_to_grasp_jnt_list[-1],
                            jnt_waypoint_dict['grasp_above'], 
                            max_time=self.max_planning_time,
                            pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                            alg=self.planning_alg)
            else:
                jnt_waypoint_dict['grasp_above'] = self.ik_helper.get_feasible_ik(
                    util.pose_stamped2list(util.pose_from_matrix(above_grasp_pose_mat)), 
                    target_link=False,
                    pcd=plan_pcd,
                    thresh=thresh)
                grasp_to_above_jnt_list = self.ik_helper.plan_joint_motion(
                        grasp_to_grasp_jnt_list[-1],
                        jnt_waypoint_dict['grasp_above'], 
                        max_time=self.max_planning_time,
                        pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                        alg=self.planning_alg)
            
        # attach the grasped object
        if attach_obj and obj_pcd is not None:
            self.attach_obj_pcd(obj_pcd, grasp_pose_mat_world=grasp_pose_mat)

        waypoint_jnt_list = None
        if grasp_to_above_jnt_list is not None:
            jnt_waypoint_dict['waypoint'] = self.waypoint_jnts
            waypoint_jnt_list = self.ik_helper.plan_joint_motion(
                    grasp_to_above_jnt_list[-1],
                    jnt_waypoint_dict['waypoint'], 
                    max_time=self.max_planning_time,
                    pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                    alg=self.planning_alg)

        place_to_offset_jnt_list = None
        if waypoint_jnt_list is not None:
            # if try_diffik_first:
            if try_diffik_first_place_offset:
                place_to_offset_jnt_list, valid_diffik = self.feasible_diffik_joint_traj(
                    pose_mat_des=place_offset_pose_mat,
                    pose_mat_start=self.waypoint_pose,
                    start_joint_pos=waypoint_jnt_list[-1],
                    coll_pcd=plan_pcd,
                    coll_pcd_thresh=self.occnet_check_thresh,
                    show_frame_name='place_to_offset',
                    from_current=False
                )

                if valid_diffik:
                    print(f'DiffIK feasible for place_to_offset')
                    jnt_waypoint_dict['place_offset'] = place_to_offset_jnt_list[-1]
                else:
                    print(f'DiffIK NOT feasible for place_to_offset')
                    jnt_waypoint_dict['place_offset'] = self.ik_helper.get_feasible_ik(
                        util.pose_stamped2list(util.pose_from_matrix(place_offset_pose_mat)), 
                        target_link=False,
                        pcd=plan_pcd,
                        thresh=thresh)
                    place_to_offset_jnt_list = self.ik_helper.plan_joint_motion(
                            waypoint_jnt_list[-1],
                            jnt_waypoint_dict['place_offset'], 
                            max_time=self.max_planning_time,
                            pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                            alg=self.planning_alg)
            else:
                jnt_waypoint_dict['place_offset'] = self.ik_helper.get_feasible_ik(
                    util.pose_stamped2list(util.pose_from_matrix(place_offset_pose_mat)), 
                    target_link=False,
                    pcd=plan_pcd,
                    thresh=thresh)
                place_to_offset_jnt_list = self.ik_helper.plan_joint_motion(
                        waypoint_jnt_list[-1],
                        jnt_waypoint_dict['place_offset'], 
                        max_time=self.max_planning_time,
                        pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                        alg=self.planning_alg)

        place_to_place_jnt_list = None
        if place_to_offset_jnt_list is not None:
            # jnt_waypoint_dict['place'] = self.ik_helper.get_feasible_ik(
            #     util.pose_stamped2list(util.pose_from_matrix(place_pose_mat)), 
            #     target_link=False,
            #     pcd=plan_pcd,
            #     thresh=thresh)
            # place_to_place_jnt_list = self.ik_helper.plan_joint_motion(
            #         place_to_offset_jnt_list[-1],
            #         jnt_waypoint_dict['place'], 
            #         max_time=self.max_planning_time,
            #         pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh)
            if try_diffik_first:
                place_to_place_jnt_list, valid_diffik = self.feasible_diffik_joint_traj(
                    pose_mat_des=place_pose_mat,
                    pose_mat_start=place_offset_pose_mat,
                    start_joint_pos=place_to_offset_jnt_list[-1],
                    coll_pcd=None,
                    coll_pcd_thresh=self.occnet_check_thresh,
                    show_frame_name='place_to_place',
                    from_current=False
                )

                if valid_diffik:
                    print(f'DiffIK feasible for place_to_place')
                    jnt_waypoint_dict['place'] = place_to_place_jnt_list[-1]
                else:
                    print(f'DiffIK NOT feasible for place_to_place')
                    jnt_waypoint_dict['place'] = self.ik_helper.get_feasible_ik(
                        util.pose_stamped2list(util.pose_from_matrix(place_pose_mat)), 
                        target_link=False,
                        pcd=None,
                        thresh=thresh)
                    place_to_place_jnt_list = self.ik_helper.plan_joint_motion(
                            place_to_offset_jnt_list[-1],
                            jnt_waypoint_dict['place'], 
                            max_time=self.max_planning_time,
                            pcd=None, occnet_thresh=self.occnet_check_thresh,
                            alg=self.planning_alg)
            else:
                jnt_waypoint_dict['place'] = self.ik_helper.get_feasible_ik(
                    util.pose_stamped2list(util.pose_from_matrix(place_pose_mat)), 
                    target_link=False,
                    pcd=None,
                    thresh=thresh)
                place_to_place_jnt_list = self.ik_helper.plan_joint_motion(
                        place_to_offset_jnt_list[-1],
                        jnt_waypoint_dict['place'], 
                        max_time=self.max_planning_time,
                        pcd=None, occnet_thresh=self.occnet_check_thresh,
                        alg=self.planning_alg)

        self.remove_all_attachments()

        place_to_offset2_jnt_list = None
        if place_to_place_jnt_list is not None:
            jnt_waypoint_dict['place_offset2'] = jnt_waypoint_dict['place_offset']
            place_to_offset2_jnt_list = place_to_place_jnt_list[::-1]
            # jnt_waypoint_dict['place_offset'] = self.ik_helper.get_feasible_ik()
            # place_to_offset2_jnt_list = self.ik_helper.plan_joint_motion(
            #         current_jnts, 
            #         grasp_jnts, 
            #         pcd=plan_pcd)

        home_jnt_list = None
        if place_to_offset2_jnt_list is not None: 
            # jnt_waypoint_dict['home'] = self.robot.home_pose.numpy()
            jnt_waypoint_dict['home'] = np.array([-0.1329, -0.0262, -0.0448, -1.3961,  0.0632,  1.9965, -0.8882])
            home_jnt_list = self.ik_helper.plan_joint_motion(
                    place_to_offset2_jnt_list[-1],
                    jnt_waypoint_dict['home'], 
                    max_time=self.max_planning_time,
                    pcd=plan_pcd, occnet_thresh=self.occnet_check_thresh,
                    alg=self.planning_alg)
        
        plan_dict_nom = {}
        plan_dict_nom['grasp_to_offset'] = grasp_to_offset_jnt_list
        plan_dict_nom['grasp_to_grasp'] = grasp_to_grasp_jnt_list
        plan_dict_nom['grasp_to_above'] = grasp_to_above_jnt_list
        plan_dict_nom['waypoint'] = waypoint_jnt_list
        plan_dict_nom['place_to_offset'] = place_to_offset_jnt_list
        plan_dict_nom['place_to_place'] = place_to_place_jnt_list
        plan_dict_nom['place_to_offset2'] = place_to_offset2_jnt_list
        plan_dict_nom['home'] = home_jnt_list

        if dense_plan:
            plan_dict = self.interpolate_plan_full(plan_dict_nom)
        else:
            for k, v in plan_dict_nom.items():
                plan_dict[k] = v
                if v is not None:
                    if len(v) == 1:
                        plan_dict[k] = np.vstack([v[0], v[0]])

        self.cached_plan = copy.deepcopy(plan_dict)

        # have_plan = sum([val is not None for val in list(plan_dict.values())]) == len(plan_dict)

        have_plan = True
        for k, v in plan_dict.items():
            have_plan = have_plan and (v is not None)
            plan_str = f'Plan segment: {k}, Valid: {v is not None}'
            print(f'{plan_str}')
        if have_plan:
            for k, v in plan_dict.items():
                print(f'Plan segment: {k}, Length: {len(v)}')

        if pb_execute and have_plan:

            # current to grasp offset
            self.execute_pb_loop(plan_dict['grasp_to_offset'])
            self.execute_pb_loop(plan_dict['grasp_to_grasp'])
            self.execute_pb_loop(plan_dict['grasp_to_above'])
            self.execute_pb_loop(plan_dict['waypoint'])
            self.execute_pb_loop(plan_dict['place_to_offset'])
            self.execute_pb_loop(plan_dict['place_to_place'])
            self.execute_pb_loop(plan_dict['place_to_offset2'])
            self.execute_pb_loop(plan_dict['home'])

        if execute:
            if not have_plan:
                print(f'Don"t have full plan! Some of plan is None')
                for k, v in plan_dict.items():
                    plan_str = f'Plan segment: {k}, Valid: {v is not None}'
                    print(f'{plan_str}')
                return
            self.gripper_open()
            self.execute_loop(plan_dict['grasp_to_offset'])
            self.execute_loop(plan_dict['grasp_to_grasp'])
            self.gripper_close()
            self.execute_loop(plan_dict['grasp_to_above'])
            self.execute_loop(plan_dict['waypoint'])
            self.execute_loop(plan_dict['place_to_offset'])
            self.execute_loop(plan_dict['place_to_place'])
            self.gripper_open()
            self.execute_loop(plan_dict['place_to_offset2'])
            self.execute_loop(plan_dict['home'])
            self.gripper_open()
        else:
            pass
        return plan_dict

    def process_full_plan_arm_gripper_combine(self, plan_dict):
        # output should be a whole numpy array, where the last column indicates the gripper values (1 for closed, 0 for open)

        arr1 = np.asarray(plan_dict['grasp_to_offset'])
        arr1 = np.hstack([arr1, np.zeros(arr1.shape[0], 1)])

        arr2 = np.asarray(plan_dict['grasp_to_grasp'])
        arr2 = np.hstack([arr2, np.zeros(arr2.shape[0], 1)])

        arr3 = np.asarray(plan_dict['grasp_to_above'])
        arr3 = np.hstack([arr3, np.ones(arr3.shape[0], 1)])

        arr4 = np.asarray(plan_dict['waypoint'])
        arr4 = np.hstack([arr4, np.ones(arr4.shape[0], 1)])

        arr5 = np.asarray(plan_dict['place_to_offset'])
        arr5 = np.hstack([arr5, np.ones(arr5.shape[0], 1)])

        arr6 = np.asarray(plan_dict['place_to_place'])
        arr6 = np.hstack([arr6, np.ones(arr6.shape[0], 1)])

        arr7 = np.asarray(plan_dict['place_to_place'])
        arr7 = np.hstack([arr7, np.ones(arr7.shape[0], 1)])

        arr8 = np.asarray(plan_dict['place_to_offset2'])
        arr8 = np.hstack([arr8, np.zeros(arr8.shape[0], 1)])

        arr9 = np.asarray(plan_dict['home'])
        arr9 = np.hstack([arr9, np.zeros(arr9.shape[0], 1)])

        full_arr = np.vstack([
            arr1, arr2, arr3, arr4, arr5, arr6, arr7, arr8, arr9
        ])

        return full_arr
