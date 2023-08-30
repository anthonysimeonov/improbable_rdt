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
from rdt.polymetis_robot_utils.plan_exec_util import PlanningHelper
from airobot import log_debug, log_warn, log_info

poly_util = PolymetisHelper()
pb_pl = PbPlUtils()


class PlanningHelperPickPlace(PlanningHelper):
    def __init__(self, mc_vis, robot, gripper, 
                 ik_helper, traj_helper, tmp_obstacle_dir,
                 gripper_type='panda'):

        super().__init__(mc_vis=mc_vis, robot=robot, gripper=gripper, 
                         ik_helper=ik_helper, traj_helper=traj_helper,
                         tmp_obstacle_dir=tmp_obstacle_dir, gripper_type=gripper_type)

        self.cached_plan = {}
        self.cached_plan['grasp_to_offset'] = None
        self.cached_plan['grasp_to_grasp'] = None
        self.cached_plan['grasp_to_above'] = None
        self.cached_plan['waypoint'] = None
        self.cached_plan['place_to_offset'] = None
        self.cached_plan['place_to_place'] = None
        self.cached_plan['place_to_offset2'] = None
        self.cached_plan['home'] = None
    
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
