import os, os.path as osp
import copy
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

sys.path.append(os.getenv('CGN_SRC_DIR'))
from cgn_eval import visualize as cgn_visualize, initialize_net as cgn_init_net, cgn_infer

from llm_robot.utils import util

class GraspHelper:
    def __init__(self, mc_vis, contactnet, cgn_args, gripper_type='panda', 
                 project_topdown=False, table_height=0.0, object_height=0.075):
        self.mc_vis = mc_vis
        self.contactnet = contactnet 
        self.cgn_args = cgn_args 
        self.gripper_type = gripper_type 

        self.grasp_translate = 0.0 if gripper_type == 'panda' else 0.125
        self.topdown_translate = 0.23
        # self.topdown_translate = 0.125

        self.table_height = table_height
        self.grasp_target_height = object_height / 2.0

        self.project_topdown = project_topdown
    
    def set_grasp_translate(self, tr):
        self.grasp_translate = tr
    
    def get_grasps(self, cgn_full_pcd, cgn_obj_mask, try_twice=True, manual_thresh=None, visualize=True, full_obj_pcd=None):

        if manual_thresh is None:
            cgn_thresh = self.cgn_args.threshold
        else:
            cgn_thresh = manual_thresh

        try:
            pred_grasps, pred_success, downsample, pred_widths = cgn_infer(
                self.contactnet, 
                cgn_full_pcd, 
                cgn_obj_mask, 
                threshold=cgn_thresh,
                return_width=True)

        except Exception as e:
            print(f'[CGN Grasp] Exception: {e}')

            # cgn_input_value = input('\nNo grasps found, do you want to try a lower threshold?\n')
            if try_twice:
                new_thresh = input(f'\nPlease enter new threshold, between 0.0 and 1.0 (current threshold is {self.cgn_args.threshold}\n')
                new_thresh = float(new_thresh)
                return self.get_grasps(cgn_full_pcd, cgn_obj_mask, try_twice=False, manual_thresh=new_thresh, visualize=visualize)
            else:
                raise Exception(e)

        if visualize:
            cgn_visualize(cgn_full_pcd, pred_grasps, mc_vis=self.mc_vis)
        
        # if self.gripper_type == 'panda':
        #     translate = 0.0
        # else:
        #     # translate = -0.125
        #     # translate = -0.08
        #     np.sqrt(r**2 - (2*r/w)**2)

        grasp_pose_all = []
        breakout = False
        # grasp_pose_nom = pred_grasps[0]
        for g_idx, grasp_pose_nom in enumerate(pred_grasps):

            # if self.gripper_type == 'panda':
            #     translate = 0.0
            # else:
            #     # translate = -0.125
            #     # translate = -0.08
            #     r = 0.105 
            #     # w = pred_widths[g_idx]
            #     # w = 0.07
            #     w = 0.075
            #     c = 0.04
            #     # translate = -1.0 * (c + np.sqrt(r**2 - (w/2)**2))

            #     translate = -0.125

            translate = self.grasp_translate

            theta = np.pi/2
            z_rot = np.eye(4)
            z_rot[2,3] = translate
            z_rot[:3,:3] = R.from_euler('z', theta).as_matrix()

            z_tf = np.matmul(z_rot, np.linalg.inv(grasp_pose_nom))
            z_tf = np.matmul(grasp_pose_nom, z_tf)
            grasp_pose = np.matmul(z_tf, grasp_pose_nom)

            if self.project_topdown and (not breakout):
                
                # get grasp pose, place translation at the mean of the object
                grasp_pose_at_obj = grasp_pose.copy()
                grasp_pose_at_obj[:-1, -1] = np.mean(full_obj_pcd, axis=0)
                # grasp_pose_at_obj[2, -1] = np.mean(full_obj_pcd, axis=0)[2]

                # util.meshcat_frame_show(self.mc_vis, f'scene/grasp_poses/g{g_idx}', grasp_pose)
                # util.meshcat_frame_show(self.mc_vis, f'scene/grasp_poses_at_obj/g{g_idx}', grasp_pose_at_obj)

                # construct via position and orientation

                # suppose the x-axis of the grasp corresponds to the axis along the can
                g_rotmat = grasp_pose_at_obj[:-1, :-1]
                cyl_axis = g_rotmat[:, 0]

                # project to plane
                cyl_axis[2] = 0.0
                cyl_axis = cyl_axis / np.linalg.norm(cyl_axis)
                td_axis = np.array([0, 0, -1.0])

                # cross product and construct orientation
                xa, za = cyl_axis, td_axis
                ya = np.cross(za, xa)
                td_rotmat = np.hstack([xa.reshape(3, 1), ya.reshape(3, 1), za.reshape(3, 1)])

                # place at object mean, and then translate up
                td_trans = np.mean(full_obj_pcd, axis=0)
                # td_trans[2] += self.topdown_translate
                td_trans[2] = self.table_height + self.grasp_target_height + self.topdown_translate

                td_pose = np.eye(4); td_pose[:-1, :-1] = td_rotmat; td_pose[:-1, -1] = td_trans
                util.meshcat_frame_show(self.mc_vis, f'scene/td_grasp_poses/g{g_idx}', td_pose)

                grasp_pose = td_pose

                if False:
                    g_rotmat = grasp_pose_at_obj[:-1, :-1]
                    g_trans = grasp_pose_at_obj[:-1, -1]
                    td_trans = g_trans.copy(); td_trans[2] += self.topdown_translate

                    # get the angle that transforms z to vertical, use this for how much we should roll about x 
                    gz_axis = g_rotmat[:, 2]
                    to_vert_theta = util.angle_from_3d_vectors(gz_axis, np.array([0, 0, -1.0]))
                    roll_mat = R.from_euler('xyz', [to_vert_theta, 0, 0]).as_matrix()

                    td_rotmat = np.matmul(g_rotmat, roll_mat)
                    td_pose1 = np.eye(4)
                    td_pose1[:-1, :-1] = td_rotmat 
                    td_pose1[:-1, -1] = np.mean(full_obj_pcd, axis=0)
                    # util.meshcat_frame_show(self.mc_vis, f'scene/td_grasp_poses_at_obj1/g{g_idx}', td_pose1)
                    td_pose1[:-1, -1] = td_trans
                    # util.meshcat_frame_show(self.mc_vis, f'scene/td_grasp_poses1/g{g_idx}', td_pose1)

                    g_rotmat = td_pose1[:-1, :-1]
                    g_trans = td_pose1[:-1, -1]
                    td_trans = g_trans.copy() #; td_trans[2] += self.topdown_translate
        
                    # repeat -- get the angle that transforms z to vertical, use this for how much we should pitch about y
                    gz_axis = g_rotmat[:, 2]
                    to_vert_theta = util.angle_from_3d_vectors(gz_axis, np.array([0, 0, -1.0]))
                    pitch_mat = R.from_euler('xyz', [0, -1.0*to_vert_theta, 0]).as_matrix()

                    td_rotmat = np.matmul(g_rotmat, pitch_mat)
                    td_pose2 = np.eye(4)
                    td_pose2[:-1, :-1] = td_rotmat 
                    td_pose2[:-1, -1] = np.mean(full_obj_pcd, axis=0)
                    # util.meshcat_frame_show(self.mc_vis, f'scene/td_grasp_poses_at_obj2/g{g_idx}', td_pose2)
                    td_pose2[:-1, -1] = td_trans
                    util.meshcat_frame_show(self.mc_vis, f'scene/td_grasp_poses2/g{g_idx}', td_pose2)

                    grasp_pose = td_pose2

            grasp_pose_all.append(grasp_pose)
        
        return grasp_pose_all 