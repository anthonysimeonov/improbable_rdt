import os, os.path as osp
import numpy as np

import sys

from rrp_robot.utils import util, path_util
#path_util.append_lcm_path()
sys.path.append(osp.join(path_util.get_rrp_src(), 'lcm_types'))
from rrp_robot.lcm_types.rrp_lcm import point_t, quaternion_t, pose_t, pose_stamped_t, point_cloud_t, start_goal_pose_stamped_t, simple_img_t, simple_depth_img_t, square_matrix_t, simple_binary_mask_t

def np2point_cloud_t(pcd_np, frame_id='world', pcd_t=None):
    if pcd_t is None:
        pcd_t = point_cloud_t()
    for i in range(pcd_np.shape[0]):
        pt_msg = point_t()
        pt_msg.x = pcd_np[i, 0]
        pt_msg.y = pcd_np[i, 1]
        pt_msg.z = pcd_np[i, 2]
        pcd_t.points.append(pt_msg)
    pcd_t.num_points = pcd_np.shape[0]
    pcd_t.header.frame_name = frame_id
    return pcd_t


def np2img_t(img_np, img_t=None, depth=False):
    if img_t is None:
        img_t = simple_depth_img_t() if depth else simple_img_t()
    img_t.width = img_np.shape[1]
    img_t.height = img_np.shape[0]

    if depth:
        img_t.size = img_np.shape[0]*img_np.shape[1]
    else:
        img_t.size = img_np.shape[0]*img_np.shape[1]*3

    # add in each RGB/depth value one at a time
    for i in range(img_np.shape[0]):
        for j in range(img_np.shape[1]):
            pixel_val = img_np[i, j]
            if depth:
                img_t.pixels.append(pixel_val)
            else:
                img_t.pixels.append(pixel_val[0])
                img_t.pixels.append(pixel_val[1])
                img_t.pixels.append(pixel_val[2])

    return img_t


def np2binary_mask_t(mask_np, mask_t=None):
    if mask_t is None:
        mask_t = simple_binary_mask_t() 

    if mask_np.ndim > 1:
        mask_np = mask_np.reshape(-1)

    mask_t.size = mask_np.shape[0]

    for i in range(mask_np.shape[0]):
        mask_val = mask_np[i]
        mask_t.mask.append(mask_val)

    return mask_t


def unpack_binary_mask_lcm(mask_t_mask, size):
    flat_mask  = np.asarray(mask_t_pixels)

    return flat_mask


def unpack_img_lcm(img_t_pixels, height, width, depth=False):
    flat_pixels = np.asarray(img_t_pixels)
    if depth:
        img_np = flat_pixels.reshape((height, width)).astype(np.uint16)
    else:
        img_np = flat_pixels.reshape((height, width, 3)).astype(np.uint8)

    return img_np


def pose_stamped2list(msg):
    return [float(msg.pose.position.x),
            float(msg.pose.position.y),
            float(msg.pose.position.z),
            float(msg.pose.orientation.x),
            float(msg.pose.orientation.y),
            float(msg.pose.orientation.z),
            float(msg.pose.orientation.w),
            ]


def list2pose_stamped_lcm(pose, frame_id='world'):
    ps_t = pose_stamped_t()
    ps_t.header.frame_name = frame_id
    ps_t.pose.position.x = pose[0]
    ps_t.pose.position.y = pose[1]
    ps_t.pose.position.z = pose[2]
    ps_t.pose.orientation.x = pose[3]
    ps_t.pose.orientation.y = pose[4]
    ps_t.pose.orientation.z = pose[5]
    ps_t.pose.orientation.w = pose[6]
    return ps_t 

def list2pose_stamped_array_lcm(pose_1, pose_2):
    ps1_t = list2pose_stamped_lcm(pose_1)
    ps2_t = list2pose_stamped_lcm(pose_2)
    ps_arr_t = start_goal_pose_stamped_t()
    ps_arr_t.start_pose = ps1_t
    ps_arr_t.goal_pose = ps2_t
    return ps_arr_t


def matrix2pose_stamped_lcm(pose_mat, frame_id='world'):
    pose_msg = util.pose_from_matrix(pose_mat)
    pose_list = util.pose_stamped2list(pose_msg)
    return list2pose_stamped_lcm(pose_list) 
    

def unpack_pointcloud_lcm(points, num_points):
    """
    Function to unpack a point cloud LCM message into a list
    Args:
        points (list): Each element is point_cloud_t type, with fields
            x, y, z
        num_points (int): Number of points in the point cloud
    """
    pt_list = []
    for i in range(num_points):
        pt = [
            points[i].x,
            points[i].y,
            points[i].z
        ]
        pt_list.append(pt)
    return pt_list


def pack_square_matrix(mat_np):
    mat_msg = square_matrix_t()
    mat_msg.n = mat_np.shape[0]
    mat_msg.n_entries = mat_np.shape[0]**2

    mat_np_flat = mat_np.reshape(-1)
    for i in range(mat_np_flat.shape[0]):
        mat_msg.entries.append(mat_np_flat[i])
    return mat_msg
