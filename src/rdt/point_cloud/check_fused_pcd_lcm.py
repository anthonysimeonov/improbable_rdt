import os, os.path as osp
import sys
import argparse
import random
import signal
import threading
import numpy as np
import copy
import meshcat
import matplotlib.pyplot as plt
import trimesh
import lcm

# import from airobot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

# from panda_rrp_utils.simple_multicam import MultiRealsense

from rrp_robot.utils import util, trimesh_util, lcm_util, plotly_save, path_util
from rrp_robot.config.default_eval_cfg import get_eval_cfg_defaults
from rrp_robot.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rrp_robot.robot.simple_multicam import MultiRealsenseLocal
from rrp_robot.segmentation.instance_segmentation import InstanceSegServer
from rrp_robot.utils.real_util import RealImageLCMSubscriber, RealCamInfoLCMSubscriber

sys.path.append(osp.join(path_util.get_rrp_src(), 'lcm_types'))
from rrp_robot.lcm_types.rrp_lcm import (
    point_t, quaternion_t, pose_t, pose_stamped_t, start_goal_pose_stamped_t, 
    point_cloud_t, point_cloud_array_t, simple_img_t, simple_depth_img_t, square_matrix_t)

import open3d
def o3d_fps(np_pcd, num_samples=1024):
    print(f'Running farthest point sampling with {num_samples} points')
    o3d_pcd = open3d.geometry.PointCloud()
    o3d_pcd.points = open3d.utility.Vector3dVector(np_pcd)

    o3d_pcd_ds = o3d_pcd.farthest_point_down_sample(num_samples=num_samples)
    return np.asarray(o3d_pcd_ds.points)

def o3d_vds(np_pcd, voxel_size=0.0025):
    print(f'Running voxel down sampling with {voxel_size} voxel_size')
    o3d_pcd = open3d.geometry.PointCloud()
    o3d_pcd.points = open3d.utility.Vector3dVector(np_pcd)

    o3d_pcd_ds = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)
    return np.asarray(o3d_pcd_ds.points)


def lcm_sub_thread(lc):
    while True:
        lc.handle_timeout(1)


def main(args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    signal.signal(signal.SIGINT, util.signal_handler)

    # create interfaces
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")

    lc_th = threading.Thread(target=lcm_sub_thread, args=(lc,))
    lc_th.daemon = True
    lc_th.start()

    mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6001')
    mc_vis['scene'].delete()

    cfg = get_eval_cfg_defaults()
    config_fname = osp.join(path_util.get_rrp_config(), 'eval_cfgs', args.config)
    if osp.exists(config_fname):
        cfg.merge_from_file(config_fname)
    else:
        log_info(f'Config file {config_fname} does not exist, using defaults')
    cfg.freeze()

    # setup camera interfaces as LCM subscribers
    rs_cfg = get_default_multi_realsense_cfg()
    serials = rs_cfg.SERIAL_NUMBERS

    rgb_topic_name_suffix = rs_cfg.RGB_LCM_TOPIC_NAME_SUFFIX
    depth_topic_name_suffix = rs_cfg.DEPTH_LCM_TOPIC_NAME_SUFFIX
    info_topic_name_suffix = rs_cfg.INFO_LCM_TOPIC_NAME_SUFFIX
    pose_topic_name_suffix = rs_cfg.POSE_LCM_TOPIC_NAME_SUFFIX

    prefix = rs_cfg.CAMERA_NAME_PREFIX
    camera_names = [f'{prefix}{i}' for i in range(len(serials))]
    cam_list = [camera_names[int(idx)] for idx in args.cam_index]

    # update the topic names based on each individual camera
    rgb_sub_names = [f'{cam_name}_{rgb_topic_name_suffix}' for cam_name in camera_names]
    depth_sub_names = [f'{cam_name}_{depth_topic_name_suffix}' for cam_name in camera_names]
    info_sub_names = [f'{cam_name}_{info_topic_name_suffix}' for cam_name in camera_names]
    pose_sub_names = [f'{cam_name}_{pose_topic_name_suffix}' for cam_name in camera_names]

    img_subscribers = []
    for i, name in enumerate(cam_list):
        img_sub = RealImageLCMSubscriber(lc, rgb_sub_names[i], depth_sub_names[i])
        info_sub = RealCamInfoLCMSubscriber(lc, pose_sub_names[i], info_sub_names[i])
        img_subscribers.append((name, img_sub, info_sub))
    
    calib_dir = osp.join(path_util.get_rrp_src(), 'robot/camera_calibration_files')
    calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in args.cam_index]

    cams = MultiRealsenseLocal(cam_list, calib_filenames)

    while True:

        pcd_pts = []
        pcd_dict_list = []
        cam_int_list = []
        cam_poses_list = []
        rgb_imgs = []
        depth_imgs = []
        proc_pcd_list = []
        keypoint_list = []
        pcd_color_list = [
            (255, 0, 0),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 0),
            (0, 255, 255),
            (0, 0, 255),
            (255, 255, 255)
        ]
            
        for idx, cam in enumerate(cams.cams):
            rgb, depth = img_subscribers[idx][1].get_rgb_and_depth(block=True)
            rgb_imgs.append(rgb)

            cam_intrinsics = img_subscribers[idx][2].get_cam_intrinsics(block=True)
            cam.cam_int_mat = cam_intrinsics
            cam._init_pers_mat()
            cam_pose_world = cam.cam_ext_mat
            cam_int_list.append(cam_intrinsics)
            cam_poses_list.append(cam_pose_world)

            depth = depth * 0.001
            valid = depth < cam.depth_max
            valid = np.logical_and(valid, depth > cam.depth_min)
            depth_valid = copy.deepcopy(depth)
            depth_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth
            depth_imgs.append(depth_valid)

            pcd_cam = cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb, depth_image=depth_valid)[0]
            pcd_cam_img = pcd_cam.reshape(depth.shape[0], depth.shape[1], 3)
            pcd_world = util.transform_pcd(pcd_cam, cam_pose_world)
            pcd_world_img = pcd_world.reshape(depth.shape[0], depth.shape[1], 3)
            pcd_dict = {
                'world': pcd_world,
                'cam': pcd_cam_img,
                'cam_img': pcd_cam,
                'world_img': pcd_world_img,
                'cam_pose_mat': cam_pose_world
            }

            pcd_pts.append(pcd_world)
            pcd_dict_list.append(pcd_dict)
            
            kp_dict = dict(rgb=rgb, pcd_world_img=pcd_world_img)
            keypoint_list.append(kp_dict)
            
            pcd_color = pcd_color_list[idx]
            util.meshcat_pcd_show(mc_vis, pcd_world, color=pcd_color, name=f'scene/indiv_pcds/pcd_world_cam_{idx}_first')
            # util.meshcat_pcd_show(mc_vis, proc_pcd_i, color=(0, 0, 100), name=f'{mc_iter_name}/pcd_world_cam_{idx}_first')

        pcd_full = np.concatenate(pcd_pts, axis=0)
        # pcd_full_ds = o3d_fps(pcd_full, num_samples=min(int(pcd_full.shape[0] * 0.5), 8192))
        pcd_full_ds = o3d_vds(pcd_full, voxel_size=0.0045)
        util.meshcat_pcd_show(mc_vis, pcd_full_ds, color=(0, 0, 0), name=f'scene/pcd_world_full', size=0.0025)
    
        table_obs = trimesh.creation.box([0.77, 1.22, 0.001]).apply_transform(util.matrix_from_list([0.15 + 0.77/2.0, 0.0015, 0.0, 0.0, 0.0, 0.0, 1.0]))
        util.meshcat_trimesh_show(mc_vis, 'scene/table', table_obs, opacity=0.3)
        from IPython import embed; embed()
        assert False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='base_config.yaml')
    parser.add_argument('--cam_index', nargs='+', help='set which cameras to get point cloud from', required=True)
    parser.add_argument('--seg_viz', action='store_true')

    args = parser.parse_args()

    main(args)
