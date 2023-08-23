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

from rrp_robot.segmentation.keypoint_select import ObjectManualSegmentation


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

    # set up instance segmentation interface
    segmentation_server = InstanceSegServer(cfg)
    manual_seg = ObjectManualSegmentation()

    while True:
        ##### get image and point cloud together, run segmentation prediction, and obtain segmented point cloud #####
        # log_info('\n\nGETTING RGB AND DEPTH IMAGES FROM LCM\n\n')
        # rgb_img, depth_img = img_sub.get_rgb_and_depth()
        # rgb_img = rgb_img.astype(np.uint8)
        # depth_img = depth_img.astype(np.uint16)
        # 
        # # once we have obtained a new camera image, reset the visualizer
        # mc_vis['scene'].delete()

        # log_info('\n\nGETTING CAMERA POSE FROM LCM\n\n')
        # cam_pose_list = cam_sub.get_cam_pose()
        # cam_pose_mat = util.matrix_from_pose(util.list2pose_stamped(cam_pose_list))
        # # cam_pose_mat = np.eye(4)
        # real_cam.set_cam_ext(cam_ext=cam_pose_mat)
        # cam_ext_mat_list = [cam_pose_mat, cam_pose_mat]

        # log_info('\n\nGETTING CAMERA INTRINSICS FROM LCM\n\n')
        # cam_int_mat = cam_sub.get_cam_intrinsics()
        # real_cam.cam_int_mat = cam_int_mat
        # real_cam._init_pers_mat()

        # log_info('\n\nGETTING CURRENT END EFFECTOR POSE FROM LCM\n\n')
        # current_ee_pose = ee_pose_sub.get_ee_pose()

        # # filter and convert this to point cloud
        # # depth_img = depth_img * real_cam.depth_scale
        # depth_img = depth_img * depth_scale_true
        # valid = depth_img < real_cam.depth_max
        # valid = np.logical_and(valid, depth_img > real_cam.depth_min)
        # depth_img_valid = copy.deepcopy(depth_img)
        # depth_img_valid[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth

        # pcd_cam = real_cam.get_pcd(in_world=False, filter_depth=False, rgb_image=rgb_img, depth_image=depth_img_valid)[0]
        # pcd_cam_img = pcd_cam.reshape(depth_img.shape[0], depth_img.shape[1], 3)
        # pcd_world = util.transform_pcd(pcd_cam, cam_pose_mat)
        # # pcd_world = copy.deepcopy(pcd_cam)
        # pcd_dict = {
        #     'world': pcd_world,
        #     'cam': pcd_cam,
        #     'cam_img': pcd_cam,
        #     'cam_pose_mat': cam_pose_mat
        #     }

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
            util.meshcat_pcd_show(mc_vis, pcd_world, color=pcd_color, name=f'scene/pcd_world_cam_{idx}_first')
            # util.meshcat_pcd_show(mc_vis, proc_pcd_i, color=(0, 0, 100), name=f'{mc_iter_name}/pcd_world_cam_{idx}_first')
    
        table_obs = trimesh.creation.box([0.77, 1.22, 0.001]).apply_transform(util.matrix_from_list([0.15 + 0.77/2.0, 0.0015, 0.0, 0.0, 0.0, 0.0, 1.0]))
        util.meshcat_trimesh_show(mc_vis, 'scene/table', table_obs, opacity=0.3)
        from IPython import embed; embed()
        assert False

        manual_seg.set_source_data_keypoints(keypoint_list)

        print("here with keypoint labels")
        from IPython import embed; embed()

        # perform segmentation and get target object point cloud
        log_info('\n\nRUNNING RGB IMAGE SEGMENTATION TO GET TARGET OBJECT POINT CLOUD\n\n')
        # obj_pcd_pts, target_obj_seg_mask = segmentation_server.get_target_pcd(
        #     pcd_dict, 
        #     rgb_image=rgb_imgs[0], 
        #     depth_image=depth_imgs[0],
        #     all_segments=False,
        #     viz=args.seg_viz)

        full_mask = segmentation_server.segmentation_interface.make_prediction(rgb_imgs[0], all_segments=True)
        uv0 = keypoint_list[0]['uv'][0]
        keypoint_mask = full_mask[:, uv0[1], uv0[0]].squeeze()
        keypoint_mask_idx = np.where(keypoint_mask)[0]
        if keypoint_mask_idx.shape[0] > 0:
            keypoint_mask_idx = keypoint_mask_idx[:1]

        obj_keypoint_mask = full_mask[keypoint_mask_idx[0]] 
        target_point_cloud = pcd_dict['world'][obj_keypoint_mask.reshape(-1)]
        target_rgb = rgb_imgs[0]*obj_keypoint_mask[:, :, None]
        target_depth = depth_imgs[0]*obj_keypoint_mask

        print("here with images and point clouds")
        from IPython import embed; embed()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--config', type=str, default='base_config.yaml')
    parser.add_argument('--cam_index', nargs='+', help='set which cameras to get point cloud from', required=True)
    parser.add_argument('--seg_viz', action='store_true')

    args = parser.parse_args()

    main(args)
