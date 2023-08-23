import os, os.path as osp
import copy
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import open3d
import pyrealsense2 as rs

from llm_robot.utils import util
from llm_robot.segmentation.pcd_utils import manually_segment_pcd


class PerceptionHelper:
    def __init__(self, mc_vis, cams, img_subscribers, manual_seg, instance_seg_server):
        self.mc_vis = mc_vis
        self.cams = cams
        self.img_subscribers = img_subscribers
        self.manual_seg = manual_seg
        self.instance_seg_server = instance_seg_server

        self.visualize_segmentation = False
        self.use_manual_seg = True

        self.cached_target_obj_pcd_obs = None
        self.cached_pcd_dict = None
    
    def set_manual_seg(self, use_manual_seg):
        self.use_manual_seg = use_manual_seg

    def set_visualize_segmentation(self, visualize_segmentation):
        self.visualize_segmentation = visualize_segmentation

    def get_seg_pcd(self, seg_viz=False, just_crop=False, surf='table', use_cached=False):
        log_info('Getting RGB/depth image from LCM and running instance segmentation...')


        if use_cached and (self.cached_target_obj_pcd_obs is not None):
            # target_obj_pcd_obs = copy.deepcopy(self.cached_target_obj_pcd_obs)
            # return target_obj_pcd_obs

            pcd_dict = copy.deepcopy(self.cached_pcd_dict)
            return pcd_dict

        mc_iter_name = 'scene/get_seg_pcd'
        self.mc_vis[mc_iter_name].delete()

        crop_dict = {}
        # crop_dict['table'] = ([0.375, 0.75], [-0.5, 0.2], [0.0075, 0.25])
        # crop_dict['table'] = ([0.2, 0.75], [-0.5, 0.5], [0.0075, 0.5])
        # crop_dict['table'] = ([0.2, 1.5], [-0.5, 0.5], [0.015, 0.5])
        crop_dict['table'] = ([0.2, 1.5], [-0.6, 0.6], [0.007, 0.8])
        # crop_dict['table'] = ([0.2, 1.5], [-0.5, 0.5], [0.007, 0.5])
        # crop_dict['table'] = ([0.2, 0.75], [-0.5, 0.5], [0.015, 0.5])
        # crop_dict['table_side'] = ([0.375, 0.75], [-0.5, 0.0], [0.01, 0.25])
        crop_dict['table_side'] = ([0.375, 0.75], [-0.5, 0.0], [0.025, 0.25])
        crop_dict['block'] = ([0.2, 0.75], [-0.5, 0.2], [0.095, 0.35])

        if surf not in crop_dict.keys():
            print(f'Specified surface: {surf} not included, defaulting to table')
            surf = 'table'
        cropx, cropy, cropz = crop_dict[surf]
        crop_note = surf

        tab_cropx, tab_cropy, tab_cropz = crop_dict['table']
        table_crop_note = 'table'

        # cropx, cropy, cropz, crop_note = [0.375, 0.75], [-0.5, 0.2], [0.0075, 0.25], 'table'
        # cropx, cropy, cropz, crop_note = [0.2, 0.75], [-0.5, 0.2], [0.095, 0.35], 'block'
        # cropx, cropy, cropz, crop_note = [0.2, 0.75], [-0.5, 0.2], [0.1, 0.35], 'block'

        dbs_eps, dbs_minp = 0.0035, 10
        # dbs_eps, dbs_minp = 0.0035, 5

        # dbs_eps, dbs_minp = 0.0035, 50
        # dbs_eps, dbs_minp = 0.005, 25
        # dbs_eps, dbs_minp = 0.01, 10

        pcd_pts = []
        pcd_dict_list = []
        cam_int_list = []
        cam_poses_list = []
        rgb_imgs = []
        depth_imgs = []
        proc_pcd_list = []
        keypoint_list = []
        target_rgb_list = []
        target_depth_list = []
        target_mask_list = []
        valid_view_list = []
        target_pcd_world_list = []
        target_pcd_world_cl_list = []
        obj_kp_mask_global_list = []
        for idx, cam in enumerate(self.cams.cams):
            rgb, depth = self.img_subscribers[idx][1].get_rgb_and_depth(block=True)
            rgb_imgs.append(rgb)

            cam_intrinsics = self.img_subscribers[idx][2].get_cam_intrinsics(block=True)
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

            if just_crop:

                # proc_pcd_i = manually_segment_pcd(pcd_world, x=cropx, y=cropy, z=cropz, note=crop_note)
                proc_pcd_i, crop_idx = manually_segment_pcd(pcd_world, x=cropx, y=cropy, z=cropz, note=crop_note, return_idx=True)
                util.meshcat_pcd_show(self.mc_vis, proc_pcd_i, color=(0, 0, 100), name=f'{mc_iter_name}/pcd_world_cam_{idx}_first')

                crop_mask = np.zeros(pcd_dict['world'].shape[0]).astype(bool); crop_mask[crop_idx] = True
                obj_kp_mask_global_list.append(crop_mask)

                pcd_o3d = open3d.geometry.PointCloud()
                pcd_o3d.points = open3d.utility.Vector3dVector(proc_pcd_i)
                # labels = np.array(pcd_o3d.cluster_dbscan(eps=0.005, min_points=50, print_progress=True))
                labels = np.array(pcd_o3d.cluster_dbscan(eps=dbs_eps, min_points=dbs_minp, print_progress=True))
                
                clusters_detected = np.unique(labels)
                pcd_clusters = []
                cluster_sizes = []
                for seg_idx in clusters_detected:
                    seg_inds = np.where(labels == seg_idx)[0]
                    cluster = proc_pcd_i[seg_inds]
                    pcd_clusters.append(cluster)
                    sz = cluster.shape[0]
                    cluster_sizes.append(sz)
                top_idx = np.argmax(cluster_sizes)

                top_clusters = pcd_clusters[top_idx]
                proc_pcd_i = copy.deepcopy(top_clusters)
                proc_pcd_list.append(proc_pcd_i)

                util.meshcat_pcd_show(self.mc_vis, proc_pcd_i, name=f'{mc_iter_name}/pcd_world_cam_{idx}')
                continue

            kp_dict = dict(rgb=rgb, pcd_world_img=pcd_world_img)
            keypoint_list.append(kp_dict)

            kp_list = [kp_dict]

            break_this_camera = False
            if self.use_manual_seg:
                while True:
                    self.manual_seg.set_source_data_keypoints(kp_list)
                    # perform segmentation and get target object point cloud
                    log_info('\n\nRUNNING RGB IMAGE SEGMENTATION TO GET TARGET OBJECT POINT CLOUD\n\n')

                    _, _ = self.instance_seg_server.get_target_pcd(
                        pcd_dict, 
                        rgb_image=rgb,
                        depth_image=depth_valid,
                        all_segments=False,
                        viz=self.visualize_segmentation)

                    full_mask = self.instance_seg_server.segmentation_interface.make_prediction(rgb, all_segments=True)

                    if 'uv' not in kp_list[0] or (len(kp_list[0]['uv']) < 1):
                        print(f'Assuming we are not using this camera, continuing to next view')
                        break_this_camera = True
                        break

                    uv0 = kp_list[0]['uv'][0]
                    kp0_idx = uv0[1]*640 + uv0[0]
                    kp0_arr = np.zeros(pcd_world.shape[0]).astype(int)
                    kp0_arr[kp0_idx] = 1

                    # if len(kp_list[0]['uv']) > 1:
                    #     uv1 = kp_list[0]['uv'][0]
                    #     kp1_idx = uv1[1]*640 + uv0[0]
                    #     kp1_arr = np.zeros(pcd_world.shape[0]).astype(int)
                    #     kp1_arr[kp1_idx] = 1

                    keypoint_mask = full_mask[:, uv0[1], uv0[0]].squeeze()
                    keypoint_mask_idx = np.where(keypoint_mask)[0]
                    n_det = keypoint_mask_idx.shape[0]
                    if n_det > 1:
                        
                        # fig, axs = plt.subplots(1, n_det)

                        cand_rgb_list = []
                        for k in range(n_det):
                            candidate_mask = full_mask[keypoint_mask_idx[k]] 
                            candidate_rgb = rgb*candidate_mask[:, :, None]
                            # axs[k].imshow(candidate_rgb)
                            # axs[k].set_title(f'Segment: {k}')
                            cand_rgb_list.append(candidate_rgb.copy().transpose(2, 0, 1))

                        vis_vis.images(cand_rgb_list, padding=1, opts=dict(title='segs')) 
                        # plt.show()

                        det_idx = input('\n\nEnter index of segment you want to use\n\n')
                        try:
                            det_idx = int(det_idx)
                            # det_idx = int(input('\n\nEnter index of segment you want to use\n\n'))
                        except ValueError as e:
                            print(f'[Det IDX select] {e}')
                            det_idx = input('\n\nEnter index of segment you want to use\n\n')
                            det_idx = int(det_idx)

                        # keypoint_mask_idx = keypoint_mask_idx[:1]
                        keypoint_mask_idx = np.array([keypoint_mask_idx[det_idx]])

                    try:
                        obj_keypoint_mask = full_mask[keypoint_mask_idx[0]] 
                        break
                    except IndexError as e:
                        print(f'[Get segmented point cloud] Exception: {e}')
                        print(f'[Get segmented point cloud] Trying again')
                        continue
                        
                if break_this_camera:
                    valid_view_list.append(False)
                    target_mask_list.append(None)
                    obj_kp_mask_global_list.append(None)
                    continue

                _, crop_idx = manually_segment_pcd(pcd_dict['world'], x=cropx, y=cropy, z=cropz, note=crop_note, return_idx=True)
                crop_mask = np.zeros(pcd_dict['world'].shape[0]).astype(bool); crop_mask[crop_idx] = True
                valid_mask = np.zeros(pcd_dict['world'].shape[0]).astype(bool); valid_idx = np.where(pcd_cam[:, 2] > 0.01)[0]; valid_mask[valid_idx] = True
                # obj_keypoint_mask = obj_keypoint_mask * crop_mask.reshape(obj_keypoint_mask.shape)
                obj_keypoint_mask = obj_keypoint_mask * crop_mask.reshape(obj_keypoint_mask.shape) * valid_mask.reshape(obj_keypoint_mask.shape)
                target_point_cloud = pcd_dict['world'][obj_keypoint_mask.reshape(-1)]
                target_point_cloud_inds = np.where(obj_keypoint_mask.reshape(-1))[0]
                obj_kp_mask_global = copy.deepcopy(obj_keypoint_mask.reshape(-1))

                # target_pcd_inds = np.where(obj_keypoint_mask.reshape(-1))[0]
                target_rgb = rgb*obj_keypoint_mask[:, :, None]
                target_depth = depth_valid*obj_keypoint_mask

                target_rgb_list.append(target_rgb)
                target_depth_list.append(target_depth)
                target_pcd_world_list.append(target_point_cloud)
                target_mask_list.append(obj_keypoint_mask)

                util.meshcat_pcd_show(self.mc_vis, target_point_cloud, (255, 0, 255), f'{mc_iter_name}/obj_pcd_cam_{idx}', size=0.001)

                pcd_o3d = open3d.geometry.PointCloud()
                pcd_o3d.points = open3d.utility.Vector3dVector(target_point_cloud)
                # labels = np.array(pcd_o3d.cluster_dbscan(eps=0.006, min_points=70, print_progress=True))
                # labels = np.array(pcd_o3d.cluster_dbscan(eps=0.01, min_points=70, print_progress=True))
                # labels = np.array(pcd_o3d.cluster_dbscan(eps=0.01, min_points=10, print_progress=True))
                labels = np.array(pcd_o3d.cluster_dbscan(eps=dbs_eps, min_points=dbs_minp, print_progress=True))

                if False:
                    clusters_detected = np.unique(labels)
                    cluster_pcds = []
                    for seg_idx in clusters_detected:
                        seg_inds = np.where(labels == seg_idx)[0]
                        cluster = target_point_cloud[seg_inds]
                        cluster_pcds.append(cluster)
                        sz = cluster.shape[0]
                        util.meshcat_pcd_show(self.mc_vis, cluster, (0, 255, 255), f'scene/clusters/cl_{seg_idx}', size=0.001)
                    
                    from rrp_robot.utils import trimesh_util
                    trimesh_util.trimesh_show(cluster_pcds)

                    print("here with clusters")
                    from IPython import embed; embed()

                kp_cluster_pts = []
                cl_iter = 0

                kp_cluster_mask_full = np.zeros(pcd_dict['world'].shape[0])
                cl_value = input('\n\nPress "c" if you want to cluster further\n\n')
                if cl_value == 'c':
                    while True:

                        cl_iter += 1
                        kp_pt = pcd_world[kp0_idx]
                        sph = trimesh.creation.uv_sphere(0.01).apply_translation(kp_pt)
                        util.meshcat_trimesh_show(self.mc_vis, f'{mc_iter_name}/kp_sph_{cl_iter}', sph, (0, 0, 255))

                        if pcd_cam[kp0_idx][2] < 0.01:
                            print(f'Invalid depth (0.0)')
                            cl_iter -= 1
                            kp_cl_dict = dict(rgb=rgb, pcd_world_img=pcd_world_img)
                            kp_cl_list = [kp_cl_dict]
                            self.manual_seg.set_source_data_keypoints(kp_cl_list)

                            if 'uv' not in kp_list[0] or (len(kp_list[0]['uv']) < 1):
                                print(f'Assuming we are not using this camera, continuing to next view')
                                break_this_camera = True
                                break

                            uv0 = kp_cl_list[0]['uv'][0]
                            kp0_idx = uv0[1]*640 + uv0[0]
                            
                            kp0_arr = np.zeros(pcd_world.shape[0]).astype(int)
                            kp0_arr[kp0_idx] = 1
                            continue

                        # kp_target_idx = target_pcd_inds[kp_idx]
                        try:
                            kp_target_idx = np.where(kp0_arr[obj_keypoint_mask.reshape(-1)])[0][0]
                        except IndexError as e:
                            print(f'[Get segmented point cloud] Exception: {e}')
                            # from IPython import embed; embed()
                            print(f'[Get segmented point cloud] Bad keypoint selection, possibly invalid 3D point. Try again\n')
                            kp_cl_dict = dict(rgb=rgb, pcd_world_img=pcd_world_img)
                            kp_cl_list = [kp_cl_dict]
                            self.manual_seg.set_source_data_keypoints(kp_cl_list)

                            if 'uv' not in kp_list[0] or (len(kp_list[0]['uv']) < 1):
                                print(f'Assuming we are not using this camera, continuing to next view')
                                break_this_camera = True
                                break

                            uv0 = kp_cl_list[0]['uv'][0]
                            kp0_idx = uv0[1]*640 + uv0[0]
                            
                            kp0_arr = np.zeros(pcd_world.shape[0]).astype(int)
                            kp0_arr[kp0_idx] = 1
                            continue

                        target_cl_idx = labels[kp_target_idx]
                        kp_cl_inds = np.where(labels == target_cl_idx)[0]
                        kp_cluster = target_point_cloud[kp_cl_inds]
                        kp_cluster_pts.append(kp_cluster)

                        kp_cluster_inds = target_point_cloud_inds[kp_cl_inds]
                        kp_cluster_mask = np.zeros(pcd_dict['world'].shape[0]); kp_cluster_mask[kp_cluster_inds] = 1
                        kp_cluster_mask_full = np.logical_or(kp_cluster_mask_full, kp_cluster_mask)

                        util.meshcat_pcd_show(self.mc_vis, kp_cluster, (0, 0, 255), f'{mc_iter_name}/clusters/cl_{cl_iter}_{idx}', size=0.002)

                        inp_value = input('Press "k" to select another keypoint')
                        if inp_value == "k":
                            kp_cl_dict = dict(rgb=rgb, pcd_world_img=pcd_world_img)
                            kp_cl_list = [kp_cl_dict]
                            self.manual_seg.set_source_data_keypoints(kp_cl_list)

                            uv0 = kp_cl_list[0]['uv'][0]
                            kp0_idx = uv0[1]*640 + uv0[0]
                            kp0_arr = np.zeros(pcd_world.shape[0]).astype(int)
                            kp0_arr[kp0_idx] = 1
                        else:
                            break

                    if break_this_camera:
                        valid_view_list.append(False)
                        target_mask_list.append(None)
                        obj_kp_mask_global_list.append(None)
                        continue
                    obj_kp_mask_global = obj_kp_mask_global * kp_cluster_mask_full
                else:
                    target_point_cloud = manually_segment_pcd(target_point_cloud, x=cropx, y=cropy, z=cropz, note=crop_note)
                    kp_cluster_pts.append(target_point_cloud)
                
                kp_cluster_full = np.concatenate(kp_cluster_pts, axis=0)
                target_pcd_world_cl_list.append(kp_cluster_full)

                obj_kp_mask_global_list.append(obj_kp_mask_global)

                util.meshcat_pcd_show(self.mc_vis, kp_cluster_full, (0, 255, 255), f'{mc_iter_name}/obj_pcd_cam_clustered_{idx}', size=0.003)
                valid_view_list.append(True)
                
                # clusters_detected = np.unique(labels)
                # pcd_clusters = []
                # cluster_sizes = []
                # for seg_idx in clusters_detected:
                #     seg_inds = np.where(labels == seg_idx)[0]
                #     cluster = proc_pcd_i[seg_inds]
                #     pcd_clusters.append(cluster)
                #     sz = cluster.shape[0]
                #     cluster_sizes.append(sz)
                # top2sz = np.argmax(cluster_sizes)

                # top2clusters = pcd_clusters[top2sz]

        scene_pcd_pts = np.concatenate(pcd_pts, axis=0)
        util.meshcat_pcd_show(self.mc_vis, scene_pcd_pts, (0, 0, 0), f'{mc_iter_name}/scene_pcd', size=0.0005)
        log_info(f'Returning object point cloud...')

        if just_crop:
            # cropped_pcd = manually_segment_pcd(scene_pcd_pts, x=cropx, y=cropy, z=cropz, note=crop_note)
            cropped_pcd = np.concatenate(proc_pcd_list, axis=0)

            util.meshcat_pcd_show(self.mc_vis, cropped_pcd, (255, 0, 0), f'{mc_iter_name}/obj_pcd', size=0.003)

            target_pcd = cropped_pcd
        else: 
            obj_pcd_pts = np.concatenate(target_pcd_world_list, axis=0)
            cl_obj_pcd_pts = np.concatenate(target_pcd_world_cl_list, axis=0)
            cl_obj_pcd_pts = manually_segment_pcd(cl_obj_pcd_pts, x=cropx, y=cropy, z=cropz, note=crop_note)
            util.meshcat_pcd_show(self.mc_vis, cl_obj_pcd_pts, (255, 0, 0), f'{mc_iter_name}/obj_pcd', size=0.003)

            target_pcd = cl_obj_pcd_pts

        self.cached_target_obj_pcd_obs = copy.deepcopy(target_pcd)

        cropped_scene_pcd_pts = manually_segment_pcd(scene_pcd_pts, x=tab_cropx, y=tab_cropy, z=tab_cropz, note=table_crop_note)
        target_obb = trimesh.PointCloud(target_pcd).bounding_box_oriented.to_mesh()
        in_idx = inside_mesh.check_mesh_contains(target_obb, cropped_scene_pcd_pts)
        cropped_scene_pcd_notarget_pts = cropped_scene_pcd_pts[np.logical_not(in_idx)]

        pcd_dict = dict(
            target_obj=target_pcd, 
            scene_cropped=cropped_scene_pcd_notarget_pts,
            scene=pcd_pts,
            target_mask=obj_kp_mask_global_list,
            valid_view=valid_view_list
        )

        self.cached_pcd_dict = copy.deepcopy(pcd_dict)

        return pcd_dict


class RealsenseInterface:
    def __init__(self, apply_scale_depth=False):
        self.depth_scale = 0.001
        self.apply_scale_depth = apply_scale_depth

    def get_rgb_and_depth_image(self, pipeline):
        align_to = rs.stream.color
        align = rs.align(align_to)

        device, pipe = pipeline

        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
            # print(e)
            print(f"Couldn't get frame for device: {device}")
            return -1

        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return -1

        scale_to_use = self.depth_scale if self.apply_scale_depth else 1.0
        # depth_image = np.asanyarray(aligned_depth_frame.get_data()) * self.depth_scale
        depth_image = np.asanyarray(aligned_depth_frame.get_data()) * scale_to_use
        color_image = np.asanyarray(color_frame.get_data())

        return color_image, depth_image
    
    def get_intrinsics_mat(self, pipeline):
        align_to = rs.stream.color
        align = rs.align(align_to)

        device, pipe = pipeline

        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
            # print(e)
            print(f"Couldn't get frame for device: {device}")
            return np.eye(3)

        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            return np.eye(3)

        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

        intrinsics_matrix = np.array(
            [[depth_intrin.fx, 0., depth_intrin.ppx],
            [0., depth_intrin.fy, depth_intrin.ppy],
            [0., 0., 1.]]
        )
        return intrinsics_matrix


def enable_devices(serials, ctx, resolution_width = 640,resolution_height = 480, frame_rate = 30):
    pipelines = []
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        cfg.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)
        pipe.start(cfg)
        time.sleep(1.0)
        pipelines.append([serial,pipe])

    return pipelines


def pipeline_stop(pipelines):
    for (device,pipe) in pipelines:
        # Stop streaming
        pipe.stop()