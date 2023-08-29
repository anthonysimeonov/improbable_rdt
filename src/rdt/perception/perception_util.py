import os, os.path as osp
import copy
import time
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d
import pyrealsense2 as rs

from rdt.common import util
from rdt.point_cloud.pcd_utils import manually_segment_pcd

from typing import List
from meshcat import Visualizer
from rdt.camera.simple_multicam import MultiRealsenseLocal
from rdt.common.real_util import RealImageLCMSubscriber
from rdt.segmentation.keypoint_select import ObjectKeypointSelection
from rdt.segmentation.instance_segmentation import InstanceSegServer
from rdt.segmentation.sam import SAMSeg
from rdt.segmentation.annotate import Annotate


class RealsenseInterface:
    def __init__(self, apply_scale_depth: bool=False):
        self.depth_scale = 0.001
        self.apply_scale_depth = apply_scale_depth

    def get_rgb_and_depth_image(self, pipeline: rs.pipeline):
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
    
    def get_intrinsics_mat(self, pipeline: rs.pipeline):
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


def enable_devices(serials: List[str], ctx: rs.context, resolution_width: int=640, resolution_height: int=480, frame_rate: int=30):
    pipelines = []
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
        cfg.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.rgb8, frame_rate)
        pipe.start(cfg)
        time.sleep(1.0)
        pipelines.append([serial,pipe])

    return pipelines


def pipeline_stop(pipelines: List[rs.pipeline]):
    for (device, pipe) in pipelines:
        # Stop streaming
        pipe.stop()


class SelectKeypointSegmentedPointCloudHelperSAM:
    def __init__(self, mc_vis: Visualizer, cams: MultiRealsenseLocal,
                 realsense_interface: RealsenseInterface, realsense_pipelines: List[rs.pipeline]):
        self.mc_vis = mc_vis
        self.sam_seg = SAMSeg(cuda=True)
        self.cams = cams
        self.realsense_interface = realsense_interface
        self.realsense_pipelines = realsense_pipelines

        self.current_full_observation_dict = None

        self.setup_scene()
    
    def setup_scene(self):
        cropx, cropy, cropz = [0.2, 0.85], [-0.45, 0.5], [-0.003, 1]
        self.scene_bound = (cropx, cropy, cropz)
    
    def get_current_full_observation_dict(self) -> dict:
        return self.current_full_observation_dict
    
    def get_img_depth_pcd(self) -> dict:
        pcd_pts = []
        pcd_dict_list = []
        cam_int_list = []
        cam_poses_list = []
        rgb_imgs = []
        depth_imgs = []
        for idx, cam in enumerate(self.cams.cams):
            rgb, depth = self.realsense_interface.get_rgb_and_depth_image(self.realsense_pipelines[idx])
            rgb_imgs.append(rgb)                
            cam_intrinsics = self.realsense_interface.get_intrinsics_mat(self.realsense_pipelines[idx])
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
            util.meshcat_pcd_show(self.mc_vis, pcd_world, name=f'scene/get_obs/pcd_world_cam_{idx}')
        
        out_dict = {}
        out_dict['pcd_dict_list'] = pcd_dict_list
        out_dict['cam_int_list'] = cam_poses_list
        out_dict['cam_poses_list'] = cam_poses_list
        out_dict['rgb_imgs'] = rgb_imgs
        out_dict['depth_imgs'] = depth_imgs

        self.current_full_observation_dict = copy.deepcopy(out_dict)

        return out_dict

    def get_segmented_object_pcd(self) -> dict:
        img_depth_pcd_dict = self.get_img_depth_pcd()
        n_imgs = len(img_depth_pcd_dict['rgb_imgs'])
        pcd_dict_list = img_depth_pcd_dict['pcd_dict_list']
        pcd_pts = [pcd_dict_list[i]['world'] for i in range(n_imgs)]
        rgb_imgs = img_depth_pcd_dict['rgb_imgs']

        full_pcd = np.concatenate(pcd_pts, axis=0)
        
        # crop the point cloud to the table
        full_proc_pcd = manually_segment_pcd(full_pcd, bounds=self.scene_bound)
        util.meshcat_pcd_show(self.mc_vis, full_proc_pcd, name='scene/get_seg_obs/cropped_scene')

        object_pts = []
        scene_pts = []

        for idx, _ in enumerate(self.cams.cams):
            rgb = rgb_imgs[idx]
            pcd_world_img = pcd_dict_list[idx]['world_img']

            object_annotator = Annotate()
            object_bb = object_annotator.select_bb(rgb, f'Select object in scene')

            if object_bb is not None:
                object_mask = self.sam_seg.mask_from_bb(object_bb, image=rgb)
                scene_mask = np.logical_not(object_mask)
                object_partial_pcd = pcd_world_img[object_mask].reshape((-1,3))
                object_partial_pcd = manually_segment_pcd(object_partial_pcd, bounds=self.scene_bound)
                object_pts.append(object_partial_pcd)

                scene_partial_pcd = pcd_world_img[scene_mask].reshape((-1,3))
                scene_partial_pcd = manually_segment_pcd(scene_partial_pcd, bounds=self.scene_bound)
                scene_pts.append(scene_partial_pcd)

        out_pcd_dict = {}
        out_pcd_dict['object_pcd'] = None
        out_pcd_dict['scene_pcd'] = None
        if len(object_pts) > 0:
            object_pcd = np.concatenate(object_pts, axis=0)
            out_pcd_dict['object_pcd'] = object_pcd

        if len(scene_pts) > 0:
            scene_pcd = np.concatenate(scene_pts, axis=0)
            out_pcd_dict['scene_pcd'] = scene_pcd

        return out_pcd_dict

