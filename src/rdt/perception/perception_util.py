import warnings
import copy
import numpy as np
import pyrealsense2 as rs
from matplotlib import pyplot as plt

from rdt.common import util
from rdt.point_cloud.pcd_utils import manually_segment_pcd

from typing import List
from meshcat import Visualizer
from rdt.camera.simple_multicam import MultiRGBDCalibrated
from rdt.perception.realsense_util import RealsenseInterface

try:
    import visdom
except ImportError as e:
    warnings.warn(f'Error with importing visdom: {e}, cannot use Visdom for image visualization until installed', ImportWarning)
    
    class DummyVisdom:
        class Visdom:
            def __init__(self):
                pass

        def __init__(self):
            self.Visdom = Visdom()
    
    visdom = DummyVisdom()

from rdt.segmentation.sam import SAMSeg
from rdt.segmentation.annotate import Annotate


class RGBDPointCloudHelper:
    def __init__(self, mc_vis: Visualizer, cams: MultiRGBDCalibrated,
                 realsense_interface: RealsenseInterface, realsense_pipelines: List[rs.pipeline]):

        self.mc_vis = mc_vis
        self.cams = cams
        self.realsense_interface = realsense_interface
        self.realsense_pipelines = realsense_pipelines

        self.current_full_observation_dict = {}

        self.vis_vis = None  # can set manually to a handler for using Visdom with self.setup_visdom
    
    def setup_visdom(self, vis_vis: visdom.Visdom):
        self.vis_vis = vis_vis

    def get_current_full_observation_dict(self) -> dict:
        return copy.deepcopy(self.current_full_observation_dict)
    
    def show_current_rgb_imgs(self, use_pyplot: bool=False, title: str='current_rgb'):
        if 'rgb_imgs' not in self.current_full_observation_dict:
            print(f'No RGB images to show currently, please call "get_img_depth_pcd"')
            return

        if use_pyplot:
            _, axs = plt.subplots(1, len(self.current_full_observation_dict['rgb_imgs']))
            
            for i, rgb_img in enumerate(self.current_full_observation_dict['rgb_imgs']):
                axs[i].imshow(rgb_img)
            
            plt.show()

            return

        if self.vis_vis is None:
            print(f'No interface to Visdom, cannot show (pass "use_pyplot" to show with matplotlib)')
        
        rgb_to_show = []
        for rgb_img in self.current_full_observation_dict['rgb_imgs']:
            rgb_to_show.append(rgb_img.copy().transpose(2, 0, 1))
        
        self.vis_vis.images(rgb_to_show, padding=1, opts=dict(title=f'{title}'))
    
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


class SelectKeypointSegmentedPointCloudHelperSAM(RGBDPointCloudHelper):
    def __init__(self, mc_vis: Visualizer, cams: MultiRGBDCalibrated,
                 realsense_interface: RealsenseInterface, realsense_pipelines: List[rs.pipeline]):
        
        super().__init__(mc_vis=mc_vis, 
                         cams=cams, 
                         realsense_interface=realsense_interface, 
                         realsense_pipelines=realsense_pipelines)

        self.sam_seg = SAMSeg(cuda=True)

        self.setup_scene()
    
    def setup_scene(self):
        cropx, cropy, cropz = [0.2, 0.85], [-0.45, 0.5], [-0.003, 1]
        self.scene_bound = (cropx, cropy, cropz)

    def get_segmented_object_pcd(self) -> dict:
        img_depth_pcd_dict = self.get_img_depth_pcd()
        n_imgs = len(img_depth_pcd_dict['rgb_imgs'])
        pcd_dict_list = img_depth_pcd_dict['pcd_dict_list']
        pcd_pts = [pcd_dict_list[i]['world'] for i in range(n_imgs)]
        rgb_imgs = img_depth_pcd_dict['rgb_imgs']

        full_pcd = np.concatenate(pcd_pts, axis=0)
        
        # crop the point cloud to the table
        full_proc_pcd = manually_segment_pcd(full_pcd, bounds=self.scene_bound)[0]
        util.meshcat_pcd_show(self.mc_vis, full_proc_pcd, name='scene/get_seg_obs/cropped_scene')

        object_pts = []
        scene_pts = []
        object_mask_list = []
        scene_mask_list = []

        for idx, _ in enumerate(self.cams.cams):
            rgb = rgb_imgs[idx]
            pcd_world_img = pcd_dict_list[idx]['world_img']
            pcd_world = pcd_dict_list[idx]['world']

            object_annotator = Annotate()
            object_bb = object_annotator.select_bb(rgb, f'Select object in scene')

            if object_bb is not None:
                object_seg_mask = self.sam_seg.mask_from_bb(object_bb, image=rgb)
                scene_seg_mask = np.logical_not(object_seg_mask)

                crop_mask = manually_segment_pcd(pcd_world, bounds=self.scene_bound)[1]
                crop_mask = crop_mask.reshape(pcd_world_img.shape[0], pcd_world_img.shape[1])

                object_mask = object_seg_mask & crop_mask
                scene_mask = scene_seg_mask & crop_mask

                object_partial_pcd = pcd_world_img[object_mask].reshape((-1,3))
                # object_partial_pcd = manually_segment_pcd(object_partial_pcd, bounds=self.scene_bound)
                object_pts.append(object_partial_pcd)

                scene_partial_pcd = pcd_world_img[scene_mask].reshape((-1,3))
                # scene_partial_pcd = manually_segment_pcd(scene_partial_pcd, bounds=self.scene_bound)
                scene_pts.append(scene_partial_pcd)

                object_mask_list.append(object_mask)
                scene_mask_list.append(scene_mask)

        out_pcd_dict = {}
        out_pcd_dict['object_pcd'] = None
        out_pcd_dict['scene_pcd'] = None
        if len(object_pts) > 0:
            object_pcd = np.concatenate(object_pts, axis=0)
            out_pcd_dict['object_pcd'] = object_pcd

        if len(scene_pts) > 0:
            scene_pcd = np.concatenate(scene_pts, axis=0)
            out_pcd_dict['scene_pcd'] = scene_pcd
        
        out_pcd_dict['object_masks'] = object_mask_list
        out_pcd_dict['scene_masks'] = scene_mask_list

        return out_pcd_dict

