import os, os.path as osp
import json
import numpy as np
import copy
import cv2
import matplotlib.pyplot as plt
from yacs.config import CfgNode as CN

from airobot.sensor.camera.rgbdcam import RGBDCamera


class MultiRealsenseLocal:
    def __init__(self, cam_names, calib_filenames, width=640, height=480, cfg=None):
        self.cams = []
        self.names = cam_names
        self.calib_filenames = calib_filenames
        self.width = width
        self.height = height
        self.cfg = cfg

        # for i in range(1, n_cam+1):
        for i, name in enumerate(self.names):
            print('Initializing camera %s' % name)
            
            cam_cfg = self._camera_cfgs(name)
            cam = RGBDCamera(cfgs=cam_cfg)
            cam.depth_scale = 1.0
            cam.img_height = cam_cfg.CAM.SIM.HEIGHT
            cam.img_width = cam_cfg.CAM.SIM.WIDTH
            cam.depth_min = cam_cfg.CAM.SIM.ZNEAR
            cam.depth_max = cam_cfg.CAM.SIM.ZFAR

            # read_cam_ext obtains extrinsic calibration from file that has previously been saved
            pos, ori = self._read_cam_ext(self.calib_filenames[i])
            cam.set_cam_ext(pos, ori)

            self.cams.append(cam)

    def _read_cam_ext(self, cal_filename):
        assert osp.exists(cal_filename), f'Calibration filename: {cal_filename} does not exist!'

        with open(cal_filename, 'r') as f:
            calib_data = json.load(f)

        cam_pos = np.array(calib_data['b_c_transform']['position'])
        cam_ori = np.array(calib_data['b_c_transform']['orientation'])

        return cam_pos, cam_ori

    def _camera_cfgs(self, name):
        """Returns set of camera config parameters

        Returns:
        YACS CfgNode: Cam config params
        """
        _C = CN()
        _C.ZNEAR = 0.01
        _C.ZFAR = 1
        _C.WIDTH = self.width
        _C.HEIGHT = self.height
        _C.FOV = 60
        _ROOT_C = CN()
        _ROOT_C.CAM = CN()
        _ROOT_C.CAM.SIM = _C
        _ROOT_C.CAM.REAL = _C
        return _ROOT_C.clone()

    def get_observation(self, depth_max=1.0, color_seg=False):
        """
        Function to get an observation from multiple realsense cameras, containing
        just the color image, depth image, and corresponding 3D point cloud 
        built using the underlying extrinsic matrix stored for each camera in airobot.

        Args:
            depth_max (float, optional): Max depth to capture in depth image.
                Defaults to 1.0.

        Returns:
            dict: Contains observation data, with keys for
            - rgb: list of np.ndarrays for each RGB image
            - depth: list of np.ndarrays for each depth image
            - pcd_pts: list of np.ndarrays for each segmented point cloud
            - pcd_colors: list of np.ndarrays for the colors corresponding
                to the points of each segmented pointcloud
        """
        rgbs = []
        depths = []
        pcd_pts = []
        pcd_colors = []

        for cam in self.cams:
            rgb, depth = cam.get_images(
                get_rgb=True,
                get_depth=True
            )

            pts_raw, colors_raw = cam.get_pcd(
                in_world=True,
                filter_depth=False,
                depth_max=depth_max
            )
            flat_seg = None
            if color_seg:
                hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

                # lower_red = np.array([0, 200, 10])
                # upper_red = np.array([15, 255, 255])            
                # seg = cv2.inRange(hsv, lower_red, upper_red)

                lower_orange = np.array([1, 160, 100])
                upper_orange = np.array([45, 255, 255])            
                seg = cv2.inRange(hsv, lower_orange, upper_orange)

                print(np.where(seg))
                # plt.imshow(rgb * seg[:, :, None])
                # plt.imshow(cv2.bitwise_and(rgb, rgb, mask=seg.astype(np.uint8)))
                # plt.show()

                flat_seg = seg.flatten()
                # print(seg.shape, rgb.shape, depth.shape)
                # # # flat_seg = np.ones(depth.shape[0]*depth.shape[1]).astype(bool)
                # from IPython import embed; embed()

                # rgb = rgb * seg[:, :, None]
                # depth = depth * seg
                rgb = cv2.bitwise_and(rgb, rgb, mask=seg.astype(np.uint8))
                depth = cv2.bitwise_and(depth, depth, mask=seg.astype(np.uint8))


                # lower_table = np.array([30, 0, 0])
                # upper_table = np.array([255, 255, 255])            

                # table_seg_color = cv2.inRange(hsv, lower_table, upper_table)  

            # pts_raw, colors_raw = cam.get_pcd(
            #     in_world=True,
            #     filter_depth=False,
            #     depth_max=depth_max
            # )

            rgbs.append(copy.deepcopy(rgb))
            depths.append(copy.deepcopy(depth))

            if flat_seg is not None:
                pts_raw = pts_raw[np.where(flat_seg)[0]]
                colors_raw = colors_raw[np.where(flat_seg)[0]]

            pcd_pts.append(pts_raw)
            pcd_colors.append(colors_raw)

        obs_dict = {}
        obs_dict['rgb'] = rgbs
        obs_dict['depth'] = depths
        obs_dict['pcd_pts'] = pcd_pts
        obs_dict['pcd_colors'] = pcd_colors
        return obs_dict
