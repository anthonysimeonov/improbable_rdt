import os, os.path as osp
import json
import numpy as np
from yacs.config import CfgNode as CN

from rdt.camera.rgbdcam import RGBDCamera


class MultiRGBDCalibrated:
    def __init__(self, cam_names, calib_filenames=None, width=640, height=480, cfg=None):
        self.cams = []
        self.names = cam_names
        self.calib_filenames = calib_filenames
        self.width = width
        self.height = height
        self.cfg = cfg

        if calib_filenames is not None:
            assert len(calib_filenames) == len(cam_names), f'Not enough calibration filenames'
            for fname in calib_filenames:
                assert osp.exists(fname), f'Calib filename {fname} does not exist!'

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

            if calib_filenames is not None:
                # read_cam_ext obtains extrinsic calibration from file that has previously been saved
                pos, ori = self._read_cam_ext(self.calib_filenames[i])
                cam.set_cam_ext(pos, ori)
            else:
                cam.set_cam_ext(np.zeros(3), np.array([0., 0., 0., 1.]))  # use identify for calibration

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
