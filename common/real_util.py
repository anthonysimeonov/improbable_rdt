import os, os.path as osp
import sys
import time
import numpy as np
import threading
import copy

from rrp_robot.utils import util, path_util, lcm_util

sys.path.append(osp.join(path_util.get_rrp_src(), 'lcm_types'))
from rrp_robot.lcm_types.rrp_lcm import (
    point_t, quaternion_t, pose_t, pose_stamped_t, start_goal_pose_stamped_t, 
    point_cloud_t, point_cloud_array_t, simple_img_t, simple_depth_img_t, square_matrix_t)


class RealImageLCMSubscriber:
    def __init__(self, lc, rgb_img_sub_name, depth_img_sub_name):
        self.lc = lc

        # rgb stuff
        self.rgb_img_sub_name = rgb_img_sub_name
        self.rgb_img_sub = self.lc.subscribe(self.rgb_img_sub_name, self.rgb_img_sub_handler)

        # depth stuff
        self.depth_img_sub_name = depth_img_sub_name
        self.depth_img_sub = self.lc.subscribe(self.depth_img_sub_name, self.depth_img_sub_handler)

        self._rgb_image = None
        self._depth_image = None
        self._image_lock = threading.RLock()
        self._depth_lock = threading.RLock()

    def rgb_img_sub_handler(self, channel, data):
        msg = simple_img_t.decode(data)
        img_np = lcm_util.unpack_img_lcm(msg.pixels, msg.height, msg.width, depth=False)
        with self._image_lock:
            self._rgb_image = img_np

    def get_rgb_img(self, block=False):
        while True:
            with self._image_lock:
                if self._rgb_image is not None:
                    rgb_img = copy.deepcopy(self._rgb_image).astype(np.uint8)
                else:
                    rgb_img = None
                
            if rgb_img is not None or not block:
                with self._image_lock:
                    self._rgb_image = None
                break
            time.sleep(0.01)
        return rgb_img

    def depth_img_sub_handler(self, channel, data):
        msg = simple_depth_img_t.decode(data)
        img_np = lcm_util.unpack_img_lcm(msg.pixels, msg.height, msg.width, depth=True)
        with self._depth_lock:
            self._depth_image = img_np

    def get_depth_img(self, block=False):
        while True:
            with self._depth_lock:
                if self._depth_image is not None:
                    depth_img = copy.deepcopy(self._depth_image).astype(np.uint16)
                else:
                    depth_img = None
            
            if depth_img is not None or not block:
                with self._depth_lock:
                    self._depth_image = None
                break
            time.sleep(0.01)
        return depth_img

    def get_rgb_and_depth(self, block=False):
        while True:
            rgb = self.get_rgb_img()
            depth = self.get_depth_img()

            if (rgb is not None and depth is not None) or not block:
                break
            time.sleep(0.01)
        return rgb, depth


class RealCamInfoLCMSubscriber:
    def __init__(self, lc, cam_pose_sub_name, cam_intrinsics_sub_name):
        self.lc = lc
        self.cam_pose_sub_name = cam_pose_sub_name
        self.cam_intrinsics_sub_name = cam_intrinsics_sub_name
        self.cam_pose_sub = self.lc.subscribe(self.cam_pose_sub_name, self.cam_pose_sub_handler)
        self.cam_int_sub = self.lc.subscribe(self.cam_intrinsics_sub_name, self.cam_intrinsics_sub_handler)

        self.cam_pose = None
        self.cam_int = None
        self._cam_int_lock = threading.RLock()
        self._cam_pose_lock = threading.RLock()

    def cam_pose_sub_handler(self, channel, data):
        msg = pose_stamped_t.decode(data)
        with self._cam_pose_lock:
            self.cam_pose = lcm_util.pose_stamped2list(msg)

    def cam_intrinsics_sub_handler(self, channel, data):
        msg = square_matrix_t.decode(data)
        with self._cam_int_lock:
            self.cam_int = np.asarray(msg.entries).reshape(3, 3)

    def get_cam_pose(self, block=False):
        while True:
            with self._cam_pose_lock:
                cam_pose = copy.deepcopy(self.cam_pose)
            if cam_pose is not None or not block:
                with self._cam_pose_lock:
                    self.cam_pose = None
                break
            time.sleep(0.01)
        return cam_pose

    def get_cam_intrinsics(self, block=False):
        while True:
            with self._cam_int_lock:
                cam_int = copy.deepcopy(self.cam_int)
            if cam_int is not None or not block:
                with self._cam_int_lock:
                    self.cam_int = None
                break
            time.sleep(0.01)
        return cam_int


class RealPCDLCMSubscriber:
    def __init__(self, lc, pcd_sub_name):
        self.lc = lc
        self.pcd_sub_name = pcd_sub_name
        self.pcd_sub = self.lc.subscribe(self.pcd_sub_name, self.pcd_sub_handler)

        self.points = None
        self._pcd_lock = threading.RLock()
        self._pcd_arr_lock = threading.RLock()

    def pcd_sub_handler(self, channel, data):
        msg = point_cloud_t.decode(data)
        points = msg.points
        num_pts = msg.num_points
        
        with self._pcd_lock:
            self.points = lcm_util.unpack_pointcloud_lcm(points, num_pts)

    def get_pcd(self, block=False):
        while True:
            with self._pcd_lock:
                pcd = copy.deepcopy(self.points)
            if pcd is not None or not block:
                self.points = None
                break
        return np.asarray(pcd).reshape(-1, 3)


class RealEEPoseLCMSubscriber:
    def __init__(self, lc, ee_pose_sub_name):
        self.lc = lc
        self.ee_pose_sub_name = ee_pose_sub_name
        self.ee_pose_sub = self.lc.subscribe(self.ee_pose_sub_name, self.ee_pose_sub_handler)

        self.ee_pose = None
        self._ee_lock = threading.RLock()
    
    def ee_pose_sub_handler(self, channel, data):
        msg = pose_stamped_t.decode(data)
        with self._ee_lock:
            self.ee_pose = lcm_util.pose_stamped2list(msg)

    def get_ee_pose(self, block=False):
        while True:
            # wait to receive commands from the optimizer
            with self._ee_lock:  
                ee_pose = copy.deepcopy(self.ee_pose)
            if ee_pose is not None or not block:
                break
            time.sleep(0.01)
        return ee_pose


class RealDualEEPoseLCMSubscriber:
    def __init__(self, lc, ee_pose_sub_name, demo_offset_z=0.0):
        self.lc = lc
        self.ee_pose_sub_name = ee_pose_sub_name
        self.ee_sub = self.lc.subscribe(self.ee_pose_sub_name, self.ee_pose_sub_handler)
        self.demo_offset_z = demo_offset_z

        self.start_pose = None
        self.goal_pose = None
        self._pose_lock = threading.RLock()
    
    def ee_pose_sub_handler(self, channel, data):
        msg = start_goal_pose_stamped_t.decode(data)
        with self._pose_lock:
            print(f'Got something on channel: {channel}')
            self.start_pose = lcm_util.pose_stamped2list(msg.start_pose)
            self.goal_pose = lcm_util.pose_stamped2list(msg.goal_pose)

    def get_ee_pose(self, block=False):
        while True:
            with self._pose_lock:
                start_pose = copy.deepcopy(self.start_pose)
                goal_pose = copy.deepcopy(self.goal_pose)
            
            if (start_pose is not None and goal_pose is not None) or not block:
                with self._pose_lock:
                    self.start_pose = None
                    self.goal_pose = None
                break
            time.sleep(0.01)
        return start_pose, goal_pose


class RealImageSubscriber:
    def __init__(self, lc, rgb_img_sub_name, depth_img_sub_name,
                 rgb_img_arr_sub_name=None, depth_img_arr_sub_name=None, timeout=120.0, use_timeout=False):
        self.lc = lc
        # rgb stuff
        self.rgb_img_sub_name = rgb_img_sub_name
        self.rgb_img_arr_sub_name = rgb_img_arr_sub_name
        self.rgb_img_sub = self.lc.subscribe(self.rgb_img_sub_name, self.rgb_img_sub_handler)
        #self.rgb_img_arr_sub = self.lc.subscribe(self.rgb_img_sub_name, self.rgb_img_arr_sub_handler)

        # depth stuff
        self.depth_img_sub_name = depth_img_sub_name
        self.depth_img_arr_sub_name = depth_img_arr_sub_name
        self.depth_img_sub = self.lc.subscribe(self.depth_img_sub_name, self.depth_img_sub_handler)
        #self.depth_img_arr_sub = self.lc.subscribe(self.depth_img_sub_name, self.depth_img_arr_sub_handler)

        self.sub_timeout = timeout
        self.use_timeout = use_timeout

    def rgb_img_sub_handler(self, channel, data):
        msg = simple_img_t.decode(data)
        img_np = lcm_util.unpack_img_lcm(msg.pixels, msg.height, msg.width, depth=False)
        self.rgb_img = img_np
        self.received_rgb_img = True

    def rgb_img_arr_sub_handler(self, channel, data):
        raise NotImplementedError

    def get_rgb_img(self):
        self.received_rgb_img = False
        while True:
            if self.received_rgb_img:
                break
            if self.use_timeout:
                self.lc.handle_timeout(self.sub_timeout)
            else:
               self.lc.handle()
        rgb_img = self.rgb_img
        return rgb_img

    def get_rgb_img_arr(self):
        raise NotImplementedError

    def depth_img_sub_handler(self, channel, data):
        msg = simple_depth_img_t.decode(data)
        img_np = lcm_util.unpack_img_lcm(msg.pixels, msg.height, msg.width, depth=True)
        self.depth_img = img_np
        self.received_depth_img = True

    def depth_img_arr_sub_handler(self, channel, data):
        raise NotImplementedError

    def get_depth_img(self):
        self.received_depth_img = False
        while True:
            if self.received_depth_img:
                break
            if self.use_timeout:
                self.lc.handle_timeout(self.sub_timeout)
            else:
                self.lc.handle()
        depth_img = self.depth_img
        return depth_img

    def get_depth_img_arr(self):
        raise NotImplementedError

    def get_rgb_and_depth(self):
        rgb = self.get_rgb_img()
        depth = self.get_depth_img()
        return rgb, depth


class RealPCDSubscriber:
    def __init__(self, lc, pcd_sub_name, pcd_arr_sub_name):
        self.lc = lc
        self.pcd_sub_name = pcd_sub_name
        self.pcd_arr_sub_name = pcd_arr_sub_name
        self.pcd_sub = self.lc.subscribe(self.pcd_sub_name, self.pcd_sub_handler)
        self.pcd_arr_sub = self.lc.subscribe(self.pcd_arr_sub_name, self.pcd_arr_sub_handler)

    def pcd_sub_handler(self, channel, data):
        msg = point_cloud_t.decode(data)
        points = msg.points
        num_pts = msg.num_points

        self.points = lcm_util.unpack_pointcloud_lcm(points, num_pts)
        self.received_pcd_data = True

    def pcd_arr_sub_handler(self, channel, data):
        msg = point_cloud_array_t.decode(data)
        points_list = []
        for i in range(msg.num_point_clouds):
            points = msg.point_clouds[i].points
            num_pts = msg.point_clouds[i].num_points
            points = lcm_util.unpack_pointcloud_lcm(points, num_pts)
            points_list.append(points)
        self.points_list = points_list
        self.received_pcd_data = True

    def get_pcd(self):
        self.received_pcd_data = False
        while True:
            if self.received_pcd_data:
                break
            self.lc.handle()
        pcd = self.points
        return pcd

    def get_pcd_arr(self):
        self.received_pcd_data = False
        while True:
            if self.received_pcd_data:
                break
            self.lc.handle()
        pcd_list = self.points_list
        return pcd_list


class RealCamInfoSubscriber:
    def __init__(self, lc, cam_pose_sub_name, cam_intrinsics_sub_name, timeout=10.0, use_timeout=False):
        self.lc = lc
        self.cam_pose_sub_name = cam_pose_sub_name
        self.cam_intrinsics_sub_name = cam_intrinsics_sub_name
        self.cam_pose_sub = self.lc.subscribe(self.cam_pose_sub_name, self.cam_pose_sub_handler)
        self.cam_int_sub = self.lc.subscribe(self.cam_intrinsics_sub_name, self.cam_intrinsics_sub_handler)

        self.sub_timeout = timeout
        self.use_timeout = use_timeout
    
    def cam_pose_sub_handler(self, channel, data):
        msg = pose_stamped_t.decode(data)
        print('got something')
        self.cam_pose = lcm_util.pose_stamped2list(msg)
        self.received_pose_message = True

    def cam_intrinsics_sub_handler(self, channel, data):
        msg = square_matrix_t.decode(data)
        self.cam_int = np.asarray(msg.entries).reshape(3, 3)
        self.received_intrinsics_message = True

    def get_cam_pose(self):
        # wait to receive commands from the optimizer
        self.received_pose_message = False
        while True:
            if self.received_pose_message:
                break
            # self.lc.handle()
            if self.use_timeout:
                self.lc.handle_timeout(self.sub_timeout)
            else:
                self.lc.handle()
            time.sleep(0.001)
        
        return self.cam_pose

    def get_cam_intrinsics(self):
        self.received_intrinsics_message = False
        while True:
            if self.received_intrinsics_message:
                break
            # self.lc.handle()
            if self.use_timeout:
                self.lc.handle_timeout(self.sub_timeout)
            else:
                self.lc.handle()
            time.sleep(0.001)
        return self.cam_int


class RealEEPoseSubscriber:
    def __init__(self, lc, ee_pose_sub_name):
        self.lc = lc
        self.ee_pose_sub_name = ee_pose_sub_name
        self.ee_pose_sub = self.lc.subscribe(self.ee_pose_sub_name, self.ee_pose_sub_handler)
    
    def ee_pose_sub_handler(self, channel, data):
        msg = pose_stamped_t.decode(data)
        print('got something')
        self.ee_pose = lcm_util.pose_stamped2list(msg)
        self.received_pose_message = True

    def get_ee_pose(self):
        # wait to receive commands from the optimizer
        self.received_pose_message = False
        while True:
            if self.received_pose_message:
                break
            self.lc.handle()
            time.sleep(0.001)
        
        return self.ee_pose
