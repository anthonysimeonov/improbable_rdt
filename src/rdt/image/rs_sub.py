import os, os.path as osp
import sys
import time
import numpy as np
import cv2 as cv
import threading

import lcm

from ndf_robot.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from ndf_robot.utils.real_util import RealImageLCMSubscriber, RealCamInfoLCMSubscriber
from ndf_robot.real.multi_realsense_publisher_visualizer import subscriber_visualize

def lc_th(lc):
    while True:
        lc.handle_timeout(1)
        time.sleep(0.001)
        # lc.handle_timeout(0.001)

def main():

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")

    rs_cfg = get_default_multi_realsense_cfg()
    serials = rs_cfg.SERIAL_NUMBERS

    rgb_topic_name_suffix = rs_cfg.RGB_LCM_TOPIC_NAME_SUFFIX
    depth_topic_name_suffix = rs_cfg.DEPTH_LCM_TOPIC_NAME_SUFFIX
    info_topic_name_suffix = rs_cfg.INFO_LCM_TOPIC_NAME_SUFFIX
    pose_topic_name_suffix = rs_cfg.POSE_LCM_TOPIC_NAME_SUFFIX

    prefix = rs_cfg.CAMERA_NAME_PREFIX
    camera_names = [f'{prefix}{i}' for i in range(len(serials))]

    # update the topic names based on each individual camera
    rgb_sub_names = [f'{cam_name}_{rgb_topic_name_suffix}' for cam_name in camera_names]
    depth_sub_names = [f'{cam_name}_{depth_topic_name_suffix}' for cam_name in camera_names]
    info_sub_names = [f'{cam_name}_{info_topic_name_suffix}' for cam_name in camera_names]
    pose_sub_names = [f'{cam_name}_{pose_topic_name_suffix}' for cam_name in camera_names]

    img_subscribers = []
    for i, name in enumerate(camera_names):
        img_sub = RealImageLCMSubscriber(lc, rgb_sub_names[i], depth_sub_names[i])
        info_sub = RealCamInfoLCMSubscriber(lc, pose_sub_names[i], info_sub_names[i])
        img_subscribers.append((name, img_sub, info_sub))

    lc_thread = threading.Thread(target=lc_th, args=(lc,))
    lc_thread.daemon = True
    lc_thread.start()

    while True:
        exit = subscriber_visualize(img_subscribers)

        if exit == True:
            print('Program closing...')
            break

if __name__ == "__main__":
    main()
