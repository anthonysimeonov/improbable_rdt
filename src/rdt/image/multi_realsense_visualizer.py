# Libraries
import os, os.path as osp
import time
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import argparse
import copy

from rdt.common import path_util, lcm_util
from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg


def find_devices():
    ctx = rs.context() # Create librealsense context for managing devices
    serials = []
    if (len(ctx.devices) > 0):
        for dev in ctx.devices:
            print ('Found device: ', \
                    dev.get_info(rs.camera_info.name), ' ', \
                    dev.get_info(rs.camera_info.serial_number))
            serials.append(dev.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")
        
    return serials, ctx


def enable_devices(serials, ctx, resolution_width=640, resolution_height=480, frame_rate=30):
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

            
def pipeline_stop(pipelines):
    for (device,pipe) in pipelines:
        # Stop streaming
        pipe.stop() 


def Visualize(pipelines):
    align_to = rs.stream.color
    align = rs.align(align_to)

    for (device,pipe) in pipelines:
        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
            # print(e)
            print(f"Couldn't get frame for device: {device}")
            continue
        # frames.get_depth_frame() is a 640x360 depth image
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # Render images
        depth_colormap = cv.applyColorMap(cv.convertScaleAbs(depth_image, alpha=0.03), cv.COLORMAP_JET)

        show_images = np.hstack((cv.cvtColor(color_image, cv.COLOR_RGB2BGR), depth_colormap))
        cv.imshow('RealSense' + device, show_images)
        key = cv.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv.destroyAllWindows()
            return True
            
        # Save images and depth maps from both cameras by pressing 's'
        if key==115:
            cv.imwrite( str(device) + '_aligned_depth.png', depth_image)
            cv.imwrite( str(device) + '_aligned_color.png', color_image)
            print('Save')
        

def main(args):

    rs_cfg = get_default_multi_realsense_cfg()
    
    if args.rs_config is not None:
        rs_config_fname = osp.join(path_util.get_rrp_config(), 'real_cam_cfgs', args.rs_config)
        if osp.exists(rs_config_fname):
            rs_cfg.merge_from_file(rs_config_fname)
        else:
            print(f'Config file {rs_config_fname} does not exist, using defaults')
    rs_cfg.freeze()

    if args.find_devices:
        serials, ctx = find_devices()
    else:
        ctx = rs.context() # Create librealsense context for managing devices
        serials = rs_cfg.SERIAL_NUMBERS
    
    prefix = rs_cfg.CAMERA_NAME_PREFIX
    camera_names = [f'{prefix}{i}' for i in range(len(serials))]

    resolution_width = rs_cfg.WIDTH # pixels
    resolution_height = rs_cfg.HEIGHT # pixels
    frame_rate = rs_cfg.FRAME_RATE # fps

    pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)

    time.sleep(1.0)

    try:
        while True:
            exit = Visualize(pipelines)
            if exit == True:
                print('Program closing...')
                break
    finally:
        pipeline_stop(pipelines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--find_devices', action='store_true', help='If True, look for whatever devices are connected. Else, use the serial numbers from the config')
    parser.add_argument('--rs_config', type=str, default=None, help='Can provide a .yaml file located in the src/rdt/config/real_cam_cfgs directory')
    args = parser.parse_args()
    main(args)
