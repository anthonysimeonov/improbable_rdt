# Libraries
import os, os.path as osp
import time
import pyrealsense2 as rs
import numpy as np
import cv2
import zlib
import struct
import multiprocessing
import argparse
import threading
import copy

import lcm

from rdt.common import path_util, lcm_util
from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.common.real_util import RealImageLCMSubscriber, RealCamInfoLCMSubscriber

from rdt.lcm_types.rdt_lcm import img_t, point_cloud_t, header_t

def subscriber_visualize(subs):
    for (name, img_sub, info_sub) in subs:
        rgb_image, depth_image = img_sub.get_rgb_and_depth(block=True)
        if rgb_image is None or depth_image is None:
            return
        # intrinsics = info_sub.get_cam_intrinsics()

        # Render images
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        show_images = np.hstack((cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR), depth_colormap))
        cv2.imshow(f'RealSense_{name}', show_images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return True
            
        # Save images and depth maps from both cameras by pressing 's'
        if key==115:
            now = time.time()
            cv2.imwrite(f'{name}_{now}_aligned_depth.png', depth_image)
            cv2.imwrite(f'{name}_{now}_aligned_color.png', rgb_image)
            print('Save')


def visualizer_process_target():
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")

    def lc_th(lc):
        while True:
            lc.handle_timeout(1)
            time.sleep(0.001)
    
    lc_thread = threading.Thread(target=lc_th, args=(lc,))
    lc_thread.daemon = True

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

    lc_thread.start()

    while True:
        exit = subscriber_visualize(img_subscribers)

        if exit == True:
            print('Program closing...')
            break

        time.sleep(0.001)

    return


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


def publish_lcm_loop(lc, rgb_topic_names, depth_topic_names, info_topic_names, pose_topic_names,
                     pipelines, publish_every_t, publish_camera_info=True):
    align_to = rs.stream.color
    align = rs.align(align_to)

    last_pub_t = time.time()
    while True:
        for (device,pipe) in pipelines:
            try:
                # Get frameset of color and depth
                frames = pipe.wait_for_frames(100)
            except RuntimeError as e:
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

            # clip depth to be valid 16-bit image (depth values that far away are wrong anyways)
            clipped_depth_image = np.clip(depth_image, -32767, 32767)

            # # publish images
            # rgb_msg = lcm_util.np2img_t(color_image.astype(np.uint8))
            # depth_msg = lcm_util.np2img_t(clipped_depth_image, depth=True)
            # # depth_msg = lcm_util.np2img_t(clipped_depth_image.astype(np.uint16), depth=True)

            if False:
                # Intrinsics & Extrinsics
                depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
                color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
                depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

                # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
                depth_sensor = pipe.get_active_profile().get_device().first_depth_sensor()
                depth_scale = depth_sensor.get_depth_scale()

                intrinsics_matrix = np.array(
                    [[depth_intrin.fx, 0., depth_intrin.ppx],
                    [0., depth_intrin.fy, depth_intrin.ppy],
                    [0., 0., 1.]]
                )

                cam_int_msg = lcm_util.pack_square_matrix(intrinsics_matrix)

            now = time.time()
            # publish_this_loop = (now - last_pub_t) >= publish_every_t
            publish_this_loop = True
            if publish_this_loop:
                
                start_pub = time.time()
                rgb_msg = img_t()
                rgb_msg.width = color_image.shape[1]
                rgb_msg.height = color_image.shape[1]

                depth_msg = img_t()
                depth_msg.width = depth_image.shape[1]
                depth_msg.height = depth_image.shape[1]

                color_retval, color_buf = cv2.imencode(".jpg", color_image)
                depth_retval, depth_buf = cv2.imencode(".jpg", depth_image)
                if color_retval and depth_retval:
                    rgb_msg.data = color_buf
                    rgb_msg.size = color_buf.size
                    depth_msg.data = depth_buf
                    depth_msg.size = depth_buf.size

                lc.publish(rgb_topic_names[device], rgb_msg.encode())
                lc.publish(depth_topic_names[device], depth_msg.encode())
                end_pub = time.time()

                if False:
                    # let's also get out the intrinsics/camera info, if we want to send that out as well
                    if publish_camera_info:
                        lc.publish(info_topic_names[device], cam_int_msg.encode())
                print(f'Delta: {now - last_pub_t}')
                print(f'Pub: {end_pub - start_pub}')
                last_pub_t = time.time()


def publish_lcm(lc, rgb_topic_names, depth_topic_names, info_topic_names, pose_topic_names,
            pipelines, publish_camera_info=True, rgbd_multicam_interface=None, publish_pointcloud=False, pointcloud_topic_names=None):
    align_to = rs.stream.color
    align = rs.align(align_to)

    for i, (device,pipe) in enumerate(pipelines):
        try:
            # Get frameset of color and depth
            frames = pipe.wait_for_frames(100)
        except RuntimeError as e:
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

        # clip depth to be valid 16-bit image (depth values that far away are wrong anyways)
        clipped_depth_image = np.clip(depth_image, -32767, 32767)

        # publish images
        # rgb_msg = lcm_util.np2img_t(color_image.astype(np.uint8))
        # depth_msg = lcm_util.np2img_t(clipped_depth_image, depth=True)

        rgb_msg = img_t()
        rgb_msg.width = color_image.shape[1]
        rgb_msg.height = color_image.shape[1]

        depth_msg = img_t()
        depth_msg.width = depth_image.shape[1]
        depth_msg.height = depth_image.shape[1]

        color_retval, color_buf = cv2.imencode(".jpg", color_image)
        # depth_retval, depth_buf = cv2.imencode(".jpg", depth_image)
        depth_retval = True
        depth_image_bytes = depth_image.tobytes()
        depth_buf = zlib.compress(depth_image_bytes)

        if color_retval and depth_retval:
            rgb_msg.data = color_buf
            rgb_msg.size = color_buf.size
            depth_msg.data = depth_buf
            depth_msg.size = len(depth_buf)

        # depth_msg = lcm_util.np2img_t(clipped_depth_image.astype(np.uint16), depth=True)
        lc.publish(rgb_topic_names[device], rgb_msg.encode())
        lc.publish(depth_topic_names[device], depth_msg.encode())

        # Intrinsics & Extrinsics
        depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        color_intrin = color_frame.profile.as_video_stream_profile().intrinsics
        depth_to_color_extrin = aligned_depth_frame.profile.get_extrinsics_to(color_frame.profile)

        # Depth scale - units of the values inside a depth frame, i.e how to convert the value to units of 1 meter
        depth_sensor = pipe.get_active_profile().get_device().first_depth_sensor()
        depth_scale = depth_sensor.get_depth_scale()

        intrinsics_matrix = np.array(
            [[depth_intrin.fx, 0., depth_intrin.ppx],
            [0., depth_intrin.fy, depth_intrin.ppy],
            [0., 0., 1.]]
        )

        if False:        
            # let's also get out the intrinsics/camera info, if we want to send that out as well
            if publish_camera_info:
                cam_int_msg = lcm_util.pack_square_matrix(intrinsics_matrix)
                lc.publish(info_topic_names[device], cam_int_msg.encode())
        if publish_pointcloud:

            # process depth into our units
            cam = rgbd_multicam_interface.cams[i]
            cam.cam_int_mat = intrinsics_matrix
            cam._init_pers_mat()

            depth = depth_image * 0.001
            valid = depth < cam.depth_max
            valid = np.logical_and(valid, depth > cam.depth_min)
            depth[np.logical_not(valid)] = 0.0 # not exactly sure what to put for invalid depth

            # get point cloud from depth image
            point_cloud_world_frame = cam.get_pcd(
                in_world=True,  # False
                filter_depth=False, 
                rgb_image=color_image, 
                depth_image=depth)[0].astype(np.float32)
            
            # pack and send point cloud message
            pcd_msg = point_cloud_t()
            # point_format = 'fff'
            # point_struct = struct.Struct(point_format)
            # point_data_bytes = bytearray()
            # for point in point_cloud_cam_frame:
            #     packed_point = point_struct.pack(point[0], point[1], point[2])
            #     point_data_bytes.extend(packed_point)
            point_data_bytes = point_cloud_world_frame.tobytes()
            pcd_msg.data = point_data_bytes
            pcd_msg.data_size = len(point_data_bytes)
            pcd_msg.num_points = point_cloud_world_frame.shape[0]
            pcd_msg.header = header_t()
            pcd_msg.header.frame_name = f'cam_{i}'
            pcd_msg.header.seq = 0
            pcd_msg.header.utime = round(time.time() * 1000)

            lc.publish(pointcloud_topic_names[device], pcd_msg.encode())

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
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        show_images = np.hstack((cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR), depth_colormap))
        cv2.imshow('RealSense' + device, show_images)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            return True
            
        # Save images and depth maps from both cameras by pressing 's'
        if key==115:
            cv2.imwrite( str(device) + '_aligned_depth.png', depth_image)
            cv2.imwrite( str(device) + '_aligned_color.png', color_image)
            print('Save')
        

def main(args):
    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=0")

    rs_cfg = get_default_multi_realsense_cfg()
    
    if args.rs_config is not None:
        rs_config_fname = osp.join(path_util.get_rrp_config(), 'real_cam_cfgs', args.rs_config)
        if osp.exists(rs_config_fname):
            rs_cfg.merge_from_file(rs_config_fname)
        else:
            print(f'Config file {rs_config_fname} does not exist, using defaults')
    rs_cfg.freeze()

    rgb_pub_name_suffix = rs_cfg.RGB_LCM_TOPIC_NAME_SUFFIX
    depth_pub_name_suffix = rs_cfg.DEPTH_LCM_TOPIC_NAME_SUFFIX
    info_pub_name_suffix = rs_cfg.INFO_LCM_TOPIC_NAME_SUFFIX
    pose_pub_name_suffix = rs_cfg.POSE_LCM_TOPIC_NAME_SUFFIX
            
    if args.find_devices:
        serials, ctx = find_devices()
    else:
        ctx = rs.context() # Create librealsense context for managing devices
        serials = rs_cfg.SERIAL_NUMBERS
    
    prefix = rs_cfg.CAMERA_NAME_PREFIX
    camera_names = [f'{prefix}{i}' for i in range(len(serials))]

    # update the topic names based on each individual camera
    camera_rgb_topics = [f'{cam_name}_{rgb_pub_name_suffix}' for cam_name in camera_names]
    camera_depth_topics = [f'{cam_name}_{depth_pub_name_suffix}' for cam_name in camera_names]
    camera_info_topics = [f'{cam_name}_{info_pub_name_suffix}' for cam_name in camera_names]
    camera_pose_topics = [f'{cam_name}_{pose_pub_name_suffix}' for cam_name in camera_names]
    camera_pointcloud_topics = [f'{cam_name}_point_cloud' for cam_name in camera_names]

    rgb_topics_by_serial = {serials[i] : camera_rgb_topics[i] for i in range(len(serials))}
    depth_topics_by_serial = {serials[i] : camera_depth_topics[i] for i in range(len(serials))}
    info_topics_by_serial = {serials[i] : camera_info_topics[i] for i in range(len(serials))}
    pose_topics_by_serial = {serials[i] : camera_info_topics[i] for i in range(len(serials))}

    resolution_width = rs_cfg.WIDTH # pixels
    resolution_height = rs_cfg.HEIGHT # pixels
    frame_rate = rs_cfg.FRAME_RATE # fps
    publish_freq = (1.0 / args.publish_freq)
    publish_every_t = publish_freq

    pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)

    time.sleep(1.0)

    calib_dir = osp.join(path_util.get_rdt_src(), 'robot/camera_calibration_files')
    calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in range(len(serials))]
    from rdt.camera.simple_multicam import MultiRGBDCalibrated
    rgbd_multicam_interface = MultiRGBDCalibrated(
        cam_names=camera_names,
        calib_filenames=calib_filenames, # None
    )
    pointcloud_topics_by_serial = {serials[i] : camera_pointcloud_topics[i] for i in range(len(serials))}

    if args.visualize:
        # create visualization process
        print('Starting to visualize camera data (in background process). Press "q" to end the visualization')
        rs_viz_proc = multiprocessing.Process(target=visualizer_process_target)
        rs_viz_proc.daemon = True
        rs_viz_proc.start()

    loop_time = time.time()

    if False:
        exit = publish_lcm_loop(
            lc, 
            rgb_topics_by_serial, 
            depth_topics_by_serial, 
            info_topics_by_serial, 
            pose_topics_by_serial, 
            pipelines,
            publish_every_t)
        assert False

    try:
        if args.just_visualize:
            print('Starting to visualize camera data')
            while True:
                exit = Visualize(pipelines)
                if exit == True:
                    print('Program closing...')
                    break
        else:
            print('Starting to publish camera data')
            while True:
                delta_t = (time.time() - loop_time)
                # if delta_t > publish_freq:
                if True:
                    start_cmd = time.time()
                    exit = publish_lcm(
                        lc, 
                        rgb_topics_by_serial, 
                        depth_topics_by_serial, 
                        info_topics_by_serial, 
                        pose_topics_by_serial, 
                        pipelines,
                        publish_camera_info=args.info_pub,
                        rgbd_multicam_interface=rgbd_multicam_interface,
                        publish_pointcloud=args.pcd_pub,
                        pointcloud_topic_names=pointcloud_topics_by_serial)
                    end_cmd = time.time()
                    # print(f'Delta time: {delta_t}')
                    # print(f'Command time: {end_cmd - start_cmd}')
                    loop_time = time.time()
                time.sleep(0.001)
                # lc.handle_timeout(1)

    finally:
        pipeline_stop(pipelines)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--find_devices', action='store_true', help='If True, look for whatever devices are connected. Else, use the serial numbers from the config')
    parser.add_argument('--publish_freq', type=float, default=10.0, help='Frequency in 1/s to send image messages over LCM')
    parser.add_argument('--visualize', action='store_true', help='If True, create windows to view the streams from each camera')
    parser.add_argument('--just_visualize', action='store_true', help='If True, don"t publish, only visualize the raw streams')
    parser.add_argument('--rs_config', type=str, default=None, help='Can provide a .yaml file located in the src/rrp_robot/config/real_cam_cfgs directory')
    parser.add_argument('--info_pub', action='store_true', help='If set, publish camera info (intrinsics)')
    parser.add_argument('--pcd_pub', action='store_true', help='If set, publish 3D point cloud')
    args = parser.parse_args()
    main(args)
