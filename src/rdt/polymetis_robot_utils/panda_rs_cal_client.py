import os, os.path as osp
import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import torch
import time
import cv2
import signal
import sys
from scipy import optimize  
import copy
import argparse
import threading
from scipy.spatial.transform import Rotation as R
import meshcat
import trimesh
import pyrealsense2 as rs

from polymetis import GripperInterface, RobotInterface

from airobot import log_info, log_warn, log_debug, log_critical, set_log_level
from rdt.common import util, path_util
from rdt.common.franka_ik import FrankaIK
from rdt.polymetis_robot_utils.plan_exec_util import PlanningHelper
from rdt.polymetis_robot_utils.polymetis_util import PolymetisHelper
from rdt.polymetis_robot_utils.traj_util import PolymetisTrajectoryUtil
from rdt.perception.perception_util import enable_devices, RealsenseInterface

from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg


poly_util = PolymetisHelper()


def get_tool_position(poly_pose, offset):
    ee_pose_world = util.list2pose_stamped(poly_util.polypose2list(poly_pose))
    offset_pose_ee = util.list2pose_stamped(offset.tolist() + [0, 0, 0, 1])
    offset_pose_world = util.convert_reference_frame(
        pose_source=offset_pose_ee,
        pose_frame_target=util.unit_pose(),
        pose_frame_source=ee_pose_world
    )
    return util.pose_stamped2np(offset_pose_world)[:3]


### !!! THIS ASSUMES THAT NOTHING IS ATTACHED TO THE ROBOT! MAKE SURE NOT TO SET THE TCP AS THE OFFSET POSE, OR ELSE IT WILL GET DOUBLED!!! ###
checkerboard_offset_from_tool = np.array([0.0, 0.0, 0.1])


cam012_home_joints = {
    'panda_joint1': -0.04935819447772545,
    'panda_joint2': -0.3469163031241724,
    'panda_joint3': 0.19999700257783215,
    'panda_joint4': -2.483996871786347,
    'panda_joint5': 0.08815259526835548,
    'panda_joint6': 2.305844914529558,
    'panda_joint7': -1.6618345036738449
}

# cam0_workspace_limits = np.asarray([[400, 425], [-100, 100], [225, 250]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates) # SET THIS

cam0_workspace_limits = np.asarray([[375, 500], [-390, -300], [250, 300]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates) # SET THIS
cam1_workspace_limits = np.asarray([[375, 500], [-390, -300], [200, 250]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates) # SET THIS
cam2_workspace_limits = np.asarray([[400, 550], [-250, 250], [250, 350]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates) # SET THIS
cam3_workspace_limits = np.asarray([[400, 550], [-250, -250], [450, 550]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates) # SET THIS


# Estimate rigid transform with SVD (from Nghia Ho)
def get_rigid_transform(A, B):
    assert len(A) == len(B)
    N = A.shape[0]; # Total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - np.tile(centroid_A, (N, 1)) # Centre the points
    BB = B - np.tile(centroid_B, (N, 1))
    H = np.dot(np.transpose(AA), BB) # Dot is matrix multiplication for array
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)
    if np.linalg.det(R) < 0: # Special reflection case
        Vt[2,:] *= -1
        R = np.dot(Vt.T, U.T)
    t = np.dot(-R, centroid_A.T) + centroid_B.T
    return R, t


def get_rigid_transform_error(z_scale):
	global measured_pts, observed_pts, observed_pix, world2camera, camera, cam_intrinsics

	# Apply z offset and compute new observed points using camera intrinsics
	observed_z = observed_pts[:,2:] * z_scale
	observed_x = np.multiply(observed_pix[:,[0]]-cam_intrinsics[0][2],observed_z/cam_intrinsics[0][0])
	observed_y = np.multiply(observed_pix[:,[1]]-cam_intrinsics[1][2],observed_z/cam_intrinsics[1][1])
	new_observed_pts = np.concatenate((observed_x, observed_y, observed_z), axis=1)
	# print("new_observed_pts", new_observed_pts)
	# print("measured_pts", measured_pts)
	
	# Estimate rigid transform between measured points and new observed points
	R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
	t.shape = (3,1)
	world2camera = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)

	# Compute rigid transform error
	registered_pts = np.dot(R,np.transpose(measured_pts)) + np.tile(t,(1,measured_pts.shape[0]))
	error = np.transpose(registered_pts) - new_observed_pts
	error = np.sum(np.multiply(error,error))
	rmse = np.sqrt(error/measured_pts.shape[0])
	return rmse


def lcm_sub_thread(lc):
    while True:
        lc.handle_timeout(0.001)


def main(args):
    if args.debug:
        set_log_level('debug')
    else:
        set_log_level('info')

    global measured_pts, observed_pts, observed_pix, world2camera, camera, cam_intrinsics

    signal.signal(signal.SIGINT, util.signal_handler)

    calib_package_path = osp.join(path_util.get_rdt_src(), 'robot/camera_calibration_files')
    assert osp.exists(calib_package_path), f'Calibration file destination {calib_package_path} doesn"t exist!'

    with_robot = args.robot


    zmq_url=f'tcp://127.0.0.1:{args.port_vis}'
    mc_vis = meshcat.Visualizer(zmq_url=zmq_url)
    mc_vis['scene'].delete()

    # setup camera interfaces 
    rs_cfg = get_default_multi_realsense_cfg()
    serials = rs_cfg.SERIAL_NUMBERS

    prefix = rs_cfg.CAMERA_NAME_PREFIX
    camera_names = [f'{prefix}{i}' for i in range(len(serials))]
    cam_list = [camera_names[int(idx)] for idx in args.cam_index]
    serials = [serials[int(idx)] for idx in args.cam_index]

    calib_dir = osp.join(path_util.get_rdt_src(), 'robot/camera_calibration_files')
    calib_filenames = [osp.join(calib_dir, f'cam_{idx}_calib_base_to_cam.json') for idx in args.cam_index]

    ctx = rs.context() # Create librealsense context for managing devices

    # Define some constants 
    resolution_width = 640 # pixels
    resolution_height = 480 # pixels
    frame_rate = 30  # fps

    pipelines = enable_devices(serials, ctx, resolution_width, resolution_height, frame_rate)

    
    done = False
    while not done:
        for idx, cam in enumerate(camera_names):
        # for idx, cam_and_subs in enumerate(img_subscribers):
            # cam, img_sub, info_sub = cam_and_subs

            save_dir = osp.join(os.getcwd(), 'calibration', cam)
            if not osp.exists(save_dir):
                os.makedirs(save_dir)

            calib = RealsenseInterface()
            cam_intrinsics = calib.get_intrinsics_mat(pipelines[idx])
            if np.array_equal(np.eye(3), cam_intrinsics):
                while True:
                    time.sleep(1.0)
                    cam_intrinsics = calib.get_intrinsics_mat(pipelines[idx])
                    if not np.array_equal(np.eye(3), cam_intrinsics):
                        break

            log_info('Camera intrinsics: ')
            print(cam_intrinsics)

            # Set workspace
            if cam == 'cam_0':
                workspace_limits = cam0_workspace_limits
                home_joints = cam012_home_joints
            elif cam == 'cam_1':
                workspace_limits = cam1_workspace_limits
                home_joints = cam012_home_joints
            elif cam == 'cam_2':
                workspace_limits = cam2_workspace_limits
                home_joints = cam012_home_joints
            elif cam == 'cam_3':
                workspace_limits = cam3_workspace_limits
                home_joints = cam012_home_joints

            # Set grid resolution
            workspace_limits = workspace_limits / 1000.0 # change to m
            # calib_grid_num = [6,5,4]
            # calib_grid_num = [8,6,4]
            calib_grid_num = [4,3,2]
            # calib_grid_num = [4,4,3]


            # Construct 3D calibration grid across workspace
            gridspace_x = np.linspace(workspace_limits[0][0], workspace_limits[0][1], num=calib_grid_num[0])
            gridspace_y = np.linspace(workspace_limits[1][0], workspace_limits[1][1], num=calib_grid_num[1])
            gridspace_z = np.linspace(workspace_limits[2][0], workspace_limits[2][1], num=calib_grid_num[2])

            # calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y, gridspace_z)
            calib_grid_x, calib_grid_y, calib_grid_z = np.meshgrid(gridspace_x, gridspace_y[::-1], gridspace_z)
            num_calib_grid_pts = calib_grid_x.shape[0]*calib_grid_x.shape[1]*calib_grid_x.shape[2]
            calib_grid_x.shape = (num_calib_grid_pts,1)
            calib_grid_y.shape = (num_calib_grid_pts,1)
            calib_grid_z.shape = (num_calib_grid_pts,1)
            calib_grid_pts = np.concatenate((calib_grid_x, calib_grid_y, calib_grid_z), axis=1)

            measured_pts = []
            observed_pts = []
            observed_pix = []

            if with_robot: 
                log_info('Connecting to robot...')

                franka_ip = "173.16.0.1" 

                robot = RobotInterface(ip_address=franka_ip)
                # robot.start_joint_impedance()
                Kx_new = robot.Kx_default.clone()
                Kxd_new = robot.Kxd_default.clone()
                # Kx_new *= 1.75
                Kx_new *= 2.5
                Kxd_new *= 1.25

                # Kq_new = robot.Kq_default.clone()
                # Kqd_new = robot.Kqd_default.clone()
                # # Kq_new *= 1.75
                # Kq_new *= 2.5
                # Kqd_new *= 1.25

                Kq_new = torch.Tensor([400.0, 400.0, 400.0, 400.0, 250.0, 150.0, 50.0])
                # Kq_new = torch.Tensor([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
                Kqd_new = torch.Tensor([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0])

                out_of_frame = torch.Tensor([-0.1329, -0.0262, -0.0448, -1.3961,  0.0632,  1.9965, -0.8882])
                # robot.go_home()

                gripper = GripperInterface(ip_address=franka_ip)

                traj_helper = PolymetisTrajectoryUtil(robot=robot)
                n_med = 500
                traj_helper.set_diffik_lookahead(int(n_med * 7.0 / 100))

                # ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], no_gripper=True, mc_vis=mc_vis)
                ik_helper = FrankaIK(gui=True, base_pos=[0, 0, 0], no_gripper=False, mc_vis=mc_vis, occnet=False)
                tmp_obstacle_dir = osp.join(path_util.get_rdt_obj_descriptions(), 'tmp_planning_obs')
                util.safe_makedirs(tmp_obstacle_dir)
                table_obs = trimesh.creation.box([0.77, 1.22, 0.001]) #.apply_transform(util.matrix_from_list([0.15 + 0.77/2.0, 0.0015, 0.0, 0.0, 0.0, 0.0, 1.0]))
                cam_obs1 = trimesh.creation.box([0.2, 0.1, 0.2]) #.apply_transform(util.matrix_from_list([0.135, 0.55, 0.1, 0.0, 0.0, 0.0, 1.0]))
                cam_obs2 = trimesh.creation.box([0.2, 0.1, 0.5]) #.apply_transform(util.matrix_from_list([0.135, -0.525, 0.25, 0.0, 0.0, 0.0, 1.0]))

                table_obs_fname = osp.join(tmp_obstacle_dir, 'table.obj')
                cam1_obs_fname = osp.join(tmp_obstacle_dir, 'cam1.obj')
                cam2_obs_fname = osp.join(tmp_obstacle_dir, 'cam2.obj')

                table_obs.export(table_obs_fname)
                cam_obs1.export(cam1_obs_fname)
                cam_obs2.export(cam2_obs_fname)

                ik_helper.register_object(
                    table_obs_fname,
                    pos=[0.15 + 0.77/2.0, 0.0, 0.0015],
                    ori=[0, 0, 0, 1],
                    name='table')
                # ik_helper.register_object(
                #     cam1_obs_fname,
                #     pos=[0.135, 0.55, 0.1],
                #     ori=[0, 0, 0, 1],
                #     name='cam1')
                ik_helper.register_object(
                    cam1_obs_fname,
                    pos=[0.135, -0.525, 0.25],
                    ori=[0, 0, 0, 1],
                    name='cam2')

                planning = PlanningHelper(
                    mc_vis=mc_vis,
                    robot=robot,
                    gripper=gripper,
                    ik_helper=ik_helper,
                    traj_helper=traj_helper,
                    tmp_obstacle_dir=tmp_obstacle_dir
                )

                cam_cal_box = trimesh.creation.box([0.12, 0.01, 0.17])
                # cam_cal_box = trimesh.creation.box([0.01, 0.12, 0.17])
                cam_cal_box.apply_translation(np.array([0.0, 0.0, 0.17/2]))
                planning.remove_all_attachments()
                planning.attach_obj(cam_cal_box, grasp_pose_mat_world=np.eye(4), name='checkerboard.obj')

                home_plan = planning.plan_home()
                if args.reset_pose:
                    home_success = planning.execute_loop(home_plan, time_scaling=1.0)  # smaller time scaling for slower
                    if home_success is not None:
                        print(f'Could not move home for some reason...')
                        from IPython import embed; embed()
                        return
                    robot.start_cartesian_impedance(Kx=torch.zeros(6), Kxd=torch.zeros(6))
                    time.sleep(2.0)

                    val = input('Please manually move the robot to a good initial configuration near the center of the workspace, and press "enter" to begin')

                robot.start_joint_impedance(Kq=Kq_new, Kqd=Kqd_new)

                # move_time = 0.5
                move_time = 1.0

                # Move robot to each calibration point in workspace
                log_info('Collecting data...')
                current_pose = copy.deepcopy(robot.get_ee_pose())
                nominal_ori_mat = poly_util.polypose2mat(current_pose)[:-1, :-1]
                for calib_pt_idx in range(num_calib_grid_pts):
                    robot_position = calib_grid_pts[calib_pt_idx, :]
                    new_pose_pos = np.asarray(robot_position)
                    pos_str = ', '.join([str(val) for val in new_pose_pos.tolist()])
                    log_info(f'Moving to new position: {pos_str}')

                    # plan path to pose target
                    # new_pose = copy.deepcopy(robot.get_ee_pose())
                    # new_pose_mat = poly_util.polypose2mat(new_pose)
                    new_pose_mat = np.eye(4)
                    new_pose_mat[:-1, -1] = new_pose_pos
                    new_pose_mat[:-1, :-1] = nominal_ori_mat.copy()

                    # motion_plan = planning.plan_pose_target(new_pose_mat)
                    # planning.execute_loop(motion_plan)

                    current_pose = robot.get_ee_pose()
                    current_pose_mat = poly_util.polypose2mat(current_pose)
                    to_new_pose_mats = planning.interpolate_pose(current_pose_mat, new_pose_mat, 500)
                    for i in range(len(to_new_pose_mats)):
                        if i % 50 == 0:
                            util.meshcat_frame_show(mc_vis, f'scene/poses/to_next/{i}', to_new_pose_mats[i])

                    motion_plan = None
                    while True:
                        if args.check_each_motion:
                            input_value = input('''
                                Press... 
                                "y" to execute
                                "r" to repeat path viz
                                "p" to plan
                                "yp" to execute plan
                                "b" to break
                            ''')
                        else:
                            joint_traj_diffik = traj_helper.diffik_traj(
                                to_new_pose_mats,
                                precompute=True,
                                execute=False,
                                total_time=move_time)

                            joint_traj_diffik_list = [val.tolist() for val in joint_traj_diffik]
                            log_info(f'Checking feasibility of DiffIK traj...')
                            planning.execute_pb_loop(joint_traj_diffik_list[::10])
                            valid_ik = True 
                            for jnt_val in joint_traj_diffik_list:
                                ik_helper.set_jpos(jnt_val)
                                in_collision = ik_helper.check_collision()
                                valid_ik = valid_ik and in_collision
                                if not valid_ik:
                                    break
                            
                            if not valid_ik:
                                log_info(f'DiffIK traj was infeasible, planning joint motion...')
                                motion_plan = planning.plan_pose_target(new_pose_mat)
                                planning.execute_pb_loop(motion_plan)
                                log_info(f'Executing planned joint motion...')
                                input_value = 'yp'
                            else:
                                log_info(f'DiffIK traj was feasible, executing...')
                                input_value = 'y'

                            break
                        
                        if input_value == 'em':
                            from IPython import embed; embed()
                            continue

                        if input_value in ['y', 'yp', 'b']:
                            break
                        
                        if input_value == 'r':
                            joint_traj_diffik = traj_helper.diffik_traj(
                                to_new_pose_mats,
                                precompute=True,
                                execute=False,
                                total_time=move_time)

                            joint_traj_diffik_list = [val.tolist() for val in joint_traj_diffik]
                            log_info(f'Showing DiffIK traj...')
                            planning.execute_pb_loop(joint_traj_diffik_list[::10])

                            if motion_plan is not None:
                                log_info(f'Showing motion planning traj...')
                                planning.execute_pb_loop(motion_plan)

                        if input_value == 'p':
                            motion_plan = planning.plan_pose_target(new_pose_mat)
                        
                        
                    if input_value == 'b':
                        continue
                    
                    if input_value == 'y':
                        # traj_helper.diffik_traj(
                        #     to_new_pose_mats,
                        #     precompute=False, 
                        #     execute=True, 
                        #     total_time=move_time)

                        planning.execute_loop(joint_traj_diffik_list)

                    if input_value == 'yp':
                        if motion_plan is not None:
                            planning.execute_loop(motion_plan)
                        else:
                            log_info(f'Planning failed')
                            continue

                    time.sleep(0.5)
                    tool_position = get_tool_position(robot.get_ee_pose(), checkerboard_offset_from_tool)  
                    time.sleep(1.0)
                    #################################################################################################################
                    
                    # Find checkerboard center
                    checkerboard_size = (3,3)
                    refine_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                    
                    # get rgb and depth image
                    rgb, depth = calib.get_rgb_and_depth_image(pipelines[idx])
                    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
                    log_info(f'Saving to {save_dir}')
                    cv2.imwrite(osp.join(save_dir, f'rgb_{calib_pt_idx}_{idx}_{time.time()}.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    cv2.imwrite(osp.join(save_dir, f'depth_{calib_pt_idx}_{idx}_{time.time()}.png'), depth_colormap)
                
                    grayscale = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

                    # normalize the grayscale image to be brighter
                    max_grayscale = float(np.amax(grayscale))
                    norm_grayscale = np.asarray((grayscale/max_grayscale)*255, dtype=np.uint8)

                    camera_depth_img = depth
                    camera_color_img = grayscale # instead of rgb -  use grayscale
                    gray_data_2 = grayscale
                    
                    ###################################################
                    
                    # Find checkerboard center
                    checkerboard_found, corners = cv2.findChessboardCorners(gray_data_2, checkerboard_size, None, cv2.CALIB_CB_ADAPTIVE_THRESH)
                    log_info(f'Found Checkerboard: {checkerboard_found}')
                    if checkerboard_found:
                        corners_refined = cv2.cornerSubPix(gray_data_2, corners, (3,3), (-1,-1), refine_criteria)

                        # Get observed checkerboard center 3D point in camera space
                        checkerboard_pix = np.round(corners_refined[4,0,:]).astype(int)
                        checkerboard_z = camera_depth_img[checkerboard_pix[1]][checkerboard_pix[0]]
                        checkerboard_x = np.multiply(checkerboard_pix[0]-cam_intrinsics[0][2],checkerboard_z/cam_intrinsics[0][0])
                        checkerboard_y = np.multiply(checkerboard_pix[1]-cam_intrinsics[1][2],checkerboard_z/cam_intrinsics[1][1])
                        log_debug(f'Checkerboard [x, y, z]: {checkerboard_x:.3f}, {checkerboard_y:.3f}, {checkerboard_z:.3f}')
                        if np.abs(checkerboard_z) < 1e-4:
                            log_info('Depth value too low, skipping')
                            continue

                        # Save calibration point and observed checkerboard center
                        observed_pts.append([checkerboard_x,checkerboard_y,checkerboard_z])

                        measured_pts.append(tool_position)
                        observed_pix.append(checkerboard_pix)
                        
                        # Draw and display the corners
                        # vis = cv2.drawChessboardCorners(robot.camera.color_data, checkerboard_size, corners_refined, checkerboard_found)
                        vis = cv2.drawChessboardCorners(gray_data_2, (1,1), corners_refined[4,:,:], checkerboard_found)
                        cv2.imwrite(osp.join(save_dir, '%06dd.png' % len(measured_pts)), vis)
                        # cv2.imshow('Calibration',vis)
                        # cv2.waitKey(0)

                if with_robot: 
                    # Move robot back to home pose
                    # panda.set_joint_position_speed(0.3)
                    # panda.move_to_joint_positions(home_joints)

                    home_plan = planning.plan_home()
                    planning.execute_loop(home_plan)

                    measured_pts = np.asarray(measured_pts)
                    observed_pts = np.asarray(observed_pts)
                    observed_pix = np.asarray(observed_pix)

                    try:
                        np.save(osp.join(save_dir, 'measured_pts'), measured_pts)
                        np.save(osp.join(save_dir, 'observed_pts'), observed_pts)
                        np.save(osp.join(save_dir, 'observed_pix'), observed_pix)
                    except IOError as e:
                        print(e)
                        from IPython import embed
                        embed()

                else:
                    measured_pts = np.load(osp.join(save_dir, 'measured_pts.npy'))
                    observed_pts = np.load(osp.join(save_dir, 'observed_pts.npy'))
                    observed_pix = np.load(osp.join(save_dir, 'observed_pix.npy') ) 

                world2camera = np.eye(4)

                # Optimize z scale w.r.t. rigid transform error
                log_info('Calibrating...')

                z_scale_init = 1
                optim_result = optimize.minimize(get_rigid_transform_error, np.asarray(z_scale_init), method='Nelder-Mead')
                camera_depth_offset = optim_result.x

                # Save camera optimized offset and camera pose
                log_info('Saving...')
                np.savetxt(osp.join(save_dir, 'camera_depth_scale.txt'), camera_depth_offset, delimiter=' ')
                get_rigid_transform_error(camera_depth_offset)
                camera_pose = np.linalg.inv(world2camera)
                np.savetxt(osp.join(save_dir, 'camera_pose.txt'), camera_pose, delimiter=' ')
                log_info('Done.')

                # DEBUG CODE -----------------------------------------------------------------------------------
                np.savetxt(osp.join(save_dir, 'measured_pts.txt'), np.asarray(measured_pts), delimiter=' ')
                np.savetxt(osp.join(save_dir, 'observed_pts.txt'), np.asarray(observed_pts), delimiter=' ')
                np.savetxt(osp.join(save_dir, 'observed_pix.txt'), np.asarray(observed_pix), delimiter=' ')
                measured_pts = np.loadtxt(osp.join(save_dir, 'measured_pts.txt'), delimiter=' ')
                observed_pts = np.loadtxt(osp.join(save_dir, 'observed_pts.txt'), delimiter=' ')
                observed_pix = np.loadtxt(osp.join(save_dir, 'observed_pix.txt'), delimiter=' ')

                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(measured_pts[:,0],measured_pts[:,1],measured_pts[:,2], c='blue')

                # print(camera_depth_offset)
                R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(observed_pts))
                t.shape = (3,1)
                camera_pose = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)
                camera2robot = np.linalg.inv(camera_pose)
                t_observed_pts = np.transpose(np.dot(camera2robot[0:3,0:3],np.transpose(observed_pts)) + np.tile(camera2robot[0:3,3:],(1,observed_pts.shape[0])))

                ax.scatter(t_observed_pts[:,0],t_observed_pts[:,1],t_observed_pts[:,2], c='red')

                new_observed_pts = observed_pts.copy()
                new_observed_pts[:,2] = new_observed_pts[:,2] * camera_depth_offset[0]
                R, t = get_rigid_transform(np.asarray(measured_pts), np.asarray(new_observed_pts))
                t.shape = (3,1)
                camera_pose = np.concatenate((np.concatenate((R, t), axis=1),np.array([[0, 0, 0, 1]])), axis=0)
                camera2robot = np.linalg.inv(camera_pose)
                t_new_observed_pts = np.transpose(np.dot(camera2robot[0:3,0:3],np.transpose(new_observed_pts)) + np.tile(camera2robot[0:3,3:],(1,new_observed_pts.shape[0])))

                ax.scatter(t_new_observed_pts[:,0],t_new_observed_pts[:,1],t_new_observed_pts[:,2], c='green')

                plt.show()

                from airobot.utils import common
                print(camera2robot[:-1, -1].tolist() + common.rot2quat(camera2robot[:-1, :-1]).tolist())

                trans = camera2robot[:-1, -1].tolist()
                quat = common.rot2quat(camera2robot[:-1, :-1]).tolist()

                ret = {
                    'b_c_transform': {
                        'position': trans,
                        'orientation': quat,
                        'T': camera2robot.tolist()
                    }
                }

                calib_file_dir = osp.join(calib_package_path, args.data_path)
                if not osp.exists(calib_file_dir):
                    os.makedirs(calib_file_dir)
                calib_file_path = osp.join(calib_file_dir, cam + '_calib_base_to_cam.json')
                log_warn(f'Saving final calibration file to {calib_file_path}')
                # calib_file_path = osp.join(calib_file_dir, args.cam + '_calib_base_to_cam.json')
                print(json.dumps(ret, indent=2))
                with open(calib_file_path, 'w') as fp:
                    json.dump(ret, fp, indent=2)    

        done = True
        break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port_vis', type=int, default=6000, help='Port for ZMQ url (meshcat visualization)')
    # parser.add_argument('--cam_index', type=int, default=0)
    parser.add_argument('--cam_index', nargs='+', help='set which cameras to get point cloud from', required=True)
    parser.add_argument('--all_cams', action='store_true')
    parser.add_argument('--robot', action='store_true')
    parser.add_argument('--data_path', type=str, default='result/panda')  
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--check_each_motion', action='store_true')
    parser.add_argument('--reset_pose', action='store_true')
    # parser.add_argument('--start_at_current_pose', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    main(args)
