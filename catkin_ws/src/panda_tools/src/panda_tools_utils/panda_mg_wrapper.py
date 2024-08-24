import os, os.path as osp
import copy
import sys
import time
import numpy as np
import rospy
import rospkg
rospack = rospkg.RosPack()
import moveit_commander
import moveit_msgs
import trimesh
from moveit_commander.exception import MoveItCommanderException

from geometry_msgs.msg import Pose, PoseStamped

from urdf_parser_py.urdf import URDF
#from trac_ik_python import trac_ik
import tf.transformations as transformations

# import pb_robot
# from pykdl_utils.kdl_kinematics import KDLKinematics

from ndf_robot.utils import util
from ndf_robot.utils.franka_ik import FrankaIK

sys.path.append(osp.join(rospack.get_path('panda_ndf'), 'src'))
from panda_ndf_utils import util as real_robot_util

def list_to_pose(pose_list):
    msg = Pose()
    msg.position.x = pose_list[0]
    msg.position.y = pose_list[1]
    msg.position.z = pose_list[2]
    msg.orientation.x = pose_list[3]
    msg.orientation.y = pose_list[4]
    msg.orientation.z = pose_list[5]
    msg.orientation.w = pose_list[6]
    return msg


def pose_to_list(pose):
    pose_list = []
    pose_list.append(pose.position.x)
    pose_list.append(pose.position.y)
    pose_list.append(pose.position.z)
    pose_list.append(pose.orientation.x)
    pose_list.append(pose.orientation.y)
    pose_list.append(pose.orientation.z)
    pose_list.append(pose.orientation.w)
    return pose_list


class FrankaMoveIt:
    def __init__(self, franka_arm, cfg=None, optimal_planning=True, pb_gui=True,
                 gripper_type='panda', include_camera_obstacles=True):
        self.cfg = cfg
        self.franka_arm = franka_arm
        self.gripper_type = gripper_type

        self.moveit_robot = moveit_commander.RobotCommander()
        self.moveit_scene = moveit_commander.PlanningSceneInterface()
        if optimal_planning:
            # self.moveit_planner = 'RRTstarkConfigDefault'
            self.moveit_planner = 'PRMstarkConfigDefault'
        else:
            self.moveit_planner = 'RRTConnectkConfigDefault'

        self.robot_description = '/robot_description'
        self.urdf_string = rospy.get_param(self.robot_description)

        self.mp = GroupPlanner(
            move_group='panda_arm',
            moveit_robot=self.moveit_robot,
            planner_id=self.moveit_planner,
            scene=self.moveit_scene,
            max_attempts=10,
            planning_time=5.0,
            goal_tol=0.005,
            eef_delta=0.01,
            jump_thresh=10,
            camera_obstacles=include_camera_obstacles
        )

        self.ee_offset_dist = 0.15

        time.sleep(1.0)

        for name in self.moveit_scene.get_known_object_names():
            self.moveit_scene.remove_world_object(name)

        if include_camera_obstacles:
            print('Adding camera obstacles to MoveIt! planning scene')
            self.moveit_scene.add_box('camera_box_1',
                            real_robot_util.list2pose_stamped([0.135, 0.55, 0.1, 0.0, 0.0, 0.0, 1.0], "panda_link0"),
                            size=(0.2, 0.1, 0.2))

            # self.moveit_scene.add_box('camera_box_2',
            #                 real_robot_util.list2pose_stamped([0.135, -0.55, 0.1, 0.0, 0.0, 0.0, 1.0], "panda_link0"),
            #                 size=(0.2, 0.1, 0.2))
            self.moveit_scene.add_box('camera_box_2',
                            real_robot_util.list2pose_stamped([0.135, -0.525, 0.25, 0.0, 0.0, 0.0, 1.0], "panda_link0"),
                            size=(0.2, 0.1, 0.50))

            self.moveit_scene.add_box('table',
                            real_robot_util.list2pose_stamped([0.15 + 0.77/2.0, 0.0015, 0.0, 0.0, 0.0, 0.0, 1.0], "panda_link0"),
                            size=(0.77, 1.22, 0.001))


            # Fake planes to limit workspace and avoid weird motions (set workspace didn't work)
            self.moveit_scene.add_plane('top',
                                real_robot_util.list2pose_stamped(
                                    [0.0, 0.0, 0.95, 0.0, 0.0, 0.0, 1.0], "panda_link0"),
                                normal=(0, 0, 1))
            self.moveit_scene.add_plane('left',
                                real_robot_util.list2pose_stamped(
                                    [0.0, 0.65, 0.0, 0.0, 0.0, 0.0, 1.0], "panda_link0"),
                                normal=(0, 1, 0))
            self.moveit_scene.add_plane('right',
                                real_robot_util.list2pose_stamped(
                                    [0.0, -0.65, 0.0, 0.0, 0.0, 0.0, 1.0], "panda_link0"),
                                normal=(0, 1, 0))

            # self.moveit_scene.add_plane('table',
            #                     real_robot_util.list2pose_stamped(
            #                         [0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 1.0], "panda_link0"),
            #                     normal=(0, 0, 1))


        #self.moveit_scene.add_box(
        #    'table',
        #    pose=util.list2pose_stamped(cfg.TABLE_POSE),
        #    size=cfg.TABLE_SIZE)

        # self.moveit_scene.add_mesh(
        #         name='rack',
        #         pose=util.list2pose_stamped(cfg.RACK_POSE),
        #         filename='objects/8020_rack_magnet.stl',
        #         size=(1.01, 1.01, 1.01)
        #     )

        self.pb_gui = pb_gui
        # self.ik_helper = FrankaIK(gui=self.pb_gui, base_pos=[0, 0, 0])

        #pb_robot.utils.connect(use_gui=self.pb_gui)
        #pb_robot.utils.disable_real_time()
        #pb_robot.utils.set_default_camera()

        # Create robot object 
        #self.pb_panda = pb_robot.panda.Panda() 

    def compute_ik(self, pose_list, seed=None, show=True):
        """Compute a set of joint angles that achieve a specified
        end-effector pose

        Args:
            pose_list (list): [x, y, z, qx, qy, qz, qw] pose of the end-effector
            seed (list, optional): Joint angles to seed the solver. Defaults to None.

        Returns:
            list: Joint angles
        """
        #pose_mat = util.matrix_from_pose(util.list2pose_stamped(pose_list))
        #jnts = self.pb_panda.arm.ComputeIK(pose_mat, seed_q=seed)
        #if show:
        #    self.pb_panda.arm.SetJointValues(jnts)
        
        raise NotImplementedError

        # jnts = self.ik_helper.get_feasible_ik(pose_list)
        # return jnts

    def compute_fk(self, joints_list):
        """Compute the end-effector pose corresponding to some specified
        joint configuration

        Args:
            joints_list (list): Joint angles in order from 1 to 7

        Returns:
            list: [x, y, z, qx, qy, qz, qw] end-effector pose
        """
        #pose_mat = self.pb_panda.arm.ComputeFK(joints_list)
        #return util.pose_stamped2list(util.pose_from_matrix(pose_mat))
        raise NotImplementedError

    def convert_plan_to_panda(self, plan):
        """Takes the direct output of MoveIt's move_group.plan() and converts
        it to a sequence of waypoints that can be directly sent to the panda
        interface

        Args:
            plan (JointTrajectory): Trajectory returned by MoveIt planner

        Returns:
            list: Each element is a dictionary of joint angles in the panda interface format
        """
        joint_trajectory_points_list = plan[1].joint_trajectory.points
        n_waypoints = len(joint_trajectory_points_list)
        joints_list = [joint_trajectory_points_list[i].positions for i in range(n_waypoints)]
        joints_panda = [self.franka_arm.convertToDict(list(jnts)) for jnts in joints_list]
        return joints_panda

    def get_joints_panda_to_moveit(self, panda_jnts=None, finger_joints=None):
        """Obtains a ROS msg that can be used to set the start state of the MoveIt
        planner. Defaults to using the current state

        Args:
            panda_jnts (dict, optional): Keys are joint names, values are joint angles
                Obtained using panda.joint_angles(). Defaults to None.
            finger_joints (list or tuple, optional): Values to set for the finger joints.
                Defaults to None.

        Returns:
            moveit_msgs.msg.RobotState: ROS message containing the joint information
        """
        if panda_jnts is None:
            panda_jnts = self.franka_arm.joint_angles()
        panda_jnts_list = self.franka_arm.convertToList(panda_jnts)
        n_arm_jnts = len(panda_jnts_list)

        state_msg = moveit_msgs.msg.RobotState()
        state_msg.joint_state.name = self.moveit_robot.get_current_state().joint_state.name
        state_msg.joint_state.position[:n_arm_jnts] = tuple(panda_jnts_list)
        
        # only fill in the finger joints if we have more than n dof
        if len(state_msg.joint_state.name) > n_arm_jnts:
            if finger_joints is not None:
                state_msg.joint_state.position[n_arm_jnts:] = tuple(finger_joints)
            else:
                if self.gripper_type == 'panda':
                    state_msg.joint_state.position[n_arm_jnts:] = (0.0, 0.0)
                else:
                    state_msg.joint_state.position[n_arm_jnts:] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        return state_msg

    def set_current_start_state(self):
        """Quick helper to directly set the start state to the current state
        """
        current_state_msg = self.get_joints_panda_to_moveit()
        self.mp.set_start_state(current_state_msg)

    def get_world_frame_ee_offset(self, ee_pose, offset, dist=0.075): #dist=0.275 #dist=0.15): # dist=0.075
        if not isinstance(offset, np.ndarray):
            offset = np.asarray(offset)
        
        offset = offset / np.linalg.norm(offset)
        offset = offset * dist

        offset_pose = util.list2pose_stamped(offset.tolist() + [0, 0, 0, 1])
        transformed_pose = util.transform_pose(
            pose_source=util.list2pose_stamped(ee_pose),
            pose_transform=offset_pose
        )
        return util.pose_stamped2list(transformed_pose)
    
    def set_ee_offset_dist(self, dist):
        self.ee_offset_dist = dist

    def get_ee_offset(self, ee_pose, dist=0.15): # dist=0.075
        """
        Gets the updated world frame normal direction of the gripper
        """
        # dist = 0.1
        dist = copy.deepcopy(self.ee_offset_dist)
        normal_x = util.list2pose_stamped([dist, 0, 0, 0, 0, 0, 1])
        normal_y = util.list2pose_stamped([0, dist, 0, 0, 0, 0, 1])
        normal_z = util.list2pose_stamped([0, 0, dist, 0, 0, 0, 1])

        normal_x = util.transform_pose(normal_x, util.list2pose_stamped(ee_pose))
        normal_y = util.transform_pose(normal_y, util.list2pose_stamped(ee_pose))
        normal_z = util.transform_pose(normal_z, util.list2pose_stamped(ee_pose))

        # dx_vec = util.pose_stamped2np(normal_x)[:3] - np.asarray(ee_pose)[:3]
        # dy_vec = util.pose_stamped2np(normal_y)[:3] - np.asarray(ee_pose)[:3]
        # dz_vec = util.pose_stamped2np(normal_z)[:3] - np.asarray(ee_pose)[:3]
        dx_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_x)[:3]
        dy_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_y)[:3]
        dz_vec = np.asarray(ee_pose)[:3] - util.pose_stamped2np(normal_z)[:3]

        # return dx_vec.tolist() + [0, 0, 0, 1]
        # return dx_vec.tolist() + [0, 0, 0, 1]
        return dz_vec.tolist() + [0, 0, 0, 1]
    
    def get_offset_pose(self, ee_pose):
        ee_offset_pose_list = self.get_ee_offset(ee_pose)
        offset_ee_pose = util.transform_pose(
            pose_source=util.list2pose_stamped(ee_pose),
            pose_transform=util.list2pose_stamped(ee_offset_pose_list)
        )
        return util.pose_stamped2list(offset_ee_pose)

    def plan_joint_target_mp(self, target_jnts, current=True, execute=False):
        """Plan a collision-free path using MoveIt, with a joint target, using
        the current state as the start state. 

        Args:
            target_jnts (dict): Dictionary of joint values to use for target. Keys
                and panda joint names, values are angles in radians. Can be obtained
                using ArmInterface joint_angles() method.
            current (bool): If True, set the start state as the current state of the system.
                If False, you must have manually called mp.set_start_state()
            execute (bool): If True, will use execute_position_path to execute the path
                if one is found
        
        Returns:
            list: Each element is a dictionary. The whole path can be easily executed
                using the ArmInterface execute_position_path() method. Alternatively,
                you can use ArmInterface move_to_joint_positions() method for each point
                one by one. 

        """
        if current:
            current_state_msg = self.get_joints_panda_to_moveit()
            self.mp.set_start_state(current_state_msg)

        self.mp.planning_group.set_joint_value_target(target_jnts)

        plan = self.mp.planning_group.plan()

        panda_joints_plan = self.convert_plan_to_panda(plan)
        if execute:
            if len(panda_joints_plan) > 0:
                # for jnts in panda_joints_plan:
                #     self.franka_arm.move_to_joint_positions(jnts)
                self.franka_arm.execute_position_path(panda_joints_plan)
            else:
                print('Feasible path not found, nothing to execute')
                pass
            
        return panda_joints_plan

    def plan_linear_pose_target_mp(self, start_pose, target_pose, eef_delta=0.01, jump_thresh=10.0, N_waypoints=3):
        """Plan a collision-free path using MoveIt, with a pose target target, using
        the current state as the start state. Different than "plan_pose_target_mp" because
        the path will be planned by first LINEARLY interpolating the pose from start to 
        goal and then using MoveIt's "compute_caretesian_path" capability.

        Args:
            start_pose (list): Cartesian end-effector pose to use for target.
            target_pose (list): Cartesian end-effector pose to use for target.
        
        Returns:
            list: Each element is a dictionary. The whole path can be easily executed
                using the ArmInterface execute_position_path() method. Alternatively,
                you can use ArmInterface move_to_joint_positions() method for each point
                one by one. 
        """
        waypoints_pose_stamped = util.interpolate_pose(
            util.list2pose_stamped(start_pose), 
            util.list2pose_stamped(target_pose),
            N=N_waypoints)
        waypoints = [wp.pose for wp in waypoints_pose_stamped]

        self.mp.planning_group.set_pose_target(target_pose)

        plan, fraction = self.mp.planning_group.compute_cartesian_path(
            waypoints,
            eef_delta,
            jump_thresh,
            avoid_collisions=True)

        # if (fraction != 1.0) or (len(plan.joint_trajectory.points) == 0):
        if (fraction < 0.9) or (len(plan.joint_trajectory.points) == 0):
            print(f'\n\nCompute cartesian path failed, fraction: {fraction}, plan length: {len(plan.joint_trajectory.points)}\n\n')
            return []

        plan = [None, plan] # why do we do this?

        panda_joints_plan = self.convert_plan_to_panda(plan)
            
        return panda_joints_plan

    def plan_pose_target_mp(self, target_pose, current=True, execute=False):
        """Plan a collision-free path using MoveIt, with a pose target target, using
        the current state as the start state. 

        Args:
            target_pose (PoseStamped): Cartesian end-effector pose to use for target.
            current (bool): If True, set the start state as the current state of the system.
                If False, you must have manually called mp.set_start_state()
            execute (bool): If True, will use execute_position_path to execute the path
                if one is found
        
        Returns:
            list: Each element is a dictionary. The whole path can be easily executed
                using the ArmInterface execute_position_path() method. Alternatively,
                you can use ArmInterface move_to_joint_positions() method for each point
                one by one. 


        """
        if current:
            current_state_msg = self.get_joints_panda_to_moveit()
            self.mp.set_start_state(current_state_msg)

        print(type(target_pose))
        self.mp.planning_group.set_pose_target(target_pose)

        plan = self.mp.planning_group.plan()

        panda_joints_plan = self.convert_plan_to_panda(plan)
        if execute:
            if len(panda_joints_plan) > 0:
                # for jnts in panda_joints_plan:
                #     self.franka_arm.move_to_joint_positions(jnts)
                self.franka_arm.execute_position_path(panda_joints_plan)
            else:
                print('Feasible path not found, nothing to execute')
                pass
            
        return panda_joints_plan

    def get_pregrasp(self, grasp_pose):
        offset = self.get_ee_offset(grasp_pose)
        pregrasp = util.transform_pose(
            pose_source=util.list2pose_stamped(grasp_pose), 
            pose_transform=util.list2pose_stamped(offset))
        return util.pose_stamped2list(pregrasp)

    def get_preplace(self, place_pose):
        offset = self.get_ee_offset(place_pose)
        preplace = util.transform_pose(
            pose_source=util.list2pose_stamped(place_pose), 
            pose_transform=util.list2pose_stamped(offset))
        return util.pose_stamped2list(preplace)

    def add_pcd_bounding_box(self, pcd, start_ee_pose):
        # get bounding box
        tpcd = trimesh.PointCloud(pcd)
        tpcd_bb = tpcd.bounding_box_oriented
        extents = tpcd_bb.extents

        # get the pose of this box in the world and attach it TODO
        obj_pose_world = self.robot.get_ee_pose()
        ee_pose_world = self.robot.get_ee_pose()
        obj_pose_ee = util.convert_reference_frame(
            pose_source=obj_pose_world,
            pose_frame_target=ee_pose_world,
            pose_frame_source=util.unit_pose()
        )

        # add box to robot
        self.scene.add_dynamic_obj(
            reference_frame='panda_link8',
            obj_name='obj_pcd_bb', 
            pos=obj_pose_ee[:3], 
            ori=obj_pose_ee[3:],
            size=extents,
            touch_links=['panda_link8', 'panda_link7', 'panda_link6'])

    def unlink_obj(self, ref_frame, obj_name=None, delete=True):
        """
        Unlink the attached object from ref_frame.
        Args:
            ref_frame (str): the parent link at which the
                object is attached to.
            obj_name (str): the object name.
            delete (bool): If True, the object will be deleted from the scene.
                Otherwise, the object will be unlinked
                from the parent link only,
                but the object is still there in the scene.
        """
        self.scene.remove_attached_object(ref_frame, obj_name)
        if delete:
            self.remove_obj(obj_name)

    def remove_all_objs(self):
        """
        Remove all the added objects in the planning scene.
        """
        objs = self.scene.get_objects()
        objs_attached = self.scene.get_attached_objects()
        # remove add objects
        for key in objs.keys():
            self.remove_obj(key)
        # remove attached objects
        for key in objs_attached.keys():
            self.unlink_obj(objs_attached[key].link_name, key)        

    def execute_plan_joint_impedance(self, plan, stiffness=[70, 70, 50, 60, 35, 35, 6], loop_time=0.2, *args, **kwargs):
        """
        Move through a sequence of joint angles with non-blocking joint impedance control commands. 

        Warning!!! Stiffness must be tuned appropriately for each joint! If stiffness is too high and
        joint targets are too far from each other then arm with vibrate. If stiffness is too low, then
        targets will not be reached and true joint configuration may drift from desired joints.
        
        Args:
            plan (list): Each element is a dict with the joint angle names for keys and joint values for values
            stiffness (list): Each element is the stiffness to use for the joint at the corresponding index. Default
                values have been somewhat tuned to allow for non-jerky motion when joint targets are about 0.1 radians
                away from each other.
            loop_time (float): Amount of time to wait between sending new commands. This value should be tuned to allow for
                smooth motion playback. If we wait too long, we'll "stop and go", if we don't wait long enough, we'll 
                execute the path very inaccurately.
        """
        plan_list = [self.franka_arm.convertToList(jnt_dict) for jnt_dict in plan]  # set_joint_impedance_config takes in lists

        for jnt in plan_list:
            self.franka_arm.set_joint_impedance_config(jnt, stiffness)
            time.sleep(loop_time)
        
    def interpolate_plan(self, plan, n_pts=None, des_step_dist=0.1):
        """
        Densely interpolate a plan that was obtained from the motion planner
        """
        if len(plan) > 0:
            plan_list = [self.franka_arm.convertToList(jnt_dict) for jnt_dict in plan]  # set_joint_impedance_config takes in lists
            plan_np = [np.asarray(jnt_list) for jnt_list in plan_list]
            if n_pts is None:
                # rough heuristic of making sure every joint doesn't take a step larger than 0.1 radians per waypoint

                max_step_dist = 0.0
                for i in range(len(plan) - 1):
                    step_ = plan_np[i]
                    step_next = plan_np[i+1]

                    dists = np.abs(step_ - step_next)

                    max_dist = np.max(dists)

                    if max_dist > max_step_dist:
                        max_step_dist = max_dist

                n_pts_per_step = np.ceil(max_step_dist / des_step_dist)
                n_pts = int(len(plan) * n_pts_per_step)
                print('Got max dist: %f, Going to make sure each step is interpolated to %d points, giving total of %d points' % (max_step_dist, n_pts_per_step, n_pts))
            else:
                n_pts_per_step = int(np.ceil(n_pts / len(plan) * 1.0))
            
            n_pts_per_step = int(n_pts_per_step)
            print('n_pts_per_step', n_pts_per_step)
            new_plan = np.asarray(plan_list[0]) 
            for i in range(len(plan) - 1):
                step_ = plan_np[i]
                step_next = plan_np[i+1]
                
                print('step_', step_)
                print('step_next', step_next)
                interp = np.linspace(step_, step_next, n_pts_per_step)
                # new_plan.extend(interp.tolist())
                new_plan = np.vstack((new_plan, interp))
            
            print('Shape of new plan: (%d, %d)' % (new_plan.shape[0], new_plan.shape[1]))
            new_plan = [self.franka_arm.convertToDict(jnt_np.tolist()) for jnt_np in new_plan]
            return new_plan
        else:
            return []


# GroupPlanner interfaces planning functions for a planning group from MoveIt!
class GroupPlanner:
    def __init__(self, move_group, moveit_robot, planner_id, scene, max_attempts, planning_time, goal_tol=0.0005,
                 eef_delta=0.01, jump_thresh=10.0, camera_obstacles=False):
        self.move_group = move_group
        self.moveit_robot = moveit_robot

        # Maximum attempts for computing a Cartesian waypoint trajectory
        self.max_attempts = max_attempts
        # Distance/step in Cartesian space for returned points in a trajectory
        self.eef_delta = eef_delta
        # Maximum jump/discontinuity (0 is any allowed)
        self.jump_thresh = jump_thresh

        # MoveGroup from MoveIt!
        self.planning_group = moveit_commander.MoveGroupCommander(self.move_group)
        self.planning_group.set_goal_tolerance(goal_tol)
        self.planning_group.set_goal_joint_tolerance(0.01)
        self.planning_group.set_planning_time(planning_time)
        self.planning_group.set_pose_reference_frame("panda_link0")
        self.planning_group.set_planner_id(planner_id)
        self.planning_group.set_num_planning_attempts(20)
        self.planning_group.allow_replanning(True)

        # Collision scene from MoveIt!
        self.scene = scene

    # Sets start state to: (a) a defined state, if it exists, or (b) current state
    def set_start_state(self, start_state_msg):
        self.planning_group.set_start_state(start_state_msg)
