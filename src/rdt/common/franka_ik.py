#!/usr/bin/env python3

from __future__ import print_function
import os, os.path as osp
import copy
import sys
import random
import numpy as np

pb_planning_src = os.environ['PB_PLANNING_SOURCE_DIR']
sys.path.append(pb_planning_src)
import pybullet as p

from pybullet_tools.utils import add_data_path, connect, dump_body, disconnect, wait_for_user, \
    get_movable_joints, get_sample_fn, set_joint_positions, get_joint_positions, get_joint_name, get_joint_info, LockRenderer, link_from_name, get_link_pose, \
    multiply, Pose, Point, interpolate_poses, HideOutput, draw_pose, set_camera_pose, load_pybullet, \
    assign_link_colors, add_line, point_from_pose, remove_handles, BLUE, pairwise_collision, set_client, get_client, pairwise_link_collision, \
    plan_joint_motion, create_attachment, enable_real_time, disable_real_time, body_from_end_effector, set_pose, set_renderer

from pybullet_tools.ikfast.franka_panda.ik import PANDA_INFO, FRANKA_URDF_2F140, FRANKA_URDF, FRANKA_URDF_NOGRIPPER
FRANKA_URDF_NOGRIPPER = osp.join(pb_planning_src, FRANKA_URDF_NOGRIPPER)
FRANKA_URDF = osp.join(pb_planning_src, FRANKA_URDF)
print('FRANKA URDF: ', FRANKA_URDF)
FRANKA_URDF_2F140 = osp.join(pb_planning_src, FRANKA_URDF_2F140)
from pybullet_tools.ikfast.ikfast import get_ik_joints, either_inverse_kinematics, check_ik_solver

from rdt.common import util


class PbPlUtils:
    def __init__(self):
        pass

    @staticmethod
    def load_pybullet(obj_path, base_pos, base_ori, scale=1.0):
        body_id = load_pybullet(obj_path, base_pos=base_pos, base_ori=base_ori, scale=scale)
        return body_id


class Attachment(object):
    def __init__(self, parent, parent_link, grasp_pose, child): # , external_pb_client):
        self.parent = parent # TODO: support no parent
        self.parent_link = parent_link
        self.grasp_pose = grasp_pose
        self.child = child
        # self.pb_client = external_pb_client
        #self.child_link = child_link # child_link=BASE_LINK
        
    @property
    def bodies(self):
        return flatten_links(self.child) | flatten_links(self.parent, get_link_subtree(
            self.parent, self.parent_link))

    def assign(self):
        parent_link_pose = get_link_pose(self.parent, self.parent_link)
        parent_link_pose_pose = util.list2pose_stamped(list(parent_link_pose[0]) + list(parent_link_pose[1]))

        child_pose = util.convert_reference_frame(
            pose_source=util.pose_from_matrix(self.grasp_pose),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=parent_link_pose_pose
            )
        # child_pose = util.matrix_from_pose(child_pose)
        child_pose = (util.pose_stamped2list(child_pose)[:3], util.pose_stamped2list(child_pose)[3:])

        # child_pose = body_from_end_effector(parent_link_pose, self.grasp_pose)
        set_pose(self.child, child_pose)
        return child_pose

    def apply_mapping(self, mapping):
        self.parent = mapping.get(self.parent, self.parent)
        self.child = mapping.get(self.child, self.child)

    def __repr__(self):
        return '{}({},{})'.format(self.__class__.__name__, self.parent, self.child)


class FrankaIK:
    def __init__(self, gui=True, base_pos=[0, 0, 1], robotiq=True, no_gripper=False, mc_vis=None):
        self.robotiq = robotiq
        if gui:
            set_client(0)
        else:
            set_client(1)
        connect(use_gui=gui)
        self.pb_client = get_client()
        add_data_path()
        draw_pose(Pose(), length=1.)
        set_camera_pose(camera_point=[1, -1, 1])

        with LockRenderer():
            with HideOutput(True):
                self.no_gripper = False
                if no_gripper:
                    self.no_gripper = True
                    self.robot = load_pybullet(FRANKA_URDF_NOGRIPPER, base_pos=base_pos, fixed_base=True)
                    assign_link_colors(self.robot, max_colors=3, s=0.5, v=1.)
                else:
                    if robotiq:
                        self.robot = load_pybullet(FRANKA_URDF_2F140, base_pos=base_pos, fixed_base=True)
                        assign_link_colors(self.robot, max_colors=3, s=0.5, v=1.)
                    else:
                        self.robot = load_pybullet(FRANKA_URDF, base_pos=base_pos, fixed_base=True)
                        assign_link_colors(self.robot, max_colors=3, s=0.5, v=1.)

        dump_body(self.robot)

        self.info = PANDA_INFO
        
        if self.no_gripper:
            self.tool_link = link_from_name(self.robot, 'panda_link8')
        else:
            if robotiq:
                self.tool_link = link_from_name(self.robot, 'panda_link8')
                self.attach_tool_link = link_from_name(self.robot, 'panda_grasptarget')
            else:
                self.tool_link = link_from_name(self.robot, 'panda_hand')
        # print(f'Tool link: {self.tool_link}, Attach tool link: {self.attach_tool_link}')
        
        draw_pose(Pose(), parent=self.robot, parent_link=self.tool_link)
        self.movable_joints = get_movable_joints(self.robot)
        print('Joints', [get_joint_name(self.robot, joint) for joint in self.movable_joints])
        check_ik_solver(self.info)

        self.ik_joints = get_ik_joints(self.robot, self.info, self.tool_link)

        self.home_joints = [-0.19000000996229238,
                            0.0799999292867887,
                            0.22998567421354038,
                            -2.4299997925910426,
                            0.030000057559800147,
                            2.519999744925224,
                            0.859999719845722]

        
        self.panda_ignore_pairs_initial = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            (4, 5),
            (5, 6),
            (6, 7),
            (6, 8),
            (7, 8), (7, 9), (7, 10), (7, 11),
            (8, 9), (8, 10), (8, 11),
            (9, 10), (9, 11),
            (10, 11)
        ]

        if self.robotiq:
            self.panda_ignore_pairs_initial += [(6,9),
                                                (8,10), (8,12), 
                                                (9,14), (9,15), (9,19),
                                                (10,14),
                                                (11,12),
                                                (12,13), (12,14),
                                                (13,14),
                                                (15,16), (15,19),
                                                (16,17),
                                                (17,18), (17,19),
                                                (18,19)
            ]

        # symmetric
        self.panda_ignore_pairs = []
        for (i, j) in self.panda_ignore_pairs_initial:
            self.panda_ignore_pairs.append((i, j))
            self.panda_ignore_pairs.append((j, i))
        self._setup_self(ignore_link_pairs=self.panda_ignore_pairs)

        set_joint_positions(self.robot, self.ik_joints, self.home_joints)
        self.obstacle_dict = {}
        self.attachment_dict = {}
        self.registered_body_ids = {}

        if self.robotiq:
            self._grasp_target_to_ee = [0, 0, -0.23, 0, 0, 0, 1]
            self._ee_to_grasp_target = [0, 0, 0.23, 0, 0, 0, 1]
        else:
            self._grasp_target_to_ee = [0, 0, -0.105, 0, 0, 0, 1]
            self._ee_to_grasp_target = [0, 0, 0.105, 0, 0, 0, 1]

        self.mc_vis = mc_vis

        self.lower_limits = [get_joint_info(self.robot, joint).jointLowerLimit for joint in self.ik_joints]
        self.upper_limits = [get_joint_info(self.robot, joint).jointUpperLimit for joint in self.ik_joints]
        
    def set_mc_vis(self, mc_vis):
        self.mc_vis = mc_vis

    def set_jpos(self, jnts):
        set_joint_positions(self.robot, self.ik_joints, jnts)
    
    def get_jpos(self):
        jpos = get_joint_positions(self.robot, self.ik_joints)
        return jpos

    def load_urdf(self, urdf_path, pos, ori, scale=1.0, collision=True, name=None):
        body_id = load_pybullet(urdf_path, base_pos=pos, base_ori=ori, scale=scale)
        if collision:
            if name is None:
                name = str(len(self.obstacle_dict) + 1)
            self.add_collision_bodies({name: body_id})
        return body_id

    def register_object(self, obj_path, pos, ori, scale=1.0, collision=True, name=None):
        # let's make sure that all the bodies are actually in the planning scene
        body_id = load_pybullet(obj_path, base_pos=pos, base_ori=ori, scale=scale)
        if not obj_path.endswith('urdf'):
            print(f'pos: {pos}')
            print(f'ori: {ori}')
            p.resetBasePositionAndOrientation(body_id, pos, ori, physicsClientId=self.pb_client)
        if collision:
            if name is None:
                name = str(len(self.obstacle_dict) + 1)
            self.add_collision_bodies({name: body_id})
        return body_id

    
    def add_collision_bodies(self, bodies={}):
        if not len(bodies):
            return
        
        for k, v in bodies.items():
            print('ADDING BODY', k, 'WITH VALUE', v)
            self.obstacle_dict[k] = v
    
    def remove_collision_bodies(self, names):
        for name in names:
            if name in self.obstacle_dict.keys() and name != 'self':
                del self.obstacle_dict[name]

    def clear_collision_bodies(self):
        self.obstacle_dict = {}

    def add_attachment_bodies(self, parent_body, parent_link, grasp_pose_mat, bodies={}):
        """
        Args:
            parent_body: body id of the parent
            parent_link: link id of the parent link
            grasp_pose_mat: 4x4 matrix representing pose of the attachment, with respect to the parent link
            bodies: dict with names as keys and body_ids as values
        """
        if not len(bodies):
            return
            
        for k, v in bodies.items():
            self.attachment_dict[k] = Attachment(parent=parent_body, parent_link=parent_link, grasp_pose=grasp_pose_mat, child=v) 
    
    def remove_attachment_bodies(self, names):
        for name in names:
            if name in self.attachment_dict.keys() and name != 'self':
                del self.attachment_dict[name]

    def clear_attachment_bodies(self):
        self.attachment_dict = {}

    def _setup_self(self, ignore_link_pairs=[]):
        """Setup the internal information regarding the robot joints, links to
        consider for self-collision checking, and

        Args:
            ignore_link_pairs (list, optional): List of tuples. Each tuple indicates
                a pair of robot links that should NOT be considered when checking
                self collisions . Defaults to [].
        """
        # setup self-collision link pairs
        self.check_self_coll_pairs = []
        for i in range(p.getNumJoints(self.robot, physicsClientId=self.pb_client)):
            for j in range(p.getNumJoints(self.robot, physicsClientId=self.pb_client)):
                # don't check link colliding with itself, and ignore specified links
                if i != j and (i, j) not in ignore_link_pairs:
                    self.check_self_coll_pairs.append((i, j))
        
    def check_self_collision(self):
        # * self-collision link check
        for link1, link2 in self.check_self_coll_pairs:
            if pairwise_link_collision(self.robot, link1, self.robot, link2):
                print(link1, link2)
                return True, 1
        return False, 0
    
    def all_between(self, lower_limits, values, upper_limits):
        assert len(lower_limits) == len(values)
        assert len(values) == len(upper_limits)
        return np.less_equal(lower_limits, values).all() and np.less_equal(values, upper_limits).all()
    
    def within_joint_limits(self, q, verbose=False):
        if self.all_between(self.lower_limits, q, self.upper_limits):
            #print('Joint limits violated')
            if verbose: 
                print(self.lower_limits, q, self.upper_limits)
            return True
        return False

    def check_collision(self, pcd=None, thresh=0.5, hand_only=False):
        # self_collision = any_link_pair_collision(self.robot, None, self.robot, None)

        within_joint_limits = self.within_joint_limits(self.get_jpos())
        if not within_joint_limits:
            return True, 'limits'

        self_collision = self.check_self_collision()[0]
        if self_collision:
            return True, 'self'
        
        for name, obstacle in self.obstacle_dict.items():
            collision = pairwise_collision(self.robot, obstacle)
            if collision:
                return True, name

        return False, None

    def _convert_to_ee(self, pose_list):
        pose_list = util.pose_stamped2list(util.convert_reference_frame(
            pose_source=util.list2pose_stamped(self._grasp_target_to_ee),
            pose_frame_target=util.unit_pose(),
            pose_frame_source=util.list2pose_stamped(pose_list)
        ))
        return pose_list

    def get_ik(self, pose_list, execute=False, target_link=True, *args, **kwargs):
        """ASSUMES THE POSE IS OF THE LINK "PANDA_GRASPTARGET"!!!
        We do an internal conversion here to express this as the pose of 
        the end effector link that this IK can use, which is panda_hand

        Args:
            pose_list (list): [x, y, z, x, y, z, w]
            execute (bool, optional): If yes, set the joints to this value. Defaults to False.

        Returns:
            list: Joint values
        """
        if target_link:
            # convert the pose we get to the pose of our EE
            old_pose_list = copy.deepcopy(pose_list)
            pose_list = self._convert_to_ee(pose_list)

        pos, ori = pose_list[:3], pose_list[3:] # check quat convention
        pose = (tuple(pos), tuple(ori))
        try:
            conf = next(either_inverse_kinematics(self.robot, self.info, self.tool_link, pose, 
                                                  max_distance=None, max_time=0.5, max_candidates=250), None)
        except:
            print('in franka_ik')
            from IPython import embed; embed()
        if conf is None:
            print('Failure!')
            return None 
        
        if execute:
            set_joint_positions(self.robot, self.ik_joints, conf)
        return conf

    def get_feasible_ik(self, pose_list, max_attempts=100, verbose=True, 
                        target_link=True, pcd=None, thresh=0.5, hand_only=False):
        if target_link:
            pose_list = self._convert_to_ee(pose_list)
        
        # and iterate over output of IK
        pos, ori = pose_list[:3], pose_list[3:] 
        pose = (tuple(pos), tuple(ori))
        try:
            confs = either_inverse_kinematics(self.robot, self.info, self.tool_link, pose, 
                                              max_distance=None, max_time=0.5, max_candidates=250)
        except:
            print('in franka_ik')
            from IPython import embed; embed()
        
        # randomize the confs to get different ones if we call this multiple times
        def yielding(ls):
            for val in ls:
                yield val
        confs = list(yielding(confs))
        random.shuffle(confs)
        for conf in confs:
            set_joint_positions(self.robot, self.ik_joints, conf)
            collision_info = self.check_collision(pcd, thresh=thresh, hand_only=hand_only)
            if not collision_info[0]:
                return conf 
            else:
                if verbose:
                    print('Collision with body: %s' % collision_info[1])
            if hand_only:
                if verbose:
                    print(f'Checking collision with hand only, just using a single conf (breaking)')
                break
        print('Failed to get feasible IK')
        return None
    
    def plan_joint_motion(self, start, goal, alg='rrt_star', max_time=5.0):
        self.set_jpos(start)
        
        if start is None:
            print(f'Start is None, returning None')
            return None
        if goal is None:
            print(f'Goal is None, returning None')
            return None

        plan = plan_joint_motion(
            self.robot, self.ik_joints, goal, obstacles=self.obstacle_dict.values(), self_collisions=True, attachments=self.attachment_dict.values(),
            disabled_collisions=set(self.panda_ignore_pairs), algorithm=alg, max_time=max_time)
        return plan

    def _retract(self):
        sample_fn = get_sample_fn(self.robot, self.movable_joints)
        for i in range(10):
            print('Iteration:', i)
            conf = sample_fn()
            set_joint_positions(self.robot, self.movable_joints, conf)
            self._test_retraction(use_pybullet=False, max_distance=0.1, max_time=0.5, max_candidates=100)

    def _test_retraction(self, distance=0.1, **kwargs):
        ik_joints = get_ik_joints(self.robot, self.info, self.tool_link)
        start_pose = get_link_pose(self.robot, self.tool_link)
        end_pose = multiply(start_pose, Pose(Point(z=-distance)))
        handles = [add_line(point_from_pose(start_pose), point_from_pose(end_pose), color=BLUE)]
        path = []
        pose_path = list(interpolate_poses(start_pose, end_pose, pos_step_size=0.01))
        for i, pose in enumerate(pose_path):
            print('Waypoint: {}/{}'.format(i+1, len(pose_path)))
            handles.extend(draw_pose(pose))
            conf = next(either_inverse_kinematics(self.robot, self.info, self.tool_link, pose, **kwargs), None)
            if conf is None:
                print('Failure!')
                path = None
                wait_for_user()
                break
            set_joint_positions(self.robot, ik_joints, conf)
            path.append(conf)
            wait_for_user()
        remove_handles(handles)
        return path

