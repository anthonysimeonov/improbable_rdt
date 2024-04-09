# to run this, must also run pip install params-proto

import os
import sys
import time
from contextlib import contextmanager
from multiprocessing.managers import SharedMemoryManager

import numpy as np
import scipy.spatial.transform as st
from params_proto import ParamsProto, Proto

from rdt.spacemouse.spacemouse_pybullet.panda_pybullet import PandaPybullet, neutral_joint_positions
from rdt.spacemouse.spacemouse_pybullet.pybullet_helpers import draw_pose, remove_handles
from rdt.spacemouse.spacemouse_shared_memory import Spacemouse


class DemoPybulletArgs(ParamsProto):
    frequency: int = Proto(10, help="Control frequency in Hz.")
    command_latency: float = Proto(
        0.01,
        help="Latency between receiving SapceMouse command to executing on Robot in Sec.",
    )
    deadzone: float = Proto(0.05, help="deadzone for the spacemouse to avoid drift")

    max_pos_speed: float = Proto(0.3, help="max translational speed. higher for faster")
    max_rot_speed: float = Proto(0.7, help="max rotational speed. higher for faster")


args = DemoPybulletArgs


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different


def main():
    frequency = args.frequency
    dt = 1 / frequency
    command_latency = args.command_latency

    # FIXME: get actual joint positions from the robot, make it go home, etc.
    init_joint_positions = neutral_joint_positions
    gripper_open = True

    env = PandaPybullet(initial_joint_positions=init_joint_positions)
    translation, quat_xyzw = env.ee_pose()
    rotvec = st.Rotation.from_quat(quat_xyzw).as_rotvec()
    target_pose = np.array([*translation, *rotvec])

    def to_pb_pose(pose_):
        target_translation = pose_[:3]
        target_quat_xyzw = st.Rotation.from_rotvec(pose_[3:]).as_quat()
        return target_translation, target_quat_xyzw

    pose_handles = []

    with SharedMemoryManager() as shm_manager:
        with Spacemouse(shm_manager=shm_manager, deadzone=args.deadzone) as sm:
            t_start = time.monotonic()
            iter_idx = 0
            stop = False

            while not stop:
                # calculate timing
                t_cycle_end = t_start + (iter_idx + 1) * dt
                t_sample = t_cycle_end - command_latency
                # t_command_target = t_cycle_end + dt
                precise_wait(t_sample)

                # get teleop command
                sm_state = sm.get_motion_state_transformed()
                dpos = sm_state[:3] * (args.max_pos_speed / frequency)
                drot_xyz = sm_state[3:] * (args.max_rot_speed / frequency)
                drot = st.Rotation.from_euler("xyz", drot_xyz)

                if sm.is_button_pressed(0) or sm.is_button_pressed(1):
                    env.gripper_close() if gripper_open else env.gripper_open()
                    gripper_open = not gripper_open

                new_target_pose = target_pose.copy()
                new_target_pose[:3] += dpos
                new_target_pose[3:] = (
                    drot * st.Rotation.from_rotvec(target_pose[3:])
                ).as_rotvec()

                # Run collision-free IK to see if we can reach the new target pose
                target_joint_positions, ik_success = env.inverse_kinematics(
                    to_pb_pose(new_target_pose)
                )
                if ik_success:
                    # update joint positions and target pose
                    env.set_joint_positions(target_joint_positions)
                    target_pose = new_target_pose

                # Draw the current target pose
                # pybullet draw is a bit messed up so some lines may linger around
                with suppress_stdout():
                    remove_handles(pose_handles)
                    pose_handles = draw_pose(to_pb_pose(target_pose))

                precise_wait(t_cycle_end)
                iter_idx += 1


if __name__ == "__main__":
    DemoPybulletArgs.frequency = 30
    main()
