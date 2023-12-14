import time
from functools import cached_property
from typing import List, Optional, Sequence, Tuple

from rdt.spacemouse.spacemouse_pybullet.pybullet_helpers import (
    LockRenderer,
    add_data_path,
    any_link_pair_collision,
    connect,
    get_joint_positions,
    get_link_pose,
    get_links,
    get_movable_joints,
    get_self_link_pairs,
    inverse_kinematics_helper,
    link_from_name,
    load_pybullet,
    pairwise_link_collision,
    set_camera_pose,
    set_joint_positions,
    wait_for_user,
)

neutral_joint_positions = (
    -0.000000,
    -0.785398,
    0.000000,
    -2.356194,
    0.000000,
    1.570796,
    0.785398,
    0.04,
    0.04,
)


# (translation, quaternion) = (x, y, z), (x, y, z, w)
PybulletPose = Tuple[Tuple[float, float, float], Tuple[float, float, float, float]]


class PandaPybullet:
    """Helper class for the Panda in Pybullet."""

    def __init__(
        self,
        initial_joint_positions: Sequence[float] = neutral_joint_positions,
        use_gui: bool = True,
    ):
        # Load the Panda into pybullet
        connect(use_gui)
        add_data_path()
        self._panda = load_pybullet("franka_panda/panda.urdf", fixed_base=True)
        self._plane = load_pybullet("plane.urdf")
        if use_gui:
            set_camera_pose((1, 0.0, 0.65))

        # Set the joint positions and make sure there is no collision
        set_joint_positions(self._panda, self._joints, initial_joint_positions)
        if self.has_collision():
            raise ValueError("Initial joint positions are in collision!")

    @cached_property
    def _ee_link(self) -> int:
        return link_from_name(self._panda, "panda_grasptarget")

    @cached_property
    def _links(self) -> Sequence[int]:
        """Panda links excluding the base."""
        return get_links(self._panda)

    @cached_property
    def _self_link_pairs(self) -> Sequence[Tuple[int, int]]:
        """All pairs of Panda links except gripper left and right fingers. Used for collision checking."""
        pairs = get_self_link_pairs(self._panda, self._joints)
        assert pairs[-1] == (9, 10)
        return pairs[:-1]

    @cached_property
    def _joints(self) -> Sequence[int]:
        """Includes arm and gripper joints."""
        return get_movable_joints(self._panda)

    def ee_pose(self) -> PybulletPose:
        return get_link_pose(self._panda, self._ee_link)

    def joint_positions(self) -> List[float]:
        """Note: this includes gripper joints."""
        return get_joint_positions(self._panda, self._joints)

    def set_arm_joint_positions(self, positions: Sequence[float]):
        set_joint_positions(self._panda, self._joints[:-2], positions)

    def set_joint_positions(self, positions: Sequence[float]):
        set_joint_positions(self._panda, self._joints, positions)

    def gripper_close(self):
        gripper_joints = self._joints[-2:]
        set_joint_positions(self._panda, gripper_joints, [0.0, 0.0])

    def gripper_open(self):
        gripper_joints = self._joints[-2:]
        set_joint_positions(self._panda, gripper_joints, [0.04, 0.04])

    def has_collision(self) -> bool:
        """Checks for collision with the plane and self-collision."""
        if any_link_pair_collision(self._panda, self._links, self._plane):
            return True
        for link_1, link_2 in self._self_link_pairs:
            if pairwise_link_collision(self._panda, link_1, self._panda, link_2):
                return True
        return False

    def inverse_kinematics(
        self, target_pose: PybulletPose
    ) -> Tuple[Optional[List[float]], bool]:
        """
        If the target pose is in collision (with the plane at z=0.0) or unreachable,
        we return None and False. Otherwise, we return the IK solution and True.
        """
        current_joint_positions = self.joint_positions()
        target_joint_positions = inverse_kinematics_helper(
            self._panda, self._ee_link, target_pose
        )

        with LockRenderer():
            set_joint_positions(self._panda, self._joints, target_joint_positions)
            # If there's a collision, set the joint positions back to the original
            if self.has_collision():
                set_joint_positions(self._panda, self._joints, current_joint_positions)
                return None, False

        # Collision free!
        return target_joint_positions, True


if __name__ == "__main__":
    import numpy as np

    panda = PandaPybullet(use_gui=True)
    init_ee = panda.ee_pose()
    print("Initial EE pose:", init_ee)

    target_ee = ((0.5, 0.15, 0.02), init_ee[-1])
    print("Target EE pose:", target_ee)

    ik_durations = []
    for _ in range(100):
        start_time = time.perf_counter()
        q_target, ik_success = panda.inverse_kinematics(target_ee)
        panda.set_joint_positions(q_target)
        duration = time.perf_counter() - start_time
        ik_durations.append(duration)

    final_ee = panda.ee_pose()
    print("Final EE pose:", final_ee)

    print("Mean IK duration per call:", np.mean(ik_durations))
    print("IK frequency (Hz):", 1 / np.mean(ik_durations))
    wait_for_user()
