import os, os.path as osp
import imageio
import numpy as np

from enum import Enum


class CollectEnum(Enum):
    DONE_FALSE = 2  # Data collection in progress.
    SUCCESS = 3  # Successful trajectory is collected.
    FAIL = 4  # Failed trajectory is collected.
    REWARD = 5  # Annotate reward +1.
    SKILL = 6  # Annotate new skill.
    RESET = 7  # Reset environment.
    TERMINATE = 8  # Terminate data collection.


def save_rgbd(rgbd_list, dirname):
    assert os.path.exists(dirname), f'Directory {dirname} does not exist, cannot save data'

    n_imgs = len(rgbd_list)
    rgb_fnames = [osp.join(dirname, f'rgb_{i}.png') for i in range(n_imgs)]
    depth_fnames = [osp.join(dirname, f'depth_{i}.png') for i in range(n_imgs)]

    # save rgb
    for i, imgs in enumerate(rgbd_list):
        imageio.imwrite(rgb_fnames[i], imgs['rgb'])

    # save depth
    for i, imgs in enumerate(rgbd_list):
        imageio.imwrite(depth_fnames[i], imgs['depth'])


def save_robot_state(robot_state_dict):
    pass
