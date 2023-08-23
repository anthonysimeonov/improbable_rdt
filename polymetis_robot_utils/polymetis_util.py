import numpy as np
import torch
from scipy.spatial.transform import Rotation as R


class PolymetisHelper:
    def __init__(self):
        pass 

    @staticmethod
    def polypose2mat(polypose):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = polypose[0].numpy()
        pose_mat[:-1, :-1] = R.from_quat(polypose[1].numpy()).as_matrix()
        return pose_mat 

    @staticmethod
    def polypose2list(polypose):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = polypose[0].numpy()
        pose_mat[:-1, :-1] = R.from_quat(polypose[1].numpy()).as_matrix()
        pose_list = pose_mat[:-1, -1].tolist() + R.from_matrix(pose_mat[:-1, :-1]).as_quat().tolist()
        return pose_list

    @staticmethod
    def polypose2np(polypose):
        pose_mat = np.eye(4)
        pose_mat[:-1, -1] = polypose[0].numpy()
        pose_mat[:-1, :-1] = R.from_quat(polypose[1].numpy()).as_matrix()
        pose_list = pose_mat[:-1, -1].tolist() + R.from_matrix(pose_mat[:-1, :-1]).as_quat().tolist()
        return np.asarray(pose_list)

    @staticmethod 
    def mat2polypose(pose_mat):
        trans = torch.from_numpy(pose_mat[:-1, -1])
        quat = torch.from_numpy(R.from_matrix(pose_mat[:-1, :-1]).as_quat())
        return trans, quat

    @staticmethod 
    def np2polypose(pose_np):
        trans = torch.from_numpy(pose_np[:3])
        quat = torch.from_numpy(pose_np[3:])
        return trans, quat
