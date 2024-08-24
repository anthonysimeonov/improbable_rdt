################################################################################

def convert_reference_frame_mat(pose_source_mat, pose_frame_target_mat, pose_frame_source_mat):
    
    # transform that maps from target to source (S = XT)
    target2source_mat = torch.matmul(pose_frame_source_mat, torch.inverse(pose_frame_target_mat))
    # target2source_mat = torch.bmm(pose_frame_source_mat, torch.inverse(pose_frame_target_mat))

    # obtain source pose in target frame
    pose_source_in_target_mat = torch.matmul(target2source_mat, pose_source_mat)
    return pose_source_in_target_mat


def to_mat(pos, quat):
    T = torch.eye(4).reshape(1, 4, 4).repeat(pos.shape[0]).to(pos.device)
    T[:-1, :-1] = matrix_from_quat(quat)
    T[:-1, -1] = pos
    return T


def to_pos_quat(mat):
    pos = mat[:, :-1, -1]
    quat = quat_from_matrix(mat[: :-1, :-1])
    return pos, quat


def convert_wrist2tip(wrist_pos, wrist_quat):
    """
    Function to convert a pose of the wrist link (nominally, panda_link8) to
    the pose of the frame in between the panda hand fingers
    """
    wrist2tip_tf = torch.eye(4); wrist2tip_tf[:-1, -1] = torch.Tensor([0.0, 0.0, 0.1034])
    
    wrist_pose_mat = to_mat(wrist_pos, wrist_quat)
    tip_pose_mat = convert_reference_frame_mat(
        pose_source_mat=wrist2tip_tf,
        pose_frame_target_mat=torch.eye(4),
        pose_frame_source_mat=wrist_pose_mat
    )

    tip_pos, tip_quat = to_pos_quat(tip_pose_mat)
    return tip_pos, tip_quat


def convert_tip2wrist(tip_pos, tip_quat):
    """
    Function to convert a pose of the wrist link (nominally, panda_link8) to
    the pose of the frame in between the panda hand fingers
    """
    tip2wrist_tf = np.eye(4); tip2wrist_tf[:-1, -1] = np.array([0.0, 0.0, -0.1034])
    tip_pose_mat = to_mat(tip_pos, tip_quat)
    wrist_pose_mat = convert_reference_frame_mat(
        pose_source_mat=tip2wrist_tf,
        pose_frame_target_mat=torch.eye(4),
        pose_frame_source_mat=tip_pose_mat
    )

    wrist_pos, wrist_quat = to_pos_quat(wrist_pose_mat)
    return wrist_pos, wrist_quat

################################################################################

################################################################################

def convert_reference_frame_mat(pose_source_mat, pose_frame_target_mat, pose_frame_source_mat):
    
    # transform that maps from target to source (S = XT)
    target2source_mat = np.matmul(pose_frame_source_mat, np.linalg.inv(pose_frame_target_mat))

    # obtain source pose in target frame
    pose_source_in_target_mat = np.matmul(target2source_mat, pose_source_mat)
    return pose_source_in_target_mat


def to_mat(pos, quat):
    T = np.eye(4)
    T[:-1, :-1] = R.from_quat(quat).as_matrix()
    T[:-1, -1] = pos
    return T


def to_pos_quat(mat):
    pos = mat[:-1, -1]
    quat = R.from_matrix(mat[:-1, :-1]).as_quat()
    return pos, quat


def convert_wrist2tip(wrist_pos_t, wrist_quat_t):
    dev = wrist_pos_t.device
    wrist_pos, wrist_quat = wrist_pos_t.cpu().numpy().squeeze(), wrist_quat_t.cpu().numpy().squeeze()
    wrist2tip_tf = np.eye(4); wrist2tip_tf[:-1, -1] = np.array([0.0, 0.0, 0.1034])
    
    wrist_pose_mat = to_mat(wrist_pos, wrist_quat)
    tip_pose_mat = convert_reference_frame_mat(
        pose_source_mat=wrist2tip_tf,
        pose_frame_target_mat=np.eye(4),
        pose_frame_source_mat=wrist_pose_mat
    )

    tip_pos, tip_quat = to_pos_quat(tip_pose_mat)
    return torch.from_numpy(tip_pos).to(dev).reshape(1, 3), convert_quat(torch.from_numpy(tip_quat).to(dev), "wxyz").reshape(1, 4)


def convert_tip2wrist(tip_pos_t, tip_quat_t):
    dev = tip_pos_t.device
    tip_pos, tip_quat = tip_pos_t.cpu().numpy().squeeze(), tip_quat_t.cpu().numpy().squeeze()
    tip2wrist_tf = np.eye(4); tip2wrist_tf[:-1, -1] = np.array([0.0, 0.0, -0.1034])
    tip_pose_mat = to_mat(tip_pos, tip_quat)
    wrist_pose_mat = convert_reference_frame_mat(
        pose_source_mat=tip2wrist_tf,
        pose_frame_target_mat=np.eye(4),
        pose_frame_source_mat=tip_pose_mat
    )

    wrist_pos, wrist_quat = to_pos_quat(wrist_pose_mat)
    return torch.from_numpy(wrist_pos).to(dev).reshape(1, 3), convert_quat(torch.from_numpy(wrist_quat).to(dev), "wxyz").reshape(1, 4)

################################################################################


