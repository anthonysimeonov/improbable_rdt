import numpy as np
import open3d as o3d

try:
    from airobot import log_debug
except ImportError as e:
    print(f"{e}")

from rdt.common import util  # , trimesh_util


def extend_pcds(cam_pcds, pcd_list, cam_scores, pcd_scores, threshold=0.08):
    """
    Given partial pcds in a camera view, finds which existing incomplete pcd it most likely belongs to
    Finds the partial pcd with the closest centroid that is at least 2cms away from the existing pcds

    Remaining partial pcds will be assumed to belong to a new obj and a new set is created

    cam_pcds: np.array (n, m, 3)
    pcd_list: np.array (j, k, 3)
    cam_scores: (n, a) list
    pcd_scores: (j, b) list

    return: updated pcd list
    """
    centroids = [np.average(partial_obj, axis=0) for partial_obj in pcd_list]

    new_pcds = []
    new_scores = []
    for i, centroid in enumerate(centroids):
        new_centroids = np.array([np.average(pcd, axis=0) for pcd in cam_pcds])
        if len(new_centroids) == 0:
            break
        diff = new_centroids - centroid
        centroid_dists = np.sqrt(np.sum(diff**2, axis=-1))
        min_idx = np.argmin(centroid_dists)
        log_debug(f"closest centroid is {centroid_dists[min_idx]} away")

        if centroid_dists[min_idx] <= threshold:
            original_pcd = pcd_list[i]
            updated_pcds = np.concatenate([original_pcd, cam_pcds[min_idx]], axis=0)
            new_score = pcd_scores[i] + [cam_scores[min_idx][0]]
            new_pcds.append(updated_pcds)
            new_scores.append(new_score)
        else:
            new_pcds.append(pcd_list[i])
            new_scores.append(pcd_scores[i])

    if new_pcds == []:
        return pcd_list, pcd_scores
    return new_pcds, new_scores


def filter_pcds(
    pcds, scores, bounds=None, mean_inliers=False, downsample=False, show=False
):
    filtered_pcds = []
    filtered_scores = []
    for i, full_pcd in enumerate(pcds):
        obj_pcd = manually_segment_pcd(
            full_pcd,
            bounds=bounds,
            mean_inliers=mean_inliers,
            downsample=downsample,
            show=show,
        )
        if obj_pcd.all():
            filtered_pcds.append(obj_pcd)
            filtered_scores.append(scores[i])
            log_debug(f"pcd of size {len(obj_pcd)} with score {scores[i]}")
    if filtered_pcds:
        pass
    else:
        log_debug("No pcds after apply masks :(")
    return filtered_pcds, filtered_scores


def manually_segment_pcd(
    full_pcd, bounds=None, mean_inliers=False, downsample=False, show=False
):
    crop_pcd = full_pcd

    if bounds:
        x, y, z = bounds
        pcd_proc_debug_str1 = f"[manually_segment_pcd] Boundary params: "
        pcd_proc_debug_str2 = (
            f"x: [{x[0]}, {x[1]}], y: [{y[0]}, {y[1]}], z: [{z[0]}, {z[1]}]"
        )
        print(pcd_proc_debug_str1 + pcd_proc_debug_str2)
        crop_pcd, crop_idx = util.crop_pcd(full_pcd, x=x, y=y, z=z, return_idx=True)

    if mean_inliers:
        pcd_mean = np.mean(crop_pcd, axis=0)
        inliers = np.where(np.linalg.norm(crop_pcd - pcd_mean, 2, 1) < 0.2)[0]
        crop_pcd = crop_pcd[inliers]

    if downsample:
        perm = np.random.permutation(crop_pcd.shape[0])
        size = int(crop_pcd.shape[0])
        crop_pcd = crop_pcd[perm[:size]]

    if show:
        # trimesh_util.trimesh_show([crop_pcd])
        print(f"Show not implemented")
        pass

    crop_bool = np.zeros(full_pcd.shape[0])
    crop_bool[crop_idx] = 1
    return crop_pcd, crop_bool.astype(bool)


def pcds_from_masks(full_pcd, depth_valid, masks, scores, clustering=False):
    masked_regions = []
    masked_scores = []
    for i, mask in enumerate(masks):
        camera_mask = depth_valid != 0
        camera_binary = np.zeros(depth_valid.shape)
        camera_binary[camera_mask] = 1
        joined_mask = np.logical_and(camera_binary, mask)

        cropped_pcd = full_pcd[joined_mask]
        flat_pcd = cropped_pcd.reshape((-1, 3))

        if clustering:
            largest = get_largest_pcd(flat_pcd)
            flat_pcd = largest

        masked_regions.append(flat_pcd)
        masked_scores.append(scores[i])
    if masked_regions:
        pass
    else:
        log_debug("No pcds after apply masks :(")
    return masked_regions, masked_scores
