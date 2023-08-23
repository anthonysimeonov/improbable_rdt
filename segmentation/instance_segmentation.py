import os, os.path as osp
import sys
import argparse
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import time

# import from airobot
from airobot import log_info, log_warn, log_debug, log_critical, set_log_level

# imports from detectron
# from detectron2.utils.logger import setup_logger
# setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# # import from UOIS
# from uois.data_augmentation import array_to_tensor, standardize_image
# from uois.segmentation import UOISNet3D 
# import uois.util.utilities as uois_util_


class UOISInstanceSeg:
    def __init__(self, cfg):
        self.cfg = cfg

        dsn_config = {

            # Sizes
            'feature_dim' : 64, # 32 would be normal

            # Mean Shift parameters (for 3D voting)
            'max_GMS_iters' : 10,
            'epsilon' : 0.05, # Connected Components parameter
            'sigma' : 0.02, # Gaussian bandwidth parameter
            'num_seeds' : 200, # Used for MeanShift, but not BlurringMeanShift
            'subsample_factor' : 5,

            # Misc
            'min_pixels_thresh' : 500,
            'tau' : 15.,

        }

        rrn_config = {

            # Sizes
            'feature_dim' : 64, # 32 would be normal
            'img_H' : 224,
            'img_W' : 224,

            # architecture parameters
            'use_coordconv' : False,

        }

        uois3d_config = {

            # Padding for RGB Refinement Network
            'padding_percentage' : 0.25,

            # Open/Close Morphology for IMP (Initial Mask Processing) module
            'use_open_close_morphology' : True,
            'open_close_morphology_ksize' : 9,

            # Largest Connected Component for IMP module
            'use_largest_connected_component' : True,

        }

        checkpoint_dir = self.cfg.segmentation_weights_dir
        # checkpoint_dir = '/home/appuser/uois/models/' # TODO: change this to directory of downloaded models
        dsn_filename = checkpoint_dir + 'DepthSeedingNetwork_3D_TOD_checkpoint.pth'
        rrn_filename = checkpoint_dir + 'RRN_OID_checkpoint.pth'
        uois3d_config['final_close_morphology'] = 'TableTop_v5' in rrn_filename
        self.uois_net_3d = UOISNet3D(uois3d_config,
                                     dsn_filename,
                                     dsn_config,
                                     rrn_filename,
                                     rrn_config
                                    )

    def make_prediction(self, rgb_img, xyz_img, viz=False, viz_fname='default_uois_output.png', target_obj_pixel=None):
        """
        Predicts segmentation masks, given RGB and "point cloud image" (depth image converted to
        [x, y, z] coordinate expressed in camera frame)

        Args:
            rgb_img (np.ndarray): Size H x W x 3
            xyz_img (np.ndarray): Size H x W x 3. 3D coordinate values computed using camera intrinsics
                and depth values
            target_obj_pixel (tuple): (int, int) tuple of the form (u, v), specifying the pixel
                coordinates for the target object. This is used to decide which returned segment
                among the possibly several should be returned
        """
        rgb_imgs = rgb_img[None, :, :, :]
        xyz_imgs = xyz_img[None, :, :, :]
        batch = {
            'rgb': array_to_tensor(rgb_imgs),
            'xyz': array_to_tensor(xyz_imgs)
        }
        fg_masks, center_offsets, initial_masks, seg_masks = self.uois_net_3d.run_on_batch(batch)

        seg_masks = seg_masks.cpu().numpy()
        fg_masks = fg_masks.cpu().numpy()
        center_offsets = center_offsets.cpu().numpy().transpose(0,2,3,1)
        initial_masks = initial_masks.cpu().numpy()

        num_objs = np.unique(seg_masks[0,...]).max() + 1
        if num_objs == 0:
            warn_str = 'No segments detected! Returning None'
            log_warn(warn_str)
            return None

        if viz:
            self.visualize_prediction(num_objs, seg_masks, rgb_imgs, xyz_imgs, fname=viz_fname)

        # get target object from among the different segments
        ut, vt = target_obj_pixel
        if target_obj_pixel is None:
            warn_str = '''
            Target object pixel not specified for class-free instance segmentation method
            UOIS -- no way to pick out a specific segment, returning random sample
            '''
            log_warn(warn_str)
            det_idx = np.random.randint(num_objs)
            det_mask = seg_masks[det_idx]
        else:
            # we know which pixel coordinate the object should correspond to
            for i in range(num_objs):
                masked_pixels = np.where(seg_masks[i])
                if ut in masked_pixels[0] and vt in masked_pixels[1]:
                    det_mask = seg_masks[i]
                    break

            # if we finish this loop it means the target pixel wasn't in the segments
            warn_str = '''
            Target object pixel not included in the predicted segments. No way to pick out a specific segment, returning random sample
            '''
            log_warn(warn_str)
            det_idx = np.random.randint(num_objs)
            det_mask = seg_masks[det_idx]
        return det_mask

    def visualize_prediction(self, num_objs, seg_masks, rgb_imgs, xyz_imgs, fname='default_uois_output.png'):
        for i in range(num_objs):

            # num_objs = max(np.unique(seg_masks[i,...]).max(), np.unique(label_imgs[i,...]).max()) + 1
            num_objs = np.unique(seg_masks[i,...]).max() + 1

            rgb = rgb_imgs[i].astype(np.uint8)
            depth = xyz_imgs[i,...,2]
            seg_mask_plot = uois_util_.get_color_mask(seg_masks[i,...], nc=num_objs)
            gt_masks = uois_util_.get_color_mask(seg_masks[i,...], nc=num_objs)

            images = [rgb, depth, seg_mask_plot, gt_masks]
            titles = [f'Image {i+1}', 'Depth',
                      f"Refined Masks. #objects: {np.unique(seg_masks[i,...]).shape[0]-1}",
                      f"Ground Truth (fake). #objects: {np.unique(seg_masks[i,...]).shape[0]-1}"
                     ]
            uois_util_.subplotter(images, titles, fig_num=i+1, save_fname=f'{i}_{fname}')


class DetectronInstanceSeg:
    def __init__(self, cfg, viz=True, vis_vis=None):
        self.cfg = cfg
        # setup segmentation model
        self.deg_cfg = get_cfg()
        # cfg.merge_from_file('/home/robot2/detectron2_repo/detectron2/config/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        self.deg_cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.deg_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        # cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
        self.deg_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.deg_cfg)

        self.target_object_classes = {
            'cup': 41
        }

        self.vis_vis = vis_vis  # visdom, if we have it

    def make_prediction(self, rgb_img, all_segments=False, viz=False, viz_fname='default_detectron_output.png'):
        # img = np.asarray(Image.open(obs_file))
        # img = imageio.imread(obs_file)
        img = rgb_img
            
        print('Got image, making prediction')
        outputs = self.predictor(img)
        masks = outputs['instances'].pred_masks.data.cpu().numpy()
        bboxes = outputs['instances'].pred_boxes.tensor.data.cpu().numpy()
        classes = outputs['instances'].pred_classes.data.cpu().numpy()
        scores = outputs['instances'].scores.data.cpu().numpy()

        if viz:
            self.visualize_prediction(img, outputs, fname=viz_fname)
        
        # print('here in make prediction to see masks, bboxes, classes, scores')
        # from IPython import embed; embed()

        n_segs_total = masks.shape[0]
        if all_segments:
            if n_segs_total > 0:
                log_warn('"all_segments" True, returning all segments found')
                return masks
            else:
                log_warn('Segmentation mask for not found for any object! Returning full point cloud')
                full_mask = np.ones((img.shape)).astype(bool)
                return full_mask[None, :, :]
        else:
            # loop through classes to find any detections of the target class
            det_iters = 0
            det_mask = None
            for k, v in self.target_object_classes.items():
                det_inds = np.where(classes == v)[0]
                if det_inds.shape[0] > 0:
                    det_idx = np.random.choice(det_inds)
                    det_class_id = classes[det_idx]
                    det_mask = masks[det_idx]

                    masked_img = img.copy()
                    masked_img[np.logical_not(det_mask)] = 0
                    # if viz:
                        # masked_fname = 'masked_%s_%d_' % (k,  det_iters) + viz_fname
                        # fig = plt.figure()
                        # plt.imshow(masked_img)
                        # plt.show()
                        # fig.savefig(masked_fname)
                    # 
                    # det_iters += 1

            if det_mask is None:
                log_warn('Segmentation mask for target class not found! Will try to return a random segmentation mask')
                if n_segs_total > 0:
                    det_idx = np.random.randint(n_segs_total)
                    det_mask = masks[det_idx]
                else:
                    log_warn('Segmentation mask for not found for any object! Returning full point cloud')
                    det_mask = np.ones((img.shape)).astype(bool)
            return det_mask

    def visualize_prediction(self, inp, outputs, fname='default_detectron_output.png'):
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(inp[:, :, ::-1], MetadataCatalog.get(self.deg_cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])

        if self.vis_vis is not None:
            self.vis_vis.image(out.transpose(2, 0, 1), opts=dict(title='detectron output'))
        else:
            print(f'Don"t have visdom interface available!')
        # fig = plt.figure()
        # plt.imshow(out.get_image()[:, :, ::-1])
        # plt.show()
        # fig.savefig(fname)


class InstanceSegServer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.segmentation_interface = DetectronInstanceSeg(cfg)
        self.seg_method = 'detectron'
        #self.seg_methd = self.cfg.METHOD
        #if self.seg_method == 'detectron':
        #    self.segmentation_interface = DetectronInstanceSeg(cfg)
        #elif self.seg_method == 'UOIS':
        #    self.segmentation_interface = UOISInstanceSeg(cfg)
        #else:
        #    raise ValueError(f'Instance segmentation method {self.seg_method} not recognized')

    def get_target_pcd(self, point_cloud_dict, rgb_image, depth_image, viz=False, target_obj_pixel=None, all_segments=False):
        """
        Function to run instance segmentation and obtain a mask for the target object in the
        full point cloud.

        Args:
            point_cloud_dict (dict): Keys "world": Full point cloud, expresesed in world frame (Nx3 np.ndarray), 
                "cam": Full point cloud, expressed in camera frame (Nx3 np.ndarray), 
                "cam_img": Full point cloud, expressed in camera frame, reshaped to be same size as depth image (HxWx3 np.ndarray),
                "cam_pose_mat": 4x4 matrix representing the world frame pose of the camera
            rgb_image (np.ndarray): Shape H x W x 3
            depth_image (np.ndarray): Shape H x W (each channel is the z-depth value)

        Returns:
            np.ndarray: Shape N' x 3 where N' is the number of pixels in the observation corresponding 
                to the target o bject  segmentation
            np.ndarray: Shape H x W, where each element is a True/False value (True indicates that
                this pixel corresponds to the target object)
        """
        pcd_world = point_cloud_dict['world']
        if self.seg_method == 'detectron':
            target_mask = self.segmentation_interface.make_prediction(rgb_image, all_segments=all_segments, viz=viz) 

        elif self.seg_method == 'UOIS':
            # uois expects the full point cloud, expressed in the camera frame
            # TODO: convert depth to xyz in cam frame 
            pcd_cam_frame = point_cloud_dict['cam']
            target_mask = self.segmentation_interface.make_prediction(rgb_image, pcd_cam_frame, target_obj_pixel, all_segments=all_segments, viz=viz) 

        if all_segments:
            out_target_mask = target_mask[np.random.choice(target_mask.shape[0])]
        else:
            out_target_mask = target_mask
            
        # target_point_cloud = pcd_world[target_mask]
        target_point_cloud = pcd_world[out_target_mask.reshape(-1)]
        return target_point_cloud, target_mask



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # main(args)

    # img_file = osp.join(os.getcwd(), 'filename.png')
    # img_file = osp.join(os.getcwd(), 'noise_img.png')
    # img_file = osp.join(os.getcwd(), 'get_images/get_images_1/imgs/0.png')
    predictor = DetectronInstanceSeg()
    for i in range(16):
        # img_file = osp.join(os.getcwd(), 'get_images/get_images_%d/imgs/0.png' % i)
        img_file = osp.join(os.getcwd(), 'get_rot_images/get_rot_images_%d/imgs/0.png' % i)
        predictor.make_prediction(img_file, viz=True, viz_fname='det_out_rot_%d.png' % i)

    from IPython import embed; embed()
