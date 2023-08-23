import os, os.path as osp
import sys
import argparse
import numpy as np
import imageio
from PIL import Image
import matplotlib.pyplot as plt
import time

# from detectron2.utils.logger import setup_logger
# setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# import lcm
# from panda_rrp_utils import lcm_util
# sys.path.append(osp.join(os.environ['NDF_SOURCE_DIR'], 'lcm_types'))
# from rrp_lcm import point_cloud_array_t, point_cloud_t, start_goal_pose_stamped_t, pose_stamped_t

# class PointCloudSegmentation:
#     def __init__(self, lc, predictor, rgb_sub_name, depth_sub_name):
#         self.lc = lc
#         self.rgb_img_sub_name = rgb_sub_name
#         self.depth_img_sub_name = depth_sub_name
#         self.rgb_sub = self.lc.subscribe(self.rgb_img_sub_name, self.rgb_img_handler)
#         self.depth_sub = self.lc.subscribe(self.depth_img_sub_name, self.depth_img_handler)

#     def rgb_img_handler(self, channel, data):
#         msg = rgb_img_t.decode(data)
#         print('got something (rgb)')
#         self.rgb_img = np.random.randint(0, 255, size=(480, 640, 3)).astype(np.uint16)
#         self.received_rgb = True

#     def depth_img_handler(self, channel, data):
#         msg = depth_img_t.decode(data)
#         print('got something (depth)')
#         # self.start_pose = lcm_util.pose_stamped2list(msg.start_pose)
#         # self.depth_img = lcm_util.convert_depth(msg.start_pose)
#         self.depth_img = np.random.randint(0, 255, size=(480, 640)).astype(np.uint16)
#         self.received_depth = True
    
#     def get_rgb_and_depth(self):
#         rgb = self.get_rgb()
#         depth = self.get_depth()
#         out = {}
#         out['rgb'] = rgb
#         out['depth'] = depth
#         return out

#     def get_depth(self):
#         # wait to receive commands from the optimizer
#         self.received_depth = False
#         while True:
#             if self.received_depth:
#                 break
#             self.lc.handle()
#             time.sleep(0.001)
        
#         return self.depth_img

#     def get_rgb(self):
#         # wait to receive commands from the optimizer
#         self.received_rgb = False
#         while True:
#             if self.received_rgb:
#                 break
#             self.lc.handle()
#             time.sleep(0.001)
        
#         return self.rgb_img

class DetectronInstanceSeg:
    def __init__(self, viz=True):
        # setup segmentation model
        self.cfg = get_cfg()
        # cfg.merge_from_file('/home/robot2/detectron2_repo/detectron2/config/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml')
        self.cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        # cfg.MODEL.WEIGHTS = 'detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl'
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg)

        self.target_object_classes = {
            'cup': 41
        }

    def make_prediction(self, obs_file, viz=False, fname='default_detectron_output.png'):
        # img = np.asarray(Image.open(obs_file))
        img = imageio.imread(obs_file)
            
        print('Got image, making prediction')
        outputs = self.predictor(img)
        masks = outputs['instances'].pred_masks.data.cpu().numpy()
        bboxes = outputs['instances'].pred_boxes.tensor.data.cpu().numpy()
        classes = outputs['instances'].pred_classes.data.cpu().numpy()
        scores = outputs['instances'].scores.data.cpu().numpy()

        if viz:
            self.visualize_prediction(img, outputs, fname=fname)
        
        print('here in make prediction to see masks, bboxes, classes, scores')
        from IPython import embed; embed()

        # loop through classes to find any detections of the target class
        det_iters = 0
        for k, v in self.target_object_classes.items():
            det_inds = np.where(classes == v)[0]
            if det_inds.shape[0] > 0:
                det_idx = np.random.choice(det_inds)
                det_class_id = classes[det_idx]
                det_mask = masks[det_idx]

                masked_img = img.copy()
                masked_img[np.logical_not(det_mask)] = 0
                # masked_img = img[det_mask]
                if viz:
                    masked_fname = 'masked_%s_%d_' % (k,  det_iters) + fname
                    fig = plt.figure()
                    plt.imshow(masked_img)
                    plt.show()
                    fig.savefig(masked_fname)
                    # self.visualize_prediction(masked_img, outputs, fname='masked_%d_w_outputs_' % det_iters + fname)
                
                det_iters += 1


        return None

    def visualize_prediction(self, input, outputs, fname='default_detectron_output.png'):
        # We can use `Visualizer` to draw the predictions on the image.
        v = Visualizer(input[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        # cv2_imshow(out.get_image()[:, :, ::-1])
        fig = plt.figure()
        plt.imshow(out.get_image()[:, :, ::-1])
        plt.show()
        fig.savefig(fname)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    # main(args)

    # img_file = osp.join(os.getcwd(), 'filename.png')
    # img_file = osp.join(os.getcwd(), 'noise_img.png')
    # img_file = osp.join(os.getcwd(), 'get_images/get_images_1/imgs/0.png')
    predictor = DetectronInstanceSeg()

    print('here with det2')
    from IPython import embed; embed()

    for i in range(16):
        # img_file = osp.join(os.getcwd(), 'get_images/get_images_%d/imgs/0.png' % i)
        img_file = osp.join(os.getcwd(), 'get_rot_images/get_rot_images_%d/imgs/0.png' % i)
        predictor.make_prediction(img_file, viz=True, fname='det_out_rot_%d.png' % i)

    from IPython import embed; embed()
