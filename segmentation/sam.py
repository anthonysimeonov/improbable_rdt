import os, os.path as osp
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

from segment_anything import sam_model_registry, SamPredictor
from rrp_robot.utils import path_util

class SAMSeg:
    def __init__(self, cuda=False):
        chkpt_path = osp.join(path_util.get_rrp_model_weights(), 'segmentation/sam_vit_h_4b8939.pth')
        if not osp.exists(chkpt_path):
            from huggingface_hub import hf_hub_download
            chkpt_path = hf_hub_download("ybelkada/segment-anything", "checkpoints/sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=chkpt_path)
        sam.eval()
        if cuda:
            sam.to(device='cuda')
        self.predictor = SamPredictor(sam)

    def masks_from_bbs(self, image, all_obj_bbs):
        self.predictor.set_image(image)
        all_obj_masks = {}
        captions = list(all_obj_bbs.keys())
        for caption in captions:
            all_obj_masks[caption] = []
            for bb in all_obj_bbs[caption]:
                combined_mask = self.get_mask_from_bb(bb)
                if combined_mask is None:
                    continue
                all_obj_masks[caption].append(combined_mask)

        return all_obj_masks

    def mask_from_bb(self, bb, image=None, show=False):
        if image is not None:
            self.predictor.set_image(image)

        bb_inpt = np.array([np.array(bb)])
        masks_np, _, _ = self.predictor.predict(box = bb_inpt)
        if len(masks_np) == 0:
            return []
        combined_mask = np.array(np.sum(masks_np, axis=0), dtype=bool)
        if show:
            show_mask(combined_mask, plt)
        return combined_mask

    def mask_from_pt(self, input_pt, pt_label=1, image=None, show=False):
        if image is not None:
            self.predictor.set_image(image)

        pt_inpt = np.array([np.array(input_pt)])
        pt_label = np.array([pt_label])
        masks_np, _, _ = self.predictor.predict(point_coords=pt_inpt, point_labels=pt_label)
        if len(masks_np) == 0:
            return []
        combined_mask = np.array(np.sum(masks_np, axis=0), dtype=bool)
        if show:
            show_mask(combined_mask, plt)
        return combined_mask

def show_mask(mask, ax):
    color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
