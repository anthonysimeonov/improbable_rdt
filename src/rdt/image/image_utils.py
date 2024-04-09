import warnings
import numpy as np
import matplotlib.pyplot as plt
# import imageio

try:
    import cv2
except ImportError as e:
    warnings.warn(f'Error with importing opencv-python: {e}, cannot use cv2 for image utils')

try:
    import visdom
except ImportError as e:
    warnings.warn(f'Error with importing visdom: {e}, cannot use Visdom for image visualization until installed', ImportWarning)
    
    class DummyVisdom:
        class Visdom:
            def __init__(self):
                pass

        def __init__(self):
            self.Visdom = Visdom()
    
    visdom = DummyVisdom()

from typing import List


def show_imgs_visdom(vis_vis: visdom.Visdom, imgs_list: List[np.ndarray], title: str='default_title'):
    """
    Shows list of images with Visdom. Assumes a three-channel image (i.e., RGB)
    """
    imgs_to_show = []
    
    for i, img in enumerate(imgs_list):
        imgs_to_show.append(img.copy().transpose(2, 0, 1))

    vis_vis.images(imgs_to_show, padding=1, opts=dict(title=title))


def show_depth_imgs_visdom(vis_vis: visdom.Visdom, depth_imgs_list: List[np.ndarray], 
                           scale: float=1000.0, title: str='depth_default_title'):
    """
    Shows list of images with Visdom. Assumes a one-channel image (i.e., depth)
    """
    imgs_to_show = []
    
    for i, img in enumerate(depth_imgs_list):
        try:
            max_grayscale = np.amax(img)  # already assume we have min value equal to zero
            norm_grayscale = img / max_grayscale

            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(img * scale, alpha=0.03), cv2.COLORMAP_JET)
            # depth_colormap = (255 * plt.get_cmap('jet')(img)[:, :, :-1]).astype(int)
            depth_colormap = (255 * plt.get_cmap('jet')(norm_grayscale)[:, :, :-1]).astype(int)

            imgs_to_show.append(depth_colormap.copy().transpose(2, 0, 1))
        except Exception as e:
            print(f'[image_utils, show_depth_imgs_visdom]\n\n{e}\n\nDefaulting to showing one-channel image')
            imgs_to_show.append(img.copy())

    vis_vis.images(imgs_to_show, padding=1, opts=dict(title=title))


def show_imgs_pyplot(imgs_list: List[np.ndarray], title: str='default_title') -> plt.figure:
    """
    Shows list of images with pyplot. Assumes a three-channel image (i.e., RGB)
    """
    fig, axs = plt.subplots(1, len(imgs_list))

    for i, img in enumerate(imgs_list):
        axs[i].imshow(img)

    plt.show(title=title)

    return fig


def show_depth_imgs_pyplot(depth_imgs_list: List[np.ndarray], title: str='depth_default_title') -> plt.figure:
    """
    Shows list of images with pyplot. Assumes a one-channel image (i.e., depth)
    """
    fig, axs = plt.subplots(1, len(depth_imgs_list))

    for i, img in enumerate(depth_imgs_list):
        try:
            max_grayscale = np.amax(img)  # already assume we have min value equal to zero
            norm_grayscale = img / max_grayscale

            # depth_colormap = (255 * plt.get_cmap('jet')(img)[:, :, :-1]).astype(int)
            depth_colormap = (255 * plt.get_cmap('jet')(norm_grayscale)[:, :, :-1]).astype(int)

            axs[i].imshow(depth_colormap)
        except Exception as e:
            print(f'[image_utils, show_depth_imgs_pyplot]\n\n{e}\n\nDefaulting to showing one-channel image')
            axs[i].imshow(img)

    plt.show(title=title)

    return fig
            