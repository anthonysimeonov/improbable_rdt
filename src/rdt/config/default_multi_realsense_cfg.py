from yacs.config import CfgNode as CN

_C = CN()

_C.CAMERA_NAME_PREFIX = "cam_"

# _C.SERIAL_NUMBERS = [
#     '143122065292',
#     '843112073228',
#     '215122255998']

# _C.SERIAL_NUMBERS = ["840412060551", "215122255998"]
# _C.SERIAL_NUMBERS = ["843112073228", "215122255998"]
# _C.SERIAL_NUMBERS = ["242522072326", "215122255998"]
_C.SERIAL_NUMBERS = ["843112073228", "242522072355"]
# _C.SERIAL_NUMBERS = ["242622071805"]

_C.RGB_LCM_TOPIC_NAME_SUFFIX = "rgb_image"
_C.DEPTH_LCM_TOPIC_NAME_SUFFIX = "depth_image"
_C.INFO_LCM_TOPIC_NAME_SUFFIX = "info"
_C.POSE_LCM_TOPIC_NAME_SUFFIX = "pose"

_C.WIDTH = 640
_C.HEIGHT = 480
# _C.WIDTH = 1280
# _C.HEIGHT = 720
_C.FRAME_RATE = 30


def get_default_multi_realsense_cfg():
    return _C.clone()
