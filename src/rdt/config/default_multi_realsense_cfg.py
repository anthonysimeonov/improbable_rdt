from yacs.config import CfgNode as CN 

_C = CN()

_C.CAMERA_NAME_PREFIX = 'cam_'

# _C.SERIAL_NUMBERS = [
#     '143122065292', 
#     '843112073228', 
#     '215122255998']

_C.SERIAL_NUMBERS = ['843112073228', '215122255998'] 
# _C.SERIAL_NUMBERS = ['215122255998'] 

_C.RGB_LCM_TOPIC_NAME_SUFFIX = 'rgb_image'
_C.DEPTH_LCM_TOPIC_NAME_SUFFIX = 'depth_image'
_C.INFO_LCM_TOPIC_NAME_SUFFIX = 'info'
_C.POSE_LCM_TOPIC_NAME_SUFFIX = 'pose'

_C.WIDTH = 640
_C.HEIGHT = 480
_C.FRAME_RATE = 30

def get_default_multi_realsense_cfg():
    return _C.clone()
