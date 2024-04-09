import sys
import time
import signal
import threading
import numpy as np
import meshcat

import lcm

from rdt.common import mc_util
from rdt.config.default_multi_realsense_cfg import get_default_multi_realsense_cfg
from rdt.common.real_util import RealCompressedPCDLCMSubscriber

if False:
    try:
        import open3d

        class PCDDS:
            def __init__(self, num_samples=1024):
                self.num_samples = num_samples
                self.o3d_pcd = open3d.geometry.PointCloud()

            def ds(self, np_pcd):
                self.o3d_pcd.points = open3d.utility.Vector3dVector(np_pcd)
                return np.asarray(self.o3d_pcd.farthest_point_down_sample(num_samples=self.num_samples).points)

    except ImportError as e:
        print(f'Import Error with open3d: {e}, open3d downsampling functions not available')

        class PCDDS:
            def __init__(self, num_samples=1024):
                self.num_samples = num_samples
            
            def ds(self, np_pcd):
                return np_pcd[::10]

class PCDDS:
    def __init__(self, num_samples=8096):
        self.num_samples = num_samples

    def ds(self, np_pcd):
        interval = int(np_pcd.shape[0] / self.num_samples)
        return np_pcd[::interval]
        # return np_pcd[np.random.permutation(np_pcd.shape[0])[:self.num_samples]]

ds_func = PCDDS().ds


def signal_handler(sig, frame):
    """
    Capture exit signal from keyboard
    """
    print('Exit')
    sys.exit(0)


def lc_th(lc):
    while True:
        lc.handle_timeout(1)
    

def pcd_subscriber_mc_visualize(subs, mc_vis):
    for (name, pcd_sub) in subs:
        pcd = pcd_sub.get_pcd(block=True)
        if pcd is None:
            return
    
        # show in meshcat
        mc_util.meshcat_pcd_show(mc_vis, ds_func(pcd), (0, 0, 0), name=f'scene/pcd_{name}', size=0.003)


def main(args):

    signal.signal(signal.SIGINT, signal_handler)
    mc_vis = meshcat.Visualizer(zmq_url=f'tcp://127.0.0.1:{args.port_vis}')
    mc_vis['scene'].delete()

    lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=1")

    rs_cfg = get_default_multi_realsense_cfg()
    serials = rs_cfg.SERIAL_NUMBERS

    prefix = rs_cfg.CAMERA_NAME_PREFIX
    camera_names = [f'{prefix}{i}' for i in range(len(serials))]

    # update the topic names based on each individual camera
    pcd_sub_names = [f'{cam_name}_point_cloud' for cam_name in camera_names]

    pcd_subscribers = []
    for i, name in enumerate(camera_names):
        pcd_sub = RealCompressedPCDLCMSubscriber(lc, pcd_sub_names[i])
        pcd_subscribers.append((name, pcd_sub))

    lc_thread = threading.Thread(target=lc_th, args=(lc,))
    lc_thread.daemon = True
    lc_thread.start()

    while True:
        pcd_subscriber_mc_visualize(pcd_subscribers, mc_vis)
        time.sleep(0.001)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port_vis', type=int, default=6000)
    args = parser.parse_args()
    main(args)
