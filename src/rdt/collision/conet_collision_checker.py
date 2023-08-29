import numpy as np
import trimesh
import torch
import sys
import os
import os.path as osp
import meshcat
from scipy.spatial.transform import Rotation as R

sys.path.append(os.getenv('CGN_SRC_DIR'))
from test_meshcat_pcd import viz_scene as V
from test_meshcat_pcd import viz_pcd as VP

# con_path = os.path.join(os.getenv('HOME'), 'graspnet/graspnet/subgoal-net/evaluation/coll_check_example/')
# sys.path.append(con_path)

from rpdiff.utils import path_util
from rpdiff.collision.conet_model.conet_coll_checker_full import LocalPoolPointnet, LocalDecoder, ConvolutionalOccupancyNetwork
# from evaluation.conet_coll_checker_full import LocalPoolPointnet, LocalDecoder, ConvolutionalOccupancyNetwork

BASE_DIR = '/home/alinasar/subgoal-net/evaluation/'

GRIPPER_PATH = 'panda_gripper.obj'
GRIPPER_2f140_PATH = 'robotiq_arg2f_base_link.stl'
GRIPPER_2f140_FINGER_PATH = 'robotiq_arg2f_140_outer_finger.stl'
GRIPPER_2f140_KNUCKLE_PATH = 'robotiq_arg2f_140_inner_knuckle.stl'

models_dir = osp.join(path_util.get_rpdiff_src(), 'collision/gripper_models')
GRIPPER_PATH = osp.join(models_dir, GRIPPER_PATH)
GRIPPER_2f140_PATH = osp.join(models_dir, GRIPPER_2f140_PATH)
GRIPPER_2f140_FINGER_PATH = osp.join(models_dir, GRIPPER_2f140_FINGER_PATH)
GRIPPER_2f140_KNUCKLE_PATH = osp.join(models_dir, GRIPPER_2f140_KNUCKLE_PATH)

# BASE_DIR = '/home/alina/graspnet/graspnet/subgoal-net/evaluation/'
# GRIPPER_PATH = '/home/alina/graspnet/graspnet/subgoal-net/gripper_models/panda_gripper/panda_gripper.obj'
class PointCollision():
    def __init__(self, model, robotiq=False):
        '''
        Pipeline for comparison. Will try and find a feasible grasp based on an initial scene and point-mesh
        checks between the gripper and the visible scene pointcloud.
        '''
        self.model = model
        self.robotiq = robotiq
        if self.model is not None:
            self.sig = torch.nn.Sigmoid().to(self.model.device)
        if robotiq:
            self.g_path = GRIPPER_2f140_PATH
            self.gripper_mesh = [trimesh.load(self.g_path), trimesh.load(GRIPPER_2f140_FINGER_PATH)] #, trimesh.load(GRIPPER_2f140_KNUCKLE_PATH)]
            self.robotiq_query_pts = []
            for m in self.gripper_mesh:
                # self.robotiq_query_pts.append(np.concatenate((trimesh.sample.volume_mesh(m, 1000), np.ones((1000,1))), axis=1))
                self.robotiq_query_pts.append(np.concatenate((m.sample(1000), np.ones((1000,1))), axis=1))
        else:
            self.g_path = GRIPPER_PATH
            self.gripper_mesh = trimesh.load(self.g_path)        
            self.gripper_query_pts = trimesh.sample.volume_mesh(self.gripper_mesh, 2000)
            self.gripper_query_pts = np.concatenate((self.gripper_query_pts, self.gripper_mesh.sample(500)), 0) # add points explicitly on the mesh surface
            self.gripper_query_pts = np.concatenate((self.gripper_query_pts, np.ones((self.gripper_query_pts.shape[0], 1))), axis=1)

        # weights_folder = BASE_DIR+'weights'

        weights_folder = osp.join(path_util.get_rpdiff_model_weights(), 'conet_weights')

        # coll_check_weights = osp.join(weights_folder, 'coll_check/table_clutter_single_scene_8192/model_270000.pt')
        # coll_check_weights = osp.join(weights_folder, 'less_islands/model_100000.pt') #'model_0_75_180000.pt')

        num = 140000
        # num = 70000
        coll_check_weights = osp.join(
            weights_folder, f'shapenetsem_tig_singleview_notable_less_islands_1-12obj_8192_pw1-0/model_{num}.pt') #'model_0_75_180000.pt')
        coll_check_state_dict = torch.load(coll_check_weights)
        pt_dim = 3
        padding = 0.1
        voxel_reso_grid = 32
        voxel_reso_grid_pt = 128
        scene_encoder_kwargs = {
            'local_coord': False,
            # encoder: pointnet_local_pool
            'c_dim': 32,
            # encoder_kwargs:
            'hidden_dim': 32,
            'plane_type': ['xz', 'xy', 'yz', 'grid'],
            # 'plane_type': ['grid']
            'plane_resolution': 128,
            'unet3d': True,
            'unet3d_kwargs': {
                    'num_levels': 3,
                    'f_maps': 32,
                    'in_channels': 32,
                    'out_channels': 32,
                    'plane_resolution': 128,
                },
            'unet': True,
            'unet_kwargs': {
                    'depth': 5,
                    'merge_mode': 'concat',
                    'start_filts': 32,
                }
        }

        mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
        mc_vis['scene'].delete()
        # Encoder
        scene_encoder = LocalPoolPointnet(
            dim=pt_dim,
            padding=padding,
            grid_resolution=voxel_reso_grid,
            mc_vis=mc_vis,
            **scene_encoder_kwargs).cuda()

        # Decoder
        scene_decoder = LocalDecoder(
            dim=pt_dim,
            c_dim=scene_encoder_kwargs['c_dim'],
            hidden_size=32
        ).cuda()

        # Full model
        self.con = ConvolutionalOccupancyNetwork(
            decoder=scene_decoder,
            encoder=scene_encoder
        ).cuda()
        self.con.load_state_dict(coll_check_state_dict['model'])
        
        self.gripper_tf = np.array([[ 9.46042344e-04,-3.23783829e-04,-1.29582650e-05, 0.00000000e+00],
                                    [-3.24043028e-04,-9.45285611e-04,-3.78316034e-05, 0.00000000e+00],
                                    [ 1.15856773e-19, 3.99893342e-05,-9.99200107e-04,-3.50000000e-02],
                                    [ 0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        
    def point_mesh_coll_check(self, grasp, pcd, thresh=0):
        '''
        args:
            grasp -- 4x4 pose matrix
            pcd -- Nx3 pointcloud of scene
            thresh -- number of points that are okay to collide with gripper (account for camera noise)
        returns:
            collides -- True if in collision, False if not
        '''
        self.gripper_mesh.apply_transform(grasp)
        pq = trimesh.proximity.ProximityQuery(self.gripper_mesh)
        signed_dists = pq.signed_distance(pcd)
        collides = sum(signed_dists > 0) > thresh
        print(sum(signed_dists > 0))
        return collides

    def transform_grasp(self, grasp, dist=0.12, theta=np.pi/2):
        offset = np.eye(4)
        if type(dist) != list:
            offset[2, 3] = -dist
        else:
            offset[:3,3] = dist
        if type(theta) != list:
            r = R.from_euler('z', theta, degrees=False)
        else:
            r = R.from_euler('xyz', theta, degrees=False)
        offset[:3,:3] = r.as_matrix()
        offset = np.matmul(offset, np.linalg.inv(grasp))
        offset = np.matmul(grasp, offset)
        grasp = np.matmul(offset, grasp)
        return grasp

    def test_robotiq_points(self):
        widths = np.linspace(0, 0.1, num=6)
        for w in widths:
            points = self.get_robotiq_points(w)
            VP(points[:,:3], 'robotiq_'+str(w))
        
        gm = trimesh.load(GRIPPER_PATH)
        qp = gm.sample(5000)
        VP(qp, 'panda')
        print('pause')
        from IPython import embed; embed()
    
    def get_robotiq_points(self, width):
        # make scene of base and fingers
        width += 0.005
        z_off = np.sqrt((0.13**2 - width**2)) + 0.05
        base_pose = self.transform_grasp(np.eye(4), dist=z_off-0.03)
        fingerbase = self.transform_grasp(base_pose, dist=[0.01, 0.0, 0.01], theta=0.0)
        knuckle_pose = self.transform_grasp(fingerbase, dist=0.0, theta=[np.pi/2, 0, 0])
        finger1_pose = self.transform_grasp(fingerbase, dist=[0, width, z_off], theta=[1.9,0,0])
        
        points = [np.matmul(base_pose, self.robotiq_query_pts[0].T).T]
        points.append(np.matmul(finger1_pose, self.robotiq_query_pts[1].T).T)
        flip = np.array([[-1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]])
        points.append(np.matmul(flip, points[1].T).T)
        points = np.concatenate(points, axis=0)        
        return points

    def con_check(self, pcd_np, query_pts_np, thresh=0.5):
        # get points that are inside each shape
        # print('in check')
        # from IPython import embed; embed()        
        occ = self.con(torch.from_numpy(query_pts_np).float().cuda().reshape(1, -1, 3), torch.from_numpy(pcd_np).float().cuda().reshape(1, -1, 3))
        occ = occ.probs.detach().cpu().numpy().reshape(-1)
        in_idx = np.where(occ > thresh)

        # print(np.max(occ))

        in_pts = query_pts_np[in_idx]
        return in_pts, np.max(occ)
    
    def learned_coll_check(self, grasp, pcd, name='grasp', floor=0):
        '''
        Uses a convolutional occupancy network to predict occupancy between scene and gripper query pts
        args:
            grasp -- 4x4 pose matrix of gripper
            pcd -- Nx3 pointcloud of scene
        '''        
        # get gripper points transformed to grasp pose
        grasp_pts = np.matmul(grasp, self.gripper_query_pts.T).T[:, :3]
        VP(grasp_pts, name)
        # VP(np.matmul(grasp, self.qp.T).T[:,:3], name+'panda')
        
        # if any gripper points are predicted to be inside the scene or below floor, grasp collides
        below_floor = False #sum(pcd[:,2] < floor) > 0
        in_pts, max_occ = self.con_check(pcd, grasp_pts)
        # print(in_pts.shape)
        VP(in_pts, 'in_pts')

        # from IPython import embed; embed()
        
        collides = (in_pts.shape[0] > 0) or below_floor
        return collides, in_pts
        
    
    def find_feasible_grasp(self, pcd, obj_mask, subgoal, downsample=None, conf_thresh=0.0, cc='con'):
        '''
        Runs forward passes on the model and checks grasps until one is found that works for the 
        start and end of the scene
        
        Note: This assumes that there exists a feasible grasp somewhere. First it'll run through all
              the predicted grasps above a confidence threshold, and if none work then it'll re-predict
              the grasps (to try and get a better random seed for point downsampling)

        collision checking options (cc) -- pm for point-mesh, con for conv occ net
        '''
        feasible_grasp = None
        feasible_success = 0
        attempts = 0

        check_pcd = pcd[np.logical_not(obj_mask)]
        
        if cc == 'pm':
            coll_checker = self.point_mesh_coll_check
        elif cc == 'con':
            coll_checker = self.learned_coll_check
        while feasible_grasp is None and attempts < 3:
            print('predicting')
            grasps, success, widths = self.infer(pcd, downsample, obj_mask, subgoal)
            grasp_set = grasps[success > conf_thresh]
            widths = widths[success > conf_thresh]
            success = success[success > conf_thresh]
            v = None
            for grasp_candidate, w, s in zip(grasp_set, widths, success):
                print('trying new grasp')
                if self.robotiq:
                    # self.test_robotiq_points()
                    self.gripper_query_pts = self.get_robotiq_points(w)
                end_grasp = np.matmul(subgoal, grasp_candidate)
                # v = V(v, [self.g_path], [end_grasp], [1], ['g'])
                colliding, in_pts = coll_checker(end_grasp, check_pcd)
                if not colliding:
                    if s > feasible_success:
                        feasible_grasp = grasp_candidate
                        feasible_success = s
                    break
            attempts += 1

        return feasible_grasp, feasible_success

    def find_feasible_grasp_set(self, pcd, obj_mask, subgoal, num, downsample=None, conf_thresh=0.3, cc='con'):
        '''
        Runs forward passes on the model and finds all grasps that work for the start and end of the scene
        
        num -- max number of grasps to return. will return ones with highest pred success that are collision free
        collision checking options (cc) -- pm for point-mesh, con for conv occ net
        '''
        gm = trimesh.load(GRIPPER_PATH)
        self.qp = gm.sample(5000)
        self.qp = np.concatenate((self.qp, np.ones((self.qp.shape[0], 1))), axis=1)
        feasible_grasp_set = []
        feasible_success = []
        if cc == 'pm':
            coll_checker = self.point_mesh_coll_check
        elif cc == 'con':
            coll_checker = self.learned_coll_check
        print('predicting')
        grasps, success, widths = self.infer(pcd, downsample, obj_mask, subgoal)
        grasp_set = grasps[success > conf_thresh]
        widths = widths[success > conf_thresh]
        success = success[success > conf_thresh]

        # grasp_set = grasp_set[np.argsort(success)][-num:]
        # widths = widths[np.argsort(success)][-num:]
        v = None
        mean = np.mean(pcd, axis=0)
        pcd -= mean
        self.test_con(pcd)
        
        for i, (grasp_candidate, s, w) in enumerate(zip(grasp_set, success, widths)):
            if i%10 == 0:
                print('trying new grasp')
                if self.robotiq:
                    self.test_robotiq_points()
                    self.gripper_query_pts = self.get_robotiq_points(w)
                end_grasp = np.matmul(subgoal, grasp_candidate)
                end_grasp[:3,3] -= mean
                # colliding = coll_checker(end_grasp, pcd, name='g'+str(i))
                colliding, in_pts = coll_checker(grasp_candidate, pcd, name=str(i)+'/g')
                if not colliding:
                    print('NOT IN COLLISION')
                    feasible_grasp_set.append(grasp_candidate)
                    feasible_success.append(s)
                else:
                    print('IN COLLISION')

        return feasible_grasp_set, feasible_success

    def filter_grasps(self, grasp_set, subgoal, pcd):
        '''
        finds all grasps that work for the start and end of the scene
        
        num -- max number of grasps to return. will return ones with highest pred success that are collision free
        collision checking options (cc) -- pm for point-mesh, con for conv occ net
        '''
        gm = trimesh.load(GRIPPER_PATH)
        self.qp = gm.sample(5000)
        self.qp = np.concatenate((self.qp, np.ones((self.qp.shape[0], 1))), axis=1)
        feasible_grasp_set = []
        feasible_success = []
        coll_checker = self.learned_coll_check

        mean = np.mean(pcd, axis=0)
        # pcd -= mean
        VP(pcd, 'pcd_centered')
        
        coll_free_mask = []
        
        for i, grasp_candidate in enumerate(grasp_set):
            # print('trying new grasp')
            print(i)
            if self.robotiq:
                self.test_robotiq_points()
                self.gripper_query_pts = self.get_robotiq_points(w)
            end_grasp = np.matmul(subgoal, grasp_candidate)
            # end_grasp[:3,3] -= mean
            colliding, in_pts = coll_checker(end_grasp, pcd, name='g'+str(i), floor=mean[2])
            # colliding = coll_checker(grasp_candidate, pcd, name=str(i)+'/g')
            if colliding:
                VP(np.matmul(end_grasp, self.qp.T).T[:,:3], 'panda/'+str(i))
                # print('NOT IN COLLISION')
                coll_free_mask.append(True)
            else:
                # VP(np.matmul(end_grasp, self.qp.T).T[:,:3], 'panda/'+str(i))
                VP(in_pts, 'panda/'+str(i)+'_in', color=(255,0,0))
                # print('IN COLLISION')
                coll_free_mask.append(False)

        return coll_free_mask

    def test_con(self, pcd):
        from evaluation.con_utils import util, three_util
        mc_vis = meshcat.Visualizer(zmq_url='tcp://127.0.0.1:6000')
        
        pt_dim = 3
        padding = 0.1
        voxel_reso_grid = 32
        voxel_reso_grid_pt = 128 #96, 128
        raster_pts = three_util.get_raster_points(voxel_reso_grid_pt, padding=padding)
        raster_pts = raster_pts.reshape(voxel_reso_grid_pt, voxel_reso_grid_pt, voxel_reso_grid_pt, 3)
        raster_pts = raster_pts.transpose(2, 1, 0, 3)
        raster_pts = raster_pts.reshape(-1, 3)

        mc_size = 0.004
        thresh = 0.5
        box_in_pts, max_occ = self.con_check(pcd, raster_pts, thresh=thresh)
        util.meshcat_pcd_show(mc_vis, box_in_pts, (255, 0, 0), name='scene/box_in_pts', size=mc_size)
        from IPython import embed; embed()
    
    def infer(self, pcd, downsample, obj_mask_big, subgoal):
        '''
        Run a single forward pass on the model.
        args:
            pcd -- Nx3 pointcloud of scene
            obj_mask -- Mx3 mask of target object
            subgoal -- 4x4 transform of object from start to end
        '''
        # if downsample is None:
        #     downsample = np.array(random.sample(range(pcd.shape[0]-1), 20000))
        # pcd = pcd[downsample, :]
        # obj_mask_big = obj_mask[downsample]

        pcd = torch.Tensor(pcd).to(dtype=torch.float32).to(self.model.device)
        batch = torch.zeros(pcd.shape[0]).to(dtype=torch.int64).to(self.model.device)
        idx = torch.linspace(0, pcd.shape[0]-1, 2048).to(dtype=torch.int64).to(self.model.device) #fps(pcd, batch, 2048/pcd.shape[0])
        obj_mask = obj_mask_big[idx.cpu().numpy()]
        if sum(obj_mask) == 0:
            print('cant see object')
            return None

        obj_mask = torch.Tensor(obj_mask)
        points, grasps, s, widths, _, _ = self.model(pcd[:,3:], pcd[:,:3], batch=batch, idx=idx, obj_mask=[obj_mask])

        print('model pass')
        
        s = self.sig(s)
        pred_grasps = torch.flatten(grasps, start_dim=0, end_dim=1)
        s = torch.flatten(s, start_dim=0, end_dim=1)
        widths = torch.flatten(widths, start_dim=0, end_dim=1)

        # width_mask = widths > 0.02

        pcd = pcd.detach().cpu().numpy()
        # VP(pcd, 'pcd', clear=True)
        # VP(pcd[obj_mask_big.astype(bool)], 'obj', clear=True)
        # from IPython import embed; embed()
        pred_grasps = pred_grasps[obj_mask.to(torch.bool)].detach().cpu().numpy()
        pred_successes = s[obj_mask.to(torch.bool)].detach().cpu().numpy()
        pred_widths = widths[obj_mask.to(torch.bool)].detach().cpu().numpy()

        # pred_grasps = pred_grasps[width_mask.to(torch.bool)].detach().cpu().numpy()
        # pred_successes = s[width_mask.to(torch.bool)].detach().cpu().numpy()
        # pred_widths = widths[width_mask.to(torch.bool)].detach().cpu().numpy()

        return pred_grasps, pred_successes, pred_widths
