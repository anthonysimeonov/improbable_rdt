import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
from torch_scatter import scatter_mean, scatter_max

from rrp_robot.collision.conet_model.con_model.layers import ResnetBlockFC
from rrp_robot.collision.conet_model.con_model.unet import UNet
from rrp_robot.collision.conet_model.con_model.unet3d import UNet3D
from rrp_robot.collision.conet_model.con_model.common import coordinate2index, normalize_coordinate, normalize_3d_coordinate, map2local


class LocalPoolPointnet(nn.Module):
    ''' PointNet-based encoder network with ResNet blocks for each point.
        Number of input points are fixed.
    
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
        scatter_type (str): feature aggregation when doing local pooling
        unet (bool): weather to use U-Net
        unet_kwargs (str): U-Net parameters
        unet3d (bool): weather to use 3D U-Net
        unet3d_kwargs (str): 3D U-Net parameters
        plane_resolution (int): defined resolution for plane feature
        grid_resolution (int): defined resolution for grid feature 
        plane_type (str): feature type, 'xz' - 1-plane, ['xz', 'xy', 'yz'] - 3-plane, ['grid'] - 3D grid volume
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
        n_blocks (int): number of blocks ResNetBlockFC layers
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128, scatter_type='max', 
                 unet=False, unet_kwargs=None, unet3d=False, unet3d_kwargs=None, 
                 plane_resolution=None, grid_resolution=None, plane_type='xz', padding=0.1, n_blocks=5, 
                 local_coord=False, pos_encoding='linear', mc_vis=None):
        super().__init__()
        self.mc_vis = mc_vis
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2*hidden_dim)
        self.blocks = nn.ModuleList([
            ResnetBlockFC(2*hidden_dim, hidden_dim) for i in range(n_blocks)
        ])
        self.fc_c = nn.Linear(hidden_dim, c_dim)

        self.actvn = nn.ReLU()
        self.hidden_dim = hidden_dim

        if unet:
            self.unet = UNet(c_dim, in_channels=c_dim, **unet_kwargs)
        else:
            self.unet = None

        if unet3d:
            self.unet3d = UNet3D(**unet3d_kwargs)
        else:
            self.unet3d = None

        self.reso_plane = plane_resolution
        self.reso_grid = grid_resolution
        self.plane_type = plane_type
        self.padding = padding

        if scatter_type == 'max':
            self.scatter = scatter_max
        elif scatter_type == 'mean':
            self.scatter = scatter_mean
        else:
            raise ValueError('incorrect scatter type')

        if local_coord:
            unit_size = 1.1 / self.reso_grid 
            self.map2local = map2local(unit_size, pos_encoding=pos_encoding)
        else:
            self.map2local = None
        
        if pos_encoding == 'sin_cos':
            self.fc_pos = nn.Linear(60, 2*hidden_dim)
        else:
            self.fc_pos = nn.Linear(dim, 2*hidden_dim)


    def generate_plane_features(self, p, c, plane='xz'):
        # acquire indices of features in plane
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        index = coordinate2index(xy, self.reso_plane)

        # scatter plane features from points
        fea_plane = c.new_zeros(p.size(0), self.c_dim, self.reso_plane**2)
        c = c.permute(0, 2, 1) # B x 512 x T
        fea_plane = scatter_mean(c, index, out=fea_plane) # B x 512 x reso^2
        fea_plane = fea_plane.reshape(p.size(0), self.c_dim, self.reso_plane, self.reso_plane) # sparce matrix (B x 512 x reso x reso)

        # process the plane features with UNet
        if self.unet is not None:
            fea_plane = self.unet(fea_plane)

        return fea_plane

    def generate_grid_features(self, index, c):
        # scatter grid features from points        
        c = c.permute(0, 2, 1)
        if index.max() < self.reso_grid**3:
            fea_grid = c.new_zeros(c.size(0), self.c_dim, self.reso_grid**3)
            fea_grid = scatter_mean(c, index, out=fea_grid) # B x c_dim x reso^3
        else:
            fea_grid = scatter_mean(c, index) # B x c_dim x reso^3
            if fea_grid.shape[-1] > self.reso_grid**3: # deal with outliers
                fea_grid = fea_grid[:, :, :-1]
        fea_grid = fea_grid.reshape(c.size(0), self.c_dim, self.reso_grid, self.reso_grid, self.reso_grid)

        if self.unet3d is not None:
            fea_grid = self.unet3d(fea_grid)

        return fea_grid

    # def pool_local(self, xy, index, c):
    def pool_local(self, index, c):
        bs, fea_dim = c.size(0), c.size(2)
        # keys = xy.keys()
        keys = index.keys()

        c_out = 0
        for key in keys:
            # scatter plane features from points
            if key == 'grid':
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_grid**3)
            else:
                fea = self.scatter(c.permute(0, 2, 1), index[key], dim_size=self.reso_plane**2)
            if self.scatter == scatter_max:
                fea = fea[0]
            # gather feature back to points
            fea = fea.gather(dim=2, index=index[key].expand(-1, fea_dim, -1))
            c_out += fea
        return c_out.permute(0, 2, 1)


    def forward(self, p):
        batch_size, T, D = p.size()

        p_coords = p[:, :, :3]

        # acquire the index for each point
        coord = {}
        index = {}

        if 'xz' in self.plane_type:
            coord['xz'] = normalize_coordinate(p_coords.clone(), plane='xz', padding=self.padding)
            index['xz'] = coordinate2index(coord['xz'], self.reso_plane)
        if 'xy' in self.plane_type:
            coord['xy'] = normalize_coordinate(p_coords.clone(), plane='xy', padding=self.padding)
            index['xy'] = coordinate2index(coord['xy'], self.reso_plane)
        if 'yz' in self.plane_type:
            coord['yz'] = normalize_coordinate(p_coords.clone(), plane='yz', padding=self.padding)
            index['yz'] = coordinate2index(coord['yz'], self.reso_plane)
        if 'grid' in self.plane_type:
            coord['grid'] = normalize_3d_coordinate(p_coords.clone(), padding=self.padding)
            index['grid'] = coordinate2index(coord['grid'], self.reso_grid, coord_type='3d')

        if self.map2local:
            pp = self.map2local(p_coords)  # only map first three coords to local (others might be non-coords)
            if p.shape[1] > 3:
                pp = torch.cat((pp, p[:, :, 3:]), dim=-1)

            net = self.fc_pos(pp)
        else:
            net = self.fc_pos(p)

        net = self.blocks[0](net)
        for block in self.blocks[1:]:
            pooled = self.pool_local(index, net)
            net = torch.cat([net, pooled], dim=2)
            net = block(net)

        c = self.fc_c(net)

        fea = {}
        if 'grid' in self.plane_type:
            fea['grid'] = self.generate_grid_features(index['grid'], c)
        if 'xz' in self.plane_type:
            fea['xz'] = self.generate_plane_features(p_coords, c, plane='xz')
        if 'xy' in self.plane_type:
            fea['xy'] = self.generate_plane_features(p_coords, c, plane='xy')
        if 'yz' in self.plane_type:
            fea['yz'] = self.generate_plane_features(p_coords, c, plane='yz')
            
        return fea


class LocalDecoder(nn.Module):
    ''' Decoder.
        Instead of conditioning on global features, on plane/volume local features.

    Args:
        dim (int): input dimension
        c_dim (int): dimension of latent conditioned code c
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of blocks ResNetBlockFC layers
        leaky (bool): whether to use leaky ReLUs
        sample_mode (str): sampling feature strategy, bilinear|nearest
        padding (float): conventional padding paramter of ONet for unit cube, so [-0.5, 0.5] -> [-0.55, 0.55]
    '''

    def __init__(self, dim=3, c_dim=128,
                 hidden_size=256, n_blocks=5, leaky=False, sample_mode='bilinear', padding=0.1):
        super().__init__()
        self.c_dim = c_dim
        self.n_blocks = n_blocks

        if c_dim != 0:
            self.fc_c = nn.ModuleList([
                nn.Linear(c_dim, hidden_size) for i in range(n_blocks)
            ])


        self.fc_p = nn.Linear(dim, hidden_size)

        self.blocks = nn.ModuleList([
            ResnetBlockFC(hidden_size) for i in range(n_blocks)
        ])

        self.fc_out = nn.Linear(hidden_size, 1)

        if not leaky:
            self.actvn = F.relu
        else:
            self.actvn = lambda x: F.leaky_relu(x, 0.2)

        self.sample_mode = sample_mode
        self.padding = padding


    def sample_plane_feature(self, p, c, plane='xz'):
        xy = normalize_coordinate(p.clone(), plane=plane, padding=self.padding) # normalize to the range of (0, 1)
        xy = xy[:, :, None].float()
        vgrid = 2.0 * xy - 1.0 # normalize to (-1, 1)
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1)
        return c

    def sample_grid_feature(self, p, c):
        p_nor = normalize_3d_coordinate(p.clone(), padding=self.padding) # normalize to the range of (0, 1)
        p_nor = p_nor[:, :, None, None].float()
        vgrid = 2.0 * p_nor - 1.0 # normalize to (-1, 1)
        # acutally trilinear interpolation if mode = 'bilinear'
        c = F.grid_sample(c, vgrid, padding_mode='border', align_corners=True, mode=self.sample_mode).squeeze(-1).squeeze(-1)
        return c


    def forward(self, p, c_plane, **kwargs):
        if self.c_dim != 0:
            plane_type = list(c_plane.keys())
            c = 0
            if 'grid' in plane_type:
                c += self.sample_grid_feature(p, c_plane['grid'])
            if 'xz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xz'], plane='xz')
            if 'xy' in plane_type:
                c += self.sample_plane_feature(p, c_plane['xy'], plane='xy')
            if 'yz' in plane_type:
                c += self.sample_plane_feature(p, c_plane['yz'], plane='yz')
            c = c.transpose(1, 2)

        p = p.float()
        net = self.fc_p(p)

        acts = []
        for i in range(self.n_blocks):
            if self.c_dim != 0:
                net = net + self.fc_c[i](c)

            net = self.blocks[i](net)
            # if i == (self.n_blocks - 1):
            #     acts.append(net)
            acts.append(net)

        out = self.fc_out(self.actvn(net))
        out = out.squeeze(-1)

        if len(kwargs.keys()):
            if kwargs['return_features']:
                return out, acts
        return out, acts


class ConvolutionalOccupancyNetwork(nn.Module):
    ''' Occupancy Network class.

    Args:
        decoder (nn.Module): decoder network
        encoder (nn.Module): encoder network
        device (device): torch device
    '''

    def __init__(self, decoder, encoder=None, device=None, proj_in_dim=32, proj_out_dim=384):
        super().__init__()

        self.decoder = decoder.to(device)

        if encoder is not None:
            self.encoder = encoder.to(device)
        else:
            self.encoder = None

        #self.proj = nn.Linear(proj_in_dim, proj_out_dim)

        self._device = device

    def forward(self, p, inputs, sample=True, **kwargs):
        ''' Performs a forward pass through the network.

        Args:
            p (tensor): sampled points
            inputs (tensor): conditioning input
            sample (bool): whether to sample for z
        '''
        #############
        if isinstance(p, dict):
            batch_size = p['p'].size(0)
        else:
            batch_size = p.size(0)
        c = self.encode_inputs(inputs)
        p_r = self.decode(p, c, **kwargs)
        return p_r

    def encode_inputs(self, inputs):
        ''' Encodes the input.

        Args:
            input (tensor): the input
        '''

        # print('encode_inputs')
        # from IPython import embed; embed()
        if self.encoder is not None:
            c = self.encoder(inputs)
        else:
            # Return inputs?
            c = torch.empty(inputs.size(0), 0)

        return c

    def decode(self, p, c, **kwargs):
        ''' Returns occupancy probabilities for the sampled points.

        Args:
            p (tensor): points
            c (tensor): latent conditioned code c
        '''

        logits = self.decoder(p, c, **kwargs)
        # from IPython import embed; embed()
        # p_r = dist.Bernoulli(logits=logits)
        p_r = dist.Bernoulli(logits=logits[0])
        return p_r

    def to(self, device):
        ''' Puts the model to the device.

        Args:
            device (device): pytorch device
        '''
        model = super().to(device)
        model._device = device
        return model

    def extract_latent(self, input):
        inputs = input['point_cloud']
        c_dict = self.encode_inputs(inputs)
        latent = {}
        for k, v in c_dict.items():
            latent[k] = v.detach()
        self.cached_latent = copy.deepcopy(latent)
        return latent

    def forward_latent(self, z, coords, return_occ=False, *args, **kwargs):
        if 'use_cached_latent' in kwargs.keys():
            z = self.cached_latent
        out_occ, acts = self.decoder(coords, z, return_features=True)
        acts = torch.cat(acts, dim=-1)
        out_features = F.normalize(acts, p=2, dim=-1)
        if return_occ:
            p_r = dist.Bernoulli(logits=out_occ)
            return out_features, p_r
        else:
            return out_features

    def set_cached_latent(self, latent):
        self.cached_latent = copy.deepcopy(latent)

    def fetch_proj_grid_feat(self, p, c):
        interp_grid = self.decoder.sample_grid_feature(p, c).permute(0, 2, 1)
        # out_feat = self.proj(interp_grid)
        # return out_feat
        return interp_grid
