import numpy as np
import torch
from .deform_model import Deform_Model
from utils.general_utils import Pytorch3dRasterizer, Embedder, face_vertices_gen


class SphereModel(Deform_Model):
    def __init__(self, device, _, use_same_shape: bool=False, lite_model: bool = False):
        super(SphereModel, self).__init__(device)
        self.device = device
        self.use_same_shape = use_same_shape

        self.mesh = np.load('assets/sphere.npz', allow_pickle=True)['mesh'][()]

        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)

        self.vertices = torch.Tensor(self.mesh['verts'])[None, ...].to(self.device) * 0.1
        self.vertices[..., 0] -= 0.05
        self.vertices[..., 1] -= 0.03

        raw_uvcoords = torch.Tensor(self.mesh['uvs'])[None, ...].clone()
        raw_uvcoords = raw_uvcoords * 2.0 - 1.0
        raw_uvcoords[..., 1] = -raw_uvcoords[..., 1]
        uvcoords = torch.cat([raw_uvcoords, torch.ones_like(raw_uvcoords[:, :, 0:1])], -1)  # [bz, ntv, 3]
        self.uvcoords = uvcoords.to(self.device)
        self.uvfaces = torch.LongTensor(self.mesh['uv_faces'])[None, ...].to(self.device)
        self.tri_faces = torch.LongTensor(self.mesh['faces'])[None, ...].to(self.device)

        self.uv_size = 128
        self.uv_rasterizer = Pytorch3dRasterizer(self.uv_size)

        self.init_networks(50, lite_model, 6)

    def example_init(self, codedict):
        codedict['B'] = batch_size = codedict['expr'].shape[0]
        vertices = self.vertices.detach().clone()

        face_vertices_shape = face_vertices_gen(vertices, self.tri_faces.expand(batch_size, -1, -1))
        rast_out, pix_to_face, bary_coords = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1),
            self.uvfaces.expand(batch_size, -1, -1),
            face_vertices_shape
        )
        self.pix_to_face_ori = pix_to_face
        self.bary_coords = bary_coords

        uvmask = rast_out[:, -1].unsqueeze(1)
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1)
        self.uvmask_flaten_idx = (uvmask_flaten[:] > 0)

        uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0)
        uv_vertices_shape = uv_vertices_shape_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        self.uv_vertices_shape = uv_vertices_shape
        self.uv_vertices_shape_embeded = self.pts_embedder(uv_vertices_shape)
        self.v_num = self.uv_vertices_shape_embeded.shape[1]

    def decode(self, codedict):
        expr_code = codedict['expr'].detach()
        batch_size = codedict['B']
        condition = expr_code.clone()

        # MLP
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        deforms = torch.tanh(deforms)
        uv_vertices_deforms = deforms[..., :3]
        r = torch.pow(2, deforms[..., 3:4] * 1.44269504)
        rot_delta = torch.cat((r, deforms[..., 4:7]), -1)
        scale_coef = torch.pow(2, deforms[..., 7:] * 1.44269504)

        if self.use_same_shape and self.default_shape is not None:
            uv_vertices = self.default_shape.clone()
        else:
            vertices = self.vertices.detach().clone()
            face_vertices = face_vertices_gen(vertices, self.tri_faces.expand(batch_size, -1, -1))

            # rasterize face_vertices to uv space
            D = face_vertices.shape[-1]  # 3
            attributes = face_vertices.clone()
            attributes = attributes.view(attributes.shape[0] * attributes.shape[1], 3, attributes.shape[-1])
            N, H, W, K, _ = self.bary_coords.shape
            idx = self.pix_to_face_ori.clone().view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)
            pixel_face_vals = attributes.gather(0, idx).view(N, H, W, K, 3, D)
            pixel_vals = (self.bary_coords[..., None] * pixel_face_vals).sum(dim=-2)
            uv_vertices = pixel_vals[:, :, :, 0].permute(0, 3, 1, 2)
            uv_vertices_flaten = uv_vertices[0].view(uv_vertices.shape[1], -1).permute(1, 0)  # batch=1
            uv_vertices = uv_vertices_flaten[self.uvmask_flaten_idx].unsqueeze(0)
            if self.default_shape is None:
                self.default_shape = uv_vertices.clone()

        verts_final = uv_vertices + uv_vertices_deforms

        return verts_final, rot_delta, scale_coef

    def capture(self):
        return (
            self.deformNet.state_dict(),
            self.optimizer.state_dict(),
            { 'default_shape', self.default_shape.clone() }
        )

    def restore(self, model_args):
        (net_dict,
         opt_dict,
         default_shape) = model_args
        
        self.deformNet.load_state_dict(net_dict)
        self.training_setup()
        self.optimizer.load_state_dict(opt_dict)
        self.default_shape = torch.Tensor(list(default_shape)[0])

    def training_setup(self):
        params_group = [
            {'params': self.deformNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
