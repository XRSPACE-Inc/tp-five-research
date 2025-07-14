import torch
# import torch.nn as nn
from .reconstruct_model import Model
from .deform_model import Deform_Model
from utils.general_utils import Pytorch3dRasterizer, Embedder, face_vertices_gen
# from src.camera_model import _compute_rotation


class ICTModel(Deform_Model):
    def __init__(self, device, _, use_same_shape: bool=False, lite_model: bool = False):
        super(ICTModel, self).__init__(device)
        self.device = device
        self.use_same_shape = use_same_shape

        num_shape = 100
        num_exp = 53-6
        self.flame_model = Model(
            file='assets/ictfacekit_1.npz',
            texture_file=None,
        ).to(self.device)
        self.default_shape_code = torch.zeros(1, num_shape, device=self.device)
        self.default_expr_code = torch.zeros(1, num_exp, device=self.device)

        # positional encoding
        self.pts_freq = 8
        self.pts_embedder = Embedder(self.pts_freq)

        self.uvcoords = self.flame_model.tdmm.uvcoords
        self.uvfaces = self.flame_model.tdmm.uv_faces[None, ...]
        self.tri_faces = self.flame_model.tdmm.faces[None, ...]

        # rasterizer
        self.uv_size = 128
        self.uv_rasterizer = Pytorch3dRasterizer(self.uv_size)

        # self.neckhead_id_list = []
        # self.neckhead_id_tensor = torch.tensor(self.neckhead_id_list, dtype=torch.int64).to(self.device)
        self.init_networks(53, lite_model)
        # self.transformNet = nn.Sequential(*[
        #     nn.Linear(6, 128),
        #     nn.Linear(128, 128),
        #     nn.Linear(128, 6),
        # ])

    def example_init(self, codedict):
        # speed up
        codedict['B'] = batch_size = codedict['expr'].shape[0]
        vertices = self.flame_model.forward_geo(codedict)

        face_vertices_shape = face_vertices_gen(vertices, self.tri_faces.expand(batch_size, -1, -1))
        rast_out, pix_to_face, bary_coords = self.uv_rasterizer(
            self.uvcoords.expand(batch_size, -1, -1),
            self.uvfaces.expand(batch_size, -1, -1),
            face_vertices_shape
        )
        self.pix_to_face_ori = pix_to_face
        self.bary_coords = bary_coords

        uvmask = rast_out[:, -1].unsqueeze(1)
        uvmask_flaten = uvmask[0].view(uvmask.shape[1], -1).permute(1, 0).squeeze(1)  # batch=1
        self.uvmask_flaten_idx = (uvmask_flaten[:] > 0)

        # pix_to_face_flaten = pix_to_face[0].clone().view(-1)  # batch=1
        # self.pix_to_face = pix_to_face_flaten[self.uvmask_flaten_idx]  # pix to face idx
        # self.pix_to_v_idx = self.tri_faces[0, self.pix_to_face, :]  # pix to vert idx

        uv_vertices_shape = rast_out[:, :3]
        uv_vertices_shape_flaten = uv_vertices_shape[0].view(uv_vertices_shape.shape[1], -1).permute(1, 0)  # batch=1
        uv_vertices_shape = uv_vertices_shape_flaten[self.uvmask_flaten_idx].unsqueeze(0)

        self.uv_vertices_shape = uv_vertices_shape  # for cano init
        self.uv_vertices_shape_embeded = self.pts_embedder(uv_vertices_shape)
        self.v_num = self.uv_vertices_shape_embeded.shape[1]

        # mask
        # self.uv_head_idx = (
        #     a_in_b_torch(self.pix_to_v_idx[:,0], self.neckhead_id_tensor)
        #     & a_in_b_torch(self.pix_to_v_idx[:,1], self.neckhead_id_tensor)
        #     & a_in_b_torch(self.pix_to_v_idx[:,2], self.neckhead_id_tensor)
        # )

    def decode(self, codedict):
        # shape_code = codedict['jaw_pose'].detach()
        expr_code = codedict['expr'].detach()
        # angle = codedict['eyes_pose'].detach()
        # trans = codedict['eyelids'].detach()
        # jaw_pose = codedict['jaw_pose'].detach()
        # eyelids = codedict['eyelids'].detach()
        # eyes_pose = codedict['eyes_pose'].detach()
        batch_size = codedict['B']
        # condition = torch.cat((expr_code, jaw_pose, eyes_pose, eyelids), dim=1)
        # condition = torch.cat((expr_code, angle, trans), dim=1)
        condition = expr_code.clone()

        # MLP
        condition = condition.unsqueeze(1).repeat(1, self.v_num, 1)
        uv_vertices_shape_embeded_condition = torch.cat((self.uv_vertices_shape_embeded, condition), dim=2)
        deforms = self.deformNet(uv_vertices_shape_embeded_condition)
        deforms = torch.tanh(deforms)
        uv_vertices_deforms = deforms[..., :3]
        # rot_delta_0 = deforms[..., 3:7]
        # rot_delta_r = torch.exp(rot_delta_0[..., 0]).unsqueeze(-1)
        # rot_delta_v = rot_delta_0[..., 1:]
        # rot_delta = torch.cat((rot_delta_r, rot_delta_v), dim=-1)
        # scale_coef = deforms[..., 7:]
        # scale_coef = torch.exp(scale_coef)
        r = torch.pow(2, deforms[..., 3:4] * 1.44269504)
        rot_delta = torch.cat((r, deforms[..., 4:7]), -1)
        scale_coef = torch.pow(2, deforms[..., 7:] * 1.44269504)

        if self.use_same_shape and self.default_shape is not None:
            uv_vertices = self.default_shape.clone()
        else:
            vertices = self.flame_model.forward_geo(codedict)
            # angle = codedict['eyes_pose'].detach()
            # trans = codedict['eyelids'].detach()
            # rt = self.transformNet(torch.cat((angle, trans), -1))
            # rt = torch.tanh(rt)
            # rmat = _compute_rotation(rt[:, :3])
            # vertices = vertices @ rmat
            # vertices = vertices + rt[:, 3:].unsqueeze(1)
            # vertices = torch.bmm(w2c[:, :3, :3], vertices.transpose(1, 2)) + w2c[:, :3, -1].unsqueeze(-1)
            # vertices = vertices.transpose(1, 2)
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

        # # conduct mask
        # verts_final = verts_final[:, self.uv_head_idx, :]
        # rot_delta = rot_delta[:, self.uv_head_idx, :]
        # scale_coef = scale_coef[:, self.uv_head_idx, :]

        return verts_final, rot_delta, scale_coef

    def capture(self):
        return (
            self.deformNet.state_dict(),
            # self.transformNet.state_dict(),
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
            # {'params': self.transformNet.parameters(), 'lr': 1e-4},
        ]
        self.optimizer = torch.optim.Adam(params_group, betas=(0.9, 0.999))
