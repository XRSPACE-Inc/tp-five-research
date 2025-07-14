import os
import sys
import copy
import random
import numpy as np
import torch
import argparse
from pathlib import Path

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from arguments import ModelParams, PipelineParams, OptimizationParams

# from utils.general_utils import quatProduct_batch
from train_custom_trans import setup_scene


def set_random_seed(seed):
    r"""Set random seeds for everything.

    Args:
        seed (int): Random seed.
        by_rank (bool):
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Model(torch.nn.Module):
    def __init__(self, deform_model: Deform_Model, gaussian: GaussianModel, full_model: bool, compress_model: bool) -> None:
        super().__init__()
        self.register_buffer('uv_vertices_shape_embeded', deform_model.uv_vertices_shape_embeded)
        self.register_buffer('default_shape', deform_model.default_shape)
        self.deformNet = deform_model.deformNet

        self.register_buffer('_rotation_base', gaussian._rotation_base)
        self.register_buffer('_scaling_base', gaussian._scaling_base)
        self.register_buffer('_features_dc', gaussian._features_dc)
        # self.register_buffer('_features_rest', gaussian._features_rest)
        self.register_buffer('_opacity', gaussian._opacity)
        self.full_model = full_model
        self.compress_model = compress_model


    def compress_data(self, _xyz, f_dc, opacities, scale, rotation) -> torch.Tensor:
        # sorted_indices = torch.argsort(
        #     -torch.exp(torch.sum(scale, 1, keepdim=True))
        #     / (1 + torch.exp(-opacities)),
        #     0
        # )
        # sorted_indices = sorted_indices[:, 0]
        # position = _xyz[sorted_indices]
        # scale = scale[sorted_indices]
        # SH_C0 = 0.28209479177387814
        # color_rgb = 0.5 + SH_C0 * f_dc[sorted_indices]
        # color_a = 1 / (1 + torch.exp(-opacities[sorted_indices]))
        # rot = rotation[sorted_indices]

        position = _xyz
        SH_C0 = 0.28209479177387814
        color_rgb = 0.5 + SH_C0 * f_dc
        color_rgb = color_rgb.clamp(min=0)
        color_a = 1 / (1 + torch.exp(-opacities))
        color = torch.cat((color_rgb, color_a), -1)

        return torch.cat((position, color, rotation, scale), -1)


        position = _xyz
        scale = torch.exp(scale)
        SH_C0 = 0.28209479177387814
        color_rgb = 0.5 + SH_C0 * f_dc
        color_a = 1 / (1 + torch.exp(-opacities))
        rot_x = rotation[:, 0:1]
        rot_y = rotation[:, 1:2]
        rot_z = rotation[:, 2:3]
        rot_w = rotation[:, 3:4]
        rot = torch.cat((rot_y, rot_z, rot_w, rot_x), 1)
        rot = (rot / torch.linalg.norm(rot))

        rot_x = rot[:, 0]
        rot_y = rot[:, 1]
        rot_z = rot[:, 2]
        rot_w = rot[:, 3]


        scale_mat = torch.eye(3, device=scale.device)[None, ...].repeat(scale.shape[0], 1, 1)
        scale_mat[:, 0, 0] *= scale[:, 0]
        scale_mat[:, 1, 1] *= scale[:, 1]
        scale_mat[:, 2, 2] *= scale[:, 2]

        rot_mat = torch.eye(3, device=scale.device)[None, ...].repeat(rot.shape[0], 1, 1)
        rot_mat[:, 0, 0] = 1 - 2 * (rot_y * rot_y + rot_z * rot_z)
        rot_mat[:, 0, 1] = 2 * (rot_x * rot_y + rot_w * rot_z)
        rot_mat[:, 0, 2] = 2 * (rot_x * rot_z + rot_w * rot_y)
        rot_mat[:, 1, 0] = 2 * (rot_x * rot_y + rot_w * rot_y)
        rot_mat[:, 1, 1] = 1 - 2 * (rot_x * rot_x + rot_z * rot_z)
        rot_mat[:, 1, 2] = 2 * (rot_y * rot_z + rot_w * rot_x)
        rot_mat[:, 2, 0] = 2 * (rot_x * rot_z + rot_w * rot_y)
        rot_mat[:, 2, 1] = 2 * (rot_y * rot_z + rot_w * rot_x)
        rot_mat[:, 2, 2] = 1 - 2 * (rot_x * rot_x + rot_y * rot_y)

        M = torch.bmm(rot_mat, scale_mat)
        M[2] *= -1
        sig = torch.bmm(M, torch.transpose(M, 1, 2))
        sigma0 = torch.cat((sig[:, 0, 0][:, None], sig[:, 0, 1][:, None], sig[:, 0, 2][:, None]), 1)
        sigma1 = torch.cat((sig[:, 1, 1][:, None], sig[:, 1, 2][:, None], sig[:, 2, 2][:, None]), 1)

        res = torch.cat((position, color_rgb, color_a, sigma0, sigma1), -1)
        return res

    def forward(self, condition: torch.Tensor):
        v_num = self.uv_vertices_shape_embeded.shape[1]
        condition = condition.unsqueeze(1).repeat(1, v_num, 1)
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
        rot_delta = deforms[..., 3:7]
        scale_coef = deforms[..., 7:]
        uv_vertices = self.default_shape.clone()

        verts_final = uv_vertices + uv_vertices_deforms

        _xyz = verts_final[0]
        rotation = rot_delta[0]
        # rotation = quatProduct_batch(self._rotation_base, rot_delta[0])
        scale = self._scaling_base * scale_coef[0]
        normals = torch.zeros_like(_xyz)
        f_dc = self._features_dc.transpose(1, 2).flatten(start_dim=1)
        # f_rest = self._features_rest.transpose(1, 2).flatten(start_dim=1)
        opacities = self._opacity

        # _xyz: torch.Size([13453, 3])
        # normals: torch.Size([13453, 3])
        # f_dc: torch.Size([13453, 3])
        # f_rest: torch.Size([13453, 45])
        # opacities: torch.Size([13453, 1])
        # scale: torch.Size([13453, 3])
        # rotation: torch.Size([13453, 4])
        if self.full_model:
            res = torch.cat((_xyz, normals, f_dc, opacities, scale, rotation), 1)
            return res
        if self.compress_model:
            return self.compress_data(_xyz, f_dc, opacities, scale, rotation)
        return torch.cat((_xyz, f_dc, opacities, scale, rotation), 1)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--logname', type=str, default='log', help='log name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument('-med', '--mediapipe', action='store_true', help='mediapipe')
    parser.add_argument('-c', '--ict', action='store_true', help='ict')
    parser.add_argument('-ss', '--use_same_shape', action='store_true', help='use same shape')
    parser.add_argument('--full', action='store_true', help='export full model')
    parser.add_argument('--compressed', action='store_true', help='export compressed model')
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    file_name = args.idname
    bs_layer = 120
    if args.mediapipe:
        file_name += f'_mediapipe_51'
        bs_layer = 51
    elif args.ict:
        file_name += '_ict_53'
        bs_layer = 53
    else:
        file_name += '_base_120'
    if args.use_same_shape:
        file_name += '_ss'
    if args.full:
        file_name += '_full'
    if args.compressed:
        file_name += '_comp'

    batch_size = 1
    set_random_seed(args.seed)

    DeformModel, scene, gaussians, logdir, train_dir, model_dir = setup_scene(
        args,
        lpt,
        opt,
        train_type=2,
    )

    DeformModel.eval()

    first_iter = 0

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint)
        print(f'checkpoint loaded: {args.checkpoint}')
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    viewpoint_cam = viewpoint[0]
    viewpoint_cam.to_device()
    frame_id = viewpoint_cam.uid

    # deform gaussians
    codedict['bs_param'] = viewpoint_cam.bs_param
    codedict['expr'] = viewpoint_cam.exp_param
    codedict['eyes_pose'] = viewpoint_cam.eyes_pose
    codedict['eyelids'] = viewpoint_cam.eyelids
    codedict['jaw_pose'] = viewpoint_cam.jaw_pose
    verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
    gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

    x = viewpoint_cam.bs_param.clone()
    x = x.to('cpu')
    model = Model(deform_model=DeformModel, gaussian=gaussians, full_model=args.full, compress_model=args.compressed)
    model = copy.deepcopy(model)
    model.to('cpu')

    output_path = Path('export/')
    output_path.mkdir(exist_ok=True)

    output = model(x)

    gaussians.save_ply(str(output_path /  f'{file_name}.ply'))

    torch.save({'state': model.state_dict()}, output_path / 'source_dict.pth')
    np.savetxt(
        str(output_path / 'input.txt'),
        x.detach().cpu().numpy(),
    )
