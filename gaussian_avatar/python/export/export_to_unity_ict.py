import sys
import copy
import random
import numpy as np
import torch
import argparse
from pathlib import Path

from scene import GaussianModel
from src.deform_model import Deform_Model
from arguments import ModelParams, PipelineParams, OptimizationParams

from train_ict import setup_scene


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
    def __init__(self, deform_model: Deform_Model, gaussian: GaussianModel, full_model: bool) -> None:
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
        return torch.cat((_xyz, f_dc, opacities, scale, rotation), 1)


def main():
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
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    # ppt = pp.extract(args)

    file_name = f'{args.idname}_ict_53'
    if args.use_same_shape:
        file_name += '_ss'
    if args.full:
        file_name += '_full'

    set_random_seed(args.seed)

    DeformModel, scene, gaussians, logdir, train_dir, model_dir = setup_scene(
        args,
        lpt,
        opt,
        train_type=2,
    )

    DeformModel.eval()

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint)
        print(f'checkpoint loaded: {args.checkpoint}')
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    codedict = {}
    codedict['expr'] = scene.bs_param.to(args.device)
    codedict['eyes_pose'] = scene.angle_param.to(args.device)
    codedict['eyelids'] = scene.trans_param.to(args.device)
    codedict['jaw_pose'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    viewpoint = scene.getCameras().copy()

    viewpoint_cam = viewpoint[0]
    viewpoint_cam.to_device()

    # deform gaussians
    codedict['expr'] = viewpoint_cam.bs_param
    codedict['eyes_pose'] = viewpoint_cam.eyes_pose
    codedict['eyelids'] = viewpoint_cam.eyelids
    codedict['jaw_pose'] = viewpoint_cam.jaw_pose
    codedict['B'] = viewpoint_cam.exp_param.shape[0]
    verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
    gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

    x = viewpoint_cam.bs_param.clone()
    x = x.to('cpu')
    model = Model(deform_model=DeformModel, gaussian=gaussians, full_model=args.full)
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


if __name__ == "__main__":
    import traceback
    try:
        main()
    except:
        traceback.print_exc()
