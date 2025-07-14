import os
import sys
import random
import numpy as np
import torch
import argparse

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from arguments import ModelParams, PipelineParams, OptimizationParams


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

    batch_size = 1
    set_random_seed(args.seed)

    # deform model
    DeformModel = Deform_Model(args.device, bs_layer, args.use_same_shape).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()

    # dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    logdir = data_dir+'/'+args.logname
    scene = Scene_mica(data_dir, mica_datadir, train_type=2, white_background=lpt.white_background, device=args.device, ict=args.ict, mediapipe=args.mediapipe)
    im_size = (scene.bg_image.shape[2], scene.bg_image.shape[1])

    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    # bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    bg_color = [0, 0, 0]
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

    gaussians.save_ply(f'exported_ply/{file_name}.ply')
    print(f'save exported_ply/{file_name}.ply done')