import os
import sys
import random
import numpy as np
import torch
import argparse
import cv2

from scene import GaussianModel, Scene_mica_1
from src.deform_model import Deform_Model
from gaussian_renderer import render
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
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    batch_size = 1
    set_random_seed(args.seed)

    # deform model
    DeformModel = Deform_Model(args.device).to(args.device)
    DeformModel.training_setup()
    DeformModel.eval()

    # dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    logdir = data_dir+'/'+args.logname
    output_dir = os.path.join(data_dir, 'transfer_data')
    os.makedirs(output_dir, exist_ok=True)
    scene = Scene_mica_1(
        data_dir,
        mica_datadir,
        train_type=0,
        white_background=lpt.white_background,
        device=args.device,
        do_not_skip=True
    )
    im_size = (scene.bg_image.shape[2], scene.bg_image.shape[1])

    first_iter = 0
    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)

    if args.checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    # viewpoint = scene.getCameras().copy()
    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    for iteration in range(len(scene)):
        viewpoint_cam = scene[iteration]
        viewpoint_cam.to_device()
        frame_id = viewpoint_cam.uid

        # deform gaussians
        if viewpoint_cam.bs_param is not None:
            codedict['bs_param'] = viewpoint_cam.bs_param
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)
        torch.save(
            {'verts_final': verts_final, 'rot_delta': rot_delta, 'scale_coef': scale_coef},
            os.path.join(output_dir, f'{frame_id}.data')
        )
