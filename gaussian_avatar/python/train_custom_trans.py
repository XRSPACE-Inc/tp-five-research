import os
import sys
import random
import numpy as np
import torch
import argparse
from PIL import Image
import lpips

from scene import GaussianModel, Scene_mica
from src.deform_model import Deform_Model
from gaussian_renderer import render
from arguments import ModelParams, PipelineParams, OptimizationParams
from utils.loss_utils import huber_loss
from utils.general_utils import normalize_for_percep


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


def setup_scene(args, lpt, opt, train_type):
    # deform model
    DeformModel = Deform_Model(args.device, 51, use_same_shape=True, lite_model=False).to(args.device)
    DeformModel.training_setup()

    # dataloader
    data_dir = os.path.join('dataset', args.idname)
    mica_datadir = os.path.join('metrical-tracker/output', args.idname)
    log_dir = os.path.join(data_dir, 'log')
    train_dir = os.path.join(log_dir, 'train')
    model_dir = os.path.join(log_dir, 'ckpt')
    # transfer_data_dir = os.path.join(data_dir, 'transfer_data')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    scene = Scene_mica(
        data_dir,
        mica_datadir,
        train_type=train_type,
        white_background=lpt.white_background,
        device=args.device,
        mediapipe=True,
        # transfer_data_path=transfer_data_dir,
    )

    gaussians = GaussianModel(lpt.sh_degree)
    gaussians.training_setup(opt)
    return DeformModel, scene, gaussians, log_dir, train_dir, model_dir


def main():
    # print('!!!!!!!!!!!!!!!!!!!!!! transfer vertex is not working for some reason?')
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--idname', type=str, default='id1_25', help='id name')
    parser.add_argument('--image_res', type=int, default=512, help='image resolution')
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.device = "cuda"
    lpt = lp.extract(args)
    opt = op.extract(args)
    ppt = pp.extract(args)

    # batch_size = 1
    set_random_seed(args.seed)

    percep_module = lpips.LPIPS(net='vgg').to(args.device)

    DeformModel, scene, gaussians, log_dir, train_dir, model_dir = setup_scene(
        args,
        lpt,
        opt,
        train_type=2 if args.debug else 0,
    )

    first_iter = 0
    if args.start_checkpoint:
        (model_params, gauss_params, first_iter) = torch.load(args.start_checkpoint)
        DeformModel.restore(model_params)
        gaussians.restore(gauss_params, opt)

    # bg_color = [1, 1, 1] if lpt.white_background else [0, 1, 0]
    bg_color = [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=args.device)

    codedict = {}
    codedict['shape'] = scene.shape_param.to(args.device)
    DeformModel.example_init(codedict)

    viewpoint_stack = None
    first_iter += 1
    mid_num = 15000
    total_iteraions = 1001 if args.debug else opt.iterations + 1
    visualize_iteraions = 50 if args.debug else 500
    print('======== start')
    for iteration in range(first_iter, total_iteraions):
        # Every 500 its we increase the levels of SH up to a maximum degree
        if iteration % 500 == 0:
            gaussians.oneupSHdegree()

        # random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getCameras().copy()
            random.shuffle(viewpoint_stack)
            if len(viewpoint_stack) > 2000:
                viewpoint_stack = viewpoint_stack[:2000]
        viewpoint_cam = viewpoint_stack.pop(random.randint(0, len(viewpoint_stack)-1))
        # frame_id = viewpoint_cam.uid

        viewpoint_cam.to_device()

        # deform gaussians
        codedict['bs_param'] = viewpoint_cam.bs_param
        codedict['expr'] = viewpoint_cam.exp_param
        codedict['eyes_pose'] = viewpoint_cam.eyes_pose
        codedict['eyelids'] = viewpoint_cam.eyelids
        codedict['jaw_pose'] = viewpoint_cam.jaw_pose
        verts_final, rot_delta, scale_coef = DeformModel.decode(codedict)

        if iteration == 1:
            gaussians.create_from_verts(verts_final[0])
            gaussians.training_setup(opt)
        gaussians.update_xyz_rot_scale(verts_final[0], rot_delta[0], scale_coef[0])

        # Render
        render_pkg = render(viewpoint_cam, gaussians, ppt, background)
        image = render_pkg["render"]

        # Loss
        gt_image = viewpoint_cam.original_image
        # mouth_mask = viewpoint_cam.mouth_mask

        loss_huber = huber_loss(image, gt_image, 0.1)  # + 40*huber_loss(image*mouth_mask, gt_image*mouth_mask, 0.1)

        loss_G = 0.
        head_mask = viewpoint_cam.head_mask
        image_percep = normalize_for_percep(image*head_mask)
        gt_image_percep = normalize_for_percep(gt_image*head_mask)
        if iteration > mid_num:
            loss_G = torch.mean(percep_module.forward(image_percep, gt_image_percep)) * 0.05

        # gaussian_vert_loss = ((viewpoint_cam.transfer_vert - verts_final) ** 2).mean() * 1000
        # gaussian_rot_loss = ((viewpoint_cam.transfer_rot - rot_delta) ** 2).mean() * 1000
        # gaussian_scale_loss = ((viewpoint_cam.transfer_scale - scale_coef) ** 2).mean() * 1000
        # gaussian_loss = gaussian_vert_loss + gaussian_rot_loss + gaussian_scale_loss

        loss = loss_huber*1 + loss_G*1

        loss.backward()

        with torch.no_grad():
            viewpoint_cam.to_cpu()
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                DeformModel.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                DeformModel.optimizer.zero_grad(set_to_none=True)

            # print loss
            if iteration % visualize_iteraions == 0:
                if iteration <= mid_num:
                    print("step: %d, huber: %.5f, gaussian vert: %.5f, rot: %.5f, scale: %.5f" % (
                        iteration, loss_huber.item(),
                        # gaussian_vert_loss.item(), gaussian_rot_loss.item(), gaussian_scale_loss.item()
                        0, 0, 0
                    ))
                else:
                    print("step: %d, huber: %.5f, percep: %.5f, gaussian vert: %.5f, rot: %.5f, scale: %.5f" % (
                        iteration, loss_huber.item(), loss_G.item(),
                        # gaussian_vert_loss.item(), gaussian_rot_loss.item(), gaussian_scale_loss.item(),
                        0, 0, 0
                    ))

            # visualize results
            if iteration % visualize_iteraions == 0 or iteration == 1:
                # print('save:', viewpoint_cam.image_name)
                # save_image = np.zeros((args.image_res, args.image_res*2, 3))
                gt_image_np = (gt_image*255.).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                image = image.clamp(0, 1)

                middle = gt_image * 0.5 + image * 0.5

                image_np = (image*255.).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)

                middle_np = (middle * 255.).permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
                # save_image[:, :args.image_res, :] = gt_image_np
                # save_image[:, args.image_res:, :] = image_np
                save_image = np.concatenate((gt_image_np, middle_np, image_np), 1)
                # save_image = cv2.flip(save_image, 0)
                Image.fromarray(save_image).save(os.path.join(train_dir, f"{iteration}.jpg"))
                # print(os.path.join(train_dir, f"{iteration}.jpg"))

            # save checkpoint
            if iteration % 5000 == 0:
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((DeformModel.capture(), gaussians.capture(), iteration), model_dir + "/chkpnt" + str(iteration) + ".pth")


if __name__ == "__main__":
    import traceback
    try:
        main()
    except:
        traceback.print_exc()
