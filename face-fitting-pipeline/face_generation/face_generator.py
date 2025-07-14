import os
import random
import datetime
from time import sleep

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image
from PIL import ImageDraw
import numpy as np

from .xravatar.rendering import Renderer, bvec_dot, gpmm_illumination_sh
from .xravatar.material import Material
from .xravatar.blendshape import _morph_pca
from .xravatar.utils import read_obj, write_obj, view_img, to_np
from .xravatar.utils import save_tensor_img
from .xravatar.MGCNet import load_datas as load_mgc
from .xravatar.MGCNet import mgc_preprocess
from .identity_trainer import IdentityTrainer
from .SBR_lm import VGG16 as sbrnet
from .insightface import iresnet100


def sample_landmarks(input_imgs,
                     fake_imgs,
                     lm_true,
                     lm_fake,
                     point_size=5,
                     token=None):

    input_imgs = input_imgs.clone()
    fake_imgs = fake_imgs.clone()
    lm_true = lm_true.clone()
    lm_fake = lm_fake.clone()

    lm_true = lm_true * 0.5 + 0.5
    lm_true[..., 1] = 1.0 - lm_true[..., 1]
    lm_fake = lm_fake * 0.5 + 0.5
    lm_fake[..., 1] = 1.0 - lm_fake[..., 1]

    # print(lm_true.cpu().detach().numpy())
    # print(lm_fake.cpu().detach().numpy())

    def drawlm(img, lms, color, size=10):
        image_size, image_size, C = img.shape
        # print(img.shape)
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for i in range(len(lms)):
            lm = lms[i]
            # print(lm)
            if lm[0] > 0 and lm[0] < image_size and lm[1] > 0 and lm[
                    1] < image_size:
                draw.arc([(int(lm[0]), int(lm[1])),
                          (int(lm[0]) + size, int(lm[1]) + size)],
                         0,
                         360,
                         fill=color,
                         width=size)
        del draw
        return np.array(img)

    def draw_line(img, lm1, lm2, color):
        image_size, image_size, C = img.shape
        img = Image.fromarray(img)
        draw = ImageDraw.Draw(img)
        for i in range(len(lm1)):
            l1 = lm1[i]
            l2 = lm2[i]
            if l1[0] > 0 and l1[0] < image_size and l1[1] > 0 and l1[
                    1] < image_size:
                draw.line((int(l1[0]), int(l1[1]), int(l2[0]), int(l2[1])),
                          fill='green')
        del draw
        return np.array(img)

    N, _, _, image_size = input_imgs.shape
    lm_true *= image_size
    lm_fake *= image_size
    if isinstance(lm_true, torch.Tensor):
        if lm_true.shape[-1] != 2:
            lm_true = lm_true.reshape(N, -1, 2).cpu().detach().numpy()
    if isinstance(lm_fake, torch.Tensor):
        if lm_fake.shape[-1] != 2:
            lm_fake = lm_fake.reshape(N, -1, 2).cpu().detach().numpy()

    for i in range(N):
        true_img = view_img(input_imgs[i])
        fake_img = view_img(fake_imgs[i])

        true_img = drawlm(true_img, lm_true[i], (0, 255, 255), point_size)
        fake_img = drawlm(fake_img, lm_fake[i], (255, 0, 0), point_size)
        line_img = draw_line(np.zeros_like(true_img), lm_true[i], lm_fake[i],
                             (0, 0, 255))
        line_img = drawlm(line_img, lm_true[i], (0, 255, 255), point_size)
        line_img = drawlm(line_img, lm_fake[i], (255, 0, 0), point_size)

        try:
            os.makedirs('./results/testing/', exist_ok=True)
            if token is not None:
                Image.fromarray(true_img).save(
                    './results/testing/{}_{}.jpg'.format(token, i))
                Image.fromarray(fake_img).save(
                    './results/testing/{}_fake_{}.jpg'.format(token, i))
                Image.fromarray(line_img).save(
                    './results/testing/{}_line_{}.jpg'.format(token, i))
            else:
                Image.fromarray(true_img).save(
                    './results/testing/test_{}.jpg'.format(i))
                Image.fromarray(fake_img).save(
                    './results/testing/test_fake_{}.jpg'.format(i))
                Image.fromarray(line_img).save(
                    './results/testing/test_line_{}.jpg'.format(i))
        except Exception:
            print()


class WeightFitter(nn.Module):
    def __init__(self, weight_size, batch_size, device, loaded=[]):
        super().__init__()
        if len(loaded) == 0:
            self.weights = nn.Parameter(torch.zeros([batch_size, weight_size, 1]).type(torch.float32).to(device))
        else:
            self.weights = nn.Parameter(torch.from_numpy(loaded).type(torch.float32).to(device))

    def forward(self):
        return self.weights


def identity_trainer_loss(trainer, img_fake):
    code_true = trainer.code_true
    content_true = trainer.content_true
    code_fake, content_fake = trainer.pred(img_fake)

    losses = ((Variable(content_true[0], requires_grad=False) - content_fake[0]) ** 2).mean()
    for i in range(1, len(content_true)):
        losses += ((Variable(content_true[i], requires_grad=False) - content_fake[i]) ** 2).mean()

    return torch.nn.CosineEmbeddingLoss()(code_fake, Variable(code_true, requires_grad=False), torch.ones_like(code_true[:, 0])), losses


def random_pose(rid=0):
    r = random.randint(0, 5)
    r = 0
    g_pos = torch.zeros([1, 3])
    g_rot = torch.zeros([1, 3])
    if rid == 0:
        g_rot[:, 1] = 0.0174532925 * -30.0
    elif rid == 1:
        g_rot[:, 1] = 0.0174532925 * 0.0
    else:
        g_rot[:, 1] = 0.0174532925 * 30.0
    if r > 3:
        fov = 35.0
        g_pos[:, 1] -= 0.14
        g_pos[:, 2] -= 4.0
        # g_rot[:, 1] = 0.0174532925 * np.random.uniform(-45.0, 45.0, 1)[0]
        g_rot[:, 0] = 0.0174532925 * np.random.uniform(-6.0, 6.0, 1)[0]
    elif r == 3:
        fov = 15.0
        g_pos[:, 1] -= 0.12
        g_pos[:, 2] -= 9.0
        # g_rot[:, 1] = 0.0174532925 * np.random.uniform(-40.0, 40.0, 1)[0]
        g_rot[:, 0] = 0.0174532925 * np.random.uniform(-6.0, 6.0, 1)[0]
    else:
        fov = 45.0
        g_pos[:, 1] -= 0.15
        g_pos[:, 2] -= 3.5
        # g_rot[:, 1] = 0.0174532925 * np.random.uniform(-50.0, 50.0, 1)[0]
        g_rot[:, 0] = 0.0174532925 * np.random.uniform(-6.0, 6.0, 1)[0]
    g_pos[:, 1] += g_rot[:, 0]
    g_pos[:, 0] -= g_rot[:, 1] * 0.8
    return fov, g_pos, g_rot


def pred_sbr_lm(model, img, mean, std):
    img = F.interpolate(img.clone(), size=256, mode='area')
    img -= mean
    img /= std
    heatmap, lm = model(img)

    return heatmap, lm


def landmark_loss(model, heatmap_true, img_fake, mean, std):
    heatmap_fake, _ = pred_sbr_lm(model, img_fake, mean, std)

    return torch.nn.SmoothL1Loss()(heatmap_fake, heatmap_true)


def get_sub_faces(full_face, sub_datas):
    out = []
    for sub in sub_datas:
        out.append(full_face[sub[0]:sub[1], :])
    out = torch.cat(out, 0)
    return out


def rgb2hsv(rgb: np.ndarray = None) -> np.ndarray:
    '''
    __author__ = 'Christopher Hahne'
    __email__ = 'inbox@christopherhahne.de'
    __license__ =
    Copyright (c) 2020 Christopher Hahne <inbox@christopherhahne.de>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Convert RGB color space to HSV color space
    :param rgb: input array in red, green and blue (RGB) space
    :type rgb: :class:`~numpy:numpy.ndarray`
    :return: array in hue, saturation and value (HSV) space
    :rtype: ~numpy:np.ndarray
    https://github.com/hahnec/color-space-converter/blob/master/color_space_converter/hsv_converter.py
    '''

    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    # slicing implementation of HSV channel definitions
    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 /
                          (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 /
                          (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 /
                          (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv


def hsv2rgb(hsv: np.ndarray = None) -> np.ndarray:
    '''
    __author__ = 'Christopher Hahne'
    __email__ = 'inbox@christopherhahne.de'
    __license__ =
    Copyright (c) 2020 Christopher Hahne <inbox@christopherhahne.de>
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    Convert HSV color space to RGB color space
    :param hsv: input array in hue, saturation and value (HSV) space
    :type hsv: :class:`~numpy:numpy.ndarray`
    :return: array in red, green and blue (RGB) space
    :rtype: ~numpy:np.ndarray
    '''

    hi = np.floor(hsv[..., 0] / 60.0) % 6
    hi = hi.astype('uint8')
    v = hsv[..., 2].astype('float')
    f = (hsv[..., 0] / 60.0) - np.floor(hsv[..., 0] / 60.0)
    p = v * (1.0 - hsv[..., 1])
    q = v * (1.0 - (f * hsv[..., 1]))
    t = v * (1.0 - ((1.0 - f) * hsv[..., 1]))

    rgb = np.zeros(hsv.shape)
    rgb[hi == 0, :] = np.dstack((v, t, p))[hi == 0, :]
    rgb[hi == 1, :] = np.dstack((q, v, p))[hi == 1, :]
    rgb[hi == 2, :] = np.dstack((p, v, t))[hi == 2, :]
    rgb[hi == 3, :] = np.dstack((p, q, v))[hi == 3, :]
    rgb[hi == 4, :] = np.dstack((t, p, v))[hi == 4, :]
    rgb[hi == 5, :] = np.dstack((v, p, q))[hi == 5, :]

    return rgb


def _morph_pca_np(base, pca, weights):
    base_out = base.reshape([-1, 3])
    morphed = np.einsum('ij,jk->ik', pca, weights).reshape(-1, 3)
    return base_out + morphed


def build_line_face(pnt1, pnt2, width=0.002):
    normals = torch.zeros_like(pnt1)
    normals[..., 0] = width
    normals[..., 1] = 0
    pnts = torch.zeros(pnt1.shape[0], pnt1.shape[1], 4, 3).type(torch.float32).to(pnt1.device)
    pnts[:, :, 0, :] = pnt1
    pnts[:, :, 1, :] = pnt1 + normals
    pnts[:, :, 2, :] = pnt2 + normals
    pnts[:, :, 3, :] = pnt2

    faces = torch.zeros([2 * pnt1.shape[1], 3]).type(torch.int64)
    index = 0
    for i in range(pnt1.shape[1] * 2):
        if i % 2 == 0:
            faces[i] = torch.tensor([0, 1, 2]).type(torch.int64) + int(index * 4)
        else:
            faces[i] = torch.tensor([2, 3, 0]).type(torch.int64) + int(index * 4)
            index += 1
    return pnts.reshape([1, -1, 3]), faces.to(pnt1.device)


def get_mat(theta, device):
    res = torch.zeros([1, 2, 3]).type(theta.type()).to(device)
    res[:, 0, 0] = torch.cos(theta)
    res[:, 0, 1] = -torch.sin(theta)
    res[:, 1, 0] = torch.sin(theta)
    res[:, 1, 1] = torch.cos(theta)
    return res


def rot_img(img, theta, device):
    rot_mat = get_mat(theta, device).type(img.type()).repeat(img.shape[0], 1, 1)
    grid = F.affine_grid(rot_mat, img.size(), align_corners=False).type(img.type())
    return F.grid_sample(img, grid, align_corners=False)


class FaceGenerator():
    def __init__(self, root_path, debug=False):
        random_seed = 4
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        self.debug = debug
        self.image_size = 512
        self.batch_size = 1
        self.device = 'cuda'

        self.pix_weight = 10.0
        self.if_weight = 2.0
        self.if2_weight = 2.5
        self.content_weight = 45.0
        self.content2_weight = 30.0
        self.lm_weight = 100.0
        self.lm3d_weight = 10000.0
        self.flip_weight = 2.0

        self.fit_lm_first = 30
        self.step_size = 50
        self.texture_size = 1024 if debug else 2048
        self.normalmap = True

        mgcnet_model, fa = load_mgc(os.path.join(root_path, 'assets/mgcnet.pth'), self.device)
        self.mgc_model = mgcnet_model
        self.fa = fa

        self.renderer = Renderer(self.image_size, self.batch_size, self.device)
        self.loaded_texture = self._load_textures(root_path)
        self.face_model = self._load_face_model(root_path)
        self.eyebrow_builder = False
        self.eyebrow_pca = True
        if self.eyebrow_builder:
            self.all_eyebrow_roots = np.load(os.path.join(root_path, 'assets/all_eyebrow_roots.npz'), allow_pickle=True)['all_roots']
            self.all_eyebrow_dirs = []
            for i in range(self.all_eyebrow_roots.shape[0]):
                roots = torch.from_numpy(self.all_eyebrow_roots[i]).type(torch.float32).to(self.device).unsqueeze(0)
                eyebrow_pnts = roots + 0.01
                eyebrow_pnts[..., 2] = 0.0
                self.all_eyebrow_dirs.append(eyebrow_pnts - roots)
                self.all_eyebrow_roots[i] = roots
        elif self.eyebrow_pca:
            d = np.load(os.path.join(root_path, 'assets/eyebrow_textures/eyebrow_albedo_512.npz'))
            self.eyebrow_mean = torch.from_numpy(d['a_mean'].reshape([1, -1, 3])).type(torch.float32).to(self.device)
            self.eyebrow_diffs = torch.from_numpy(d['a_diffs']).type(torch.float32).to(self.device)
            self.eyebrow_weight_count = self.eyebrow_diffs.shape[-1]
            self.eyebrow_diffs = self.eyebrow_diffs.reshape([1, -1, self.eyebrow_weight_count])
            d = np.load(os.path.join(root_path, f'assets/eyebrow_textures/eyebrow_albedo_{self.texture_size}.npz'))
            self.eyebrow_full_mean = torch.from_numpy(d['a_mean'].reshape([1, -1, 3])).type(torch.float32).to(self.device)
            self.eyebrow_full_diffs = torch.from_numpy(d['a_diffs']).type(torch.float32).to(self.device).reshape([1, -1, self.eyebrow_weight_count])
            d = np.load(os.path.join(root_path, f'assets/eyebrow_textures/eyebrow_normal_{self.texture_size}.npz'))
            self.eyebrow_nor_mean = torch.from_numpy(d['a_mean'].reshape([1, -1, 3])).type(torch.float32).to(self.device)
            self.eyebrow_nor_diffs = torch.from_numpy(d['a_diffs']).type(torch.float32).to(self.device).reshape([1, -1, self.eyebrow_weight_count])

        self.insightface_model = iresnet100(pretrained=False)
        self.insightface_model.load_state_dict(torch.load(os.path.join(root_path, 'assets/iresnet100-73e07ba7.pth')))
        self.insightface_model.eval()
        self.insightface_model.to(self.device)
        self.if_mean = torch.tensor([0.5] * 3).type(torch.float32).to(self.device).view([1, 3, 1, 1])
        self.if_std = torch.tensor([0.5 * 256.0 / 255.0] * 3).type(torch.float32).to(self.device).view([1, 3, 1, 1])

        self.sbr_model = sbrnet(False)
        self.sbr_model.load_state_dict(torch.load(os.path.join(root_path, 'assets/SBR_lm.pth')))
        self.sbr_model.eval()
        self.sbr_model.to(self.device)
        self.sbr_mean = torch.tensor([0.485, 0.456, 0.406]).type(torch.float32).to(self.device).view([1, 3, 1, 1])
        self.sbr_std = torch.tensor([0.229, 0.224, 0.225]).type(torch.float32).to(self.device).view([1, 3, 1, 1])

    def _load_textures(self, root_path):
        texture_size = self.texture_size
        color_pca_size = 48
        d = np.load(os.path.join(root_path, 'assets/textures/Color_512.npz'))
        small_a_mean = torch.from_numpy(d['a_mean'].reshape([1, -1, 3])).type(torch.float32).to(self.device)
        small_a_diffs = torch.from_numpy(d['a_diffs'][..., :color_pca_size].reshape([1, -1, color_pca_size])).type(torch.float32).to(self.device)
        small_n_mean = []
        small_n_diffs = []
        if self.normalmap:
            d = np.load(os.path.join(root_path, 'assets/textures/Normal_512.npz'))
            small_n_mean = torch.from_numpy(d['a_mean'].reshape([1, -1, 3])).type(torch.float32).to(self.device)
            small_n_diffs = torch.from_numpy(d['a_diffs'][..., :color_pca_size].reshape([1, -1, color_pca_size])).type(torch.float32).to(self.device)
        small_a2_mean = []
        small_a2_diffs = []
        if self.normalmap:
            d = np.load(os.path.join(root_path, 'assets/textures/Cavity_512.npz'))
            small_a2_mean = torch.from_numpy(d['a_mean'].reshape([1, -1, 3])).type(torch.float32).to(self.device)
            small_a2_diffs = torch.from_numpy(d['a_diffs'][..., :color_pca_size].reshape([1, -1, color_pca_size])).type(torch.float32).to(self.device)
        d = np.load(os.path.join(root_path, f'assets/textures/Color_{texture_size}.npz'))
        a_mean = d['a_mean']
        a_diffs = d['a_diffs'][..., :color_pca_size]
        d = np.load(os.path.join(root_path, f'assets/textures/Normal_{texture_size}.npz'))
        n_mean = d['a_mean']
        n_diffs = d['a_diffs'][..., :color_pca_size]
        d = np.load(os.path.join(root_path, f'assets/textures/Cavity_{texture_size}.npz'))
        s_mean = d['a_mean']
        s_diffs = d['a_diffs'][..., :color_pca_size]
        return small_a_mean, small_a_diffs, small_n_mean, small_n_diffs, small_a2_mean, small_a2_diffs, a_mean, a_diffs, n_mean, n_diffs, s_mean, s_diffs

    def _load_face_model(self, root_path):
        color_pca_size = 48
        shape_pca_size = 100
        batch_size = 1
        uv_faces = None
        lm_3ds = None
        lm_indices = [
            8, 17, 19, 21, 22, 24, 26, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
            47, 48, 50, 51, 52, 54, 56, 57, 58, 60, 61, 62, 63, 64, 65, 66, 67
        ]
        materials = {}
        flip_idx = None
        mtl_index = {}
        mesh = None

        # ict identities
        d = np.load(os.path.join(root_path, 'assets/ict_model.npz'))
        id_mean = d['verts']
        faces = d['faces']
        uv_faces = d['uv_faces']
        id_pca = d['id_pca']
        ex_pca = d['ex_pca']
        a_mean, a_pca, n_mean, n_pca, a2_mean, a2_pca, _, _, _, _, _, _ = self.loaded_texture

        # UV for ICT
        m = read_obj(os.path.join(root_path, 'assets/ict_head_only.obj'))
        uvs = m['uvs']

        mesh = read_obj(os.path.join(root_path, 'assets/generic_neutral_mesh.obj'))
        # [0, 18460, 28068, 33876, 42572, 44108, 45704, 47240, 48836, 49488, 49548, 50060]
        # M_Face : 0, 18460
        # M_BackHead : 18460, 28068
        # M_GumsTongue : 28068, 33876
        # M_Teeth : 33876, 42572
        # M_ScleraLeft : 42572, 44108
        # M_IrisLeft : 44108, 45704
        # M_ScleraRight : 45704, 47240
        # M_IrisRight : 47240, 48836
        # M_LacrimalFluid : 48836, 49488
        # M_EyeBlend : 49488, 49548
        # M_EyeOcclusion : 49548, 50060
        # M_EyeLashes : 50060, 52220

        # print(mesh['mtl_index'])
        id_mean = torch.from_numpy(mesh['verts']).unsqueeze(0).to(self.device).type(torch.float32)

        flip_idx = np.load(os.path.join(root_path, 'assets/flip_idx.npy'))
        flip_idx = torch.from_numpy(flip_idx).to(self.device).type(torch.int64)

        lm_3ds = torch.tensor([
            1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390,
            3927, 3924, 2608, 3272, 4088, 3443, 268, 493, 1914, 2044, 1401,
            3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075,
            1147, 4269, 3360, 1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751,
            3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200, 6213,
            6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470,
            5517, 5966
        ]).to(self.device).type(torch.int64)

        faces = torch.from_numpy(mesh['faces']).to(self.device).type(torch.int64)

        full_uvs = torch.from_numpy(mesh['uvs']).unsqueeze(0).to(self.device).type(torch.float32)
        uv_faces = torch.from_numpy(mesh['uv_faces']).to(self.device).type(torch.int64)

        uvs = torch.from_numpy(uvs).unsqueeze(0).to(self.device).type(torch.float32)
        face_index = 13120
        full_uvs[:, :9409, :] = uvs[:, :9409, :]
        uv_faces[:face_index] = faces[:face_index].clone()
        uvs = full_uvs
        mtl_index[0] = 'M_Face'
        mtl_index[face_index] = 'M_Eyes'
        mesh['face_index'] = face_index

        # Find ICT eye faces (left)
        eye_left_faces = get_sub_faces(faces, [(42572, 44108)]).cpu().detach().numpy().reshape([-1]).tolist()
        eye_left_faces = list(set(eye_left_faces))
        eye_left_faces.sort()
        eye_left_faces = np.array(eye_left_faces).astype(int)
        # Find ICT eye faces (right)
        eye_right_faces = get_sub_faces(faces, [(45704, 47240)]).cpu().detach().numpy().reshape([-1]).tolist()
        eye_right_faces = list(set(eye_right_faces))
        eye_right_faces.sort()
        eye_right_faces = np.array(eye_right_faces).astype(int)

        faces = get_sub_faces(faces, [(0, face_index), (42572, 44108), (45704, 47240)])
        uv_faces = get_sub_faces(uv_faces, [(0, face_index), (42572, 44108), (45704, 47240)])

        uvs[..., 1] = 1.0 - uvs[..., 1]
        id_pca = torch.from_numpy(id_pca).type(torch.float32).to(self.device).unsqueeze(0)
        ex_pca = torch.from_numpy(ex_pca).type(torch.float32).to(self.device).unsqueeze(0)
        default_vc = np.ones([batch_size, color_pca_size, 1]) * 0.5
        id_pca = id_pca[..., :shape_pca_size]
        id_mean *= 10.0

        without_ex_shape = id_mean.clone()
        orig_shape = id_mean.clone()
        materials['face'] = Material(uv_faces, uvs=uvs, face_range=(0, face_index))
        materials['eyes'] = Material(uv_faces, uvs=uvs, face_range=(face_index, face_index + 1536 * 2))
        eye_tex = np.array(Image.open(os.path.join(root_path, 'assets/eye.png')).convert('RGB')) / 255.0
        eye_tex = torch.from_numpy(eye_tex).type(torch.float32).to(self.device).unsqueeze(0).permute(0, 3, 1, 2)
        materials['eyes'].textures['albedo'] = eye_tex.clone()

        uv_to_mesh = read_obj(os.path.join(root_path, 'assets/uv_target.obj'))
        uv_to = uv_to_mesh['uvs'].copy()

        uv_reference_mesh = read_obj(os.path.join(root_path, 'assets/uv_reference.obj'))
        uv_from = uv_reference_mesh['uvs']
        uv_to = torch.from_numpy(uv_to).type(torch.float32).to('cuda')
        uv_from = torch.from_numpy(uv_from).type(torch.float32).to('cuda')
        # uv_verts = -torch.cat((uv_to * 2.0 - 1.0, torch.zeros_like(uv_to[:, 0]).unsqueeze(-1)), -1).unsqueeze(0)
        # uv_faces = torch.from_numpy(out_mesh['faces']).type(torch.int64).to('cuda')
        uv_to[..., 1] = 1.0 - uv_to[..., 1]
        uv_verts = -torch.cat((uv_from * 2.0 - 1.0, torch.zeros_like(uv_from[:, 0]).unsqueeze(-1)), -1).unsqueeze(0)
        uv_reference_faces = torch.from_numpy(uv_reference_mesh['faces']).type(torch.int64).to('cuda')

        uv_verts[..., 0] = -uv_verts[..., 0]
        uv_to = uv_to.unsqueeze(0)

        # from ICT to our Nonscaled mapping
        d = np.load(os.path.join(root_path, 'assets/head_corresponding.npz'))
        corr_bc = d['bc']
        corr_tri = d['tri']
        # Our head for exporting.
        out_mesh = read_obj(os.path.join(root_path, 'assets/NonscaledHead.obj'))
        out_mesh['norms'] = None
        out_mesh['norm_faces'] = None

        # Our eyes.
        eye_mesh = read_obj(os.path.join(root_path, 'assets/eyeball_world.obj'))
        eye_mesh['orig_verts'] = eye_mesh['verts'].copy()
        eyev = eye_mesh['orig_verts']
        eyeleft = []
        eyeright = []
        for i in range(eyev.shape[0]):
            if eyev[i, 0] > 0:
                eyeleft.append(i)
            else:
                eyeright.append(i)
        # Finding Left eye & right eye vertices
        eye_mesh['left_index'] = np.array(eyeleft).astype(int)
        eye_mesh['right_index'] = np.array(eyeright).astype(int)
        # Do all for eye lens too.
        eyelen_mesh = read_obj(os.path.join(root_path, 'assets/eyelen_world.obj'))
        eyelen_mesh['orig_verts'] = eyelen_mesh['verts'].copy()
        eyev = eyelen_mesh['orig_verts']
        eyeleft = []
        eyeright = []
        for i in range(eyev.shape[0]):
            if eyev[i, 0] > 0:
                eyeleft.append(i)
            else:
                eyeright.append(i)
        eyelen_mesh['left_index'] = np.array(eyeleft).astype(int)
        eyelen_mesh['right_index'] = np.array(eyeright).astype(int)

        # skin_color_texture = np.array(Image.open(os.path.join(root_path, 'assets/skin_color.png')).convert("RGB")).astype(np.float32) / 255.0
        # skin_color_texture = torch.from_numpy(skin_color_texture).type(torch.float32).unsqueeze(0).to(self.device).permute(0, 3, 1, 2)

        return {
            'uv_faces': uv_faces,
            'mtl_index': mtl_index,
            'flip_idx': flip_idx,
            'orig_shape': orig_shape,
            'without_ex_shape': without_ex_shape,
            'faces': faces,
            'uvs': uvs,
            'id_pca': id_pca,
            'ex_pca': ex_pca,
            'default_vc': default_vc,
            'a_mean': a_mean,
            'a_pca': a_pca,
            'n_mean': n_mean,
            'n_pca': n_pca,
            'a2_mean': a2_mean,
            'a2_pca': a2_pca,
            'lm_indices': lm_indices,
            'lm_3ds': lm_3ds,
            'materials': materials,
            'uv_reference_faces': uv_reference_faces,
            'uv_verts': uv_verts,
            'uv_to': uv_to,
            'corr_bc': corr_bc,
            'corr_tri': corr_tri,
            'mesh': mesh,
            'out_mesh': out_mesh,
            'eye_left_faces': eye_left_faces,
            'eye_right_faces': eye_right_faces,
            'eye_mesh': eye_mesh,
            'eyelen_mesh': eyelen_mesh,
            # 'skin_color_texture': skin_color_texture,
        }

    def _build_eyebrow_texture(self, eyebrow_pnts, eyebrow_color, eyebrow_dir, rotate=None, image_size=1024):
        pnt2 = eyebrow_pnts + eyebrow_dir
        pnts, faces = build_line_face(eyebrow_pnts, pnt2)
        mats = {}
        mats['eyebrow_face'] = Material(faces, vc=eyebrow_color)
        self.renderer(verts=pnts, faces=faces, materials=mats, render_size=image_size)
        half_size = int(image_size / 2)
        eyebrow_texture = mats['eyebrow_face'].get_albedo()
        left_eyebrow = eyebrow_texture[:, :, 0:half_size, half_size:image_size].clone()
        if rotate is not None:
            left_eyebrow = rot_img(left_eyebrow, rotate, self.device)
        eyebrow_texture[:, :, 0:half_size, 0:half_size] = torch.flip(left_eyebrow.clone(), [3])
        return eyebrow_texture

    def _fitter(
            self, step_count,
            best_data, optimizer,
            gen_render_data, renderer_fn,
            loss_fn, log_fn=None, random_identity_fn=None, random_identity_loss_fn=None
    ):
        st1 = datetime.datetime.now()

        best_loss = 1e10
        best_data['i'] = 0
        best_data.pop('recorded', None)
        ifr = None
        for i in range(step_count):
            optimizer.zero_grad()
            render_data = gen_render_data(best_data, i)

            renderer_res = renderer_fn(render_data)

            loss, loss_log = loss_fn(renderer_res)

            if random_identity_fn is not None:
                if2_losses = None
                if2_content_losses = None
                for rid in range(3):
                    if2_losses, if2_content_losses = random_identity_fn(rid, if2_losses, if2_content_losses, render_data)

                loss, loss_log = random_identity_loss_fn(loss, loss_log, if2_losses, if2_content_losses)
                ifr = if2_losses.item() / 3

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_data['i'] = i
                if ifr is not None:
                    best_data['ifr'] = ifr
                if 'recorded' not in best_data:
                    raise Exception('need recorded in best_data!')
                recorded = best_data['recorded']
                for key in recorded:
                    best_data[key] = recorded[key]

            if log_fn is not None:
                log_fn(i, (datetime.datetime.now() - st1).total_seconds(), loss.item(), loss_log, best_data['i'], best_loss)

        if log_fn is not None:
            print()

        return loss, (datetime.datetime.now() - st1).total_seconds(), best_loss

    def _search_eyebrow_index(self, img, best_data, faces, image_size, insightface_trainer):
        materials = self.face_model['materials']
        t_weight = Variable(best_data['tweight'].clone(), requires_grad=False)
        sh_weight = Variable(best_data['shweight'].clone(), requires_grad=False)

        best_index = 0
        best_loss = 1e10
        st1 = datetime.datetime.now()

        for i in range(len(self.all_eyebrow_roots)):
            eyebrow_roots = self.all_eyebrow_roots[i].clone()
            eyebrow_vertex_colors = torch.ones_like(eyebrow_roots)
            eyebrow_vertex_colors = eyebrow_vertex_colors.unsqueeze(-2).repeat([1, 1, 4, 1]).reshape([1, -1, 3])
            eyebrow_background = self._build_eyebrow_texture(eyebrow_roots, eyebrow_vertex_colors, self.all_eyebrow_dirs[i])

            materials['face'].textures['eyebrow'] = eyebrow_background.clone()

            shape = Variable(best_data['shape'].clone(), requires_grad=False)
            res = self.renderer(
                verts=shape.clone(),
                faces=Variable(faces.clone(), requires_grad=False),
                render_size=image_size,
                sh=sh_weight,
                trans=t_weight,
                materials=materials,
                get_norm=True)

            img_fake = materials['face'].get_albedo()
            eyebrow_out = materials['face'].textures_out['eyebrow']
            eyebrow = eyebrow_out.clamp(0.0, 1.0)
            eyebrow[eyebrow > 0.5] = 1.0
            eyebrow = 1.0 - eyebrow

            img_fake *= eyebrow
            img_fake += materials['eyes'].get_albedo()
            img_fake *= res['sh']

            img_fake = torch.clamp(img_fake, min=0.0, max=1.0)

            index = res['norm'] != 0
            combined = img.clone()
            combined[index] = img_fake[index]

            if_loss, if_content_loss = identity_trainer_loss(insightface_trainer, combined)
            # img_ = img.clone()
            # pix_loss = torch.nn.L1Loss()(img_fake[index], Variable(img_[index].clone(), requires_grad=False)).item()
            if if_content_loss.item() < best_loss:
                best_loss = if_content_loss.item()
                best_index = i
            print('    {:4.2f}s eyebrow search {}. [if:{:3.4f}, if_c:{:3.4f}] best_{}:{:3.4f}'.format(
                (datetime.datetime.now() - st1).total_seconds(), i, if_loss.item(), if_content_loss.item(), best_index, best_loss), end='\r')
        print()
        return best_index

    def _fit_face(self, input_image, lms_gt, coef):
        # ======== init ========
        device = self.device
        image_size = self.image_size
        batch_size = self.batch_size
        fit_lm_first = self.fit_lm_first
        renderer = self.renderer
        face_model = self.face_model

        pix_weight = self.pix_weight
        if_weight = self.if_weight
        if2_weight = self.if2_weight
        content_weight = self.content_weight
        content2_weight = self.content2_weight
        lm_weight = self.lm_weight
        lm3d_weight = self.lm3d_weight
        flip_weight = self.flip_weight

        # ======== load bfm ========
        # uv_faces = face_model['uv_faces']
        # mtl_index = face_model['mtl_index']
        flip_idx = face_model['flip_idx']
        orig_shape = face_model['orig_shape']
        without_ex_shape = face_model['without_ex_shape']
        faces = face_model['faces']
        # uvs = face_model['uvs']
        id_pca = face_model['id_pca']
        ex_pca = face_model['ex_pca']
        default_vc = face_model['default_vc'].copy()
        a_mean = face_model['a_mean']
        a_pca = face_model['a_pca']
        n_mean = face_model['n_mean']
        n_pca = face_model['n_pca']
        lm_indices = face_model['lm_indices']
        lm_3ds = face_model['lm_3ds']
        materials = face_model['materials']
        # skin_color_texture = face_model['skin_color_texture']

        # ======== load images ========
        # orig, lms_gt, mask = gen_input_image(path=None, idx=idx)
        lms_gt = lms_gt.to(device)
        lms_gt[..., 1] = 1.0 - lms_gt[..., 1]
        lms_gt -= 0.5
        lms_gt *= 2.0
        # img = orig.resize((image_size, image_size))
        input_image = input_image.to(device)
        img = F.interpolate(input_image, size=image_size)
        if self.debug:
            save_tensor_img(img, self.debug_output + 'i_src.png')

        # ======== model loading ========
        insightface_trainer = IdentityTrainer(self.insightface_model, self.if_mean, self.if_std, img, 112, 1.0, 1.0)
        with torch.no_grad():
            heatmap_true, _ = pred_sbr_lm(self.sbr_model, img, self.sbr_mean, self.sbr_std)

        # ======== init fitters ========
        coloroffsetFitter = WeightFitter(0, batch_size, device, np.ones([batch_size, 3]))
        skin_color_Fitter = WeightFitter(0, batch_size, device, np.zeros([batch_size, 1, 1, 2]) * 0.5)
        colorFitter = WeightFitter(0, batch_size, device, np.zeros_like(default_vc))
        shapeFitter = WeightFitter(id_pca.shape[-1], batch_size, device)
        exFitter = WeightFitter(ex_pca.shape[-1], batch_size, device)
        shFitter = WeightFitter(0, batch_size, device, coef['sh'].cpu().numpy() * 0.5)
        transFitter = WeightFitter(0, batch_size, device, coef['trans'].cpu().numpy())

        lr = 1e-2

        # eyeUVFitter = WeightFitter(0, batch_size, device, np.zeros([batch_size, 1, 2]))
        # eye_weight = eyeUVFitter().clamp(-0.5, 0.5)
        eye_uv_base = materials['eyes'].uvs.clone()

        lm3d_optimizer = torch.optim.Adam([
            {'params': exFitter.parameters(), 'lr': lr * 2.5, 'tag': 'ex'},
            {'params': transFitter.parameters(), 'lr': lr * 3.0, 'tag': 't'}
        ], lr=lr, betas=(0.8, 0.999))
        # ======== best data ========
        best_data = {}

        best_data['color_offset'] = coloroffsetFitter()
        best_data['shweight'] = shFitter()
        best_data['weight'] = colorFitter() * 0.1
        best_data['eweight'] = exFitter()
        s_weight = shapeFitter()
        shape_base = _morph_pca(orig_shape.clone(), id_pca, s_weight, None)
        shape = _morph_pca(Variable(orig_shape.clone(), requires_grad=False), Variable(id_pca, requires_grad=False), s_weight, None)
        best_data['shape'] = _morph_pca(shape, Variable(ex_pca, requires_grad=False), Variable(best_data['eweight'], requires_grad=False), None)
        eye_uv_base = materials['eyes'].uvs.clone()
        vc = _morph_pca(a_mean, a_pca, Variable(best_data['weight'].clone(), requires_grad=False), None)
        vc *= Variable(best_data['color_offset'], requires_grad=False)
        albedo = vc.permute(0, 2, 1).reshape([batch_size, 3, 512, 512])
        materials['face'].textures['albedo'] = albedo.clone()

        # ======================== define fns ========================
        # ====== fit lm fns ======
        def fit_lm_render_data(best_data, i):
            t_weight = transFitter()
            e_weight = exFitter()
            shape = _morph_pca(Variable(shape_base, requires_grad=False), Variable(ex_pca, requires_grad=False), e_weight, None)
            best_data['recorded'] = {'tweight': t_weight.clone()}
            return shape, t_weight

        def fit_lm_renderer(render_data):
            shape, t_weight = render_data
            return renderer.transform_mat(shape.clone(), t_weight.clone()), shape

        def fit_lm_loss_fn(renderer_res):
            trans_verts, shape = renderer_res
            shape_flip = shape.clone()
            shape_flip_sub = shape_flip[:, :9409]
            shape_flip_sub = shape_flip_sub[:, flip_idx]
            shape_flip[:, :9409] = shape_flip_sub
            shape_flip = Variable(shape_flip, requires_grad=False)
            flip_loss = torch.nn.SmoothL1Loss()(shape, shape_flip)
            lms = trans_verts[:, lm_3ds, :2]

            # if save_process_lm:
            #     sample_landmarks(img, img, lms_gt[:, lm_indices], lms[:, lm_indices])
            lm3d_loss = torch.nn.MSELoss()(lms[:, lm_indices], lms_gt[:, lm_indices])

            return 1.0 * flip_loss + 1000.0 * lm3d_loss, None

        # ===== fit eyebrow fns =====
        def fit_eyebrow_render_data(best_data, i):
            t_weight = Variable(best_data['tweight'].clone(), requires_grad=False)
            c_weight = Variable(best_data['weight'].clone(), requires_grad=False)
            sh_weight = Variable(best_data['shweight'].clone(), requires_grad=False)
            color_offset = Variable(best_data['color_offset'].clone(), requires_grad=False)
            eyebrow_color = eyebrowColorFitter()
            eyebrow_vertex_offset = eyebrowVertexPosFitter()
            # eyebrow_roots = Variable(best_data['eyebrow_roots'].clone(), requires_grad=False)
            # eyebrow_offset = eyebrowOffsetFitter()
            # eyebrow_roots = eyebrow_roots + eyebrow_offset + eyebrow_vertex_offset
            eyebrow_roots = Variable(best_data['eyebrow_root_offset'].clone(), requires_grad=False)
            # eyebrow_roots = eyebrow_roots + eyebrow_vertex_offset
            eyebrow_offset = eyebrowOffsetFitter()
            eyebrow_roots[..., :2] = eyebrow_roots[..., :2] + eyebrow_offset[..., :2]
            eyebrow_roots = eyebrow_roots + eyebrow_vertex_offset
            eyebrow_dirs = eyebrowVertexDirFitter()
            eyebrow_dirs = torch.cat((eyebrow_dirs, torch.zeros_like(eyebrow_dirs[..., 0]).unsqueeze(-1)), -1)
            eyebrow_vertex_colors_ = eyebrowVertexColorsFitter()
            eyebrow_vertex_colors = eyebrow_vertex_colors_.repeat([1, 1, 3])
            eyebrow_vertex_colors = eyebrow_vertex_colors * eyebrow_color
            # reshape to face
            eyebrow_vertex_colors = eyebrow_vertex_colors.unsqueeze(-2).repeat([1, 1, 4, 1]).reshape([1, -1, 3])
            eyebrow_background = self._build_eyebrow_texture(eyebrow_roots, eyebrow_vertex_colors, eyebrow_dirs.clone(), eyebrow_offset[..., 2])

            best_data['recorded'] = {
                'eyebrow_color': eyebrow_color.clone(),
                'eyebrow_roots': eyebrow_roots.clone(),
                'eyebrow_dirs': eyebrow_dirs.clone(),
                'eyebrow_vertex_colors': eyebrow_vertex_colors_.clone()
            }
            # best_data['eyebrow'] = eyebrow_background.clone()

            vc = _morph_pca(a_mean, a_pca, c_weight, None)
            vc *= color_offset

            if 'shape' in best_data:
                shape = Variable(best_data['shape'].clone(), requires_grad=False)
            else:
                shape = _morph_pca(
                    Variable(shape_base, requires_grad=False),
                    Variable(ex_pca, requires_grad=False),
                    Variable(best_data['eweight'].clone(), requires_grad=False),
                    None)
            materials['face'].textures['eyebrow'] = eyebrow_background.clone()
            # vc : [1, -1, 3] => [1, 3, -1]
            albedo = vc.permute(0, 2, 1).reshape([batch_size, 3, 512, 512])
            materials['face'].textures['albedo'] = albedo.clone()
            materials['eyes'].uvs = Variable(eye_uv_base.clone(), requires_grad=False)
            return shape, sh_weight, t_weight

        def fit_eyebrow_renderer_fn(render_data):
            shape, sh_weight, t_weight = render_data

            res = renderer(
                verts=shape.clone(),
                faces=Variable(faces.clone(), requires_grad=False),
                render_size=image_size,
                sh=sh_weight,
                trans=t_weight,
                materials=materials,
                get_norm=True)

            img_fake = materials['face'].get_albedo()
            eyebrow_out = materials['face'].textures_out['eyebrow']

            eyebrow = eyebrow_out.clamp(0.0, 1.0)
            img_fake = torch.lerp(img_fake, eyebrow, eyebrow)

            img_fake += materials['eyes'].get_albedo()
            img_fake *= res['sh']

            img_fake = torch.clamp(img_fake, min=0.0, max=1.0)

            index = res['norm'] != 0
            combined = img.clone()
            combined[index] = img_fake[index]

            index = materials['face'].textures_out['eyebrow'] > 0

            if self.debug:
                try:
                    combined_ = img.clone()
                    combined_[index] = img_fake[index]
                    save_tensor_img(combined, self.debug_output + 'i_eyebrow.jpg')
                    save_tensor_img(combined_, self.debug_output + 'i_eyebrow_area.jpg')
                except Exception:
                    print()
            return img_fake, index, combined

        def fit_eyebrow_loss_fn(renderer_res):
            img_fake, index, combined = renderer_res
            img_ = img.clone()
            pix_loss = torch.nn.L1Loss()(img_fake[index], Variable(img_[index].clone(), requires_grad=False))

            if_loss, if_content_loss = identity_trainer_loss(insightface_trainer, combined)

            loss = pix_loss * pix_weight * 10.0 + content_weight * if_content_loss

            return loss, (pix_loss.item() * pix_weight * 10.0, content_weight * if_content_loss.item())

        def fit_eyebrow_log_fn(i, time_sec, loss, loss_log, best_index, best_loss):
            print('    {:4.2f}s eyebrow {}. [pix:{:3.4f}, if_c:{:3.4f}] best_{}:{:3.4f}'.format(
                time_sec, i, loss_log[0], loss_log[1], best_index, best_loss), end='\r')

        # ===== fit color fns =====
        def fit_color_render_data(best_data, i):
            t_weight = Variable(best_data['tweight'].clone(), requires_grad=False)
            c_weight = Variable(best_data['weight'].clone(), requires_grad=False)
            sh_weight = shFitter()
            color_offset = coloroffsetFitter()

            if self.eyebrow_builder:
                eyebrow_color = eyebrowColorFitter()
                # eyebrow_color = Variable(best_data['eyebrow_color'].clone(), requires_grad=False)
                eyebrow_roots = Variable(best_data['eyebrow_roots'].clone(), requires_grad=False)
                eyebrow_dirs = Variable(best_data['eyebrow_dirs'].clone(), requires_grad=False)
                eyebrow_vertex_colors = Variable(best_data['eyebrow_vertex_colors'].clone(), requires_grad=False)
                eyebrow_vertex_colors = eyebrow_vertex_colors.repeat([1, 1, 3])
                eyebrow_vertex_colors = eyebrow_vertex_colors * eyebrow_color
                eyebrow_vertex_colors = eyebrow_vertex_colors.unsqueeze(-2).repeat([1, 1, 4, 1]).reshape([1, -1, 3])
                eyebrow_background = self._build_eyebrow_texture(eyebrow_roots, eyebrow_vertex_colors, eyebrow_dirs.clone())
            elif self.eyebrow_pca:
                eyebrow_color = eyebrowColorFitter()
                best_data['eyebrow_color_'] = eyebrow_color + torch.ones_like(eyebrow_color) * 0.5
                # print(eyebrow_color)
                eyebrow_background = Variable(best_data['eyebrow_background'].clone(), requires_grad=False)
                # eyebrow_background = eyebrow_background * eyebrow_color
                eyebrow_background = eyebrow_background.permute(0, 2, 1).reshape([batch_size, 3, 512, 512])

            # skin_uv = skin_color_Fitter()
            # skin_uv = skin_uv.clamp(-0.99, 0.99)
            # skin_color = F.grid_sample(Variable(skin_color_texture, requires_grad=False), skin_uv, align_corners=False)
            # print()
            # print(f'skin_uv : {skin_uv}, skin_color : {skin_color}')
            # color_offset = skin_color.reshape([batch_size, -1])
            d = {
                'color_offset': color_offset.clone(),
                'shweight': sh_weight.clone(),
            }
            if self.eyebrow_builder or self.eyebrow_pca:
                d['eyebrow_color'] = eyebrow_color.clone()

            best_data['recorded'] = d

            vc = _morph_pca(a_mean, a_pca, c_weight, None)
            vc *= color_offset

            if 'shape' in best_data:
                shape = Variable(best_data['shape'].clone(), requires_grad=False)
            else:
                shape = _morph_pca(
                    Variable(shape_base, requires_grad=False),
                    Variable(ex_pca, requires_grad=False),
                    Variable(best_data['eweight'].clone(), requires_grad=False),
                    None)
            albedo = vc.permute(0, 2, 1).reshape([batch_size, 3, 512, 512])
            materials['face'].textures['albedo'] = albedo.clone()
            materials['eyes'].uvs = Variable(eye_uv_base.clone(), requires_grad=False)
            if self.eyebrow_builder or self.eyebrow_pca:
                materials['face'].textures['eyebrow'] = eyebrow_background.clone()
            return shape, t_weight, sh_weight, best_data

        def fit_color_renderer_fn(render_data):
            shape, t_weight, sh_weight, best_data = render_data

            res = renderer(verts=shape.clone(),
                           faces=Variable(faces.clone(), requires_grad=False),
                           render_size=image_size,
                           sh=sh_weight,
                           trans=t_weight,
                           materials=materials,
                           get_norm=True)
            img_fake = materials['face'].get_albedo()

            # if 'eyebrow' in materials['face'].textures_out:
            #     eyebrow_out = materials['face'].textures_out['eyebrow']

            #     eyebrow = eyebrow_out.clamp(0.0, 1.0)
            #     img_fake = torch.lerp(img_fake, eyebrow, eyebrow)

            if self.eyebrow_builder or self.eyebrow_pca:
                eyebrow_mask = materials['face'].textures_out['eyebrow']
                eyebrow_mask = eyebrow_mask.clamp(0.0, 1.0)
                if self.eyebrow_pca:
                    eyebrow_color = best_data['eyebrow_color_'].clone().reshape([1, -1, 1, 1])
                    eyebrow = eyebrow_mask * eyebrow_color
                    # eyebrow_mask = eyebrow_mask * 0.5
                else:
                    eyebrow = eyebrow_mask
                img_fake = torch.lerp(img_fake, eyebrow, eyebrow_mask)

            img_fake += materials['eyes'].get_albedo()
            img_fake *= res['sh']

            img_fake = torch.clamp(img_fake, min=0.0, max=1.0)

            index = res['norm'] != 0

            if self.debug:
                try:
                    combined = img.clone()
                    combined[index] = img_fake[index]
                    save_tensor_img(combined, self.debug_output + 'i_color.jpg')
                except Exception:
                    print()
            return img_fake, index

        def fit_color_loss_fn(renderer_res):
            img_fake, index = renderer_res
            img_ = img.clone()
            pix_loss = torch.nn.L1Loss()(img_fake[index], Variable(img_[index].clone(), requires_grad=False))
            return pix_loss * pix_weight * 3.0, pix_loss.item() * pix_weight * 3.0

        def fit_color_log_fn(i, time_sec, loss, loss_log, best_index, best_loss):
            print(
                '    {:4.2f}s color {}. [pix:{:3.4f}] best_{}:{:3.4f}'.format(
                    time_sec, i, loss_log, best_index, best_loss),
                end='\r')

        # ===== fit face fns =====
        def fit_face_render_data(best_data, i):
            s_weight = shapeFitter()
            t_weight = transFitter()
            # t_weight = Variable(best_data['tweight'].clone(), requires_grad=False)
            # color_offset = coloroffsetFitter()
            color_offset = Variable(best_data['color_offset'].clone(), requires_grad=False)
            # eyebrow_background = Variable(best_data['eyebrow'].clone(), requires_grad=False)
            sh_weight = Variable(best_data['shweight'].clone(), requires_grad=False)

            if self.eyebrow_builder:
                eyebrow_roots = Variable(best_data['eyebrow_roots'].clone(), requires_grad=False)
                eyebrow_dirs = Variable(best_data['eyebrow_dirs'].clone(), requires_grad=False)
                eyebrow_offset = eyebrowOffsetFitter()
                # eyebrow_dirs = torch.cat((eyebrow_dirs, torch.zeros_like(eyebrow_dirs[..., 0]).unsqueeze(-1)), -1)
                eyebrow_roots[..., :2] = eyebrow_roots[..., :2] + eyebrow_offset[..., :2]
                eyebrow_vertex_colors_ = Variable(best_data['eyebrow_vertex_colors'].clone(), requires_grad=False)
                eyebrow_vertex_colors = eyebrow_vertex_colors_.repeat([1, 1, 3])
                # reshape to face
                eyebrow_vertex_colors = eyebrow_vertex_colors.unsqueeze(-2).repeat([1, 1, 4, 1]).reshape([1, -1, 3])
                eyebrow_background = self._build_eyebrow_texture(eyebrow_roots, eyebrow_vertex_colors, eyebrow_dirs.clone(), eyebrow_offset[..., 2])
            elif self.eyebrow_pca:
                eyebrow_weights = eyebrowWeightFitter()
                eyebrow_background = _morph_pca(self.eyebrow_mean.clone(), self.eyebrow_diffs.clone(), eyebrow_weights, None)
                eyebrow_background_ = eyebrow_background.clone()
                eyebrow_background = eyebrow_background.permute(0, 2, 1).reshape([batch_size, 3, 512, 512])

            # e_weight = exFitter()
            if i > self.step_size - 20:
                # sh_weight = shFitter()
                c_weight = colorFitter() * 0.1
                c_weight = torch.clamp(c_weight, 0.0, 1.0)
                # color_offset = coloroffsetFitter()
                e_weight = exFitter()
                # e_weight = Variable(best_data['eweight'].clone(), requires_grad=False)
            else:
                c_weight = Variable(best_data['weight'].clone(), requires_grad=False)
                # e_weight = exFitter()
                e_weight = Variable(best_data['eweight'].clone(), requires_grad=False)
                # color_offset = Variable(best_data['color_offset'].clone(), requires_grad=False)
                # t_weight = Variable(best_data['tweight'].clone(), requires_grad=False)
                # sh_weight = Variable(best_data['shweight'].clone(), requires_grad=False)
                # color_offset = Variable(best_data['color_offset'].clone(), requires_grad=False)

            vc = _morph_pca(a_mean, a_pca, c_weight, None)
            vc *= color_offset
            if self.normalmap:
                nmap = _morph_pca(n_mean, n_pca, c_weight, None).permute(0, 2, 1).reshape([batch_size, 3, 512, 512])

            shape = _morph_pca(Variable(orig_shape.clone(), requires_grad=False), Variable(id_pca, requires_grad=False), s_weight, None)
            shape = _morph_pca(shape, Variable(ex_pca, requires_grad=False), e_weight, None)

            random_ex_shape = _morph_pca(Variable(without_ex_shape.clone(), requires_grad=False), id_pca, s_weight, None)

            # vc : [1, -1, 3] => [1, 3, -1]
            albedo = vc.permute(0, 2, 1).reshape([batch_size, 3, 512, 512])
            materials['face'].textures['albedo'] = albedo.clone()
            if self.eyebrow_builder or self.eyebrow_pca:
                materials['face'].textures['eyebrow'] = eyebrow_background.clone()
            if self.normalmap:
                materials['face'].textures['normalmap'] = nmap.clone()

            materials['eyes'].uvs = Variable(eye_uv_base.clone(), requires_grad=False)
            best_data['recorded'] = {
                'weight': c_weight.clone(),
                'shape': shape.clone(),
                'shape_ex': random_ex_shape.clone(),
                'sweight': s_weight.clone(),
                'eweight': e_weight.clone(),
                'tweight': t_weight.clone(),
                'texture': albedo.clone()
            }
            if self.eyebrow_builder:
                # best_data['recorded']['eyebrow_color'] = eyebrow_color.clone()
                # best_data['recorded']['eyebrow_roots'] = eyebrow_roots.clone()
                # best_data['recorded']['eyebrow_dirs'] = eyebrow_dirs.clone()
                # best_data['recorded']['eyebrow_vertex_colors'] = eyebrow_vertex_colors.clone()
                best_data['recorded']['eyebrow_offset'] = eyebrow_offset.clone()
                best_data['recorded']['eyebrow_root_offset'] = eyebrow_roots.clone()
            elif self.eyebrow_pca:
                best_data['recorded']['eyebrow_weights'] = eyebrow_weights.clone()
                best_data['recorded']['eyebrow_background'] = eyebrow_background_.clone()

            return shape, sh_weight, t_weight, e_weight, random_ex_shape, best_data

        def fit_face_renderer_fn(render_data):
            shape, sh_weight, t_weight, e_weight, random_ex_shape, best_data = render_data

            res = renderer(
                verts=shape.clone(),
                faces=Variable(faces.clone(), requires_grad=False),
                render_size=image_size,
                sh=sh_weight.clone(),
                trans=t_weight,
                materials=materials,
                get_norm=True,
                get_tangent=True,
                get_vert_proj=True,
            )

            img_fake = materials['face'].get_albedo()
            if self.eyebrow_builder or self.eyebrow_pca:
                eyebrow_mask = materials['face'].textures_out['eyebrow']
                eyebrow_mask = eyebrow_mask.clamp(0.0, 1.0)
                if self.eyebrow_pca:
                    eyebrow_color = Variable(best_data['eyebrow_color'].clone(), requires_grad=False).reshape([1, -1, 1, 1])
                    eyebrow = eyebrow_mask * eyebrow_color
                    # eyebrow_mask = eyebrow_mask * 0.5
                else:
                    eyebrow = eyebrow_mask
                img_fake = torch.lerp(img_fake, eyebrow, eyebrow_mask)
            sh_light = res['sh'].clone()
            if self.normalmap:
                normalmap = renderer.normalmapping(materials['face'], res['norm'], materials['face'].textures_out['normalmap']).reshape([batch_size, -1, 3])
                sh_light *= gpmm_illumination_sh(normalmap, sh_weight, self.device).reshape(batch_size, image_size, image_size, 3).permute(0, 3, 1, 2)
            img_fake *= sh_light
            img_fake += materials['eyes'].get_albedo()
            img_fake = torch.clamp(img_fake, min=0.0, max=1.0)

            index = res['norm'] != 0
            combined = img.clone()
            combined[index] = img_fake[index]

            if self.debug:
                try:
                    save_tensor_img(combined, self.debug_output + 'i_sopt.jpg')
                except Exception:
                    print()

            return shape, img_fake, combined, index, res

        def fit_face_loss_fn(renderer_res):
            shape, img_fake, combined, index, res = renderer_res
            shape_flip = shape.clone()
            shape_flip_sub = shape_flip[:, :9409]
            shape_flip_sub = shape_flip_sub[:, flip_idx]
            shape_flip[:, :9409] = shape_flip_sub
            shape_flip = Variable(shape_flip, requires_grad=False)
            flip_loss = torch.nn.SmoothL1Loss()(shape, shape_flip)

            lm_loss = landmark_loss(
                self.sbr_model,
                Variable(heatmap_true, requires_grad=False),
                img_fake.clone(),
                Variable(self.sbr_mean, requires_grad=False),
                Variable(self.sbr_std, requires_grad=False))

            lms = res['vp'][:, lm_3ds, :2]
            lm3d_loss = torch.nn.MSELoss()(lms, lms_gt)

            if_loss, if_content_loss = identity_trainer_loss(insightface_trainer, combined)

            img_ = img.clone()
            pix_loss = torch.nn.L1Loss()(combined[index], Variable(img_[index].clone(), requires_grad=False))

            loss = pix_weight * pix_loss + if_weight * if_loss + content_weight * if_content_loss + \
                lm_weight * lm_loss + flip_weight * flip_loss + lm3d_weight * lm3d_loss
            return loss, (
                pix_weight * pix_loss.item(), if_weight * if_loss.item(),
                content_weight * if_content_loss.item(), flip_weight * flip_loss.item(),
                lm_weight * lm_loss.item(), lm3d_weight * lm3d_loss.item())

        def fit_face_random_identity(rid, if2_losses, if2_content_losses, render_data):
            shape, sh_weight, t_weight, e_weight, random_ex_shape, best_data = render_data
            random_ex_shape_ = random_ex_shape.clone()
            rand_weight = torch.rand(e_weight.shape).type(torch.float32).to(device)
            rand_weight = Variable(rand_weight * 2.0 - 1.0, requires_grad=False)
            random_ex_shape_ = _morph_pca(random_ex_shape_, ex_pca, rand_weight, None)

            rand_sh_weight = torch.ones_like(sh_weight) * 0.1
            rand_sh_weight[:, 0, :] = 0.5
            rand_sh_weight[:, 9, :] = 0.5
            rand_sh_weight[:, 18, :] = 0.5
            # rand_sh_weight = torch.rand(sh_weight.shape).type(torch.float32).to(device)
            # rand_sh_weight = Variable(rand_sh_weight * 0.06 - 0.03, requires_grad=False)
            # rand_sh_weight += 0.1
            fov, g_pos, g_rot = random_pose(rid)
            g_pos = Variable(g_pos, requires_grad=False)
            g_rot = Variable(g_rot, requires_grad=False)
            if self.normalmap:
                random_sh = None
            else:
                random_sh = Variable(rand_sh_weight, requires_grad=False)
            res = renderer(
                verts=(random_ex_shape_ * 0.01),
                faces=Variable(faces.clone(), requires_grad=False),
                render_size=112,
                pos=g_pos,
                rot=g_rot,
                FOV=fov,
                sh=random_sh,
                materials=materials,
                get_norm=True,
                get_tangent=True,
            )
            random_fake = materials['face'].get_albedo()
            if self.eyebrow_builder or self.eyebrow_pca:
                eyebrow_mask = materials['face'].textures_out['eyebrow']
                eyebrow_mask = eyebrow_mask.clamp(0.0, 1.0)
                if self.eyebrow_pca:
                    eyebrow_color = Variable(best_data['eyebrow_color'].clone(), requires_grad=False).reshape([1, -1, 1, 1])
                    eyebrow = eyebrow_mask * eyebrow_color
                    # eyebrow_mask = eyebrow_mask * 0.5
                else:
                    eyebrow = eyebrow_mask
                random_fake = torch.lerp(random_fake, eyebrow, eyebrow_mask)
            if self.normalmap:
                random_sh = Variable(rand_sh_weight, requires_grad=False)
                normalmap = renderer.normalmapping(materials['face'], res['norm'], materials['face'].textures_out['normalmap']).reshape([batch_size, -1, 3])
                light = Variable(torch.zeros_like(normalmap), requires_grad=False)
                light[..., 2] = 1
                sh_light = bvec_dot(normalmap, light).reshape(batch_size, 1, 112, 112)
                # sh_light *= gpmm_illumination_sh(normalmap, random_sh, self.device, True).reshape(batch_size, 112, 112, 3).permute(0, 3, 1, 2)
            else:
                sh_light = res['sh']
            random_fake *= sh_light
            random_fake += materials['eyes'].get_albedo()
            random_fake = torch.clamp(random_fake, min=0.0, max=1.0)

            if self.debug:
                try:
                    save_tensor_img(random_fake, f'{self.debug_output}i_random_{rid}.jpg')
                except Exception:
                    print()

            if2_loss, if2_content_loss = identity_trainer_loss(insightface_trainer, random_fake)
            if rid == 0:
                if2_losses = if2_loss
                if2_content_losses = if2_content_loss
            else:
                if2_losses += if2_loss
                if2_content_losses += if2_content_loss
            return if2_losses, if2_content_losses

        def fit_face_random_identity_loss_fn(loss, loss_log, if2_losses, if2_content_losses):
            pix_loss, if_loss, if_content_loss, flip_loss, lm_loss, lm3d_loss = loss_log
            loss += if2_weight * if2_losses + content2_weight * if2_content_losses
            return loss, (pix_loss, if_loss, if_content_loss, flip_loss, lm_loss, lm3d_loss, if2_weight * if2_losses.item(), content2_weight * if2_content_losses.item())

        def fit_face_log_fn(i, time_sec, loss, loss_log, best_index, best_loss):
            pix_loss, if_loss, if_content_loss, flip_loss, lm_loss, lm3d_loss, if2_losses, if2_content_losses = loss_log
            print(
                '    {:4.2f}s face {}. [loss:{:3.4f} pix:{:3.4f} if:{:3.4f} ifr:{:3.4f} if_c:{:3.4f} if2_c:{:3.4f}, flip:{:3.4f}, lm:{:3.4f}, lm3d:{:3.4f}] best_{}:{:3.4f}'
                .format(time_sec, i, loss, pix_loss, if_loss, if2_losses,
                        if_content_loss, if2_content_losses, flip_loss,
                        lm_loss, lm3d_loss, best_index, best_loss),
                end='\r')

        # wait for gpu memory release
        sleep(0.1)
        # ======================== main ========================
        st = datetime.datetime.now()
        # ======== fit lm ========
        loss, time_sec, best_loss = self._fitter(
            fit_lm_first, best_data, lm3d_optimizer,
            fit_lm_render_data, fit_lm_renderer, fit_lm_loss_fn)

        print('    {:4.2f}s lm  {}. [loss:{:3.4f}] best_{}:{:3.4f}'.format(
            time_sec, fit_lm_first, loss.item(), best_data['i'], best_loss))
        if self.eyebrow_builder:
            image_size = 112
            img = F.interpolate(input_image, size=image_size)
            best_eyebrow_index = self._search_eyebrow_index(img, best_data, faces, image_size, insightface_trainer)
            eyebrow_shape = self.all_eyebrow_roots[best_eyebrow_index].shape
            best_data['eyebrow_roots'] = self.all_eyebrow_roots[best_eyebrow_index]
            best_data['eyebrow_dirs'] = self.all_eyebrow_dirs[best_eyebrow_index]
            eyebrowColorFitter = WeightFitter(0, batch_size, device, np.ones([3]) * 0.2)
            eyebrowVertexPosFitter = WeightFitter(0, batch_size, device, np.zeros([batch_size, eyebrow_shape[1], 3]))
            eyebrowVertexDirFitter = WeightFitter(0, batch_size, device, np.ones([batch_size, eyebrow_shape[1], 2]) * 0.01)
            eyebrowVertexColorsFitter = WeightFitter(0, batch_size, device, np.ones([batch_size, eyebrow_shape[1], 1]) * 0.2)
            eyebrowOffsetFitter = WeightFitter(0, batch_size, device, np.zeros([batch_size, 1, 3]))

        # ======== fitting ========
        image_size = self.image_size
        img = F.interpolate(input_image, size=image_size)

        param_datas = [
            {'params': shapeFitter.parameters(), 'lr': lr * 40.0, 'tag': 's'},
            {'params': exFitter.parameters(), 'lr': lr * 4.0, 'tag': 'ex'},
            {'params': shFitter.parameters(), 'lr': lr * 0.01, 'tag': 'sh'},
            {'params': transFitter.parameters(), 'lr': lr * 3.0, 'tag': 't'},
        ]
        param_datas.append({'params': colorFitter.parameters(), 'lr': lr * 8.0, 'tag': 'a'})
        param_datas.append({'params': coloroffsetFitter.parameters(), 'lr': lr * 15.0, 'tag': 'aoff'})
        if self.eyebrow_builder:
            param_datas.append({'params': eyebrowOffsetFitter.parameters(), 'lr': lr * 0.1, 'tag': 'eyebrowOffset'})
            # param_datas.append({'params': eyebrowVertexPosFitter.parameters(), 'lr': lr * 0.1, 'tag': 'eyebrowVertexPos'})
            # param_datas.append({'params': eyebrowVertexDirFitter.parameters(), 'lr': lr * 0.1, 'tag': 'eyebrowVertexDir'})
            # param_datas.append({'params': eyebrowVertexColorsFitter.parameters(), 'lr': lr * 2.0, 'tag': 'eyebrowVertexColor'})
            best_data['eyebrow_vertex_colors'] = eyebrowVertexColorsFitter()
        elif self.eyebrow_pca:
            eyebrowColorFitter = WeightFitter(0, batch_size, device, np.ones([3]) * 0.1)
            best_data['eyebrow_color'] = eyebrowColorFitter()
            eyebrowWeightFitter = WeightFitter(self.eyebrow_weight_count, batch_size, device)
            param_datas.append({'params': eyebrowWeightFitter.parameters(), 'lr': lr * 0.2, 'tag': "eyebrowWeight"})

        optimizer = torch.optim.Adam(param_datas, lr=lr, betas=(0.8, 0.999))

        _, _, best_loss = self._fitter(
            self.step_size, best_data, optimizer,
            fit_face_render_data, fit_face_renderer_fn, fit_face_loss_fn,
            fit_face_log_fn, fit_face_random_identity, fit_face_random_identity_loss_fn)

        print('    best index : {}, best loss : {:3.4f}'.format(best_data['i'], best_loss))
        print('    random identity loss:', best_data['ifr'])

        # ===== fit eyebrow =====
        if self.eyebrow_builder:
            # image_size = 112
            # img = F.interpolate(input_image, size=image_size)
            # best_eyebrow_index = self._search_eyebrow_index(img, best_data, faces, image_size, insightface_trainer)

            image_size = 512
            img = F.interpolate(input_image, size=image_size)
            eyebrowOffsetFitter = WeightFitter(0, batch_size, device, np.zeros([batch_size, 1, 3]))

            param_datas = [
                {'params': eyebrowColorFitter.parameters(), 'lr': lr * 5.0, 'tag': 'eyebrowColor'},
                {'params': eyebrowOffsetFitter.parameters(), 'lr': lr * 1e-8, 'tag': 'eyebrowOffset'},
                {'params': eyebrowVertexPosFitter.parameters(), 'lr': lr * 0.01, 'tag': 'eyebrowVertexPos'},
                {'params': eyebrowVertexDirFitter.parameters(), 'lr': lr * 0.05, 'tag': 'eyebrowVertexDir'},
                {'params': eyebrowVertexColorsFitter.parameters(), 'lr': lr * 0.3, 'tag': 'eyebrowVertexColor'},
            ]
            eyebrow_optimizer = torch.optim.Adam(param_datas, lr=lr, betas=(0.8, 0.999))

            _, _, _ = self._fitter(50, best_data, eyebrow_optimizer,
                                   fit_eyebrow_render_data,
                                   fit_eyebrow_renderer_fn,
                                   fit_eyebrow_loss_fn, fit_eyebrow_log_fn)

        # ===== fit color =====
        image_size = 112
        img = F.interpolate(input_image, size=image_size)

        # eyebrowColorFitter = WeightFitter(0, batch_size, device, np.ones([3]) * 0.5)
        param_datas = [
            {'params': coloroffsetFitter.parameters(), 'lr': lr * 15.0, 'tag': 'aoff'},
            {'params': skin_color_Fitter.parameters(), 'lr': lr * 15.0, 'tag': 'skincolor'},
            {'params': shFitter.parameters(), 'lr': lr * 0.01, 'tag': 'sh'},
        ]
        if self.eyebrow_builder:
            param_datas.append({'params': eyebrowColorFitter.parameters(), 'lr': lr * 15.0, 'tag': 'eyebrowColor'})
        elif self.eyebrow_pca:
            param_datas.append({'params': eyebrowColorFitter.parameters(), 'lr': lr * 0.5, 'tag': 'eyebrowColor'})
        color_optimizer = torch.optim.Adam(param_datas, lr=lr, betas=(0.8, 0.999))

        self._fitter(
            50, best_data, color_optimizer, fit_color_render_data,
            fit_color_renderer_fn, fit_color_loss_fn,
            fit_color_log_fn)

        print('    totoal time : {:4.2f}s'.format((datetime.datetime.now() - st).total_seconds()))

        eyebrow_norm = None
        eyebrow_color = None
        if self.eyebrow_builder:
            eyebrow_color = best_data['eyebrow_color'].clone()
            eyebrow_roots = best_data['eyebrow_roots'].clone()
            eyebrow_dirs = best_data['eyebrow_dirs'].clone()
            eyebrow_vertex_colors = best_data['eyebrow_vertex_colors'].clone()
            eyebrow_vertex_colors = eyebrow_vertex_colors.repeat([1, 1, 3])
            eyebrow_vertex_colors = eyebrow_vertex_colors * eyebrow_color
            eyebrow_vertex_colors = eyebrow_vertex_colors.unsqueeze(-2).repeat([1, 1, 4, 1]).reshape([1, -1, 3])
            eyebrow_out = self._build_eyebrow_texture(eyebrow_roots, eyebrow_vertex_colors, eyebrow_dirs.clone(), None, self.texture_size)
            normal = eyebrow_dirs.clone() * 100.0
            # normal[..., 2] = 1.0
            normal = F.normalize(normal, eps=1e-6, dim=-1)
            normal = normal.unsqueeze(-2).repeat([1, 1, 4, 1]).reshape([1, -1, 3])
            eyebrow_norm = self._build_eyebrow_texture(eyebrow_roots, normal, eyebrow_dirs.clone(), None, self.texture_size)
            # save_tensor_img(eyebrow_norm.clone(), './results/output/eyebrow_norm.jpg')
            eyebrow_norm = eyebrow_norm.permute(0, 2, 3, 1)
            eyebrow_out = eyebrow_out.permute(0, 2, 3, 1)
        elif self.eyebrow_pca:
            eyebrow_color = best_data['eyebrow_color'].clone()
            eyebrow_out = _morph_pca(self.eyebrow_full_mean.clone(), self.eyebrow_full_diffs.clone(), best_data['eyebrow_weights'].clone(), None)
            eyebrow_norm = _morph_pca(self.eyebrow_nor_mean.clone(), self.eyebrow_nor_diffs.clone(), best_data['eyebrow_weights'].clone(), None)
        else:
            eyebrow_out = None

        return {
            'color': best_data['weight'],
            'color_offset': to_np(best_data['color_offset']),
            'shape': best_data['sweight'],
            'ex': to_np(best_data['eweight']),
            'sh': to_np(best_data['shweight']),
            'tran': to_np(best_data['tweight']),
            'eyebrow': eyebrow_out,
            'eyebrow_color': eyebrow_color,
            'eyebrow_norm': eyebrow_norm,
        }

    def _convert_uv_space(self, tensor):
        texture_size = self.texture_size
        uv_faces = self.face_model['uv_reference_faces']
        uv_verts = self.face_model['uv_verts']
        uv_to = self.face_model['uv_to']

        textures = {}
        textures['albedo'] = tensor.to(self.device).reshape([1, texture_size, texture_size, 3]).permute(0, 3, 1, 2)
        materials = {}
        materials['face'] = Material(uv_faces, uv_to, textures)
        self.renderer(render_size=texture_size,
                      verts=uv_verts,
                      faces=uv_faces,
                      materials=materials)
        tensor = materials['face'].get_albedo().permute(0, 2, 3, 1)
        return torch.flip(tensor, [1])

    def _save_texture(self, nparr, path):
        tensor = torch.from_numpy(nparr).type(torch.float32).to(self.device)
        tensor = self._convert_uv_space(tensor)
        tensor = tensor.reshape([self.texture_size, self.texture_size, 3]).cpu().detach().numpy()
        tensor = (tensor * 255.0).astype(np.uint8)
        Image.fromarray(tensor).save(path)

    def _gen_texture(self, d, output_path, output_dict):
        weights = d['color']
        color_offset = d['color_offset']
        eyebrow = d['eyebrow']
        if eyebrow is not None:
            eyebrow = eyebrow.reshape([-1, 3])
        small_a_mean, small_a_diffs, small_n_mean, small_n_diffs, small_a2_mean, small_a2_diffs, a_mean, a_diffs, n_mean, n_diffs, s_mean, s_diffs = self.loaded_texture

        weights = weights[0].cpu().detach().numpy()
        albedo = _morph_pca_np(a_mean, a_diffs, weights)

        albedo *= color_offset
        if self.eyebrow_builder:
            albedo = torch.from_numpy(albedo).to(self.device)
            eyebrow = eyebrow.clamp(0.0, 1.0)
            eyebrow_mask = eyebrow
            albedo = torch.lerp(albedo, eyebrow,
                                eyebrow).cpu().detach().numpy()
        elif self.eyebrow_pca:
            albedo = torch.from_numpy(albedo).to(self.device)
            eyebrow_mask = eyebrow.clamp(0.0, 1.0)
            eyebrow_color = d['eyebrow_color']
            eyebrow = eyebrow_mask * eyebrow_color
            albedo = torch.lerp(albedo, eyebrow, eyebrow_mask).cpu().detach().numpy()
        albedo = albedo.reshape([self.texture_size, self.texture_size, 3])  # .cpu().detach().numpy()

        normal = _morph_pca_np(n_mean, n_diffs, weights)

        if self.eyebrow_builder:
            eyebrow_norm = d['eyebrow_norm'].reshape([-1, 3])
            eyebrow_area = eyebrow_norm[eyebrow_norm != 0]
            eyebrow_area = eyebrow_area * 0.5 + 0.5
            normal = torch.from_numpy(normal).to(self.device)
            normal = torch.lerp(normal, eyebrow_norm, eyebrow_mask).cpu().detach().numpy()
        elif self.eyebrow_pca:
            eyebrow_norm = d['eyebrow_norm'].reshape([-1, 3])
            eyebrow_norm = eyebrow_norm.cpu().detach().numpy()
            eyebrow_mask = eyebrow_mask.cpu().detach().numpy()
            normal[eyebrow_mask != 0] = eyebrow_norm[eyebrow_mask != 0]

        normal_path = os.path.join(output_path, 'normal.png')
        self._save_texture(np.clip(normal, 0.0, 1.0), normal_path)
        output_dict[normal_path] = 'normal.png'
        specular = _morph_pca_np(s_mean, s_diffs, weights)

        albedo_path = os.path.join(output_path, 'albedo.png')
        self._save_texture(np.clip(albedo.reshape([-1, 3]), 0.0, 1.0), albedo_path)
        output_dict[albedo_path] = 'albedo.png'
        if self.eyebrow_builder:
            specular = torch.from_numpy(specular).to(self.device)
            specular = torch.lerp(specular, torch.ones_like(specular) * 0.9, eyebrow).cpu().detach().numpy()
        specular_path = os.path.join(output_path, 'cavity.png')
        self._save_texture(np.clip(specular, 0.0, 1.0), specular_path)
        output_dict[specular_path] = 'cavity.png'

    def _make_eye(self, orig_shape, shape, output_path, output_dict):
        eye_mesh = self.face_model['eye_mesh']
        eyelen_mesh = self.face_model['eyelen_mesh']
        eye_left_faces = self.face_model['eye_left_faces']
        eye_right_faces = self.face_model['eye_right_faces']
        eye_verts = eye_mesh['orig_verts'].copy()
        eyelen_verts = eyelen_mesh['orig_verts'].copy()
        orig_shape *= 0.001
        shape *= 0.001
        # left eye
        # where x is pos
        # print(shape[eye_left_faces])
        # print(shape[eye_left_faces].shape)
        move = (shape[eye_left_faces] - orig_shape[eye_left_faces]).mean(0)
        eye_verts[eye_mesh['left_index']] += move
        eyelen_verts[eyelen_mesh['left_index']] += move

        # right eye
        # where x is nei
        # print(shape[eye_right_faces])
        # print(shape[eye_right_faces].shape)
        move = (shape[eye_right_faces] - orig_shape[eye_right_faces]).mean(0)
        eye_verts[eye_mesh['right_index']] += move
        eyelen_verts[eyelen_mesh['right_index']] += move

        eye_mesh['verts'] = eye_verts
        eyelen_mesh['verts'] = eyelen_verts

        mesh_path = os.path.join(output_path, 'eye.obj')
        write_obj(mesh_path, eye_mesh)
        output_dict[mesh_path] = 'eye.obj'
        mesh_path = os.path.join(output_path, 'eyelen.obj')
        write_obj(mesh_path, eyelen_mesh)
        output_dict[mesh_path] = 'eyelen.obj'

    def _make_mesh(self, d, output_path, output_dict):
        id_mean = self.face_model['orig_shape']
        id_pca = self.face_model['id_pca']
        out_mesh = self.face_model['out_mesh']
        corr_bc = self.face_model['corr_bc']
        corr_tri = self.face_model['corr_tri']
        s_weight = d['shape']
        shape = _morph_pca(id_mean, id_pca, s_weight, None).cpu().detach().numpy()[0]

        out_verts = []
        for i in range(corr_bc.shape[0]):
            bc = corr_bc[i]
            tris = corr_tri[i]
            v0 = shape[tris[0]] * bc[0]
            v1 = shape[tris[1]] * bc[1]
            v2 = shape[tris[2]] * bc[2]
            out_verts.append(v0 + v1 + v2)
        out_verts = np.array(out_verts)
        out_verts *= 0.001
        out_mesh['verts'][:out_verts.shape[0], ] = out_verts
        mesh_path = os.path.join(output_path, 'mesh.obj')
        write_obj(mesh_path, out_mesh)
        output_dict[mesh_path] = 'mesh.obj'

        self._make_eye(id_mean.cpu().detach().numpy()[0], shape, output_path, output_dict)

    def preprocess(self, img_path):
        datas = mgc_preprocess(img_path, self.fa)
        if datas is None:
            return ('no face detected',)
        with torch.no_grad():
            coef = self.mgc_model(F.interpolate(datas['img'].clone().to(self.device), size=224))
        lm68 = torch.from_numpy(datas['lm']).unsqueeze(0).type(torch.float32)

        return None, lm68, datas['img'], coef

    def generate(self, img_path):
        data = self.preprocess(img_path)
        if data[0] is not None:
            return {'err': data[0]}
        _, lm68, img, coef = data
        _N, _C, H, W = img.shape
        lm68[..., 0] /= W
        lm68[..., 1] /= H
        result = self._fit_face(img, lm68, coef)

        return {'result': result}

    def save_results(self, output_path, datas):
        output_dict = {'parent': output_path}
        d = datas['result']
        self._make_mesh(d, output_path, output_dict)
        self._gen_texture(d, output_path, output_dict)
        return output_dict


def gen_face_datas(root_path, device, logger, debug=False):
    return FaceGenerator(root_path, debug)


def test_generation(root_path, img_path, output_path='./results/output/'):
    import sys
    debug = True
    from .xravatar.utils import create_logger
    logger = create_logger(None)
    if '-p' in sys.argv:
        logger.info('production mode')
        debug = False
    if '-t' in sys.argv:
        logger.info('generate default texture')
        gen_default_texture(root_path, logger)
        return
    if '-convert' in sys.argv:
        logger.info('convert texture (input_texture.jpg)')
        convert_texture(root_path, logger)
        return

    os.makedirs(output_path, exist_ok=True)

    st_time = datetime.datetime.now()
    face_generator = gen_face_datas(root_path, 'cuda', logger, debug=debug)
    if debug:
        face_generator.debug_output = output_path
    results = face_generator.generate(img_path)
    # output_path = './results/output/'
    dict_data_out = face_generator.save_results(output_path, results)
    logger.info('dict data out :{}'.format(dict_data_out))
    logger.info('total time : {}s'.format((datetime.datetime.now() - st_time).total_seconds()))


def test_generation_list(root_path, img_paths, output_paths):
    import sys
    debug = True
    from .xravatar.utils import create_logger
    logger = create_logger(None)
    if '-p' in sys.argv:
        logger.info('production mode')
        debug = False
    if '-t' in sys.argv:
        logger.info('generate default texture')
        gen_default_texture(root_path, logger)
        return
    if '-convert' in sys.argv:
        logger.info('convert texture (input_texture.jpg)')
        convert_texture(root_path, logger)
        return

    st_time = datetime.datetime.now()
    face_generator = gen_face_datas(root_path, 'cuda', logger, debug=debug)

    for i in range(len(img_paths)):
        img_path = img_paths[i]
        output_path = output_paths[i]
        print(f'img: {img_path}\noutput: {output_path}')
        os.makedirs(output_path, exist_ok=True)

        if debug:
            face_generator.debug_output = output_path
        results = face_generator.generate(img_path)
        # output_path = './results/output/'
        dict_data_out = face_generator.save_results(output_path, results)
        logger.info('dict data out :{}'.format(dict_data_out))
        logger.info('total time : {}s'.format((datetime.datetime.now() - st_time).total_seconds()))


def gen_default_texture(root_path, logger):
    face_generator = gen_face_datas(root_path, 'cuda', logger, debug=False)
    d = {}
    d['color'] = torch.zeros([1, 48, 1]).to('cuda')
    d['color_offset'] = np.ones([3])
    face_generator._gen_texture(d, './', {})


def convert_texture(root_path, logger):
    image = Image.open('./input_texture.png').convert('RGB')
    tex = np.array(image).astype(np.float32) / 255.0
    face_generator = gen_face_datas(root_path, 'cuda', logger, debug=False)
    face_generator._save_texture(tex, './output_texture.png')
