from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pathlib import Path
from typing import Union


def vertex_normals(vertices: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute vertex normal
    reference: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/structures/meshes.py#L868

    Parameters
    ----------
        vertices: torch.Tensor (B, V, 3)
        faces: torch.LongTensor (B, F, 3)

    Returns
    -------
        vertex normals: torch.Tensor (B, V, 3)
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    assert (faces.dtype == torch.int64)

    bs, nv = vertices.shape[:2]
    bs, _nf = faces.shape[:2]
    device = vertices.device
    normals = torch.zeros(bs * nv, 3).to(device)

    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]  # expanded faces
    vertices_faces = vertices.reshape((bs * nv, 3))[faces]

    faces = faces.view(-1, 3)
    vertices_faces = vertices_faces.view(-1, 3, 3)

    normals.index_add_(
        0,
        faces[:, 1],
        torch.cross(vertices_faces[:, 2] - vertices_faces[:, 1], vertices_faces[:, 0] - vertices_faces[:, 1])
    )
    normals.index_add_(
        0,
        faces[:, 2],
        torch.cross(vertices_faces[:, 0] - vertices_faces[:, 2], vertices_faces[:, 1] - vertices_faces[:, 2])
    )
    normals.index_add_(
        0,
        faces[:, 0],
        torch.cross(vertices_faces[:, 1] - vertices_faces[:, 0], vertices_faces[:, 2] - vertices_faces[:, 0])
    )

    normals = F.normalize(normals, eps=1e-6, dim=1)
    normals = normals.reshape((bs, nv, 3))
    return normals


def face_vertices(vertices: torch.Tensor, faces: torch.LongTensor) -> torch.Tensor:
    """Compute face vertices.

    Parameters
    ----------
        vertices: torch.Tensor (B, V, 3)
        faces: torch.LongTensor (B, F, 3)

    Returns
    -------
        face_vertices: torch.Tensor (B, F, 3, 3)
    """
    assert (vertices.ndimension() == 3)
    assert (faces.ndimension() == 3)
    assert (vertices.shape[0] == faces.shape[0])
    assert (vertices.shape[2] == 3)
    assert (faces.shape[2] == 3)
    assert (faces.dtype == torch.int64)

    bs, nv = vertices.shape[:2]
    bs, _nf = faces.shape[:2]
    device = vertices.device
    faces = faces + (torch.arange(bs, dtype=torch.int32).to(device) * nv)[:, None, None]
    vertices = vertices.reshape((bs * nv, 3))
    return vertices[faces]


def morph_vertex(
    base: torch.Tensor,
    bs: torch.Tensor,
    coeff: torch.Tensor,
) -> torch.Tensor:
    '''
    Morph blendshapes.
    Parameters:
    ----------
        base: torch.Tensor, base shape [N, 3] or [B, N, 3]
        bs: torch.Tensor, blendshapes [BS, N, 3]
        coeff: torch.Tensor, blendshape coefficient [B, BS]

    Returns:
    --------
        vertices: torch.Tensor, morphed vertices [B, N, 3]
    '''
    B = coeff.shape[0]
    if len(base.shape) != 3:
        vertices = base.clone().unsqueeze(0).expand(B, -1, -1)
    else:
        vertices = base
    vertices = vertices + torch.einsum('bi,bijk->bjk', coeff, bs.unsqueeze(0).expand(B, -1, -1, -1))  # noqa:E501
    return vertices


def morph_vertex_BCN(
    base: torch.Tensor,
    bs: torch.Tensor,
    coeff: torch.Tensor,
) -> torch.Tensor:
    '''
    Morph blendshape with vertex shape [B, 3, N].
    Parameters:
    ----------
        base: torch.Tensor, base shape [3, N] or [B, 3, N]
        bs: torch.Tensor, blendshapes [BS, 3, N]
        coeff: torch.Tensor, blendshape coefficient [B, BS]

    Returns:
    --------
        vertices: torch.Tensor, morphed vertices [B, 3, N]
    '''
    B = coeff.shape[0]
    if len(base.shape) != 3:
        vertices = base.clone().unsqueeze(0).expand(B, -1, -1)
    else:
        vertices = base
    vertices = vertices + torch.sum(bs.unsqueeze(0).expand(B, -1, -1, -1) * coeff.view(B, -1, 1, 1), dim=1)
    return vertices


def morph_texture(
    base: torch.Tensor,
    bs: torch.Tensor,
    coeff: torch.Tensor,
) -> torch.Tensor:
    '''
    Morph blendshape with vertex shape [B, 3, N].
    Parameters:
    ----------
        base: torch.Tensor, base texture [3, 512, 512] or [1, 3 * 512 * 512]
        bs: torch.Tensor, blendshapes [BS, 3 * 512 * 512]
        coeff: torch.Tensor, blendshape coefficient [B, BS]
    Returns:
    --------
        texture: torch.Tensor, morphed texture [B, 3, 512, 512]
    '''
    B = coeff.shape[0]
    if len(base.shape) != 2:
        texture = base.reshape([1, -1]).repeat(B, 1)
    elif base.shape[0] == 1:
        texture = base.repeat(B, 1)
    else:
        texture = base
    if len(bs.shape) != 3:
        bs = bs.reshape([1, texture.shape[1], -1]).expand(B, -1, -1)
    texture = texture + torch.einsum('bi,bji->bj', coeff, bs)
    texture = texture.reshape(B, 3, 512, 512).clamp(0.0, 1.0)
    return texture


class TDMM(nn.Module):
    '''
    3DMM base class
    '''
    @abstractmethod
    def __init__(self):
        super().__init__()
        # self.base = None
        # self.faces = None
        # self.bs_shape = None
        # self.exp_base = None
        # self.bs_exp = None

        # self.base_color = None
        # self.bs_color = None

        # self.skin_mask = None
        # self.lm_idx = None

        self.shape_morph = morph_vertex
        self.exp_morph = morph_vertex
        self.color_morph = morph_vertex
        self.preprocess_exp = None

    def compute_shape(
        self,
        id_coef: torch.Tensor,
        exp_coef: torch.Tensor,
    ) -> torch.Tensor:
        vertices = self.shape_morph(self.base, self.bs_shape, id_coef)
        if self.exp_base is not None:
            vertices = vertices + self.exp_base.clone().unsqueeze(0)
        vertices = self.exp_morph(vertices, self.bs_exp, exp_coef)
        return vertices

    def compute_color(self, color_coef: torch.Tensor) -> torch.Tensor:
        color = self.color_morph(self.base_color, self.bs_color, color_coef)
        return color

    @abstractmethod
    def get_landmarks(self, face_proj_vertices: torch.Tensor) -> torch.Tensor:
        landmarks = face_proj_vertices[:, self.lm_idx]
        return landmarks

    def preprocess_coeff(self, coef_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if self.preprocess_exp is not None:
            coef_dict['exp'] = self.preprocess_exp(coef_dict['exp'])
        return coef_dict

    def forward(self, coef_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        res = {}
        res['vertices'] = self.compute_shape(coef_dict['shape'], coef_dict['exp'])
        res['orig_verts'] = res['vertices'].clone()

        res['faces'] = self.faces[None, ...].expand(coef_dict['B'], -1, -1)

        res['face_norm'] = vertex_normals(res['vertices'], res['faces'])

        if hasattr(self, 'base_color') and 'tex' in coef_dict:
            res['color'] = self.compute_color(coef_dict['tex'])

        if self.face_uvs is not None:
            res['face_uvs'] = self.face_uvs

        if self.uvs is not None:
            res['uvs'] = self.uvs

        if self.uv_faces is not None:
            res['uv_faces'] = self.uv_faces

        return res


class ICTFaceKit(TDMM):
    def __init__(
        self,
        file: Union[str, Path] = None,
        texture_file: Union[str, Path] = None,
        head_type: str = 'with_ear',
        add_inner_mouth: bool = False,
        add_eye_ball: bool = False,
        add_teeth: bool = False,
        add_exp_atan: bool = False,
        recenter: bool = True,
        add_eye_lm: bool = False,
        num_coef_shape: int = 100,
        num_coef_exp: int = 53 - 6,
        num_coef_color: int = 80,
        orig_uv: bool = False,
        is_train: bool = False,
    ) -> None:
        super().__init__()

        self.add_exp_atan = add_exp_atan
        self.add_inner_mouth = add_inner_mouth
        self.add_teeth = add_teeth
        self.add_eye_ball = add_eye_ball
        self.head_type = head_type

        if self.head_type == 'full':
            self.num_head = 28068
        elif self.head_type == 'with_ear':
            self.num_head = 18460
        else:
            self.num_head = 13120

        self.num_faces = self.num_head
        if self.add_eye_ball:
            self.num_faces = self.num_faces + 1536 + 1536
        if self.add_inner_mouth:
            self.num_faces = self.num_faces + 5808
        if self.add_teeth:
            self.num_faces = self.num_faces + 8696
        self.num_verts = 26719
        num_lms = 68
        if add_eye_lm:
            num_lms = 70

        self.num_shape = 100
        self.num_exp = 53
        self.num_color = 200

        self.num_coef_shape = num_coef_shape
        self.num_coef_exp = num_coef_exp
        self.num_coef_color = num_coef_color

        self.register_buffer('faces', torch.zeros(self.num_faces, 3, dtype=torch.int64))
        self.register_buffer('base', torch.zeros(self.num_verts, 3))

        self.register_buffer('bs_shape', torch.zeros(self.num_coef_shape, self.num_verts, 3))
        self.exp_base = None
        self.register_buffer('bs_exp', torch.zeros(self.num_exp, self.num_verts, 3))

        self.register_buffer('lm_idx', torch.zeros(num_lms, dtype=torch.int64))

        if orig_uv:
            self.num_uvs = 28911
        else:
            self.num_uvs = self.num_verts
        self.register_buffer('uvs', torch.zeros([self.num_uvs, 2]))
        self.register_buffer('uvcoords', torch.zeros([1, self.num_uvs, 3]))
        self.register_buffer('uv_faces', torch.zeros([self.num_faces, 3], dtype=torch.int64))
        self.register_buffer('face_uvs', torch.zeros([1, self.num_faces, 3, 3]))

        if texture_file is not None:
            tex_size = 512
            self.register_buffer('base_color', torch.zeros(1, tex_size * tex_size * 3))
            self.register_buffer('bs_color', torch.zeros(tex_size * tex_size * 3, self.num_coef_color))

        if is_train:
            self.register_buffer('skin_mask', torch.zeros([512, 512]))
            self.register_buffer('eye_face', torch.zeros([512, 3], dtype=torch.int64))
            self.register_buffer('eye_ball_face', torch.zeros([1536 + 1536, 3], dtype=torch.int64))
            self.register_buffer('mouth_face', torch.zeros([5808, 3], dtype=torch.int64))

        self.color_morph = morph_texture
        self.preprocess_exp = self.process_exp

        if file is not None and Path(file).suffix == '.npz':
            self.load_original_file(
                file,
                texture_file,
                recenter,
                is_train,
                add_eye_lm,
                orig_uv,
            )
            print('ICTFaceKit loaded')

    def load_original_file(
        self,
        file: Union[str, Path],
        texture_file: Union[str, Path],
        recenter: bool,
        is_train: bool,
        add_eye_lm: bool,
        orig_uv: bool,
    ) -> None:
        print(f'loading: {file}')
        file = Path(file).expanduser()
        content = np.load(str(file), allow_pickle=True)

        face_ls = []
        face_ls.append(content['faces'][:self.num_head, :])
        if self.add_inner_mouth:
            face_ls.append(content['faces'][28068:33876, :])
        if self.add_teeth:
            face_ls.append(content['faces'][33876:42572, :])
        if self.add_eye_ball:
            face_ls.append(content['faces'][42572:44108, :])
            face_ls.append(content['faces'][45704:47240, :])
        faces = np.concatenate(face_ls, 0)

        base = content['verts'][:self.num_verts, :]
        base = base * 0.1
        if recenter:
            base = base - base.mean(0)

        bs_shape = content['bs_shape'][:self.num_coef_shape, :self.num_verts, :]
        bs_shape = bs_shape * 0.05

        bs_exp = content['bs_exp'][:self.num_exp, :self.num_verts, :]
        bs_exp = bs_exp * 0.1

        if orig_uv and 'orig_uv' in content and 'orig_uv_faces' in content:
            uvs = content['orig_uv']
            face_ls = []
            face_ls.append(content['orig_uv_faces'][:self.num_head, :])
            if self.add_inner_mouth:
                face_ls.append(content['orig_uv_faces'][28068:33876, :])
            if self.add_teeth:
                face_ls.append(content['orig_uv_faces'][33876:42572, :])
            if self.add_eye_ball:
                face_ls.append(content['orig_uv_faces'][42572:44108, :])
                face_ls.append(content['orig_uv_faces'][45704:47240, :])
            uv_faces = np.concatenate(face_ls, 0)
        else:
            uv_faces = faces
            uvs = content['uvs'][:self.num_verts, :]

        lm = content['lms']
        if add_eye_lm:
            eye_lm = np.zeros([70], dtype=np.int64)
            eye_lm[:68] = lm
            eye_lm[68] = 23021
            eye_lm[69] = 21451
            lm = eye_lm

        state_dict = {
            'faces': torch.LongTensor(faces),
            'base': torch.Tensor(base),
            'bs_shape': torch.Tensor(bs_shape),
            'bs_exp': torch.Tensor(bs_exp),
            'lm_idx': torch.LongTensor(lm),
            'uvs': torch.Tensor(uvs),
            'uv_faces': torch.LongTensor(uv_faces),
        }

        if texture_file is not None:
            print(f'loading: {texture_file}')
            texture_file = Path(texture_file).expanduser()
            tex_space = np.load(str(texture_file), allow_pickle=True)
            eye_tex = content['eye_tex']
            eye_mask = eye_tex != 0
            if 'mean' in tex_space:
                color_base = tex_space['mean'][..., ::-1] / 255.0
            elif 'MU' in tex_space:
                color_base = tex_space['MU'].reshape([512, 512, 3])[..., ::-1]
            if 'tex_dir' in tex_space:
                self.num_color = 200
                bs_color = tex_space['tex_dir'][..., ::-1, :] / 255.0
            elif 'PC' in tex_space:
                self.num_color = 199
                bs_color = tex_space['PC'].reshape([512, 512, 3, -1])[..., ::-1, :]
            color_base[eye_mask] = eye_tex[eye_mask]
            bs_color[eye_mask, :] = 0
            color_base = np.transpose(color_base, (2, 0, 1))
            bs_color = np.transpose(bs_color, (2, 0, 1, 3))
            color_base = color_base.reshape(1, -1)
            bs_color = bs_color.reshape(-1, self.num_color)[:, :self.num_coef_color]
            state_dict['base_color'] = torch.Tensor(color_base)
            state_dict['bs_color'] = torch.Tensor(bs_color)

        if is_train:
            skin_mask = torch.Tensor(content['skin_mask_texture'])
            # skin_mask = F.interpolate(skin_mask[None, None, ...], [256, 256])[0, 0, ...]
            state_dict['skin_mask'] = skin_mask
            eye_faces = np.concatenate([
                content['faces'][49548:50060, :]
            ], 0)
            state_dict['eye_face'] = torch.LongTensor(eye_faces)
            eye_ball_faces = np.concatenate([
                content['faces'][42572:44108, :],
                content['faces'][45704:47240, :]
            ], 0)
            state_dict['eye_ball_face'] = torch.LongTensor(eye_ball_faces)

            state_dict['mouth_face'] = torch.LongTensor(content['faces'][28068:33876, :])

        raw_uvcoords = state_dict['uvs'][None, ...].clone()
        raw_uvcoords = raw_uvcoords * 2.0 - 1.0
        raw_uvcoords[..., 1] = -raw_uvcoords[..., 1]
        uvcoords = torch.cat([raw_uvcoords, torch.ones_like(raw_uvcoords[:, :, 0:1])], -1)  # [bz, ntv, 3]
        state_dict['uvcoords'] = uvcoords
        uvfaces = state_dict['uv_faces'][None, ...]
        face_uvs = face_vertices(uvcoords, uvfaces)
        state_dict['face_uvs'] = face_uvs

        self.load_state_dict(state_dict)

    def process_exp(self, exp_coeffs: torch.Tensor) -> torch.Tensor:
        if self.add_exp_atan:
            exp_coeffs = torch.atan(exp_coeffs)

        res = torch.zeros([exp_coeffs.shape[0], self.num_exp]).to(exp_coeffs.device)
        res_idx = [
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
            20, 21, 22, 23, 24, 26, 28, 29, 30, 31,
            32, 33, 35, 36, 37, 38, 39, 41, 42, 43,
            44, 45, 46, 47, 48, 49, 50, 51, 52
        ]
        res[:, res_idx] = exp_coeffs[:, :41]
        res_neg_idx = [
            12, 13, 14, 15, 25, 34
        ]
        res_pos_idx = [
            18, 19, 16, 17, 27, 40
        ]
        combined = exp_coeffs[:, 41:]
        neg = -combined
        pos = combined
        res[:, res_neg_idx] = neg.clamp(min=0.0)
        res[:, res_pos_idx] = pos.clamp(min=0.0)
        res = res.clamp(min=-0.25)

        return res
