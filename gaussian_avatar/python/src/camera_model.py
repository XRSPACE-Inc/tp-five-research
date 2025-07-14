from typing import Dict, Tuple

import torch
from torch import nn
import numpy as np


def perspective_projection(focal: float, center: float) -> np.ndarray:
    '''
    Compute perspective projection from focal and center position.

    Parameters:
    -----------
        focal: float, focal length.
        center: float, center of image.'

    Returns:
    --------
        projection: np.ndarray, projection matrix. [3, 3]
    '''
    return np.array([
        focal, 0, center,
        0, focal, center,
        0, 0, 1
    ]).reshape([3, 3]).astype(np.float32).transpose()


def _compute_rotation(angles: torch.Tensor) -> torch.Tensor:
    '''
    Rotation coefficient (euler angles in radian) to rotation matrix.
    Parameters:
    -----------
    angles: torch.Tensor, euler angles in radian. [B, 3]

    Returns:
    --------
    rot: torch.Tensor, rotation matrix. [B, 3, 3]
    '''
    batch_size = angles.shape[0]
    ones = torch.ones([batch_size, 1]).to(angles.device)
    zeros = torch.zeros([batch_size, 1]).to(angles.device)
    x, y, z = angles[:, :1], angles[:, 1:2], angles[:, 2:],

    rot_x = torch.cat([
        ones, zeros, zeros,
        zeros, torch.cos(x), -torch.sin(x),
        zeros, torch.sin(x), torch.cos(x)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_y = torch.cat([
        torch.cos(y), zeros, torch.sin(y),
        zeros, ones, zeros,
        -torch.sin(y), zeros, torch.cos(y)
    ], dim=1).reshape([batch_size, 3, 3])

    rot_z = torch.cat([
        torch.cos(z), -torch.sin(z), zeros,
        torch.sin(z), torch.cos(z), zeros,
        zeros, zeros, ones
    ], dim=1).reshape([batch_size, 3, 3])

    rot = rot_z @ rot_y @ rot_x
    return rot.permute(0, 2, 1)


def world_to_camera(
    vertices: torch.Tensor,
    rot_coef: torch.Tensor,
    tran_coef: torch.Tensor,
    camera_distance: float,
) -> torch.Tensor:
    '''
    Transfrom vertices from world space to camera space.
    Parameters:
    -----------
        vertices: torch.Tensor, vertices to transform. [B, N, 3]
        rot_coef: torch.Tensor, rotation coefficient (euler angles in radian). [B, 3]
        tran_coef: torch.Tensor, translate coefficient. [B, 3]
        camera_distance: float, camera distance.

    Returns:
    --------
        vertices: torch.Tensor, transformed vertices. [B, N, 3]
    '''
    rmat = _compute_rotation(rot_coef)
    vertices = vertices @ rmat
    vertices = vertices + tran_coef.unsqueeze(1)

    vertices[..., -1] = camera_distance - vertices[..., -1]

    return vertices


def projection(
    vertices: torch.Tensor,
    proj_mat: torch.Tensor,
) -> torch.Tensor:
    '''
    Project vertices into image space.

    Parameters:
    -----------
        vertices: torch.Tensor, vertices to project. [B, N, 3] (after world_to_camera)
        proj_mat: torch.Tensor, projection matrix. [3, 3]

    Returns:
    --------
        vertices: torch.Tensor, projected vertices. [B, N, 2] y direction is opposite to v direction
    '''
    vertices = vertices.clone() @ proj_mat.clone()
    vertices = vertices[..., :2] / vertices[..., 2:]
    return vertices


class CameraModel(nn.Module):
    '''
        Camera model compute vertices from world to camera space and projection.
    '''
    def __init__(
        self,
        image_size: int,
        focal: float,
        camera_distance: float,
    ) -> None:
        # CameraModel(image_size=224, focal=1015.0, camera_distance=10.0)
        super().__init__()
        self.image_size = image_size
        self.focal = focal
        self.camera_distance = camera_distance
        self.center = int(image_size / 2)
        proj_mat = perspective_projection(self.focal, self.center)

        self.register_buffer('proj_mat', torch.Tensor(proj_mat))

    def forward(
        self, model_dict: Dict, coef_dict: Dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        '''
            forward

            Parameters:
            -----------
                model_dict: Dict, tdmm output.
                coef_dict: Dict, encoder output.

            Returns:
            --------
                c_verts: torch.Tensor, camera space vertices. [B, N, 3]
                projected_verts: torch.Tensor, projected vertices. [B, N, 2]
                face_norm_rotated: torch.Tensor, Optional if face_norm is not None. [B, N, 3]
        '''
        c_verts = world_to_camera(
            model_dict['vertices'],
            coef_dict['angle'],
            coef_dict['trans'],
            self.camera_distance,
        )
        projected_verts = projection(c_verts.clone(), self.proj_mat.clone())

        return c_verts, projected_verts
