import torch
import numpy as np
def build_train_graph_3dmm_frustrum_batch(intrinsic, near=0001.0, far=10000.0):
    batch_size = intrinsic.shape[0]
    # def build_train_graph_3dmm_frustrum(self, intrinsic, near=0.1, far=2000.0):
    # intrinsic

    focal_len_x = intrinsic[:, 0, 0].unsqueeze(-1)

    focal_len_y = intrinsic[:, 1, 1].unsqueeze(-1)

    u = intrinsic[:, 0, -1].unsqueeze(-1)

    v = intrinsic[:, 1, -1].unsqueeze(-1)

    #
    near = torch.tensor([near]).to("cuda")
    far = torch.tensor([far]).to("cuda")
    near = near.view(1,-1).repeat(batch_size,1)
    far = far.view(1,-1).repeat(batch_size,1)

    #
    mtx_frustrum = projectionFrustrumMatrix_batch(focal_len_x, focal_len_y, u, v, near, far)

    return mtx_frustrum

def projectionFrustrumMatrix_batch(focal_len_x, focal_len_y, u, v,  near, far):
    image_width_batch = 2 * u
    image_height_batch = 2 * v

    # From triangle similarity
    width = image_width_batch * near / focal_len_x
    height = image_height_batch * near / focal_len_y

    right = width - (u * near / focal_len_x)
    left = right - width

    top = v * near / focal_len_y
    bottom = top - height

    vertical_range = right - left
    p00 = 2 * near / vertical_range
    p02 = (right + left) / vertical_range

    horizon_range = top-bottom
    p11 = 2 * near / horizon_range
    p12 = (top + bottom) / horizon_range

    depth_range = far - near
    p_22 = -(far + near) / depth_range
    p_23 = -2.0 * (far * near / depth_range)

    zero_fill = torch.zeros_like(p00).to("cuda")
    minus_one_fill = torch.ones_like(p00).to("cuda")

    r1 = torch.stack([p00, zero_fill, p02, zero_fill], axis=2)
    r2 = torch.stack([zero_fill, p11, p12, zero_fill], axis=2)
    r3 = torch.stack([zero_fill, zero_fill, p_22, p_23], axis=2)
    r4 = torch.stack([zero_fill, zero_fill, -minus_one_fill, zero_fill], axis=2)

    P = torch.cat([r1, r2, r3, r4], 1)

    return P


def euler2mat(z, y, x, clip=False):
    """[summary] Converts euler angles to rotation matrix
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
    
    Arguments:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """

    B = z.shape[0]
    N = 1
    if clip:
        z = torch.clamp(z, -np.pi, np.pi)
        y = torch.clamp(y, -np.pi, np.pi)
        x = torch.clamp(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    z = z.view([B,-1,1,1])
    y = y.view([B,-1,1,1])
    x = x.view([B,-1,1,1])
    
    zeros = torch.zeros([B, N, 1, 1]).to("cuda")
    ones = torch.ones([B, N, 1, 1]).to("cuda")

    cosz = torch.cos(z).to("cuda")
    sinz = torch.sin(z).to("cuda")

    rotz_1 = torch.cat([cosz, -sinz, zeros], 3)
    rotz_2 = torch.cat([sinz,  cosz, zeros], 3)
    rotz_3 = torch.cat([zeros, zeros, ones], 3)
    zmat = torch.cat([rotz_1, rotz_2, rotz_3], 2)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    roty_1 = torch.cat([cosy, zeros, siny], 3)
    roty_2 = torch.cat([zeros, ones, zeros], 3)
    roty_3 = torch.cat([-siny, zeros, cosy], 3)
    ymat = torch.cat([roty_1, roty_2, roty_3], 2)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    rotx_1 = torch.cat([ones, zeros, zeros], 3)
    rotx_2 = torch.cat([zeros, cosx, -sinx], 3)
    rotx_3 = torch.cat([zeros, sinx, cosx], 3)
    xmat = torch.cat([rotx_1, rotx_2, rotx_3], 2)

    rotMat = torch.matmul(torch.matmul(xmat, ymat), zmat)
    return rotMat

    """[summary] Converts euler angles to rotation matrix
    Reference: https://github.com/pulkitag/pycaffe-utils/blob/master/rot_utils.py#L174
     TODO: remove the dimension for 'N' (deprecated for converting all source
           poses altogether)
    
    Arguments:
        z: rotation angle along z axis (in radians) -- size = [B, N]
        y: rotation angle along y axis (in radians) -- size = [B, N]
        x: rotation angle along x axis (in radians) -- size = [B, N]
    
    Returns:
        Rotation matrix corresponding to the euler angles -- size = [B, N, 3, 3]
    """

    B = z.shape[0]
    N = 1
    if clip:
        z = torch.clamp(z, -np.pi, np.pi)
        y = torch.clamp(y, -np.pi, np.pi)
        x = torch.clamp(x, -np.pi, np.pi)

    # Expand to B x N x 1 x 1
    # print(z.shape)
    z = z.view([B,-1,1,1])
    y = y.view([B,-1,1,1])
    x = x.view([B,-1,1,1])

    zeros = torch.zeros([B, N, 1, 1]).to("cuda")
    ones = torch.ones([B, N, 1, 1]).to("cuda")

    cosz = torch.cos(z).to("cuda")
    sinz = torch.sin(z).to("cuda")

    rotz_1 = torch.cat([cosz, -sinz, zeros], 3)
    rotz_2 = torch.cat([sinz,  cosz, zeros], 3)
    rotz_3 = torch.cat([zeros, zeros, ones], 3)
    zmat = torch.cat([rotz_1, rotz_2, rotz_3], 2)

    cosy = torch.cos(y)
    siny = torch.sin(y)
    roty_1 = torch.cat([cosy, zeros, siny], 3)
    roty_2 = torch.cat([zeros, ones, zeros], 3)
    roty_3 = torch.cat([-siny, zeros, cosy], 3)
    ymat = torch.cat([roty_1, roty_2, roty_3], 2)

    cosx = torch.cos(x)
    sinx = torch.sin(x)
    rotx_1 = torch.cat([ones, zeros, zeros], 3)
    rotx_2 = torch.cat([zeros, cosx, -sinx], 3)
    rotx_3 = torch.cat([zeros, sinx, cosx], 3)
    xmat = torch.cat([rotx_1, rotx_2, rotx_3], 2)

    rotMat = torch.matmul(torch.matmul(xmat, ymat), zmat)
    return rotMat


def pose_vec2rt_batch(vec, clip=False):
    """Converts 6DoF parameters to rotation matrix (bs,3,3) and translation vector (bs,3,1)"""
    batch_size = vec.shape[0]
    # translation = vec[:,3:]

    translation = vec[:,3:].view(batch_size,3,1)
    rz = vec[:,0].unsqueeze(-1)
    ry = vec[:,1].unsqueeze(-1)
    rx = vec[:,2].unsqueeze(-1)

    rot_mat = euler2mat(rz, ry, rx, clip)
    rot_mat = rot_mat.squeeze(1)
    return rot_mat, translation


def pose_vec2mat_batch(vec, clip=False):
    """Converts 6DoF parameters to transformation matrix
    Args:
        vec: 6DoF parameters in the order of [rz, ry, rx, tx, ty, tz] -- [B, 6]
        (NOT the original SfMLearner: tx, ty, tz, rx, ry, rz -- [B, 6])
    Returns:
        A transformation matrix -- [B, 4, 4]
    """
    batch_size = vec.shape[0]
    rot_mat, translation = pose_vec2rt_batch(vec, clip)

    filler = torch.tensor([0.0, 0.0, 0.0, 1.0]).to("cuda")
    filler = filler.view(1,1,-1).repeat(batch_size,1,1)
    transform_mat = torch.cat([rot_mat, translation], 2)
    transform_mat = torch.cat([transform_mat, filler], 1)
    return transform_mat

def project_without_batch(mtx_intrinsic, rot_batch, t_batch):
    # batch_size = mtx_intrinsic.shape[0]

    M = torch.matmul(mtx_intrinsic, rot_batch)

    p4 = torch.matmul(mtx_intrinsic, t_batch)
    proj = torch.cat([M, p4], 2)

    r4 = torch.tensor([0., 0., 0., 1.]).view(1,1,-1).to("cuda")
    proj = torch.cat([proj, r4], 1)
    return proj


def project_batch(mtx_intrinsic, rot_batch, t_batch):
    batch_size = mtx_intrinsic.shape[0]

    M = torch.matmul(mtx_intrinsic, rot_batch)

    p4 = torch.matmul(mtx_intrinsic, t_batch)
    proj = torch.cat([M, p4], 2)

    r4 = torch.tensor([0., 0., 0., 1.]).view(1,1,-1).repeat(batch_size,1,1).to("cuda")
    proj = torch.cat([proj, r4], 1)
    return proj

def ext_to_eye_batch(rot_batch, t_batch):
    #mtx_t_trans = tf_render.expand_dims(t_batch, 1)
    t_batch = t_batch.permute(0, 2, 1)
    eye_trans = - torch.matmul(t_batch, rot_batch)
    eye = torch.squeeze(eye_trans, 1)
    return eye

def modelViewMatrix_batch(rot_batch, t_batch):
    batch_size = rot_batch.shape[0]

    mtx_inv = torch.tensor(
        [
            [1.,  0.,  0.],
            [0., -1.,  0.],
            [0.,  0., -1.]
        ])
    mtx_inv = mtx_inv.view([1, 3, 3]).repeat(batch_size,1,1).to("cuda")

    # Inv rotate

    rot_inv = torch.matmul(mtx_inv, rot_batch)
    c4 = torch.tensor([0., 0., 0.]).view(1,3,1).repeat(batch_size,1,1).to("cuda")
    rot_inv = torch.cat([rot_inv, c4], 2)

    r4 = torch.tensor([0., 0., 0., 1.]).to("cuda")
    r4 = r4.view([1, 1, 4]).repeat(batch_size,1,1)
    rot_inv = torch.cat([rot_inv, r4], 1)

    eye_inv = -ext_to_eye_batch(rot_batch, t_batch)
    eye_inv_trans = eye_inv.unsqueeze(-1)
    trans_id_inv = torch.eye(3).view(1,3,3).repeat(batch_size,1,1).to("cuda")
    trans_inv = torch.cat([trans_id_inv, eye_inv_trans], 2)
    trans_inv = torch.cat([trans_inv, r4], 1)

    mv = torch.matmul(rot_inv, trans_inv)

    return mv


def build_train_graph_3dmm_camera_batch(intrinsic, pose_6dof):
    mtx_ext = pose_vec2mat_batch(pose_6dof, False)
    mtx_rot = mtx_ext[:, :3, :3].to("cuda")
    mtx_t = pose_6dof[:, 3:6].unsqueeze(-1)

    # ext
    mtx_proj = project_batch(intrinsic, mtx_rot, mtx_t)
    # #
    mtx_mv = modelViewMatrix_batch(mtx_rot, mtx_t)
    # #
    mtx_eye = ext_to_eye_batch(mtx_rot, mtx_t)

    list_ext = mtx_ext
    list_proj = mtx_proj
    list_mv = mtx_mv
    list_eye = mtx_eye

    return list_ext, list_proj, list_mv, list_eye


def project_cam_mat_batch(intrinsics_single, _3dmm_pos):
    frustrum = build_train_graph_3dmm_frustrum_batch(intrinsics_single)    
    pos = _3dmm_pos


    _list_ext, _list_proj, list_mv, _list_eye = build_train_graph_3dmm_camera_batch(intrinsics_single, pos)

    clip_space_transforms = torch.bmm(frustrum, list_mv)
    return clip_space_transforms

if __name__ == "__main__":
    intrinsics_single = torch.tensor([[4700.000000, 0., 112.000000], [0., 4700.000000, 112.000000], [0., 0., 1.]])
    frustrum = build_train_graph_3dmm_frustrum_batch(intrinsics_single)
    pos = torch.tensor([ 1.8445775e-04, -2.2915000e-02,  3.3279083e+00,  2.4627767e+00, -6.1398830e+00,  5.0882715e+03])
    pos = pos.view(1,-1)

    list_ext, list_proj, list_mv, list_eye = build_train_graph_3dmm_camera_batch(intrinsics_single, pos)
    print(list_ext[0])
    print(list_proj[0])
    print(list_mv[0])
    print(list_eye[0])
    clip_space_transforms = torch.matmul(frustrum, list_mv[0])

    print(clip_space_transforms)
    print(project_cam_mat_batch(intrinsics_single, pos))
    # np.save("clip_space_transforms", clip_space_transforms)

