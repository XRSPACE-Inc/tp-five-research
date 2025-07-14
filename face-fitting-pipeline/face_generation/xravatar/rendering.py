import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

from .utils import to_device
from .projection import project_cam_mat_batch
from .material import Material
try:
    from xrrasterizer import rasterizer
except Exception:
    from .xrrasterizer_op import rasterizer


def gpmm_illumination_sh(norm, gamma, device, zero_lit=False):
    # compute vertex color using face_texture and SH function lighting approximation
    # input: face_texture with shape [1,N,3]
    #        norm with shape [1,N,3]
    #        gamma with shape [1,27]
    # output: face_color with shape [1,N,3], RGB order, range from 0-1
    #        lighting with shape [1,N,3], color under uniform texture, range from 0-1
    batch_size = norm.shape[0]
    num_vertex = norm.shape[1]

    init_lit = torch.tensor([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).to(device)
    if zero_lit:
        init_lit = torch.zeros_like(init_lit)
    init_lit = init_lit.view([1, 1, 9]).repeat(batch_size, 1, 1)

    gamma = gamma.reshape([-1, 3, 9])
    gamma = gamma + init_lit

    # parameter of 9 SH function
    a0 = torch.from_numpy(np.array(np.pi)).to(device)
    a1 = torch.from_numpy(np.array(2 * np.pi / np.sqrt(3.0))).to(device)
    a2 = torch.from_numpy(np.array(2 * np.pi / np.sqrt(8.0))).to(device)
    c0 = torch.from_numpy(np.array(1 / np.sqrt(4 * np.pi))).to(device)
    c1 = torch.from_numpy(np.array(np.sqrt(3.0) /
                                   np.sqrt(4 * np.pi))).to(device)
    c2 = torch.from_numpy(np.array(3 * np.sqrt(5.0) /
                                   np.sqrt(12 * np.pi))).to(device)

    Y0 = (a0 * c0).reshape([1, 1, 1]).repeat([batch_size, num_vertex, 1])
    Y0 = Y0.type(torch.float32)

    Y1 = (-a1 * c1 * norm[:, :, 1]).unsqueeze(-1)
    Y2 = (a1 * c1 * norm[:, :, 2]).unsqueeze(-1)
    Y3 = (-a1 * c1 * norm[:, :, 0]).unsqueeze(-1)
    Y4 = (a2 * c2 * norm[:, :, 0] * norm[:, :, 1]).unsqueeze(-1)
    Y5 = (-a2 * c2 * norm[:, :, 1] * norm[:, :, 2]).unsqueeze(-1)
    Y6 = (a2 * c2 * 0.5 / np.sqrt(3.0) *
          (3 * (norm[:, :, 2] ** 2) - 1)).unsqueeze(-1)
    Y7 = (-a2 * c2 * norm[:, :, 0] * norm[:, :, 2]).unsqueeze(-1)
    Y8 = (a2 * c2 * 0.5 * ((norm[:, :, 0] ** 2) -
                           (norm[:, :, 1] ** 2))).unsqueeze(-1)

    Y = torch.cat([Y0, Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8], 2)

    # Y shape:[batch,N,9].
    lit_r = torch.squeeze(torch.bmm(Y, gamma[:, 0, :].unsqueeze(-1)),
                          2)  # [batch,N,9] * [batch,9,1] = [batch,N]
    lit_g = torch.squeeze(torch.bmm(Y, gamma[:, 1, :].unsqueeze(-1)), 2)
    lit_b = torch.squeeze(torch.bmm(Y, gamma[:, 2, :].unsqueeze(-1)), 2)

    # shape:[batch,N,3]
    shade_color = torch.stack([lit_r, lit_g, lit_b], 2)

    # face_color = torch.clip_by_value(face_color, 0.0, 1.0)
    # shade_color = torch.clip_by_value(shade_color, 0.0, 1.0)

    return shade_color


def _primary_matrix(axis: int, angle):
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis 0 = X or 1 = Y or 2 = Z.
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    cos = torch.cos(angle).type(torch.float32)
    sin = torch.sin(angle).type(torch.float32)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == 0:
        o = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    if axis == 1:
        o = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    if axis == 2:
        o = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    return torch.stack(o, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles):
    '''
    euler_angles shape : [batch_size, 3]
    return [batch_size, 3, 3]
    '''
    x = _primary_matrix(0, euler_angles[:, 0])
    y = _primary_matrix(1, euler_angles[:, 1])
    z = _primary_matrix(2, euler_angles[:, 2])
    zy = torch.matmul(z, y)
    return torch.transpose(torch.matmul(zy, x), 2, 1)
    # return torch.transpose(torch.matmul(y, x), 2, 1)


def transform_verts(verts, translation, rotation, device):
    # batch_size = verts.shape[0]
    rot_mat = euler_angles_to_matrix(rotation)
    rot_mat = to_device(rot_mat, device)
    translation = to_device(translation, device)

    if len(translation) != 3:
        translation = translation.view([translation.shape[0], 1, -1])

    verts = torch.bmm(verts, rot_mat)
    verts = verts + translation
    # verts = torch.bmm(verts, scale_mat)
    return verts, rot_mat

# def transform_verts(verts, transfrom, device):
#     rot_mat = transfrom[:, :9].view([-1, 3, 3])

#     verts = torch.bmm(verts, rot_mat)
#     verts = verts + transfrom[:, 9:].view([-1, 1, 3])
#     return verts, rot_mat


def projection_matrix(FOV, batch_size, device):
    znear = 0.01
    zfar = 100.0
    fov = FOV
    aspect_ratio = 1.0

    P = torch.zeros(
        (batch_size, 4, 4), dtype=torch.float32
    ).to(device)
    ones = torch.ones((batch_size), dtype=torch.float32).to(device)
    fov = (np.pi / 180) * fov

    if not torch.is_tensor(fov):
        fov = torch.tensor(fov).to(device)
    tanHalfFov = torch.tan((fov / 2))
    top = tanHalfFov * znear
    bottom = -top
    right = top * aspect_ratio
    left = -right

    z_sign = -1.0

    P[:, 0, 0] = 2.0 * znear / (right - left)
    P[:, 1, 1] = 2.0 * znear / (top - bottom)
    P[:, 0, 2] = (right + left) / (right - left)
    P[:, 1, 2] = (top + bottom) / (top - bottom)
    P[:, 3, 2] = z_sign * ones

    P[:, 2, 2] = z_sign * zfar / (zfar - znear)
    P[:, 2, 3] = -(zfar * znear) / (zfar - znear)

    return P.transpose(1, 2).contiguous()


def bvec_dot(a, b):
    '''
    batched vector3 dot product
    '''
    N, V, C = a.shape
    a = a.view(N, -1)
    b = b.view(N, -1)
    c = a * b
    c = c.view(N, V, C)
    return torch.sum(c, dim=-1)


class Renderer(nn.Module):
    def __init__(self, image_size, batch_size, device):
        super().__init__()
        self.device = device
        self.batch_size = batch_size
        self.image_size = image_size
        self._P = None
        self._fov = 0.0
        self.projection_matrix(15.0, batch_size)
        self.cam_vector = torch.FloatTensor(np.array([[[0.0], [0.0], [1.0]]])).to(device).repeat(batch_size, 1, 1)
        # self.cam_vector_base = torch.from_numpy(np.array([[[0.0], [0.0], [1.0]]])).type(torch.float32).to(device)
        self.defined_pose_main = torch.tensor([0.000000, 0.000000, 3.141593, 0.17440447, 9.1053238, 4994.3359]).unsqueeze(0).to(device).repeat(batch_size, 1)
        self.intrinsics_single = torch.tensor([[4700.000000, 0., 112.000000], [0., 4700.000000, 112.000000], [0., 0., 1.]]).to(device).view(1, 3, 3)
        self.intrinsics_single = self.intrinsics_single.repeat(batch_size, 1, 1)
        self.vertex_mat = Material(None, need_normal=True, need_sh=True)

    def set_image_size(self, new_size):
        if self.image_size != new_size:
            self.image_size = new_size

    def compute_norm(self, verts, faces):
        faces_packed = faces
        verts_packed = verts
        verts_normals = torch.zeros_like(verts_packed)

        vertices_faces = verts_packed[:, faces_packed]

        verts_normals = verts_normals.index_add(
            1,
            faces_packed[:, 1],
            torch.cross(
                vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
                vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
                dim=-1,
            ),
        )
        verts_normals = verts_normals.index_add(
            1,
            faces_packed[:, 2],
            torch.cross(
                vertices_faces[:, :, 0] - vertices_faces[:, :, 2],
                vertices_faces[:, :, 1] - vertices_faces[:, :, 2],
                dim=-1,
            ),
        )
        verts_normals = verts_normals.index_add(
            1,
            faces_packed[:, 0],
            torch.cross(
                vertices_faces[:, :, 1] - vertices_faces[:, :, 0],
                vertices_faces[:, :, 2] - vertices_faces[:, :, 0],
                dim=-1,
            ),
        )

        return F.normalize(verts_normals, eps=1e-6, dim=-1)

    def compute_tangent(self, verts, faces, uv, norms):
        face_verts = verts[:, faces]
        face_uv = uv[:, faces]
        tan1 = torch.zeros_like(verts)
        tan2 = torch.zeros_like(verts)
        tan_xyz = torch.zeros_like(verts)
        tan_w = torch.zeros_like(verts[..., 0])

        v1 = face_verts[:, :, 0, :]
        v2 = face_verts[:, :, 1, :]
        v3 = face_verts[:, :, 2, :]

        w1 = face_uv[:, :, 0, :]
        w2 = face_uv[:, :, 1, :]
        w3 = face_uv[:, :, 2, :]

        x1 = v2[..., 0] - v1[..., 0]
        x2 = v3[..., 0] - v1[..., 0]
        y1 = v2[..., 1] - v1[..., 1]
        y2 = v3[..., 1] - v1[..., 1]
        z1 = v2[..., 2] - v1[..., 2]
        z2 = v3[..., 2] - v1[..., 2]
        s1 = w2[..., 0] - w1[..., 0]
        s2 = w3[..., 0] - w1[..., 0]
        t1 = w2[..., 1] - w1[..., 1]
        t2 = w3[..., 1] - w1[..., 1]
        div = (s1 * t2 - s2 * t1)
        div[div == 0] = 1e-6
        r = 1.0 / div

        sdir = torch.cat([((t2 * x1 - t1 * x2) * r).unsqueeze(-1), ((t2 * y1 - t1 * y2) * r).unsqueeze(-1), ((t2 * z1 - t1 * z2) * r).unsqueeze(-1)], -1)
        tdir = torch.cat([((s1 * x2 - s2 * x1) * r).unsqueeze(-1), ((s1 * y2 - s2 * y1) * r).unsqueeze(-1), ((s1 * z2 - s2 * z1) * r).unsqueeze(-1)], -1)

        tan1[:, faces[..., 0]] += sdir
        tan1[:, faces[..., 1]] += sdir
        tan1[:, faces[..., 2]] += sdir

        tan2[:, faces[..., 0]] += tdir
        tan2[:, faces[..., 1]] += tdir
        tan2[:, faces[..., 2]] += tdir

        tan_xyz = F.normalize(tan1 - norms * bvec_dot(norms, tan1).unsqueeze(-1), eps=1e-6, dim=-1)
        tan_w = bvec_dot(torch.cross(norms, tan1, -1), tan2)
        mask = tan_w < 0
        tan_w.masked_fill_(mask, -1.0)
        tan_w.masked_fill_(~mask, 1.0)
        return tan_xyz, tan_w.unsqueeze(-1)

    def compute_face_norm(self, verts, faces):
        faces_packed = faces
        verts_packed = verts
        vertices_faces = verts_packed[:, faces_packed]
        faces_normals = torch.cross(
            vertices_faces[:, :, 2] - vertices_faces[:, :, 1],
            vertices_faces[:, :, 0] - vertices_faces[:, :, 1],
            dim=-1)
        return F.normalize(faces_normals, eps=1e-6, dim=-1)

    def projection_matrix(self, FOV, batch_size):
        fov = FOV
        # if self._fov == fov:
        #     return self.P.clone()

        # self._fov = fov
        self.P = projection_matrix(fov, batch_size, self.device)
        return self.P

    def transform(self, verts, pos, rot, fov):
        verts, rot_mat = transform_verts(verts, pos, rot, self.device)
        N, P, _3 = verts.shape

        # from utils import write_obj
        # mesh = {}
        # mesh["verts"] = verts[0].cpu().detach().numpy()
        # mesh["faces"] = faces.cpu().detach().numpy()
        # write_obj("./results/testing/rendering/out.obj", mesh)

        # ones = torch.ones(N, P, 1).to(self.device)
        ones = torch.ones_like(verts[:, :, 0].view(N, P, 1))
        points_batch = torch.cat([verts, ones], dim=2)

        p_mat = self.projection_matrix(fov, points_batch.shape[0])

        points_out = torch.bmm(points_batch, p_mat)
        denom = points_out[..., 3:]
        points_out = points_out[..., :3] / denom
        return points_out, rot_mat

    def transform_mat(self, trans_verts, trans):
        homogeneous_coord = torch.ones_like(trans_verts[..., 0].unsqueeze(-1))
        vertices_homogeneous = torch.cat([trans_verts, homogeneous_coord], -1)

        transform = self.defined_pose_main.clone()
        transform += trans.clone()
        transform = project_cam_mat_batch(self.intrinsics_single.clone(), transform)
        transform = transform.permute(0, 2, 1)

        clip_space_points = torch.bmm(vertices_homogeneous, transform)

        _MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD = 1e-6
        threshold_ = torch.zeros_like(homogeneous_coord).fill_(_MINIMUM_PERSPECTIVE_DIVIDE_THRESHOLD)
        clip_space_points_w = torch.max(
            torch.abs(clip_space_points[:, :, 3:4]), threshold_) * torch.sign(clip_space_points[:, :, 3:4])

        return clip_space_points[:, :, 0:3] / clip_space_points_w

    def mask_pix_to_face(self, pix_to_face):
        pix_to_face = pix_to_face.clone()
        mask = pix_to_face == -1
        pix_to_face = pix_to_face.masked_fill(mask, 0)
        # pix_to_face[mask] = 0
        mask = mask.unsqueeze(-1)
        return pix_to_face, mask

    def interpolate_face_attributes(self,
                                    pix_to_face,
                                    mask,
                                    bary_coords,
                                    face_data):
        N, F, FV, D = face_data.shape
        if FV != 3:
            raise ValueError("Faces can only have three vertices; got %r" % FV)
        N, H, W, K, _ = bary_coords.shape
        if pix_to_face.shape != (N, H, W, K):
            msg = "pix_to_face must have shape (batch_size, H, W, K); got %r"
            raise ValueError(msg % pix_to_face.shape)

        idx = pix_to_face.view(N * H * W * K, 1, 1).expand(N * H * W * K, 3, D)

        pixel_face_vals = face_data.reshape([-1, 3, D]).gather(0, idx).view(N, H, W, K, 3, D)
        pixel_vals = (bary_coords[..., None] * pixel_face_vals).sum(dim=-2)

        # pixel_vals[mask] = 0  # Replace masked values in output.
        pixel_vals = pixel_vals.masked_fill(mask, 0)

        return pixel_vals.view([N, H, W, D])

    def interpolate(self, pix_to_face, bary_coords, materials):
        pix_to_face, mask = self.mask_pix_to_face(pix_to_face)

        for key in materials:
            material = materials[key]
            att_dict = material.get_attribute()
            mask_face = material.mask_face(pix_to_face, mask)

            for k in att_dict:
                texels = self.interpolate_face_attributes(pix_to_face, mask_face,
                                                          bary_coords, att_dict[k])
                att_dict[k] = torch.flip(texels, [1])
            material.set_attributes(att_dict)
            materials[key] = material

        return materials
        # for key in data_dict:
        #     item = data_dict[key]
        #     if item is None:
        #         continue
        #     if key in face_map:
        #         faces = face_map[key]
        #     else:
        #         faces = face_map["faces"]
        #     item = item[:, faces]
        #     texels = self.interpolate_face_attributes(pix_to_face, mask,
        #                                               bary_coords, item)
        #     data_dict[key] = torch.flip(texels, [1])
        # return data_dict

    def rasterize(self, verts, faces, face_mask):
        '''
        return pix_to_face, zbuf, barycentric_coords, dists
        '''
        verts = verts.type(torch.float32).to(self.device)
        faces = faces.type(torch.int64).to(self.device)
        face_verts = verts[:, faces]
        if face_mask is not None:
            face_mask = face_mask.unsqueeze(-1).repeat(1, 1, 1, 3)
            face_verts.masked_fill_(face_mask, -1e5)
        return rasterizer(face_verts, self.batch_size, self.image_size, blur_radius=0.0, faces_per_pixel=1)

    def reflect(self, L, N, LdotN):
        return 2 * LdotN * N - L

    def sample_texture(self, texture, uvs):
        N, C, _, _ = texture.shape
        uvs = uvs.clone()

        uvs = uvs * 2.0 - 1.0

        tex = F.grid_sample(texture, uvs, align_corners=False)
        tex = tex.view([N, C, self.image_size, self.image_size])

        # texture = texture.permute(0, 2, 3, 1).clone()
        # N, H, W, C = texture.shape
        # uvs = uvs.clone()
        # uvs = to_device(uvs.type(torch.in64), self.device)

        # uvs = (uvs[..., 1] * W) + uvs[..., 0]

        # texture = texture.view([-1, C])
        # uvs = uvs.view(-1, 1)
        # uvs = uvs.expand(uvs.shape[0], C)

        # tex = texture.gather(0, uvs)

        # tex = tex.view([-1, self.image_size, self.image_size, C])
        # tex = tex.permute(0, 3, 1, 2)

        return tex

    def from_3d_to_image(self, verts, face_mask, faces, mat_list):
        pix_to_face, zbuf, bary_coord, pix_dists = self.rasterize(verts, faces, face_mask)
        # print("pix_to_face: {}".format(pix_to_face.shape))
        # print("zbuf: {}".format(zbuf.shape))
        # print("bary_coord: {}".format(bary_coord.shape))
        # print("pix_dists: {}".format(pix_dists.shape))
        return self.interpolate(pix_to_face, bary_coord, mat_list)

    def illumination(self, tex_dict, to_2d_list):
        normalmap = tex_dict["normalmap"]

        # dJ
        n = 2.0 * normalmap - 1.0
        dJ = to_2d_list["v_tang"] * (n[:, :, :, 0].unsqueeze(-1))
        dJ += to_2d_list["v_bitan"] * (n[:, :, :, 1].unsqueeze(-1))
        dJ += to_2d_list["v_normal"] * (n[:, :, :, 2].unsqueeze(-1))
        normalmap = F.normalize(dJ, eps=1e-6, dim=-1)

        # dO
        viewDir = F.normalize(-to_2d_list["v"], eps=1e-6, dim=-1)

        # reflect
        NV = normalmap * viewDir
        NV = torch.sum(NV, -1).unsqueeze(-1)
        NV = -viewDir - 2.0 * NV * normalmap

        # el = (vec3)en(ek, dC);
        el = NV * to_2d_list["v_normal"]
        el = torch.sum(el, -1).unsqueeze(-1)
        el = torch.clamp(1.0 + el, 0.0, 1.0)
        el = torch.pow(el, 2).repeat([1, 1, 1, 3])

        # ei, _ = dY.eQ(viewDir, normalmap)

        # eT
        eS = torch.zeros_like(viewDir)
        VN = viewDir * normalmap
        VN = torch.sum(VN, -1).unsqueeze(-1)
        C = 1.0 - torch.clamp(VN, 0.0, 1.0)
        specular = tex_dict["specular"] ** 2
        eS = specular - C * specular + C

        el2 = tex_dict["lightmap_spec"] * eS
        el *= eS
        return el + el2

    def normalmapping(self, mat, vertexNormal, normalmap):
        n = 2.0 * normalmap - 1.0
        n = n.permute(0, 2, 3, 1)
        vertexNormal = vertexNormal.permute(0, 2, 3, 1)
        tangents = mat.attributes['tangent']
        bitangent = mat.attributes['bitangent']
        res = tangents * (n[:, :, :, 0].unsqueeze(-1))
        res += bitangent * (n[:, :, :, 1].unsqueeze(-1))
        res += vertexNormal * (n[:, :, :, 2].unsqueeze(-1))
        return F.normalize(res, eps=1e-6, dim=-1)

    def forward(self, **kwargs):
        if "render_size" in kwargs:
            self.image_size = int(kwargs["render_size"])
        verts = kwargs["verts"]
        faces = kwargs["faces"]

        if "norm" in kwargs:
            norm = kwargs["norm"]
        else:
            norm = self.compute_norm(verts, faces)

        fov = kwargs["FOV"] if "FOV" in kwargs else 15.0

        if "pos" in kwargs and "rot" in kwargs:
            # [face,3]
            trans_verts, rot_mat = self.transform(verts, kwargs["pos"], kwargs["rot"], fov)
            norm = torch.bmm(norm, rot_mat)
        elif "trans" in kwargs:
            trans_verts = self.transform_mat(verts, kwargs["trans"].clone())
        else:
            trans_verts = verts
        if "face_mask" in kwargs:
            face_mask = kwargs["face_mask"]
        else:
            cullback = kwargs["cullback"] if "cullback" in kwargs else True
            if cullback:
                ##################################################
                # Remove dot > 0 faces
                #################################################
                face_norm = self.compute_face_norm(trans_verts, faces)
                # faces_batched = faces.unsqueeze(0).repeat(verts.shape[0], 1, 1)
                cam_vector = self.cam_vector.clone()
                # cam_vector = self.cam_vector_base.clone().repeat(face_norm.shape[0], 1, 1)
                dot = torch.bmm(face_norm, cam_vector)
                dot = dot < 0
                # [2, face, 1]
                face_mask = dot.repeat(1, 1, 3)

                ##################################################
                # Remove dot > 0 vertices
                #################################################
                # cam_vector = torch.from_numpy(np.array([[[0.0], [0.0], [1.0]]])).repeat(verts.shape[1], 1, 1).view(verts.shape).type(torch.float32).to(self.device)
                # dot = cam_vector[:, :, 0] * norm[:, :, 0] + cam_vector[:, :, 1] * norm[:, :, 1] + cam_vector[:, :, 2] * norm[0, :, 2]
                # dot = dot > 0
                # dot = dot.view(-1, 1)
                # face_pack = dot[faces].view([-1, 3])
                # face_pack = face_pack[:, 0] & face_pack[:, 1] & face_pack[:, 2]
                # face_pack = face_pack.view(verts.shape[0], -1, 1).repeat(1, 1, 3)
                # faces = faces[face_pack].view([-1, 3])
            else:
                face_mask = None

        if "materials" in kwargs and kwargs["materials"] is not None:
            mats = kwargs["materials"]
        else:
            mats = {}
        mats["__vertex_mat"] = self.vertex_mat
        do_sh = "sh" in kwargs and kwargs["sh"] is not None
        if do_sh:
            sh = gpmm_illumination_sh(norm, kwargs["sh"], self.device)
        do_tangent = 'get_tangent' in kwargs and kwargs['get_tangent']
        for key in mats:
            mat = mats[key]
            if mat.need_update_faces:
                mat.faces = faces
            mat.set_normal(norm)
            if do_sh:
                mat.set_sh(sh)
            if do_tangent and mat.uvs is not None:
                tangents, tangents_w = self.compute_tangent(verts, faces, mat.uvs, norm)
                bitangents = torch.cross(norm, tangents) * tangents_w
                mat.set_additional_attrs('tangent', tangents)
                mat.set_additional_attrs('bitangent', bitangents)
            mats[key] = mat

        # if "textures" in kwargs and "normalmap" in kwargs["textures"]:
        #     sky_matrix = kwargs["sky_matrix"]
        #     to_2d_list["v"] = torch.bmm(verts, sky_matrix)
        #     to_2d_list["v_normal"] = torch.bmm(norm, sky_matrix)
        #     tangents, tangents_w = self.compute_tangent(verts, faces, uvs, norm)
        #     to_2d_list["v_tang"] = torch.bmm(tangents, sky_matrix)
        #     bitangents = torch.cross(norm, tangents) * tangents_w
        #     to_2d_list["v_bitan"] = torch.bmm(bitangents, sky_matrix)

        # uv = uvs.clone()
        # uv[..., 1] = 1.0 - uv[..., 1]
        # trans_verts[..., :2] = uv * 2 - 1
        # trans_verts[..., 2] = 0

        mats = self.from_3d_to_image(trans_verts, face_mask, faces, mats)

        # if "textures" in kwargs:
        #     tex_dict = kwargs["textures"]
        #     for key in tex_dict:
        #         tex_dict[key] = self.sample_texture(tex_dict[key].clone(), to_2d_list["uv"]).permute(0, 2, 3, 1)
        #     if "normalmap" in kwargs["textures"]:
        #         res["specular"] = self.illumination(tex_dict, to_2d_list)
        #     res["textures"] = tex_dict

        res = {}
        res["mats"] = mats
        for key in mats:
            mats[key].sample_image(self.image_size, do_sh)

        if "get_norm" in kwargs and kwargs["get_norm"]:
            res["norm"] = self.vertex_mat.get_normal()
        if do_sh:
            res["sh"] = self.vertex_mat.get_sh()
        if "get_vert_proj" in kwargs and kwargs["get_vert_proj"]:
            res["vp"] = trans_verts

        return res


def main_rendering():
    test_output_path = "./results/testing/rendering/"
    os.makedirs(test_output_path, exist_ok=True)
    import config
    import utils

    cuda = True if torch.cuda.is_available() else False
    device = "cuda" if cuda else "cpu"

    batch_size = 1
    image_size = 224

    from blendshape import load_bfm2009
    blendshape = load_bfm2009(config, batch_size, device, False)

    renderer = Renderer(image_size, batch_size, device)
    shape_coef = torch.zeros([80]).type(torch.float32).to(device)
    ex_coef = torch.zeros([64]).type(torch.float32).to(device)
    col_coef = torch.zeros([80]).type(torch.float32).to(device)
    sh_coef = torch.ones([27]).type(torch.float32).to(device) * 0.3

    bpos = torch.tensor([0.0, 0.0, -9.0]).type(torch.float32).to(device)
    brot = torch.tensor([0.0, 180.0 * 0.0174532925, 0.0]).type(torch.float32).to(device)  # 0.0174532925

    # FOV = 15.0
    coef_full = torch.from_numpy(np.load("coeff_all.npy")).to(device)
    print(coef_full)
    print(coef_full.shape)
    shape_coef = coef_full[:, 0:80].unsqueeze(0).to(device)
    print(shape_coef.shape)
    col_coef = coef_full[:, 80:160].unsqueeze(0).to(device)
    ex_coef = coef_full[:, 160:224].unsqueeze(0).to(device)
    sh_coef = coef_full[:, 230:257].unsqueeze(0).to(device)

    defined_pose_main = torch.tensor([0.000000, 0.000000, 3.141593, 0.17440447, 9.1053238, 4994.3359]).unsqueeze(0).to(device).repeat(batch_size, 1)
    gmm_trans = coef_full[:, 224:230].unsqueeze(0).to(device)
    gmm_trans = gmm_trans + defined_pose_main
    shape_coef = shape_coef.view([1, -1, 1]).repeat([batch_size, 1, 1])

    # shape_coef = torch.zeros_like(shape_coef)

    ex_coef = ex_coef.view([1, -1, 1]).repeat([batch_size, 1, 1])
    # ex_coef = torch.zeros_like(ex_coef)
    col_coef = to_device(col_coef, device).view([1, -1, 1]).repeat([batch_size, 1, 1])
    sh_coef = to_device(sh_coef, device).reshape([1, 27, 1]).repeat([batch_size, 1, 1])

    bpos = to_device(bpos, device).view([1, -1]).repeat([batch_size, 1])
    brot = to_device(brot, device).view([1, -1]).repeat([batch_size, 1])

    g_shape, verts, vert_colors = blendshape.get_morphed(shape_coef, ex_coef, col_coef)

    # sky_matrix = np.array([
    #     [0.3090210556983948, 0, -0.9510552287101746, 0],
    #     [0, 1, 0, 0],
    #     [0.9510552287101746, 0, 0.3090210556983948, 0],
    #     [0, 0, 0, 1]
    # ])
    # sky_matrix = to_device(sky_matrix[:3, :3], device).view(1, 3, 3).repeat([batch_size, 1, 1])

    # textures = {}
    # textures["albedo"] = utils.load_texture(
    #     "./dataset/datas_head48/textures/female01/Colour_8k.jpg", batch_size,
    #     1024, "cuda")
    # textures["normalmap"] = utils.load_texture(
    #     "./dataset/datas_head48/textures/female01/Normal.jpg", batch_size,
    #     1024, "cuda")
    # textures["specular"] = utils.load_texture(
    #     "./dataset/datas_head48/textures/female01/Spec.jpg", batch_size,
    #     1024, "cuda")

    # textures["lightmap"] = torch.ones_like(textures["albedo"]) * 1.0
    # textures["lightmap_spec"] = torch.ones_like(textures["albedo"]) * 1.0

    # mgc_vert = np.load("vertex_3D.npy")
    # mgc_vert = torch.from_numpy(mgc_vert).to("cuda").view(1,35709,3)
    # mgc_vert = mgc_vert.repeat(batch_size,1,1)

    intrinsics_single = torch.tensor([[4700.000000, 0., 112.000000], [0., 4700.000000, 112.000000], [0., 0., 1.]]).to("cuda").view(1, 3, 3)
    intrinsics_single = intrinsics_single.repeat(batch_size, 1, 1)

    pos = gmm_trans.view(1, 6)
    pos = pos.repeat(batch_size, 1)
    import time
    start_time = time.time()
    from projection import project_cam_mat_batch

    # clip_space_transforms = project_cam_mat(intrinsics_single, pos)
    # clip_space_transforms_test = clip_space_transforms
    # clip_space_transforms = torch.stack(clip_space_transforms)
    # clip_space_transforms_test = clip_space_transforms.cpu().detach().numpy()
    # print(clip_space_transforms_test.shape)
    # print("--- %s seconds ---" % (time.time() - start_time))

    start_time = time.time()
    clip_space_transforms = project_cam_mat_batch(intrinsics_single, pos)

    clip_space_transforms_test = clip_space_transforms.cpu().detach().numpy()
    print(clip_space_transforms_test.shape)
    print("--- %s seconds ---" % (time.time() - start_time))

    projection_matrices = clip_space_transforms

    res = renderer(verts=verts,
                   faces=blendshape.tri_g,
                   uvs=blendshape.uv2.clone(),
                   trans=projection_matrices,
                   sh=sh_coef,
                   vertex_colors=vert_colors,
                   get_norm=True,
                   )
    vc = res["vc"]
    sh = res["sh"]
    # print(sh)
    vc = sh * vc
    # res = res["textures"]
    utils.save_tensor_img(torch.clamp(vc, min=0.0, max=1.0),
                          os.path.join(test_output_path, "test_alb.jpg"))
    # utils.save_tensor_img(
    #     torch.clamp(res["lightmap"], min=0.0, max=1.0),
    #     os.path.join(test_output_path, "test_lig.png"))
    # output = res["albedo"] * res["lightmap"]
    # utils.save_tensor_img(
    #     torch.clamp(res["specular"], min=0.0, max=1.0),
    #     os.path.join(test_output_path, "test_spec.png"))
    # output += res["specular"]
    # utils.save_tensor_img(
    #     torch.clamp(output, min=0.0, max=1.0),
    #     os.path.join(test_output_path, "test.png"))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if "-p" in sys.argv:
            from memory_profiler import profile
            main_rendering = profile(main_rendering)
    main_rendering()
