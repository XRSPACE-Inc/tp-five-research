import os
import torch
import numpy as np
from .utils import read_obj, remove_eye_faces


def _morph_pca(base, pca, weights, indexmap):
    batch_size, N, C = base.shape
    base_out = base.clone()
    base_out = base_out.view([batch_size, -1])
    morphed = torch.bmm(pca.clone(), weights).view([batch_size, -1])
    if base_out.shape[1] != morphed.shape[1]:
        base_out = base_out.index_add(1, indexmap, morphed)
    else:
        base_out = base_out.add(morphed)
    return base_out.view([batch_size, N, C])


def get_morphed(
        shape_base, shape_pca, shape_coef,
        ex_base, ex_pca, ex_coef,
        c_base, c_pca, c_coef,
        indexmap, add_base_ex=True):
    out_g_shape = _morph_pca(shape_base, shape_pca, shape_coef, indexmap)
    out_shape = _morph_pca(out_g_shape, ex_pca, ex_coef, indexmap)
    if add_base_ex:
        if out_shape.shape[1] != ex_base.shape[1]:
            out_shape = out_shape.index_add(1, indexmap, ex_base)
        else:
            out_shape = out_shape.add(ex_base)
    if c_coef is not None:
        out_color = _morph_pca(c_base, c_pca, c_coef, indexmap)
        return out_g_shape, out_shape, out_color
    return out_g_shape, out_shape, None


def get_uv_to_vertex(uv, batch_size, texture_size, device):
    uv *= texture_size

    return torch.clamp(uv, min=0, max=texture_size - 1).type(torch.int64).to(device)


class BFM(object):
    def __init__(
            self,
            verts, tri, uvs, indexmap,
            id_pca, ex_base, ex_pca, c_base, c_pca,
            texture_g, ao_tex_g,
            bscount,
            device, normals=None, tangs=None):
        super(BFM, self).__init__()
        self._verts_g = verts
        self._tri_g = tri
        self._uvs_g = uvs
        self._normals_g = normals
        self._tangs_g = tangs
        self.indexmap = indexmap
        self.id_pca_g = id_pca
        self.ex_base_g = ex_base
        self.ex_pca_g = ex_pca
        self.c_base_g = c_base
        self.c_pca_g = c_pca
        self._texture_g = texture_g
        self._ao_tex_g = ao_tex_g
        self.bscount = bscount
        self.vertex_uv = None
        self.uv_to_vertex = None
        self.bfm_space = False

        self.device = device

    @property
    def verts(self):
        return self._verts_g.clone()

    @property
    def tri_g(self):
        return self._tri_g.clone()

    @property
    def uvs_g(self):
        return None if self._uvs_g is None else self._uvs_g.clone()

    @property
    def normals_g(self):
        return None if self._normals_g is None else self._normals_g.clone()

    @property
    def tangs_g(self):
        return None if self._tangs_g is None else self._tangs_g.clone()

    @property
    def vc_g(self):
        return self.c_base_g.clone()

    @property
    def texture(self):
        return None if self._texture_g is None else self._texture_g.clone()

    @property
    def ao_texture(self):
        return None  # self._ao_tex_g.clone()

    @property
    def bsw_range(self):
        return (-4.0, 4.0)

    def to_bfm_space(self, yes):
        if self.bfm_space != yes:
            self.bfm_space = yes
            diff = torch.tensor([-0.00322628, 0.04506069, 0.75983165]).type(torch.float32).to(self.device)
            if yes:
                self._verts_g += diff
                self._verts_g *= 100.0
                self.id_pca_g *= 100.0
                self.ex_pca_g *= 100.0
            else:
                self._verts_g *= 0.01
                self._verts_g -= diff
                self.id_pca_g *= 0.01
                self.ex_pca_g *= 0.01

    def get_uv_to_vertex(self, texture_size, device):
        self.uv_to_vertex = get_uv_to_vertex(self.vertex_uv.clone(),
                                             self._verts_g.shape[0],
                                             texture_size, device)

    def morph(self, weights):
        return _morph_pca(self._verts_g, self.id_pca_g, weights, self.indexmap)

    def morph_ex(self, verts, weights, add_base=True):
        ex_out = _morph_pca(verts, self.ex_pca_g, weights, self.indexmap)
        if add_base:
            if ex_out.shape[1] != self.ex_base_g.shape[1]:
                ex_out = ex_out.index_add(1, self.indexmap, self.ex_base_g)
            else:
                ex_out = ex_out.add(self.ex_base_g)
        return ex_out

    def morph_color(self, weights):
        return _morph_pca(self.c_base_g, self.c_pca_g, weights, self.indexmap)

    def get_morphed(self, shape_coef, ex_coef, col_coef=None, add_base_ex=True):
        if not add_base_ex:
            ex_coef[:, 0] = 0
            ex_coef[:, 2] = 0
            ex_coef[:, 24] = 0
            ex_coef[:, 46] = 0
            ex_coef[:, 1] = 0
            ex_coef[:, 3] = 0
            ex_coef[:, 11] = 0
            ex_coef[:, 28] = 0
        return get_morphed(self._verts_g, self.id_pca_g, shape_coef, self.ex_base_g, self.ex_pca_g, ex_coef, self.c_base_g, self.c_pca_g, col_coef, self.indexmap, add_base_ex)

    def convert_morpher(self, weights):
        return []


def load_bfm2009(
        asset_path,
        batch_size,
        device,
        remove_eyes=False,
        fixed_uv=True,
        to_bfm_space=False):
    mesh = read_obj(os.path.join(asset_path, 'bfm2009_fixed_uv.obj' if fixed_uv else 'bfm2009.obj'))
    verts = mesh["verts"]
    tri = mesh["faces"]
    uv = mesh["uvs"]

    d = np.load(os.path.join(asset_path, 'databfm2009.npz'))

    scale = 1.0
    verts[..., 0] = -verts[..., 0]
    tri = d["triangles"]

    # shape = np.load("E:/Files/3d_face_reconstruction/MGCNet/shape.npy")
    # shape *= 0.01
    # shape -= np.array([-0.00322628, 0.04506069, 0.75983165])
    # from utils import write_obj
    # tri = d["triangles"]
    # mesh = {}
    # mesh["verts"] = shape
    # mesh["faces"] = tri
    # write_obj("./shape.obj", mesh)
    # exit()

    # verts = d["meanshape"]
    ex_base = d["meanex"]
    if remove_eyes:
        tri = remove_eye_faces(asset_path, tri)
    color = d["meancolor"] / 255.0
    id_pca = np.reshape(d["id_pca"], [1, -1, 80])
    ex_pca = np.reshape(d["ex_pca"], [1, -1, 64])
    color_pca = np.reshape(d["c_pca"], [1, -1, 80]) / 255.0
    verts = torch.from_numpy(verts).type(torch.float32).to(device).view([1, -1, 3]).repeat([batch_size, 1, 1])
    ex_base = torch.from_numpy(ex_base).type(torch.float32).to(device).view([1, -1, 3]).repeat([batch_size, 1, 1])
    color = torch.from_numpy(color).type(torch.float32).to(device).view([1, -1, 3]).repeat([batch_size, 1, 1])
    id_pca = torch.from_numpy(id_pca).type(torch.float32).to(device).repeat([batch_size, 1, 1]) * scale
    ex_pca = torch.from_numpy(ex_pca).type(torch.float32).to(device).repeat([batch_size, 1, 1]) * scale
    color_pca = torch.from_numpy(color_pca).type(torch.float32).to(device).repeat([batch_size, 1, 1])

    texture_g = None

    uv[:, 1] = 1.0 - uv[:, 1]
    uv = torch.from_numpy(uv).type(torch.float32).to(device)
    uv = uv.clone().view([1, -1, 2]).repeat(batch_size, 1, 1)

    uv2 = np.load(os.path.join(asset_path, 'bfm2zb_uv.npy'))
    uv2[:, 1] = 1.0 - uv2[:, 1]

    tri = torch.from_numpy(tri).type(torch.int64).to(device)

    uv2 = torch.from_numpy(uv2).type(torch.float32).to(device)
    uv2_batched = uv2.clone().view([1, -1, 2]).repeat(batch_size, 1, 1)

    bfm = BFM(verts, tri, uv, None, id_pca, ex_base, ex_pca, color, color_pca, texture_g, None, 80, device)
    bfm.vertex_uv = uv2
    bfm.uv2 = uv2_batched
    bfm.to_bfm_space(to_bfm_space)
    return bfm
