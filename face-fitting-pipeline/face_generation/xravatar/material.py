import torch
import torch.nn.functional as F


class Material:
    def __init__(
            self,
            faces=None,
            uvs=None,
            textures=None,
            vc=None,
            additional_attrs=None,
            need_sh=False,
            need_normal=False,
            face_range=None):
        self.faces = faces
        self.face_range = face_range
        self.need_update_faces = faces is None
        self.uvs = uvs
        self.textures = textures if textures is not None else {}
        self.vc = vc
        self.additional_attrs = additional_attrs if additional_attrs is not None else {}
        self.attributes = {}
        self.image_size = 512
        self.textures_out = None
        self.need_sh = need_sh
        self.need_normal = need_normal

    def sample_texture(self, texture, uvs):
        N, C, _, _ = texture.shape
        uvs = uvs.clone()

        uvs = uvs * 2.0 - 1.0

        tex = F.grid_sample(texture, uvs, align_corners=False)
        tex = tex.view([N, C, self.image_size, self.image_size])

        return tex

    def set_normal(self, normals):
        self.additional_attrs['norm'] = normals if self.need_normal else None

    def set_sh(self, sh):
        self.additional_attrs['sh'] = sh if self.need_sh else None

    def set_additional_attrs(self, key, value):
        self.additional_attrs[key] = value

    def get_attribute(self):
        self.attributes = {}
        if self.vc is not None:
            self.attributes['vc'] = self.vc.clone()
        elif self.uvs is not None:
            self.attributes["uvs"] = self.uvs.clone()

        for k in self.additional_attrs:
            if self.additional_attrs[k] is not None:
                self.attributes[k] = self.additional_attrs[k].clone()

        for k in self.attributes:
            self.attributes[k] = self.attributes[k][:, self.faces]
        return self.attributes

    def mask_face(self, pix_to_face, mask):
        if self.face_range is not None:
            shape = mask.shape
            face_mask = torch.logical_or(pix_to_face < self.face_range[0], pix_to_face >= self.face_range[1])
            face_mask = face_mask.reshape(shape)
            mask = torch.logical_or(mask.clone(), face_mask).reshape(shape)
        return mask

    def set_attributes(self, attributes):
        self.attributes = attributes

    def sample_image(self, image_size, do_sh):
        self.image_size = image_size

        if self.need_normal:
            n = self.attributes['norm']
            # change normal index
            n_zero = n == 0
            # n[..., [1, 0]] = n[..., [0, 1]]
            n = n * 0.5 + 0.5
            # n[..., 0] = 1.0 - n[..., 0]
            n = n.masked_fill(n_zero, 0)
            # n = F.normalize(n, eps=1e-6, dim=-1)
            n = n.view(-1, self.image_size, self.image_size, 3).permute(0, 3, 1, 2)
            self.attributes['norm'] = n

        if self.need_sh and do_sh:
            self.attributes['sh'] = self.attributes['sh'].permute(0, 3, 1, 2)

        if self.vc is not None:
            self.attributes["vc"] = self.attributes["vc"].permute(0, 3, 1, 2)

        if len(self.textures) > 0:
            self.textures_out = {}
            for key in self.textures:
                self.textures_out[key] = self.sample_texture(
                    self.textures[key].clone(), self.attributes["uvs"])

    def get_albedo(self):
        if self.vc is not None:
            return self.attributes["vc"]
        if self.textures_out is not None:
            return self.textures_out["albedo"]
        raise Exception("material.get_albedo error: no result")

    def get_sh(self):
        return self.attributes["sh"]

    def get_normal(self):
        return self.attributes["norm"]
