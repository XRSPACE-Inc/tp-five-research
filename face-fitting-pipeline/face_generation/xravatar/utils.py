import torch
# from torchvision.utils import save_image
# import torch.nn as nn
# import torch.nn.functional as F
import torchvision.transforms as tfs
import os
import numpy as np
# from math import cos, sin, radians
# import random
from PIL import Image
import struct
import math
# from config import image_size, asset_path


root_path_name = "avatar_generation_v3"


def nan_check(t, n=None):
    if isinstance(t, dict):
        for k, v in t.items():
            nan_check(v, k)
    elif isinstance(t, list):
        for v in t:
            nan_check(v)
    else:
        if (t != t).any():
            raise Exception("t (name:{}) contains nan".format(n))


def create_logger(logger_path='./logs'):
    import logging
    import datetime
    '''
    Create logger.
    '''
    logger = logging.getLogger('XRSpace Avatar 4.0')

    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)10s][%(levelname)s] %(message)s',
                                  datefmt='%Y/%m/%d %H:%M:%S')
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if logger_path is not None:
        os.makedirs(logger_path, exist_ok=True)
        logging_name = os.path.join(
            logger_path,
            "{}.txt".format(datetime.datetime.now().strftime("%Y_%m_%d__%H_%M")))
        file_handler = logging.FileHandler(logging_name)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger.info("logger created")
    return logger


def print_configs(config, logger):
    logger.info("=== configs ===")
    params = dir(config)
    for p in params:
        if p.count("__") != 2:
            logger.info("{} : {}".format(p, config.__dict__[p]))
    logger.info("=== configs ===")


def to_tensor(obj, tensorType):
    if not isinstance(obj, tensorType):
        if isinstance(obj, np.ndarray):
            obj = torch.from_numpy(obj).type(tensorType)
        elif isinstance(obj, tuple) or isinstance(obj, list):
            obj = tensorType(obj)
        elif isinstance(obj, torch.Tensor):
            obj = obj.type(tensorType)
        elif isinstance(obj, dict):
            for key, val in obj.items():
                obj[key] = to_tensor(val, tensorType)
    return obj


def to_device(obj, device):
    if isinstance(obj, np.ndarray):
        obj = torch.from_numpy(obj).to(device)
    elif isinstance(obj, tuple) or isinstance(obj, list):
        istuple = isinstance(obj, tuple)
        if istuple:
            obj = list(obj)
        res = []
        for o in obj:
            res.append(to_device(o, device))
        obj = res
        if istuple:
            obj = tuple(obj)
    elif isinstance(obj, torch.Tensor):
        obj = obj.to(device)
    elif isinstance(obj, dict):
        for key, val in obj.items():
            obj[key] = to_device(val, device)
    return obj


def view_img(
        image_g,
        convert_normal=False,
        combine_rows: int = -1,
        combine_cols: int = None):
    '''
    image_g : images (torch.tensor), shape [N, H, W, C] or [N, C, H, W] or [H, W, C] or [C, H, W]
    output : numpy array with shape [H2, W2, C]
    '''
    if len(image_g.shape) == 3:
        image_g = image_g.unsqueeze(0)

    S0, S1, S2, S3 = image_g.shape
    if S1 < 5:
        # [N, C, H, W]
        image_NHWC = image_g.permute(0, 2, 3, 1)
        img = image_NHWC.cpu().detach().numpy()
    elif S3 < 5:
        # [N, H, W, C]
        img = image_g.cpu().detach().numpy()
    else:
        # Not a image tensor?
        raise Exception("view_img error : wrong dim input is not a image tensor?")

    if convert_normal:
        img = img * 0.5 + 0.5

    val_max = np.amax(img)
    if val_max < 2.0:
        img = img * 255.0

    if combine_rows is not None or combine_cols is not None:
        batch_size, h, w, c = img.shape
        if combine_cols == -1:
            combine_cols = min(batch_size, 4)
        if combine_rows == -1:
            combine_rows = min(batch_size, 4)
        if combine_rows is not None and combine_cols is not None:
            if combine_cols * combine_rows >= batch_size:
                res = np.zeros([h * combine_rows, w * combine_cols, c])
                for j in range(combine_cols):
                    for i in range(combine_rows):
                        if (j * combine_rows + i) < batch_size:
                            res[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = img[j * combine_rows + i, :, :, :]
            else:
                raise Exception("view_img error : combine_cols * combine_rows >= batch_size")
        elif combine_cols is None:
            combine_cols = math.ceil(batch_size / combine_rows)
            res = np.zeros([h * combine_rows, w * combine_cols, c])
            for j in range(combine_cols):
                for i in range(combine_rows):
                    if (j * combine_rows + i) < batch_size:
                        res[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = img[j * combine_rows + i, :, :, :]
        else:
            combine_rows = math.ceil(batch_size / combine_cols)
            res = np.zeros([h * combine_rows, w * combine_cols, c])
            for i in range(combine_rows):
                for j in range(combine_cols):
                    if (i * combine_cols + j) < batch_size:
                        res[i * h:(i + 1) * h, j * w:(j + 1) * w, :] = img[i * combine_cols + j, :, :, :]
        img = res

    return img.astype(np.uint8)


def save_tensor_img(tensor, path):
    img = view_img(tensor)
    if ".png" in path:
        img = np.concatenate([img, np.clip(img[..., :3].mean(-1)[..., np.newaxis], 0, 1) * 255], -1)
        Image.fromarray(img.astype(np.uint8)).convert("RGBA").save(path)
    else:
        Image.fromarray(img).save(path)


def combine_images(images, axis=1):
    '''
    images are array or [H2, W2, C] image. (output from view_img)
    output one combined image.
    '''
    res = images[0]
    for i in range(1, len(images)):
        res = np.concatenate((res, images[i]), axis=axis)
    return res


def load_texture(path, batch_size, texture_size, device):
    img = Image.open(path)
    img = img.resize((texture_size, texture_size))

    img = tfs.ToTensor()(img)[0:3, :, :]
    img = img.to(device)

    return img.view([1, 3, texture_size, texture_size]).repeat(batch_size, 1, 1, 1)


def load_texture_cached(path, texture_size, file_ext=".jpg"):
    resized = path.replace(file_ext, ".npy")
    try:
        img = np.load(resized)
    except Exception:
        img = Image.open(path).convert("RGB")
        img = img.resize((texture_size, texture_size), Image.BILINEAR)
        img = np.array(img)
        np.save(resized, img)

    return img


def _load_mesh(asset_path, gender):
    mesh = read_obj(os.path.join(asset_path, '{}_out_mesh.obj'.format(gender)))
    return [
        mesh['verts'].astype(np.float32), mesh['uvs'].astype(np.float32),
        mesh['faces'].astype(np.int32), mesh['norms'].astype(np.float32)
    ]


def init_mesh(mesh_params, batch_size, texture_size, device, testing):
    texture_size = texture_size - 1
    verts, uv, tri, norms = mesh_params
    verts_g = torch.from_numpy(verts).repeat(batch_size, 1, 1).to(device)
    uncalc_uv = uv.copy()
    crop = not testing
    uv = uv.astype(np.float32)

    # if crop:
    #     # crop uv same as crop texture
    #     uv[:, 0] -= 0.25
    #     uv[:, 1] = 1.0 - uv[:, 1] - 0.125
    #     uv *= texture_size * 2
    # else:
    #     uv[:, 1] = 1.0 - uv[:, 1]
    #     uv *= texture_size

    # uv[uv < 0] = 0.0
    # uv[uv > texture_size] = texture_size

    # uvs = np.reshape(uv, [1, -1, 2])
    # uvs_g = torch.from_numpy(uvs).to(device)
    # uvs_g = uvs_g.repeat(batch_size, 1, 1)
    # for i in range(batch_size):
    #     uvs_g[i] += texture_size * i
    #     uvs_g[i, :, 1] -= i

    # uvs_g = torch.clamp(uvs_g, max=texture_size * batch_size)

    if crop:
        # crop uv same as crop texture
        uv[:, 0] -= 0.25
        uv[:, 1] = 1.0 - uv[:, 1] - 0.125
        uv *= 2
    else:
        uv[:, 1] = 1.0 - uv[:, 1]

    uv[uv < 0] = 0.0
    uv[uv > 1] = 1.0

    uvs = np.reshape(uv, [1, -1, 2])
    uvs_g = torch.from_numpy(uvs).to(device)
    uvs_g = uvs_g.repeat(batch_size, 1, 1)

    tri_g = torch.from_numpy(tri.astype(np.int64)).to(device)
    norms_g = torch.from_numpy(norms).repeat(batch_size, 1, 1).to(device)
    return verts_g, uncalc_uv, uvs_g, tri, tri_g, norms_g


def load_standard_mesh(gender, batch_size, texture_size, device, testing):
    return init_mesh(_load_mesh(gender), batch_size, texture_size, device, testing)


def load_bs(asset_path, gender, batch_size, device):
    with open(os.path.join(asset_path, '{}_bs_out.bin'.format(gender)), "rb") as f:
        bscount = struct.unpack('i', bytearray(f.read(4)))[0]
        vc = struct.unpack('i', bytearray(f.read(4)))[0]
        mds = []
        nds = []
        for i in range(bscount):
            mds.append([])
            for _ in range(vc * 3):
                mds[i].append(struct.unpack('f', bytearray(f.read(4))))
            nds.append([])
            for _ in range(vc * 3):
                nds[i].append(struct.unpack('f', bytearray(f.read(4))))

    mds = np.array(mds).astype(np.float32).reshape(bscount, -1)
    mds = np.swapaxes(mds, 0, 1)
    nds = np.array(nds).astype(np.float32).reshape(bscount, -1)
    nds = np.swapaxes(nds, 0, 1)
    mds_g = torch.from_numpy(mds).repeat(batch_size, 1, 1).to(device)
    nds_g = torch.from_numpy(nds).repeat(batch_size, 1, 1).to(device)
    return mds_g, nds_g


def load_mesh_lm(asset_path, gender, device):
    with open(os.path.join(asset_path, '{}_mesh_lm68.txt'.format(gender)), "r+") as f:
        line = f.read()
    line = line.replace(" ", "")
    each = line.split(',')
    lms = [0] * 68
    for i in range(68):
        lms[i] = int(each[i])
    return torch.from_numpy(np.array(lms).astype(np.int64)).to(device)


def load_hbfm2019_lm(asset_path, device):
    with open(os.path.join(asset_path, 'bfm2019_lm.txt'), "r+") as f:
        line = f.read()
    line = line.replace(" ", "")
    each = line.split(',')
    lms = [0] * 68
    for i in range(68):
        lms[i] = int(each[i])
    return torch.from_numpy(np.array(lms).astype(np.int64)).to(device)


def flip_lm(lm):
    def swap(lst, i1, i2):
        t = lst[i1].copy()
        lst[i1] = lst[i2]
        lst[i2] = t
        return lst

    for i in range(8):
        lm = swap(lm, i, 16 - i)

    for i in range(5):
        lm = swap(lm, 17 + i, 26 - i)

    lm = swap(lm, 31, 35)
    lm = swap(lm, 32, 34)

    lm = swap(lm, 36, 45)
    lm = swap(lm, 37, 44)
    lm = swap(lm, 38, 43)
    lm = swap(lm, 39, 42)
    lm = swap(lm, 40, 47)
    lm = swap(lm, 41, 46)

    lm = swap(lm, 48, 54)
    lm = swap(lm, 49, 53)
    lm = swap(lm, 50, 52)
    lm = swap(lm, 55, 59)
    lm = swap(lm, 56, 58)
    lm = swap(lm, 60, 64)
    lm = swap(lm, 61, 63)
    lm = swap(lm, 65, 67)

    return lm


def load_hbfm2009_lm(asset_path, device, flip=True):
    with open(os.path.join(asset_path, 'bfm2009_lm.txt'), "r+") as f:
        line = f.read()
    line = line.replace(" ", "")
    each = line.split(',')
    lms = [0] * 68
    for i in range(68):
        lms[i] = int(each[i])
    lms = np.array(lms).astype(np.int64)
    if flip:
        lms = flip_lm(lms)
    return torch.from_numpy(lms).to(device)


def remove_indices(faces, idx):
    res = []
    for t in faces:
        if t[0] in idx or t[1] in idx or t[2] in idx:
            continue
        res.append(t)
    return np.array(res)


def keep_indices(faces, idx):
    res = []
    for t in faces:
        if t[0] in idx or t[1] in idx or t[2] in idx:
            res.append(t)
    return np.array(res)


def remove_eye_faces(asset_path, faces):
    return remove_indices(faces, np.load(os.path.join(asset_path, "eidx_face.npy")))


def read_as_list(file_name):
    res = []
    with open(file_name, "r") as f:
        lines = f.readline().split(",")
    f.close()
    for line in lines:
        res.append(int(line))
    return res


def read_as_hash(file_name):
    res = set()
    with open(file_name, "r") as f:
        lines = f.readline().split(",")
    f.close()
    for line in lines:
        res.add(int(line))

    return res


def read_texture_as_list(img, uvs):
    res = []
    h, w = img.shape
    for i in range(len(uvs)):
        x = int(uvs[i][0] * w)
        y = h - int(uvs[i][1] * h) - 1
        col = img[y][x]
        if col > 0.0:
            res.append(i)
    return res


def read_texture_file_as_list(file_name, uvs):
    img = np.array(Image.open(file_name).convert("L"))
    return read_texture_as_list(img, uvs)


def to_np(array_like, squeeze=False):
    if isinstance(array_like, torch.Tensor):
        if squeeze and array_like.shape[0] == 1:
            array_like = array_like.squeeze(0)
        array_like = array_like.cpu().detach().numpy()
    elif isinstance(array_like, np.ndarray):
        if squeeze and array_like.shape[0] == 1:
            array_like = array_like.squeeze(0)
    elif isinstance(array_like, list) or isinstance(array_like, tuple) or isinstance(array_like, int):
        array_like = np.array(array_like)
    elif array_like is None:
        array_like = np.array([])
    elif array_like is not None:
        print(array_like)
        print(type(array_like))
        raise Exception("array_like cannot convert to numpy array")
    return array_like


def read_obj(file_name, to_triangle_face=True):
    with open(file_name, "r") as f:
        lines = f.readlines()

    d = {}

    vertices = []
    normals = []
    uvs = []
    faces = []
    uv_faces = []
    norm_faces = []
    colors = []
    mtl_index = {}
    do_uv_face = True
    do_norm_face = True

    def read_face(faces, vals, face_type=0):
        face = []
        for i in range(1, len(vals)):
            face.append(int(vals[i].split('/')[face_type]) - 1)
        faces.append(face)

    def read_face_triangle(faces, vals, face_type=0):
        for i in range(3, len(vals)):
            face = []
            face.append(int(vals[1].split('/')[face_type]) - 1)
            face.append(int(vals[i - 1].split('/')[face_type]) - 1)
            face.append(int(vals[i].split('/')[face_type]) - 1)
            faces.append(np.array(face))

    read_face_fn = read_face_triangle if to_triangle_face else read_face

    for line in lines:
        vals = line.split(' ')
        if vals[0] == "v":
            if (len(vals) > 4):
                colors.append([float(vals[4]), float(vals[5]), float(vals[6])])
            vertices.append([float(vals[1]), float(vals[2]), float(vals[3])])
        elif vals[0] == "vn":
            normals.append([float(vals[1]), float(vals[2]), float(vals[3])])
        elif vals[0] == "vt":
            uvs.append([float(vals[1]), float(vals[2])])
        elif vals[0] == "f":
            read_face_fn(faces, vals)
            if do_uv_face:
                try:
                    read_face_fn(uv_faces, vals, 1)
                except Exception:
                    do_uv_face = False
                    uv_faces = []
            if do_norm_face:
                try:
                    read_face_fn(norm_faces, vals, 2)
                except Exception:
                    do_norm_face = False
                    norm_faces = []
        elif vals[0] == "usemtl":
            mtl_index[len(faces)] = vals[1].replace('\n', '')

    d["verts"] = np.array(vertices)
    d["faces"] = np.array(faces)
    d["uv_faces"] = np.array(uv_faces) if do_uv_face else uv_faces
    d["norm_faces"] = np.array(norm_faces) if do_norm_face else norm_faces
    d["uvs"] = np.array(uvs) if len(uvs) > 0 else []
    d["norms"] = np.array(normals) if len(normals) > 0 else []
    d["vcs"] = np.array(colors) if len(colors) > 0 else []
    d["mtl_index"] = mtl_index

    return d


def write_obj(file, mesh):
    def write_face(i, mlt_dict, arr, arr2=None, arr3=None):
        res = "f"
        if i in mlt_dict:
            res = "usemtl {}\nf".format(mlt_dict[i])
        if arr2 is not None and arr3 is not None:
            for i in range(len(arr)):
                res += " {}/{}/{}".format(arr[i] + 1, arr2[i] + 1, arr3[i] + 1)
        elif arr2 is not None:
            for i in range(len(arr)):
                res += " {}/{}/".format(arr[i] + 1, arr2[i] + 1)
        elif arr3 is not None:
            for i in range(len(arr)):
                res += " {}//{}".format(arr[i] + 1, arr3[i] + 1)
        else:
            for i in range(len(arr)):
                res += " {}//".format(arr[i] + 1)
        return res + "\n"

    verts = to_np(mesh["verts"], True).astype(np.float32)
    indices = to_np(mesh["faces"]) if "faces" in mesh else []
    uvs = to_np(mesh["uvs"], True) if "uvs" in mesh else []
    normals = to_np(mesh["norms"], True) if "norms" in mesh else []
    colors = to_np(mesh["vcs"], True) if "vcs" in mesh else []
    uv_faces = to_np(mesh["uv_faces"]) if "uv_faces" in mesh else []
    norm_faces = to_np(mesh["norm_faces"]) if "norm_faces" in mesh else []

    uv_indices = uv_faces if len(uv_faces) > 0 else indices
    norm_indices = norm_faces if len(norm_faces) > 0 else indices

    mtl_index = mesh["mtl_index"] if "mtl_index" in mesh else {}

    with open(file, "w") as f:
        if len(colors) == len(verts):
            for i in range(len(verts)):
                f.write("v {} {} {} {} {} {}\n".format(
                    verts[i][0], verts[i][1], verts[i][2], colors[i][0],
                    colors[i][1], colors[i][2]))
        else:
            for v in verts:
                f.write("v {} {} {}\n".format(v[0], v[1], v[2]))
        f.write("\n")
        withuv = False
        withnor = False
        if len(uvs) > 0:
            for uv in uvs:
                f.write("vt {} {}\n".format(uv[0], uv[1]))
            withuv = True
            f.write("\n")
        if len(normals) > 0:
            for n in normals:
                f.write("vn {} {} {}\n".format(n[0], n[1], n[2]))
            withnor = True
            f.write("\n")

        if len(indices) > 0:
            if not withuv and not withnor:
                for i in range(len(indices)):
                    f.write(write_face(i, mtl_index, indices[i]))
            else:
                if withuv and withnor:
                    for i in range(len(indices)):
                        f.write(write_face(i, mtl_index, indices[i], uv_indices[i], norm_indices[i]))
                elif withuv:
                    for i in range(len(indices)):
                        f.write(write_face(i, mtl_index, indices[i], uv_indices[i]))
                else:
                    for i in range(len(indices)):
                        f.write(write_face(i, mtl_index, indices[i], None, norm_indices[i]))


def next_data(train_datas, loader_tag="train_loader"):
    try:
        it = train_datas[loader_tag + "it"]
        return to_device(next(it), train_datas["device"])
    except Exception:
        dataloader = train_datas[loader_tag]
        it = iter(dataloader)
        train_datas[loader_tag + "it"] = it
        return to_device(next(it), train_datas["device"])
