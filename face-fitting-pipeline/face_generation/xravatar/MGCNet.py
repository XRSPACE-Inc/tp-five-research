import torch
import torch.nn as nn
import torchvision
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import face_alignment
import os
from skimage import transform as trans
import cv2


def convert_weight(w):
    if len(w.shape) == 4:
        return w.transpose(3, 2, 0, 1)
    return w


def make_state_dict(name_list, weights):
    d = {}
    for i in range(len(name_list)):
        name = name_list[i]
        if len(name) > 0:
            d[name] = convert_weight(weights[i])
    return d


def load_from_np(m, name_file, weight_file):
    name_file = np.load(name_file)
    weight_file = np.load(weight_file, allow_pickle=True)
    state_dict = make_state_dict(name_file, weight_file)
    for _, module in enumerate(m.named_modules()):
        name = module[0]
        item = module[1]
        if len(name) > 0 and isinstance(item, nn.Sequential) is False:
            sd = item.state_dict()
            for key in sd.keys():
                full = "{}.{}".format(name, key)
                if full in state_dict and sd[key].shape == state_dict[full].shape:
                    sd[key] = torch.from_numpy(state_dict[full])
                else:
                    if full in state_dict:
                        print("{} in state_dict but not same size got : {} and loaded size: {}".format(
                            full, sd[key].shape, state_dict[full].shape))
                    else:
                        print("{} not in state_dict".format(full))
            item.load_state_dict(sd)
    print("model loaded")
    # torch.save(m.state_dict(), "mgcnet.pth")
    # exit()
    return m


def init_model(m, last_file_path, map_location="cuda", print_log=False):
    if os.path.isfile(last_file_path):
        print("== loading", last_file_path, " ...")
        try:
            # m.load_state_dict(torch.load(last_file_path))
            addModule = False
            if map_location is not None:
                state_dict = torch.load(last_file_path, map_location=map_location)
                # if map_location != "cuda":
                #     addModule = True
            else:
                state_dict = torch.load(last_file_path, map_location=torch.device("cpu"))
            for _, module in enumerate(m.named_modules()):
                name = module[0]
                item = module[1]
                # if print_log:
                #     print("name: {}, module: {}".format(name, item))
                if len(name) > 0 and isinstance(item, nn.Sequential) is False:
                    sd = item.state_dict()
                    deep_check = False
                    for key in sd.keys():
                        full = "{}.{}".format(name, key)
                        if addModule:
                            full = "module." + full
                        if full in state_dict and sd[key].shape == state_dict[full].shape:
                            # if print_log:
                            #     print("{}:{}".format(full, state_dict[full].shape))
                            sd[key] = state_dict[full]
                        else:
                            if full in state_dict:
                                print("{} in state_dict but not same size got : {} and loaded size: {}".format(
                                    full, sd[key].shape, state_dict[full].shape))
                            else:
                                deep_check = True
                                print("{} not in state_dict".format(full))
                    if deep_check and print_log:
                        print("st : {}".format(sd))
                    item.load_state_dict(sd)
            print("model loaded")
        except Exception:
            print("load model with error, recreate a new one.")
    else:
        print("model not existed, create a new one.")

    return m


def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes, out_channels=places, kernel_size=7, stride=stride, padding=3, bias=True),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
    )


class Bottleneck(nn.Module):
    def __init__(self, in_places, places, stride=1, downsampling=False, maxpooling=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling
        self.maxpooling = maxpooling

        self.conv1 = nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=False)

        if self.downsampling:
            self.downsample = nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride, bias=True)
        if self.maxpooling:
            self.maxpool = nn.MaxPool2d(kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        # print(out.shape, "=== conv1")

        out = self.conv2(out)
        out = self.relu(out)
        # print(out.shape, "=== conv2")

        out = self.conv3(out)
        # print(out.shape, "=== conv3")

        if self.downsampling:
            residual = self.downsample(x)
            # print(out.shape, "=== down")
        if self.maxpooling:
            residual = self.maxpool(x)
            # print(out.shape, "=== maxpool")

        out += residual
        out = self.relu(out)
        # print(out.shape, "=== add")
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes=257, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion
        # self.conv1 = Conv1(in_planes=3, places=64)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)

        self.layer1 = self.make_layer(in_places=64, places=64, block=blocks[0], stride=2)
        self.layer2 = self.make_layer(in_places=256, places=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=512, places=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=1024, places=512, block=blocks[3], stride=1, pool=False)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Conv2d(2048, 257, kernel_size=1, stride=1, bias=True)

    def make_layer(self, in_places, places, block, stride, pool=True):
        layers = []
        layers.append(Bottleneck(in_places, places, downsampling=True))
        for _ in range(1, block - 1):
            layers.append(Bottleneck(places * self.expansion, places))
        if pool:
            layers.append(Bottleneck(places * self.expansion, places, stride, maxpooling=True))
        else:
            layers.append(Bottleneck(places * self.expansion, places))

        return nn.Sequential(*layers)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print(x.shape)
        # print("layer1=========")
        x = self.layer1(x)
        # print(x.shape)
        # print("layer2=========")
        x = self.layer2(x)
        # print(x.shape)
        # print("layer3=========")

        x = self.layer3(x)
        # print(x.shape)
        # print("layer4=========")
        x = self.layer4(x)
        # print(x.shape)
        # exit()

        code = self.avgpool(x)
        # print(x.shape)
        # code = torch.mean(x, [2, 3], keepdim=True)

        res = self.fc(code).view([-1, 257])
        # print(x.shape)
        results = {}
        results["code"] = code
        results["coef"] = res
        results["shape"] = res[..., :80].unsqueeze(-1)
        results["vc"] = res[..., 80:160].unsqueeze(-1)
        results["ex"] = res[..., 160:224].unsqueeze(-1)
        results["trans"] = res[..., 224:230]
        results["sh"] = res[..., 230:257].unsqueeze(-1)
        return results


def parse_coef(coef):
    results = {}
    results["shape"] = coef[..., :80].unsqueeze(-1)
    results["vc"] = coef[..., 80:160].unsqueeze(-1)
    results["ex"] = coef[..., 160:224].unsqueeze(-1)
    results["trans"] = coef[..., 224:230]
    results["sh"] = coef[..., 230:257].unsqueeze(-1)
    return results


def ResNet50():
    return ResNet([3, 4, 6, 3])


def load_datas_from_np(name_file="./net_name_pytorch.npy", weight_file="./net_weights.npy", device="cuda"):
    return load_from_np(ResNet50(), name_file, weight_file), face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device=device, flip_input=False, face_detector='sfd')


def load_datas(model_file="./datas/MGCNet/mgcnet.pth", device="cuda"):
    model = init_model(ResNet50(), model_file)
    model.to(device)
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.THREE_D, device=device, flip_input=False, face_detector='sfd')
    return model, fa


def get_lm5(lm68):
    return np.array([np.mean(lm68[36:42], 0), np.mean(lm68[42:48], 0), lm68[33], lm68[48], lm68[54]])


def crop_align_affine_transform(lm2d, image, crop_size, std_landmark):
    '''
    Crop align affine transform from:
        https://github.com/jiaxiangshang/MGCNet/blob/master/tools/preprocess/crop_image_affine.py
    '''
    lm5 = get_lm5(lm2d)
    # Transform
    std_points = std_landmark * crop_size

    tform = trans.SimilarityTransform()
    tform.estimate(lm5, std_points)
    M = tform.params[0:2, :]

    # rot_angle = tform.rotation * 180.0 / np.pi

    img_warped = cv2.warpAffine(image, M, (crop_size, crop_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    h_lm2d_home = np.concatenate([lm2d, np.ones([lm2d.shape[0], 1])], axis=1)
    lm_trans = np.matmul(M, np.array(np.transpose(h_lm2d_home)))
    lm_trans = np.transpose(lm_trans)

    # Image.fromarray(img_warped).save("./1_1.jpg")

    return lm_trans, img_warped, tform


def detect_landmark(image, fa):
    '''
    Detect landmark using fa. (face alignment)
    '''
    # h, w, c = image.shape

    detected_faces = fa.face_detector.detect_from_image(image.copy())
    lm_howfar = fa.get_landmarks(image[..., ::-1], detected_faces=detected_faces)

    if lm_howfar is not None:
        list_hf = []
        list_size_detected_face = []
        for i in range(len(lm_howfar)):
            l_hf = lm_howfar[i]
            # l_hf = l_hf * scale
            list_hf.append(l_hf)

            bbox = detected_faces[i]
            list_size_detected_face.append(bbox[2] - bbox[0] + bbox[3] - bbox[1])

        list_size_detected_face = np.array(list_size_detected_face)
        idx_max = np.argmax(list_size_detected_face)
        return list_hf[idx_max]
    return None


def get_imgae_meta(img):
    exif = img.getexif()
    meta = {}
    for tag_id in exif:
        tag = TAGS.get(tag_id, tag_id)
        data = exif.get(tag_id)
        try:
            if isinstance(data, bytes):
                data = data.decode()
        except Exception:
            continue
        meta[tag] = data
    return meta


def mgc_preprocess(img_path, fa):
    '''
    Load image and detect landmark.
    Returns:
        None if no face detected.
        lm, img
    '''
    image = Image.open(img_path).convert("RGB")
    meta = get_imgae_meta(image)
    if 'Orientation' in meta:
        ori = meta['Orientation']
        if ori == 8:
            image = image.rotate(90, expand=1)
        elif ori == 3:
            image = image.rotate(180)
        elif ori == 6:
            image = image.rotate(270, expand=1)

    W, H = image.size
    if H != W:
        d = abs(H - W)
        half = int(d / 2)
        image = np.array(image)
        if H > W:
            temp = np.zeros([H, H, 3], dtype=np.uint8)
            temp[:, half:half + W, :] = image
            image = temp
        else:
            temp = np.zeros([W, W, 3], dtype=np.uint8)
            temp[half:half + H, :, :] = image
            image = temp
        image = Image.fromarray(image)
        W, H = image.size

    landmark_input = image

    if H > 1000:
        ratio = 1000.0 / W
        newsize = (int(1000), int(ratio * H))
        landmark_input = image.resize(newsize, Image.NEAREST)

    landmark_input = np.array(landmark_input)

    with torch.no_grad():
        detected_result = detect_landmark(landmark_input, fa)
    if detected_result is None:
        return None
    lm68 = detected_result[..., :2]

    std_224_bfm09 = np.array([
        81.672401, 88.470589,
        141.862671, 88.462921,
        112.000000, 132.863434,
        87.397392, 153.562943,
        136.007263, 153.552078
    ])
    std_224_bfm09 = np.reshape(std_224_bfm09, [-1, 2]) / 224.0

    h, _w, _c = landmark_input.shape
    if h != H:
        lm68 = lm68 / h * H

    lm_trans, img_warped, _ = crop_align_affine_transform(lm68, np.array(image), 1024, std_224_bfm09)

    return {
        "lm": lm_trans,
        "img": torchvision.transforms.ToTensor()(img_warped).unsqueeze(0),
        "img_orig": Image.fromarray(img_warped)
    }


if __name__ == '__main__':
    from utils import save_tensor_img, write_obj
    from blendshape import load_bfm2009
    from rendering import Renderer

    os.makedirs("./results/MGCNet/", exist_ok=True)
    model, fa = load_datas()
    model.to("cuda")
    datas = preprocess("E:/Files/3d_face_reconstruction/MGCNet/data/test_/image00008.jpg", fa)
    if datas is None:
        print("no face detected")
        exit(1)
    img = datas["img_warped"]
    save_tensor_img(img, "./results/MGCNet/src.jpg")
    # print(img.shape)
    with torch.no_grad():
        coef = model(img.to("cuda"))

    bfm = load_bfm2009("./datas/BFM/", 1, "cuda", False, True)

    _, out_shape, vc = bfm.get_morphed(coef["shape"], coef["ex"], coef["vc"])

    write_obj(
        "./results/recon_out.obj",
        out_shape[0].cpu().detach().numpy(),
        bfm.tri_g.cpu().detach().numpy(),
        colors=vc[0].cpu().detach().numpy(),
    )

    renderer = Renderer(224, 1, "cuda")

    res = renderer(
        verts=out_shape,
        faces=bfm.tri_g,
        trans=coef["trans"],
        sh=coef["sh"],
        vc=vc,
    )

    sh = res["sh"]
    vc = res["vc"]

    save_tensor_img(torch.clamp(sh, min=0.0, max=1.0), "./results/MGCNet/sh.jpg")
    save_tensor_img(torch.clamp(vc, min=0.0, max=1.0), "./results/MGCNet/vc.jpg")
    save_tensor_img(torch.clamp(sh * vc, min=0.0, max=1.0), "./results/MGCNet/srcout.jpg")
