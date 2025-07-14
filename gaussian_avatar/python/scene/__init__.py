import os
from PIL import Image
from pathlib import Path
import torch
import numpy as np
import json
from tqdm import tqdm

from scene.gaussian_model import GaussianModel  # noqa
from scene.cameras import Camera
from arguments import ModelParams  # noqa
from utils.general_utils import PILtoTensor, PILtoTensorCropped
from utils.graphics_utils import focal2fov
from src.camera_model import _compute_rotation
import base64
from io import BytesIO


def base64_to_image(base64_str: str) -> Image.Image:
    bio = BytesIO(base64.b64decode(base64_str.encode(encoding='utf-8')))
    return Image.open(bio)
    


class Scene_mica:
    def __init__(
        self,
        datadir,
        mica_datadir,
        train_type,
        white_background,
        device,
        ict: bool=False,
        mediapipe: bool=False,
        transfer_data_path: str=None,
        landmark: bool=False,
    ):
        self.device = device
        # train_type: 0 for train, 1 for test, 2 for eval
        frame_delta = 1  # default mica-tracking starts from the second frame
        images_folder = os.path.join(datadir, "imgs")
        parsing_folder = os.path.join(datadir, "parsing")
        alpha_folder = os.path.join(datadir, "alpha")

        image_size = (540, 720)

        self.bg_image = torch.zeros((3, image_size[1], image_size[0]))
        # if white_background:
        #     self.bg_image[:, :, :] = 1
        # else:
        #     self.bg_image[1, :, :] = 1

        mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint')
        ict_ckpt_dir = os.path.join(datadir, 'checkpoint')
        mediapipe_ckpt_dir = os.path.join(datadir, 'mediapipe_bs')

        self.N_frames = len(os.listdir(mica_ckpt_dir))
        self.cameras = []
        test_num = 500
        eval_num = 50
        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames - test_num)

        ckpt_path = os.path.join(mica_ckpt_dir, '00000.frame')
        payload = torch.load(ckpt_path)

        if ict:
            ict_ckpt_path = os.path.join(ict_ckpt_dir, '00000.npz')
            ict_payload = np.load(ict_ckpt_path, allow_pickle=True)['data'][()]
            self.shape_param = torch.as_tensor(ict_payload['id'])[None, ...]
            self.bs_param = torch.as_tensor(ict_payload['exp'])[None, ...]
            self.angle_param = torch.as_tensor(ict_payload['angle'])[None, ...]
            self.trans_param = torch.as_tensor(ict_payload['trans'])[None, ...]
        else:
            flame_params = payload['flame']
            self.shape_param = torch.as_tensor(flame_params['shape'])
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        FovY = focal2fov(fl_x, orig_w)
        FovX = focal2fov(fl_y, orig_h)
        if train_type == 0:
            range_down = 0
            range_up = train_num
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames

        for frame_id in tqdm(range(range_down, range_up)):
            if frame_id % 4 != 0:
                continue
            image_name_mica = str(frame_id).zfill(5)  # obey mica tracking
            image_name_ori = str(frame_id+frame_delta).zfill(5)
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path)

            if ict:
                ict_ckpt_path = os.path.join(ict_ckpt_dir, image_name_ori+'.npz')
                ict_payload = np.load(ict_ckpt_path, allow_pickle=True)['data'][()]
                shape_param = torch.as_tensor(ict_payload['id'])[None, ...]
                angle_param = torch.as_tensor(ict_payload['angle'])[None, ...]
                trans_param = torch.as_tensor(ict_payload['trans'])[None, ...]
                eyes_pose = angle_param
                eyelids = trans_param
                jaw_pose = shape_param
                exp_param = torch.as_tensor(ict_payload['exp'])[None, ...]
            else:
                flame_params = payload['flame']
                exp_param = torch.as_tensor(flame_params['exp'])
                eyes_pose = torch.as_tensor(flame_params['eyes'])
                eyelids = torch.as_tensor(flame_params['eyelids'])
                jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0]
            R = np.transpose(w2cR)  # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            image_path = os.path.join(images_folder, image_name_ori+'.jpg')
            image = Image.open(image_path).resize(image_size, Image.Resampling.BICUBIC)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]

            # alpha
            alpha_path = os.path.join(alpha_folder, image_name_ori+'.png')
            alpha = Image.open(alpha_path).resize(image.size, Image.Resampling.BICUBIC)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path).resize(image.size, Image.Resampling.BICUBIC)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(parsing_folder, image_name_ori+'_mouth.png')
            mouth_mask = Image.open(mouth_mask_path).resize(image.size, Image.Resampling.BICUBIC)
            mouth_mask = PILtoTensor(mouth_mask)

            bs_param = None
            if ict:
                bs_param = exp_param
            elif mediapipe:
                mediapipe_ckpt_path = os.path.join(mediapipe_ckpt_dir, image_name_ori+'.npy')
                mediapipe_payload = np.load(mediapipe_ckpt_path).astype(np.float32)
                bs_param = torch.as_tensor(mediapipe_payload)[None, ...]

            transfer_data = None
            if transfer_data_path is not None:
                transfer_data = torch.load(
                    os.path.join(transfer_data_path, f'{frame_id}.data'),
                    map_location='cpu',
                )

            camera_indiv = Camera(
                colmap_id=frame_id, R=R, T=T,
                FoVx=FovX, FoVy=FovY,
                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
                bs_param=bs_param,
                transfer_data=transfer_data,
                image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)

    def getCameras(self):
        return self.cameras


class Scene_mica_1:
    def __init__(
        self,
        datadir,
        mica_datadir,
        train_type,
        white_background,
        device,
        ict: bool=False,
        mediapipe: bool=False,
        do_not_skip: bool=False,
        transfer_data_path: str=None,
    ):
        self.device = device
        # train_type: 0 for train, 1 for test, 2 for eval
        self.frame_delta = 1  # default mica-tracking starts from the second frame
        self.images_folder = os.path.join(datadir, "imgs")
        self.parsing_folder = os.path.join(datadir, "parsing")
        self.alpha_folder = os.path.join(datadir, "alpha")
        self.ict = ict
        self.mediapipe = mediapipe
        self.do_not_skip = do_not_skip
        self.transfer_data_path = transfer_data_path

        self.image_size = (540, 720)

        self.bg_image = torch.zeros((3, self.image_size[1], self.image_size[0]))
        # if white_background:
        #     self.bg_image[:, :, :] = 1
        # else:
        #     self.bg_image[1, :, :] = 1

        self.mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint')
        self.ict_ckpt_dir = os.path.join(datadir, 'checkpoint')
        self.mediapipe_ckpt_dir = os.path.join(datadir, 'mediapipe_bs')

        self.N_frames = len(os.listdir(self.mica_ckpt_dir))
        self.cameras = []
        ckpt_path = os.path.join(self.mica_ckpt_dir, '00000.frame')
        payload = torch.load(ckpt_path)
        flame_params = payload['flame']
        self.shape_param = torch.as_tensor(flame_params['shape'])
        orig_w, orig_h = payload['img_size']
        K = payload['opencv']['K'][0]
        fl_x = K[0, 0]
        fl_y = K[1, 1]
        self.FovY = focal2fov(fl_x, orig_w)
        self.FovX = focal2fov(fl_y, orig_h)

    def __getitem__(self, frame_id):
            image_name_mica = str(frame_id).zfill(5)  # obey mica tracking
            image_name_ori = str(frame_id+self.frame_delta).zfill(5)
            ckpt_path = os.path.join(self.mica_ckpt_dir, image_name_mica+'.frame')
            payload = torch.load(ckpt_path)

            flame_params = payload['flame']
            exp_param = torch.as_tensor(flame_params['exp'])
            eyes_pose = torch.as_tensor(flame_params['eyes'])
            eyelids = torch.as_tensor(flame_params['eyelids'])
            jaw_pose = torch.as_tensor(flame_params['jaw'])

            oepncv = payload['opencv']
            w2cR = oepncv['R'][0]
            w2cT = oepncv['t'][0]
            R = np.transpose(w2cR)  # R is stored transposed due to 'glm' in CUDA code
            T = w2cT

            image_path = os.path.join(self.images_folder, image_name_ori+'.jpg')
            image = Image.open(image_path).resize(self.image_size, Image.Resampling.BICUBIC)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]

            # alpha
            alpha_path = os.path.join(self.alpha_folder, image_name_ori+'.png')
            alpha = Image.open(alpha_path).resize(image.size, Image.Resampling.BICUBIC)
            alpha = PILtoTensor(alpha)

            # # if add head mask
            head_mask_path = os.path.join(self.parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path).resize(image.size, Image.Resampling.BICUBIC)
            head_mask = PILtoTensor(head_mask)
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(self.parsing_folder, image_name_ori+'_mouth.png')
            mouth_mask = Image.open(mouth_mask_path).resize(image.size, Image.Resampling.BICUBIC)
            mouth_mask = PILtoTensor(mouth_mask)

            bs_param = None
            if self.ict:
                ict_ckpt_path = os.path.join(self.ict_ckpt_dir, image_name_ori+'.npz')
                ict_payload = np.load(ict_ckpt_path, allow_pickle=True)['data'][()]
                bs_param = torch.as_tensor(ict_payload['exp'])[None, ...]
            elif self.mediapipe:
                mediapipe_ckpt_path = os.path.join(self.mediapipe_ckpt_dir, image_name_ori+'.npy')
                mediapipe_payload = np.load(mediapipe_ckpt_path).astype(np.float32)
                bs_param = torch.as_tensor(mediapipe_payload)[None, ...]

            transfer_data = None
            if self.transfer_data_path is not None:
                transfer_data = torch.load(
                    os.path.join(self.transfer_data_path, f'{frame_id}.data'),
                    map_location='cpu',
                )

            return Camera(
                colmap_id=frame_id, R=R, T=T,
                FoVx=self.FovX, FoVy=self.FovY,
                image=gt_image, head_mask=head_mask, mouth_mask=mouth_mask,
                exp_param=exp_param, eyes_pose=eyes_pose, eyelids=eyelids, jaw_pose=jaw_pose,
                bs_param=bs_param,
                transfer_data=transfer_data,
                image_name=image_name_mica, uid=frame_id, data_device=self.device)

    def __len__(self):
        return self.N_frames


class Scene_ict:
    def __init__(self, datadir: str, mica_datadir: str, train_type: int, white_background: bool, device: str):
        self.device = device
        # train_type: 0 for train, 1 for test, 2 for eval
        root = Path(datadir)
        self.images_folder = str(root / f'{root.stem}')
        self.parsing_folder = os.path.join(datadir, "parsing")
        self.alpha_folder = os.path.join(datadir, "alpha")

        self.mica_ckpt_dir = os.path.join(datadir, 'checkpoint')
        self.N_frames = len(os.listdir(self.mica_ckpt_dir))
        self.cameras = []
        test_num = 500
        eval_num = 50
        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames - test_num)
        ckpt_path = os.path.join(self.mica_ckpt_dir, '00000.npz')
        payload = np.load(ckpt_path, allow_pickle=True)['data'][()]
        self.shape_param = torch.as_tensor(payload['id'])[None, ...]
        self.exp_param = torch.as_tensor(payload['exp'])[None, ...]
        self.angle_param = torch.as_tensor(payload['angle'])[None, ...]
        self.trans_param = torch.as_tensor(payload['trans'])[None, ...]

        image_path = os.path.join(self.images_folder, '00001.jpg')
        image = Image.open(image_path)
        self.orig_w, self.orig_h = image.size
        self.render_size = (224, 224)

        self.bg_image = torch.zeros((3, self.render_size[1], self.render_size[0]))
        # if white_background:
        #     self.bg_image[:, :, :] = 1
        # else:
        #     self.bg_image[1, :, :] = 1

        # self.FovY = focal2fov(1015.0 * 2.0, self.orig_h)
        # self.FovX = focal2fov(1015.0 * 2.0, self.orig_w)
        self.FovX = self.FovY = 0.10782261406415006 * 2.7
        # print(self.FovX, self.FovY)
        # exit()
        if train_type == 0:
            range_down = 0
            range_up = train_num
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames

        self.range_down = range_down
        self.range_up = range_up
        self.total_size = range_up - range_down
        self.bluk_index = 0
        self.bluk_index_max = 0
        self.bluk_size = 0

    def get_bulk(self, bulk_size: int = 1500, skip: int = 5, debug: bool = False):
        print('get next bulk')
        if bulk_size > self.total_size or bulk_size < 0:
            bulk_size = self.total_size
        if self.bluk_size != bulk_size:
            self.bluk_size = bulk_size
            self.bluk_index_max = int(self.total_size / bulk_size)
            self.bluk_index = -1
        self.bluk_index += 1
        if self.bluk_index >= self.bluk_index_max:
            self.bluk_index = 0
        range_down = self.range_down + bulk_size * self.bluk_index
        range_up = range_down + bulk_size
        if range_up > self.total_size:
            range_up = self.total_size

        self.cameras.clear()

        for frame_id in tqdm(range(range_down, range_up)):
            if frame_id % skip != 0:
                continue
            image_name_ori = str(frame_id + 1).zfill(5)
            npy_name = str(frame_id).zfill(5)
            ckpt_path = os.path.join(self.mica_ckpt_dir, npy_name+'.npz')
            payload = np.load(ckpt_path, allow_pickle=True)['data'][()]

            # flame_params = payload['flame']
            # B
            # exp
            # angle
            # trans
            # light
            # id
            # tex
            # w2c
            # cropped_img
            # crop_data
            shape_param = torch.as_tensor(payload['id'])[None, ...]
            exp_param = torch.as_tensor(payload['exp'])[None, ...]
            angle_param = torch.as_tensor(payload['angle'])[None, ...]
            angle_param[..., 1] *= -1
            trans_param = torch.as_tensor(payload['trans'])[None, ...]

            # w2c = torch.Tensor(payload['w2c'])
            # scale = 2 / min(self.orig_h, self.orig_w)
            # w2c[:3, :4] = w2c[:3, :4] * scale
            # w2c = w2c[None, ...]

            crop_data = np.linalg.inv(payload['crop_data'])

            R = _compute_rotation(angle_param)[0].cpu().numpy()
            R = R.T
            R[1, 1] *= -1
            T = payload['trans'] + np.array([0, 0, 10], dtype=np.float32)

            image_path = os.path.join(self.images_folder, image_name_ori+'.jpg')
            image = Image.open(image_path)
            resized_image_rgb = PILtoTensorCropped(image, crop_data, self.render_size)
            gt_image = resized_image_rgb[:3, ...]

            # alpha
            alpha_path = os.path.join(self.alpha_folder, image_name_ori+'.png')
            alpha = Image.open(alpha_path).resize(image.size)
            alpha = PILtoTensorCropped(alpha, crop_data, self.render_size)

            # # if add head mask
            head_mask_path = os.path.join(self.parsing_folder, image_name_ori+'_neckhead.png')
            head_mask = Image.open(head_mask_path)
            head_mask = PILtoTensorCropped(head_mask, crop_data, self.render_size)
            head_mask[head_mask > 0] = 1
            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            # mouth mask
            mouth_mask_path = os.path.join(self.parsing_folder, image_name_ori+'_mouth.png')
            mouth_mask = Image.open(mouth_mask_path)
            mouth_mask = PILtoTensorCropped(mouth_mask, crop_data, self.render_size)

            camera_indiv = Camera(
                colmap_id=frame_id, R=R, T=T,
                FoVx=self.FovX, FoVy=self.FovY,
                image=gt_image,
                head_mask=head_mask,
                mouth_mask=mouth_mask,
                exp_param=exp_param,
                eyes_pose=angle_param,
                eyelids=trans_param,
                jaw_pose=shape_param,
                image_name=npy_name,
                uid=frame_id,
                data_device=self.device,
            )
            # camera_indiv.to_device()
            self.cameras.append(camera_indiv)
            if debug and len(self.cameras) >= 100:
                break

    def getCameras(self):
        return self.cameras


class Scene_landmark:
    def __init__(
        self,
        datadir,
        mica_datadir,
        train_type,
        white_background,
        device,
    ):
        self.device = device
        # train_type: 0 for train, 1 for test, 2 for eval

        image_size = (512, 384)

        self.bg_image = torch.zeros((3, image_size[1], image_size[0]))

        # mica_ckpt_dir = os.path.join(mica_datadir, 'checkpoint')
        mica_ckpt_dir = os.path.join(datadir, 'data')

        self.N_frames = len(os.listdir(mica_ckpt_dir)) - 1
        self.cameras = []
        test_num = 500
        eval_num = 50
        max_train_num = 10000
        train_num = min(max_train_num, self.N_frames - test_num)

        ckpt_path = os.path.join(mica_ckpt_dir, '00001_pose.json')

        self.FovX = self.FovY = 0.10782261406415006 * 2.7
        if train_type == 0:
            range_down = 1
            range_up = train_num
        if train_type == 1:
            range_down = self.N_frames - test_num
            range_up = self.N_frames
        if train_type == 2:
            range_down = self.N_frames - eval_num
            range_up = self.N_frames

        R = np.array([
            [ 0.9971675,   0.06307777,  0.04096416],
            [ 0.07184628, -0.9599736,  -0.27071917],
            [ 0.02224815,  0.2728955,  -0.9617864],
        ], dtype=np.float32)
        # R = torch.from_numpy(R)
        T = np.array([0.00583798, -0.03207922, 0.76190805], dtype=np.float32)
        # T = torch.from_numpy(T)

        for frame_id in tqdm(range(range_down, range_up)):
            image_name_mica = str(frame_id).zfill(5)  # obey mica tracking
            ckpt_path = os.path.join(mica_ckpt_dir, image_name_mica+'_pose.json')
            # payload = torch.load(ckpt_path)
            with open(ckpt_path, 'r') as f:
                data = json.loads(f.read())
            joints = data['joints']
            res_lm = []
            for k in joints:
                pos = joints[k]['pos']
                # s = joints[k]['score']
                # res_lm.append(np.array([pos[0], pos[1], s]))
                res_lm.append(np.array([pos[0], pos[1]]))

            res_lm = np.array(res_lm, dtype=np.float32)
            bs_param = torch.as_tensor(res_lm).reshape(1, -1)

            image = base64_to_image(data['image']).resize(image_size, Image.Resampling.BICUBIC)
            resized_image_rgb = PILtoTensor(image)
            gt_image = resized_image_rgb[:3, ...]

            # alpha
            alpha = base64_to_image(data['mask']).resize(image.size, Image.Resampling.BICUBIC)
            alpha = PILtoTensor(alpha)

            gt_image = gt_image * alpha + self.bg_image * (1 - alpha)
            # gt_image = gt_image * head_mask + self.bg_image * (1 - head_mask)

            camera_indiv = Camera(
                colmap_id=frame_id, R=R.copy(), T=T.copy(),
                FoVx=self.FovX, FoVy=self.FovY,
                image=gt_image, head_mask=None, mouth_mask=None,
                exp_param=None, eyes_pose=None, eyelids=None, jaw_pose=None,
                bs_param=bs_param,
                image_name=image_name_mica, uid=frame_id, data_device=device)
            self.cameras.append(camera_indiv)

        self.bs_param = bs_param.clone()

    def getCameras(self):
        return self.cameras
