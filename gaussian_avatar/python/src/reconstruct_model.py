import torch

from .ict import ICTFaceKit
# from .camera_model import CameraModel
# from renderer import NVDiffrast3DMeshRenderer


class Model(torch.nn.Module):
    def __init__(
        self,
        file: str,
        texture_file: str,
        orig_uv: bool = False,
        # image_size: int = 224,
    ) -> None:
        super().__init__()
        # self.cameraModel = CameraModel(image_size=image_size, focal=1015.0, camera_distance=10.0)
        self.tdmm = ICTFaceKit(
            file=file,
            texture_file=texture_file,
            head_type='full',
            add_eye_ball=True,
            add_exp_atan=True,
            orig_uv=orig_uv,
        )
        # self.decoder = NVDiffrast3DMeshRenderer(focal=1015.0)

    def forward_geo(self, coef_dict: dict) -> dict:
        return self.tdmm.compute_shape(
            torch.zeros_like(coef_dict['jaw_pose']),
            torch.zeros_like(coef_dict['expr'])
        ) * 0.1
