from pathlib import Path
import torch
from torch import nn
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

from PIL import Image
import numpy as np


def main():
    # convenience expression for automatically determining device
    device = (
        "cuda"
        # Device for NVIDIA or AMD GPUs
        if torch.cuda.is_available()
        else "mps"
        # Device for Apple Silicon (Metal Performance Shaders)
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # load models
    image_processor = SegformerImageProcessor.from_pretrained("jonathandinu/face-parsing")
    model = SegformerForSemanticSegmentation.from_pretrained("jonathandinu/face-parsing")
    model.to(device)

    output_path = Path('./parsing/')
    output_path.mkdir(exist_ok=True)

    # # expects a PIL.Image or torch.Tensor
    # url = "https://images.unsplash.com/photo-1539571696357-5a69c17a67c6"
    # image = Image.open(requests.get(url, stream=True).raw)
    def process(image: Image.Image):
        # run inference on image
        inputs = image_processor(images=image, return_tensors="pt").to(device)
        outputs = model(**inputs)
        logits = outputs.logits  # shape (batch_size, num_labels, ~height/4, ~width/4)

        # resize output to match input image dimensions
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=image.size[::-1],  # H x W
            mode='bilinear',
            align_corners=False)

        # get label masks
        labels = upsampled_logits.argmax(dim=1)[0]

        # move to CPU to visualize in matplotlib
        return labels.cpu().numpy()

    # id    label       note
    # 0     background
    # 1     skin
    # 2     nose
    # 3     eye_g       eyeglasses
    # 4     l_eye       left eye
    # 5     r_eye       right eye
    # 6     l_brow      left eyebrow
    # 7     r_brow      right eyebrow
    # 8     l_ear       left ear
    # 9     r_ear       right ear
    # 10    mouth       area between lips
    # 11    u_lip       upper lip
    # 12    l_lip       lower lip
    # 13    hair
    # 14    hat
    # 15    ear_r       earring
    # 16    neck_l      necklace
    # 17    neck
    # 18    cloth       clothing

    mouth_idx = [10, 11, 12]
    neckhead_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 16, 17]

    root = Path('~/Datasets/livelink/myslate_11_css_iphone/myslate_11_css_iphone').expanduser()
    for file in root.glob('*.jpg'):
        image = Image.open(str(file))
        res = process(image)

        mouth = np.zeros_like(res)
        for idx in mouth_idx:
            mouth = np.logical_or(mouth, res == idx)

        Image.fromarray(mouth).save(str(output_path / f'{file.stem}_mouth.png'))

        neckhead = np.zeros_like(res)
        for idx in neckhead_idx:
            neckhead = np.logical_or(neckhead, res == idx)

        Image.fromarray(neckhead).save(str(output_path / f'{file.stem}_neckhead.png'))
        # exit()


if __name__ == '__main__':
    main()
