from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
from avatar2d.salient_object_detection import SalientObjectDetector
from avatar2d.landmark_detection import LandmarkDetector
import urllib.parse


def parse_args():
    parser = ArgumentParser(description='Step1 Create training data (mask & joint) for the model.')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory where the training data is stored.',
    )
    return parser.parse_args()


def load_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data


def fix_path(text: str):
    # Remove the prefix 'data/local-files/?d=data/' from the text
    if text.startswith('/data/local-files/?d=data/'):
        text = text[len('/data/local-files/?d=data/'):]
    if '%' in text:
        # Decode URL-encoded characters
        text = urllib.parse.unquote(text)
    return text


def fix_image_paths(ls: list[dict]):
    for i in range(len(ls)):
        if 'img' in ls[i]:
            ls[i]['img'] = fix_path(ls[i]['img'])
    return ls


def make_label(
    image_path: Path,
    salient_object_detector: SalientObjectDetector,
    landmark_detector: LandmarkDetector,
    joint_map: dict,
    parent_map: dict,
    raw_json: dict,
) -> dict:
    with open(image_path / raw_json['img'], 'rb') as f:
        meta_str = salient_object_detector.inference(f, 'working/')

    pose_str = landmark_detector.inference(meta_str, 'working/', False)

    meta = json.loads(meta_str)
    meta['image_name'] = raw_json['img']

    meta['joints'] = json.loads(pose_str)

    return meta


def debug_label(label: dict) -> bool:
    from avatar2d.utils import base64_to_BytesIO
    from PIL import Image
    import numpy as np

    im = np.array(Image.open(base64_to_BytesIO(label['resized'])))
    for joint in label['joints'].values():
        pos = joint['pos']
        x = int(pos[0] * im.shape[1])
        y = int(pos[1] * im.shape[0])
        im[y-10:y+10, x-10:x+10] = [255, 0, 0]  # Mark joint position in red
    im = Image.fromarray(im)
    im.save('debug_label.png')
    exit()


def main():
    args = parse_args()

    joint_map = {
        'Pelvis': 'Bip001_Pelvis',
        'Spine2': 'Bip001_Spine2',
        'Head': 'Bip001_Head',
        'left_eye': 'LEye',
        'right_eye': 'REye',
        'left_ear': 'LEar',
        'right_ear': 'REar',
        'left_shoulder': 'Bip001_LUpArmTwist',
        'left_elbow': 'Bip001_L_ForeTwist',
        'left_wrist': 'Bip001_L_Hand',
        'right_shoulder': 'Bip001_RUpArmTwist',
        'right_elbow': 'Bip001_R_ForeTwist',
        'right_wrist': 'Bip001_R_Hand',
        'left_hip': 'Bip001_LThighTwist',
        'left_knee': 'Bip001_LCalfTwist',
        'left_foot': 'Bip001_L_Foot',
        'right_hip': 'Bip001_RThighTwist',
        'right_knee': 'Bip001_RCalfTwist',
        'right_foot': 'Bip001_R_Foot',
    }
    parent_map = {
        'Bip001_Pelvis': '-1',
        'Bip001_Spine2': 'Bip001_Pelvis',
        'Bip001_Head': 'Bip001_Spine2',
        'LEye': 'Bip001_Head',
        'REye': 'Bip001_Head',
        'LEar': 'Bip001_Head',
        'REar': 'Bip001_Head',
        'Bip001_LUpArmTwist': 'Bip001_Spine2',
        'Bip001_L_ForeTwist': 'Bip001_LUpArmTwist',
        'Bip001_L_Hand': 'Bip001_L_ForeTwist',
        'Bip001_RUpArmTwist': 'Bip001_Spine2',
        'Bip001_R_ForeTwist': 'Bip001_RUpArmTwist',
        'Bip001_R_Hand': 'Bip001_R_ForeTwist',
        'Bip001_LThighTwist': 'Bip001_Pelvis',
        'Bip001_LCalfTwist': 'Bip001_LThighTwist',
        'Bip001_L_Foot': 'Bip001_LCalfTwist',
        'Bip001_RThighTwist': 'Bip001_Pelvis',
        'Bip001_RCalfTwist': 'Bip001_RThighTwist',
        'Bip001_R_Foot': 'Bip001_RCalfTwist',
    }
    output_dir = Path(args.data_dir) / 'labels'
    output_dir.mkdir(parents=True, exist_ok=True)

    salient_object_detector = SalientObjectDetector('mps', False, True, 'working/')
    landmark_detector = LandmarkDetector('mps', False, True, 'working/')

    image_path = Path(args.data_dir) / 'images'
    label_path = Path(args.data_dir) / 'label_raw_data'

    json_files = list(label_path.glob('*.json'))
    raw_json_ls = []
    for json_file in json_files:
        d = load_json_file(json_file)
        raw_json_ls += d

    raw_json_ls = fix_image_paths(raw_json_ls)

    missing_images = []
    for raw_json in tqdm(raw_json_ls):

        path = image_path / raw_json['img']
        if not path.exists():
            print(f'Image {path} does not exist.')
            missing_images.append(f'{raw_json["img"]}\n')
            continue

        image_file = Path(raw_json['img'])
        output_path = output_dir / f'{image_file.stem}.json'
        if output_path.exists():
            print(f'Label {output_path} already exists.')
            continue
        label = make_label(
            image_path,
            salient_object_detector,
            landmark_detector,
            joint_map,
            parent_map,
            raw_json,
        )
        with open(output_path, 'w') as f:
            json.dump(label, f, indent=4)

    with open('missing_images.txt', 'w') as f:
        f.writelines(missing_images)
    print(f'done missing images: {len(missing_images)}')


if __name__ == '__main__':
    main()
