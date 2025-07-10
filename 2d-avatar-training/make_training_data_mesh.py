from argparse import ArgumentParser
import json
import shutil
import numpy as np
from pathlib import Path
from avatar2d.vrm_generator import VRMGenerator
from avatar2d.auto_skinning.train_arap import arap
from avatar2d.utils import base64_to_BytesIO
from tqdm import tqdm
import torch
from PIL import Image
import os


def parse_args():
    parser = ArgumentParser(description='Step2 Create training data (mesh) for the model.')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory where the training data is stored.',
    )
    parser.add_argument(
        'working_dir',
        type=str,
        help='Directory where the output training data will be saved.',
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory where the output training data will be saved.',
    )
    parser.add_argument(
        '-r',
        '--reverse',
        action='store_true',
        help='run the script in reverse mode, i.e., process the data in reverse order.',
    )
    parser.add_argument(
        '--check_data_mode',
        action='store_true',
        help='check the data mode, i.e., whether the data is in the correct format.',
    )
    parser.add_argument(
        '--check_npz',
        action='store_true',
        help='check the data mode, i.e., whether the data is in the correct format.',
    )
    parser.add_argument(
        '--start_index',
        type=int,
        default=0,
        help='Start index for processing files. Useful for resuming from a specific point.',
    )
    return parser.parse_args()


def inverse_joints(joints: dict) -> dict:
    for key in joints.keys():
        pos = joints[key]['pos']
        pos[0] = 1 - pos[0]
        pos[1] = 1 - pos[1]
        joints[key]['pos'] = pos
    return joints


def check_npz(input_file: Path) -> None:
    path = input_file.parent.parent / f'training_data_trimed/{input_file.stem}.npz'
    if not path.exists():
        print(f'File {path} does not exist, skipping check.')
        return
    data = np.load(path, allow_pickle=True)
    for k in data.keys():
        if isinstance(data[k], np.ndarray):
            if np.isnan(data[k]).any():
                os.remove(path)
                print(f'Found NaN in {path}, removed it.')
                return


def check_data(input_file: Path) -> None:
    path = input_file.parent.parent / f'pth/{input_file.stem}.pth'
    if not path.exists():
        print(f'File {path} does not exist, skipping check.')
        return

    pth = torch.load(path, map_location='cpu', weights_only=False)
    error = False
    for k in pth.keys():
        if isinstance(pth[k], np.ndarray):
            if np.isnan(pth[k]).any():
                error = True
                break
        elif isinstance(pth[k], torch.Tensor):
            if torch.isnan(pth[k]).any():
                error = True
                break

    if error:
        os.remove(path)
        print(f'Found NaN in {path}, removed it.')


def make_training_data_mesh(
    input_file: Path,
    vrm_generator: VRMGenerator,
    working_dir: Path,
    output_dir: Path,
    debug: bool = False,
) -> None:
    working_dir = working_dir / input_file.stem
    working_dir = vrm_generator.prepare_working_dir(working_dir)
    output_path = output_dir / f'{input_file.stem}.npz'
    if output_path.exists():
        print(f'Skipping {input_file}, output already exists.')
        return

    with open(input_file, 'r') as f:
        input_json = f.read()

    data_input = json.loads(input_json)

    data_input['joints'] = data_input['joints']['joints']
    resized_io = base64_to_BytesIO(data_input['resized'])
    resized_img = Image.open(resized_io)

    (
        thin_mesh,
        v_to_joint,
        mesh_generator,
        largest_mesh_idx,
        joint_dict,
        image,
    ) = vrm_generator.preprocess_mesh(data_input, working_dir)

    model_dict, _, data_outputs = arap(
        Path(vrm_generator.asset_folder),
        working_dir,
        largest_mesh_idx,
        joint_dict,
        v_to_joint,
        resized_img,
        device=vrm_generator.device,
        debug=False,
    )

    if model_dict is None:
        print(f'Failed to process {input_file}, model_dict is None.')
        shutil.rmtree(working_dir, ignore_errors=True)
        return

    input_poses = []
    output_verts = []
    output_crts = []
    for (input_pose, output_vert, output_crt) in data_outputs:
        input_poses.append(input_pose)
        output_verts.append(output_vert)
        output_crts.append(output_crt)
        if len(input_poses) >= 256:
            break

    model_dict['input_poses'] = np.array(input_poses, dtype=np.float32)
    model_dict['output_verts'] = np.array(output_verts, dtype=np.float32)
    model_dict['output_crts'] = np.array(output_crts, dtype=np.float32)
    np.savez(output_path, **model_dict)
    print(f'Saving output to {output_path}')
    shutil.rmtree(working_dir, ignore_errors=True)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available()
            else 'cpu')
    generator = VRMGenerator(device, True, True, 'working/')

    print(f'working_dir: {working_dir}')
    print(f'output_dir: {output_dir}')

    ls = data_dir.glob('*.json')
    ls = sorted(ls, key=lambda x: x.stem)

    if args.reverse:
        ls.reverse()

    failed = []

    for i, input_file in enumerate(tqdm(ls)):
        if i < args.start_index:
            continue
        try:
            if args.check_data_mode:
                if args.check_npz:
                    check_npz(input_file)
                else:
                    check_data(input_file)
                continue

            make_training_data_mesh(input_file, generator, working_dir, output_dir, i == 0)
        except KeyboardInterrupt:
            print('KeyboardInterrupt detected, exiting...')
            break
        except Exception as e:
            print(f'Failed to process {input_file}: {e}')
            failed.append(input_file)
            # exit()

    print('Failed to process the following files:')
    for f in failed:
        print(f)


if __name__ == '__main__':
    main()
