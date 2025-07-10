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
        '-i',
        '--start_index',
        type=int,
        default=0,
        help='Index to start processing from, useful for resuming interrupted runs.',
    )
    return parser.parse_args()


def make_training_data_arap(
    input_file: Path,
    vrm_generator: VRMGenerator,
    working_dir: Path,
    output_dir: Path,
    debug: bool = False,
) -> None:
    working_dir = working_dir / input_file.stem
    working_dir = vrm_generator.prepare_working_dir(working_dir)
    output_path = output_dir / f'{input_file.stem}.pth'
    if output_path.exists():
        print(f'Skipping {input_file}, output already exists.')
        return
    # print(f'Processing {input_file}')

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

    ARAP = arap(
        Path(vrm_generator.asset_folder),
        working_dir,
        largest_mesh_idx,
        joint_dict,
        v_to_joint,
        resized_img,
        device=vrm_generator.device,
        debug=False,
    )

    output_dict = {}
    output_dict['tA1xA1'] = ARAP.tA1xA1.toarray()
    output_dict['tA2xA2'] = ARAP.tA2xA2.toarray()
    output_dict['tA1'] = ARAP.tA1.toarray()
    output_dict['tA2'] = ARAP.tA2.toarray()
    output_dict['G'] = ARAP.G.toarray()

    torch.save(output_dict, output_path)
    print(f'Saving output to {output_path}')
    shutil.rmtree(working_dir, ignore_errors=True)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    working_dir = Path(args.working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)
    generator = VRMGenerator('cuda', True, True, 'working/')

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
            make_training_data_arap(input_file, generator, working_dir, output_dir, i == 0)
        except KeyboardInterrupt:
            print('KeyboardInterrupt detected, exiting...')
            break
        except Exception as e:
            print(f'Failed to process {input_file}: {e}')
            print(e.with_traceback(e.__traceback__))
            failed.append(input_file)
            # exit()

    print('Failed to process the following files:')
    for f in failed:
        print(f)


if __name__ == '__main__':
    main()
