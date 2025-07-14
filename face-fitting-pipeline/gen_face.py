import os
from argparse import ArgumentParser
from pathlib import Path
from shutil import copyfile

from face_generation.face_generator import test_generation


def parse_args():
    parser = ArgumentParser(description="Face Generation Script")
    parser.add_argument('path', help='Path to the input image for face generation')
    parser.add_argument('--root_path', type=str, default='./face_generation/',
                        help='Root path for the face generation module')
    parser.add_argument('--output_path', type=str, default='./results/output/',
                        help='Output path for generated files')
    parser.add_argument('--webresources_path', type=str, default='./webresources/',
                        help='Path for web resources to store generated files')
    return parser.parse_args()


def main():
    args = parse_args()

    path = args.path
    root_path = args.root_path
    output_path = args.output_path
    webresources_path = args.webresources_path

    assert os.path.exists(path), f"Input path '{path}' does not exist."
    assert os.path.exists(root_path), f"Root path '{root_path}' does not exist."
    assert os.path.exists(output_path), f"Output path '{output_path}' does not exist."

    try:
        test_generation(root_path, path, output_path)
    except KeyboardInterrupt:
        print()
        print("KeyboardInterrupt: exit")
    
    os.makedirs('./webresources/output/', exist_ok=True)
    output_path = Path(output_path)
    webresources_path = Path(webresources_path)
    webresources_path.mkdir(parents=True, exist_ok=True)

    copyfile(output_path / 'i_src.png', webresources_path / 'output/i_src.png')
    copyfile(output_path / 'albedo.png', webresources_path / 'output/albedo.png')
    copyfile(output_path / 'normal.png', webresources_path / 'output/normal.png')
    copyfile(output_path / 'mesh.obj', webresources_path / 'output/mesh.obj')
    copyfile(output_path / 'eye.obj', webresources_path / 'output/eye.obj')
    copyfile(output_path / 'eyelen.obj', webresources_path / 'output/eyelen.obj')
    try:
        copyfile(output_path / 'specular.png', webresources_path / 'output/specular.png')
    except Exception:
        copyfile(output_path / 'cavity.png', webresources_path / 'output/cavity.png')


if __name__ == '__main__':
    main()
