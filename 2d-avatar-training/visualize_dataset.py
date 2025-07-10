from argparse import ArgumentParser
from pathlib import Path

import numpy as np
from avatar2d.utils import base64_to_BytesIO
from PIL import Image
from tqdm import tqdm

from pyrenderer import Pyrenderer


def parse_args():
    parser = ArgumentParser(description="Visualize dataset")
    parser.add_argument(
        "dataset",
        type=str,
        help="Path to the dataset to visualize",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualization_output/",
        help="Path to save the visualization output",
    )
    return parser.parse_args()

def load_json(json_path: Path):
    import json
    with open(json_path, 'r') as f:
        return json.load(f)


def render_mesh(
    vertices: np.ndarray,
    triangles: np.ndarray,
    uvs: np.ndarray,
    texture: np.ndarray,
):
    if vertices.shape[-1] == 2:
        vertices = np.concatenate([vertices[..., :2], np.zeros_like(vertices[..., :1])], -1)

    try:
        wire, _ = Pyrenderer(texture=texture).render(vertices, triangles, uvs=uvs)
        return wire
    except Exception as e:
        print(f'Error rendering mesh: {e}')
        return None


def main():
    args = parse_args()
    print(f"Visualizing dataset: {args.dataset}")
    print(f"Output will be saved to: {args.output}")
    dataset_path = Path(args.dataset)
    if not dataset_path.exists():
        print(f"Dataset path {dataset_path} does not exist.")
        return
    
    output_path = Path(args.output)
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    np_data_files = []
    for file in (dataset_path / 'data').glob('*.npz'):
        np_data_files.append(file.stem)
    json_data_files = []
    for file in (dataset_path / 'labels').glob('*.json'):
        json_data_files.append(file.stem)
    data_files = list(set(np_data_files) & set(json_data_files))

    render_failed = False
    means = []
    bad_files = []
    for name in tqdm(data_files):
        npz_path = dataset_path / 'data' / f'{name}.npz'

        zfile = np.load(npz_path, allow_pickle=True)
        deformed_batch = zfile['output_verts'].astype(np.float32)

        vertices = zfile['base_vert']
        triangles = zfile['triangles']
        uvs = zfile['uvs']
        vertices2d = vertices[:, :2]
        shape_matched = True

        for i in range(deformed_batch.shape[0]):
            deformed = deformed_batch[i]
            if deformed.shape[0] != vertices.shape[0]:
                print(f"Deformed shape {deformed.shape} does not match base vertices shape {vertices.shape}. Skipping.")
                shape_matched = False
                break
            deformed_aabb = np.array([
                deformed[:, 0].min(), deformed[:, 1].min(),
                deformed[:, 0].max(), deformed[:, 1].max()
            ])
            vertices_aabb = np.array([
                vertices2d[:, 0].min(), vertices2d[:, 1].min(),
                vertices2d[:, 0].max(), vertices2d[:, 1].max()
            ])

            comparer = np.abs(deformed_aabb - vertices_aabb).max()
            means.append(comparer)

            if comparer > 1:
                print(f"Mesh {name} ({i}) has high normal consistency loss: {comparer}. Skipping.")
                bad_files.append(npz_path)
                if render_failed:
                    break
                json_path = dataset_path / 'labels' / f'{name}.json'
                json_dict = load_json(json_path)
                image_base64_str = json_dict['resized']
                tex_im = Image.open(base64_to_BytesIO(image_base64_str))
                texture = np.array(tex_im)

                image = render_mesh(vertices, triangles, uvs, texture)
                deformed_image = render_mesh(deformed, triangles, uvs, texture)
                if image is not None and deformed_image is not None:
                    combined = np.concatenate([image, deformed_image], axis=1)
                    visualization_path = output_path / f'{name}.png'
                    Image.fromarray(combined).save(visualization_path)
                    print(f"Saved visualization to {visualization_path}")
                else:
                    render_failed = True
                    print(f"Failed to render mesh from {name}")
                break

        if not shape_matched:
            tri_fromz = zfile['triangles'].astype(np.int32)
            print('tri_fromz:', tri_fromz, tri_fromz.shape)
            print('triangles:', triangles, triangles.shape)

    print("Visualization completed.")
    print(f"Mean differences: {np.mean(means)}, min: {np.min(means)}, max: {np.max(means)}")
    with open('bad_files.txt', 'w') as f:
        for bad_file in bad_files:
            f.write(f"{bad_file}\n")
    print(f"Bad files saved to {'bad_files.txt'}")


if __name__ == "__main__":
    main()
