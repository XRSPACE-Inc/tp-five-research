from argparse import ArgumentParser
import numpy as np
from pathlib import Path
from tqdm import tqdm
import open3d as o3d

from utils import get_tpl_edges
from utils import calc_surface_geodesic
from utils import get_geo_edges

def parse_args():
    parser = ArgumentParser(description='Step2 Create training data (mesh) for the model.')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory where the training data is stored.',
    )
    parser.add_argument(
        'output_dir',
        type=str,
        help='Directory where the output training data will be saved.',
    )
    return parser.parse_args()


def make_training_data_trim(
    input_file: Path,
    output_dir: Path,
) -> None:
    output_file = output_dir / input_file.name
    if output_file.exists():
        print(f'Output file {output_file} already exists, skipping.')
        return

    zfile = np.load(input_file, allow_pickle=True)
    vertices = zfile['verts'].astype(np.float32)
    triangles = zfile['triangles'].astype(np.int32)

    if (
        np.isnan(vertices).any()
        or np.isnan(triangles).any()
    ):
        print(f'NaN data detected in {input_file}, skipping.')
        return

    data_ls = list(
        zip(
            zfile['input_poses'].astype(np.float32),
            zfile['output_verts'].astype(np.float32),
            zfile['output_crts'].astype(np.float32)
        )
    )

    checked_input = []
    checked_output_verts = []
    checked_output_crts = []
    for input_pose, output_verts, output_crts in data_ls:
        if (
            np.isnan(input_pose).any()
            or np.isnan(output_verts).any()
            or np.isnan(output_crts).any()
        ):
            continue

        checked_input.append(input_pose)
        checked_output_verts.append(output_verts)
        checked_output_crts.append(output_crts)
        if len(checked_input) > 50:
            break

    if len(checked_input) == 0:
        print(f'No valid data found in {input_file}, skipping.')
        return

    checked_input = np.array(checked_input, dtype=np.float32)
    checked_output_verts = np.array(checked_output_verts, dtype=np.float32)
    checked_output_crts = np.array(checked_output_crts, dtype=np.float32)

    mesh = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(vertices),
        triangles=o3d.utility.Vector3iVector(triangles),
    )
    mesh.compute_vertex_normals()
    vertex_normals = np.array(mesh.vertex_normals).astype(np.float32)

    tpl_e = get_tpl_edges(vertices, triangles).T

    surface_geodesic = calc_surface_geodesic(mesh)

    geo_e = get_geo_edges(surface_geodesic, vertices).T
    np.savez(
        output_file,
        verts=vertices,
        triangles=triangles,
        input_poses=checked_input,
        output_verts=checked_output_verts,
        output_crts=checked_output_crts,
        tpl_edge_index=tpl_e,
        geo_edge_index=geo_e,
        vertex_normals=vertex_normals,
    )


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f'output_dir: {output_dir}')

    ls = data_dir.glob('*.npz')
    ls = sorted(ls, key=lambda x: x.stem)

    failed = []

    for input_file in tqdm(ls):
        try:
            make_training_data_trim(input_file, output_dir)
        except KeyboardInterrupt:
            print('KeyboardInterrupt detected, exiting...')
            break
        except Exception as e:
            print(f'Failed to process {input_file}: {e}')
            failed.append(input_file)

    print('Failed to process the following files:')
    for f in failed:
        print(f)


if __name__ == '__main__':
    main()
