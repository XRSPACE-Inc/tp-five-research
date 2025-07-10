from argparse import ArgumentParser
from pathlib import Path

from train import get_model
from model import runtime_model
from utils import write_obj
from dataset import load_npz
import torch
from tqdm import tqdm


def parse_args():
    parser = ArgumentParser(description='Create training data for the model.')
    parser.add_argument(
        'input',
        type=str,
        help='input pth file',
    )
    parser.add_argument(
        'pretrained_model',
        type=str,
        help='pretrained model path',
    )
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available()
            else 'cpu',
        help='Device to run the training on (default: cuda or mps if available, else cpu).',
    )
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()

    device = torch.device(args.device)
    trainingModel = get_model()

    pretrained_model_path = Path(args.pretrained_model)
    state_dict = torch.load(pretrained_model_path, map_location='cpu')
    trainingModel.load_state_dict(state_dict)
    trainingModel.to(device)
    trainingModel.eval()

    input_dir = Path(args.input)
    ls = list(input_dir.glob('*.npz'))
    if len(ls) == 0:
        raise FileNotFoundError(f'No .npz files found in {input_dir}.')

    start = 0
    end = min(len(ls), start + 100)

    with tqdm(total=end - start, desc='Processing files') as pbar:
        for i in range(start, end):
            input_file = ls[i]
            if not input_file.exists():
                pbar.update()
                continue

            output_path = Path('visualization_output/test/')
            output_path.mkdir(parents=True, exist_ok=True)


            data = load_npz(input_file)
            vertices = data['base_vert'][None, ...]
            triangles = data['triangles']
            input_pose = data['input_poses'][None, ...]
            gt_verts = data['output_verts'][None, ...]
            gt_crts = data['output_crts'][None, ...]

            vertices = torch.Tensor(vertices).to(device)
            triangles = torch.LongTensor(triangles).to(device)

            input_pose = torch.Tensor(input_pose).to(device)
            gt_verts = torch.Tensor(gt_verts).to(device)
            gt_crts = torch.Tensor(gt_crts).to(device)

            if torch.isnan(gt_verts).any() or torch.isnan(gt_crts).any():
                pbar.write(f'NaN data detected at file: {input_file}')
                pbar.update()
                continue

            mesh_loss_fn = torch.nn.SmoothL1Loss()
            base_vertices_xy = vertices[..., :2].clone()
            input_layer_weights, hidden_layer_weights, crt_layer_weights, output_weights = trainingModel(base_vertices_xy)

            diff, _ = runtime_model(
                input_arr=input_pose,
                input_layer=input_layer_weights,
                hidden_layer=hidden_layer_weights,
                mesh_layer=output_weights,
                crt_layer=crt_layer_weights,
            )
            base_vertices_xy = base_vertices_xy.repeat(diff.shape[0], 1, 1)

            mesh_output = diff + base_vertices_xy
            gt_diff = gt_verts - base_vertices_xy[:, None, ...]
            loss_mesh = mesh_loss_fn(diff, gt_diff)
            pbar.set_description(f'{input_file.name} - Loss: {loss_mesh.item():.4f}')

            mesh_vertices = vertices.clone()
            mesh_output = mesh_output[0]
            mesh_vertices[..., :2] = mesh_output[:1]

            m = {}
            m['verts'] = mesh_vertices[0].cpu().numpy()
            m['faces'] = triangles
            write_obj(f'{output_path}/{input_file.stem}_pred.obj', m)

            gt_verts = gt_verts[0]
            mesh_vertices[..., :2] = gt_verts[:1]

            m = {}
            m['verts'] = mesh_vertices[0].cpu().numpy()
            m['faces'] = triangles
            write_obj(f'{output_path}/{input_file.stem}_gt.obj', m)
            pbar.update()


if __name__ == '__main__':
    main()
