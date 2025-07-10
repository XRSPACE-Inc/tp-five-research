from argparse import ArgumentParser
import base64
import datetime
from io import BytesIO
import json
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from model import TrainingModel, runtime_model, FakeFullLayer
from utils import AvgLosses, to_np
from torch.utils.data import DataLoader
from dataset import ARAPDataset
import matplotlib.pyplot as plt
from utils import write_obj
from utils import copy_src
from utils import to_tensor
from utils import is_nan
from visualizer import MyVisualizer
try:
    from pytorch3d.loss import mesh_edge_loss # type: ignore
    from pytorch3d.structures import Meshes # type: ignore
    from pytorch3d.loss import chamfer_distance
    from pytorch3d.loss import mesh_laplacian_smoothing
    from pytorch3d.loss import mesh_normal_consistency
    has_pytorch3d_loss = True
except ImportError:
    has_pytorch3d_loss = False
    print('pytorch3d is not installed, mesh edge loss will not be used.')


def parse_args():
    parser = ArgumentParser(description='Create training data for the model.')
    parser.add_argument(
        'data_dir',
        type=str,
        help='Directory where the training data is stored.',
    )
    parser.add_argument(
        'checkpoint_path',
        type=str,
        help='Directory where the training data is stored.',
    )
    parser.add_argument(
        '-f',
        '--folder_name',
        type=str,
        default='train',
        help='Directory where the training data is stored.',
    )
    parser.add_argument(
        '-d',
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else
            'mps' if torch.backends.mps.is_available()
            else 'cpu',
        help='Device to run the training on (default: cuda if available, else cpu).',
    )
    parser.add_argument(
        '--log_iterations',
        type=int,
        default=1000,
        help='Number of iterations to log the training progress (default: 1000).',
    )
    parser.add_argument(
        '--save_iterations',
        type=int,
        default=5000,
        help='Number of iterations to save the model (default: 5000).',
    )
    parser.add_argument(
        '--mesh_loss_weight',
        type=float,
        default=10.0,
        help='Weight for the mesh loss (default: 10.0).',
    )
    parser.add_argument(
        '--edge_loss_weight',
        type=float,
        default=0.0,
        help='Weight for the edge loss (default: 0.01).',
    )
    parser.add_argument(
        '--normal_loss_weight',
        type=float,
        default=0.01,
        help='Weight for the edge loss (default: 0.01).',
    )
    parser.add_argument(
        '--laplacian_loss_weight',
        type=float,
        default=0.0,
        help='Weight for the laplacian loss (default: 0.01).',
    )
    parser.add_argument(
        '--crt_loss_weight',
        type=float,
        default=1.0,
        help='Weight for the CRT loss (default: 1.0).',
    )
    parser.add_argument(
        '--weights_loss_weight',
        type=float,
        default=0.0,
        help='Weight for the weight loss (default: 0.0).',
    )
    return parser.parse_args()


def save_model(model: TrainingModel, path: Path):
    torch.save(model.state_dict(), path)
    print(f'Model saved to {path}')


def base64_to_bytes(
    base64_str: str,
) -> bytes:
    return base64.b64decode(base64_str.encode(encoding='utf-8'))


def base64_to_BytesIO(
    base64_str: str,
) -> BytesIO:
    return BytesIO(base64_to_bytes(base64_str))


def save_debug_mesh(vertices: torch.Tensor, triangles: torch.LongTensor, uvs: torch.Tensor, path: Path, pbar):
    if len(vertices.shape) == 3:
        vertices = vertices[0]
    if vertices.shape[1] == 2:
        vertices = torch.cat([vertices[..., :2], torch.zeros_like(vertices[..., :1])], dim=-1)
    m = {}
    m['verts'] = vertices
    m['faces'] = triangles
    if uvs is not None:
        if len(uvs.shape) == 3:
            uvs = uvs[0]
        m['uvs'] = uvs
    write_obj(path, m)
    vertices = to_np(vertices, True)
    log = f'vertices.min: {vertices.min(0)}, ' \
          f'vertices.max: {vertices.max(0)}, '
    if pbar is not None:
        pbar.write(log)
    else:
        print(log)


def render_mesh(
    vertices: torch.Tensor,
    triangles: torch.Tensor,
):
    from pyrenderer import Pyrenderer
    if vertices.shape[1] == 2:
        vertices = torch.cat([vertices[..., :2], torch.zeros_like(vertices[..., :1])], dim=-1)

    vertices = to_np(vertices)
    triangles = to_np(triangles)
    try:
        wire, _ = Pyrenderer().render(vertices, triangles, wireframe=True)
        return wire
    except Exception as e:
        print(f'Error rendering mesh: {e}')
        return None


def plot_vertex_and_pose(image_size, image_joints, target_joints):
    dpi = 96
    margin = 120
    figsize = ((margin + image_size[0]) / dpi, (margin + image_size[1]) / dpi)
    fig = plt.figure(figsize=figsize, dpi=dpi)

    plt.clf()
    ax = fig.add_subplot(111)
    ax.scatter(image_joints[:, 0], image_joints[:, 1], c='r', marker='x')
    ax.scatter(target_joints[:, 0], target_joints[:, 1], c='g', marker='o')

    buf = BytesIO()
    plt.savefig(buf, format='jpg')
    return np.array(Image.open(buf))[None, ...]


def plot_batch_vertex_and_pose(
    label_path: Path,
    names: tuple[str],
    vertices: torch.Tensor,
    poses: torch.Tensor,
) -> dict:
    images = {}
    for i in range(len(names)):
        name = names[i]
        path = label_path / f'{name}.json'
        if not path.exists():
            # raise FileNotFoundError(f'File {path} does not exist.')
            continue
        with open(path, 'r') as f:
            data = json.load(f)
        resized = Image.open(base64_to_BytesIO(data['resized']))

        vertices_i = to_np(vertices[i])
        poses_i = to_np(poses[i])
        image_size = resized.size
        im = plot_vertex_and_pose(
            image_size,
            vertices_i,
            poses_i,
        )
        images[name] = im
    return images


def do_normal_consistency_loss(
    mesh_output: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:
    mesh_output = mesh_output[0]
    mesh_output = torch.cat([mesh_output[..., :2], torch.zeros_like(mesh_output[..., :1])], dim=-1)
    tri = triangles.clone().repeat(mesh_output.shape[0], 1, 1)

    mesh = Meshes(
        verts=mesh_output,
        faces=tri,
    )
    return mesh_normal_consistency(mesh)


def do_edge_loss(
    mesh_output: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:

    mesh_output = mesh_output[0]
    mesh_output = torch.cat([mesh_output[..., :2], torch.zeros_like(mesh_output[..., :1])], dim=-1)
    tri = triangles.clone().repeat(mesh_output.shape[0], 1, 1)

    mesh = Meshes(
        verts=mesh_output,
        faces=tri,
    )
    return mesh_edge_loss(mesh)


def do_laplacian_smoothing(
    mesh_output: torch.Tensor,
    triangles: torch.Tensor,
) -> torch.Tensor:

    mesh_output = mesh_output[0]
    mesh_output = torch.cat([mesh_output[..., :2], torch.zeros_like(mesh_output[..., :1])], dim=-1)
    tri = triangles.clone().repeat(mesh_output.shape[0], 1, 1)

    mesh = Meshes(
        verts=mesh_output,
        faces=tri,
    )
    return mesh_laplacian_smoothing(mesh)


def pbar_description(pbar, loss_dict: dict, tag: str) -> None:
    log = tag
    for k, v in loss_dict.items():
        log += f'{k}: {v.item():.4f}, '
    pbar.set_description(log)


def visualizer_print(visualizer: MyVisualizer, pbar, loss_dict: dict, tag: str) -> None:
    log = tag
    for k, v in loss_dict.items():
        log += f'{k}: {v.item():.4f}, '
    visualizer.print_current_losses(log, pbar)


def get_dataset(dataset_path: Path):
    return ARAPDataset(dataset_path, False)


def weight_loss(
    input_layer_weights: FakeFullLayer,
    hidden_layer_weights: FakeFullLayer,
    crt_layer_weights: FakeFullLayer,
    output_weights: FakeFullLayer,
    pth: dict,
    loss_fn: callable,
) -> dict:
    input_loss = input_layer_weights.loss_fn(
        loss_fn,
        pth['input.weight'],
        pth['input.bias'],
        pth['input_bn.alpha'],
        pth['input_bn.beta'],
    )
    h_loss = hidden_layer_weights.loss_fn(
        loss_fn,
        pth['h.weight'],
        pth['h.bias'],
        pth['h_bn.alpha'],
        pth['h_bn.beta'],
    )
    crt_loss = crt_layer_weights.loss_fn(
        loss_fn,
        pth['crt.weight'],
        pth['crt.bias'],
    )
    o_loss = output_weights.loss_fn(
        loss_fn,
        pth['output.weight'],
        pth['output.bias'],
    )

    return input_loss, h_loss, crt_loss, o_loss


def get_model() -> TrainingModel:
    return TrainingModel(input_channel=2)


def mesh_loss(
    mesh_output: torch.Tensor,
    gt_verts: torch.Tensor,
    vert_count: torch.LongTensor,
    loss_fn: callable,
) -> torch.Tensor:
    losses = 0.0
    if mesh_output.shape != gt_verts.shape:
        return losses

    gt_verts = gt_verts.clone()

    mesh_output = mesh_output.reshape(mesh_output.shape[0] * mesh_output.shape[1], mesh_output.shape[2], mesh_output.shape[3])
    gt_verts = gt_verts.reshape(gt_verts.shape[0] * gt_verts.shape[1], gt_verts.shape[2], gt_verts.shape[3])
    for i in range(mesh_output.shape[0]):
        loss = loss_fn(mesh_output[i, :vert_count[i], :], gt_verts[i, :vert_count[i], :])
        losses += loss
    return losses


def main():
    global has_pytorch3d_loss
    args = parse_args()

    dataset = get_dataset(Path(args.data_dir))

    dataset_debug_dir = Path(args.data_dir).expanduser() / 'debug'
    dataset_debug_dir.mkdir(exist_ok=True)

    checkpoint_path = Path(args.checkpoint_path)
    checkpoint_path.mkdir(exist_ok=True)

    now = datetime.datetime.now()
    time_tag = now.strftime('%m_%d_%H_%M')
    folder = f'{time_tag}_{args.folder_name}'
    checkpoint_path = checkpoint_path.expanduser() / folder
    checkpoint_path.mkdir(exist_ok=True)
    debug_path = checkpoint_path / 'debug'
    debug_path.mkdir(exist_ok=True)
    copy_src(checkpoint_path)

    device = torch.device(args.device)
    trainingModel = get_model()
    trainingModel.to(device)

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )

    # milestones = [10000, 20000, 30000, 40000, 50000]
    milestones = None

    iteration = 0
    total_epoch = 150
    each_epoch = len(dataloader)
    each_size = 1
    total_size = each_epoch * each_size * total_epoch
    print(f'Total size: {total_size}, each_epoch: {each_epoch}, each_size: {each_size}, total_epoch: {total_epoch}')

    optimizer = torch.optim.AdamW(trainingModel.parameters(), lr=1e-3, betas=(0.5, 0.9))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 1e-3, total_size, total_epoch, len(dataloader))

    optim_name = type(optimizer).__name__
    visualizer = MyVisualizer(f'2d-avatar/{optim_name}/{time_tag}', checkpoints_dir=checkpoint_path)

    trainingModel.train()
    losses = AvgLosses()

    mesh_loss_weight = args.mesh_loss_weight
    edge_loss_weight = args.edge_loss_weight
    laplacian_loss_weight = args.laplacian_loss_weight
    normal_loss_weight = args.normal_loss_weight
    if not has_pytorch3d_loss:
        edge_loss_weight = 0.0
        laplacian_loss_weight = 0.0
        normal_loss_weight = 0.0
    crt_loss_weight = args.crt_loss_weight
    weights_loss_weight = args.weights_loss_weight
    mesh_loss_key = f'mesh(x{mesh_loss_weight})'
    edge_loss_key = f'edge(x{edge_loss_weight})'
    laplacian_loss_key = f'lap(x{laplacian_loss_weight})'
    crt_loss_key = f'crt(x{crt_loss_weight})'
    normal_loss_key = f'norm(x{normal_loss_weight})'
    mesh_output = None
    diff = None

    def mesh_loss_fn(mesh_output, gt_verts):
        return ((gt_verts - mesh_output) ** 2).mean()

    weight_loss_fn = torch.nn.MSELoss()
    crt_loss_fn = torch.nn.MSELoss()
    get_mesh = None

    with tqdm(total=total_size, dynamic_ncols=True) as pbar:
        while True:
            for full_data in dataloader:
                path, data = full_data

                data = to_tensor(data, device)
                input_pose = data['input_poses']
                gt_verts = data['output_verts']
                gt_crts = data['output_crts']

                vertices = data['base_vert']
                triangles = data['triangles']
                uvs = data['uvs']

                if is_nan(vertices) or is_nan(data) or is_nan(gt_verts) or is_nan(gt_crts):
                    raise ValueError(f'NaN data detected at file: {path}')

                optimizer.zero_grad()
                base_vertices_xy = vertices[..., :2].clone()
                base_vertices_xy.requires_grad = False

                (
                    input_layer_weights,
                    hidden_layer_weights,
                    crt_layer_weights,
                    output_weights
                ) = trainingModel(base_vertices_xy)

                diff, crt_output = runtime_model(
                    input_arr=input_pose,
                    input_layer=input_layer_weights,
                    hidden_layer=hidden_layer_weights,
                    mesh_layer=output_weights,
                    crt_layer=crt_layer_weights,
                )

                mesh_output = diff + base_vertices_xy[:, None, ...]
                if mesh_loss_weight > 0.0:
                    if has_pytorch3d_loss:
                        loss_mesh, _ = chamfer_distance(mesh_output[0], gt_verts[0])
                        loss_mesh = (loss_mesh + mesh_loss_fn(mesh_output, gt_verts)) * mesh_loss_weight
                    else:
                        loss_mesh = mesh_loss_fn(mesh_output, gt_verts) * mesh_loss_weight
                else:
                    loss_mesh = torch.tensor(0.0, device=device)
                loss_edge = do_edge_loss(mesh_output, triangles) * edge_loss_weight if edge_loss_weight > 0.0 else torch.tensor(0.0, device=device)
                loss_normal = do_normal_consistency_loss(mesh_output, triangles) * normal_loss_weight if normal_loss_weight > 0.0 else torch.tensor(0.0, device=device)
                loss_laplacian = do_laplacian_smoothing(mesh_output, triangles) * laplacian_loss_weight if laplacian_loss_weight > 0.0 else torch.tensor(0.0, device=device)
                loss_crt = crt_loss_fn(crt_output, gt_crts) * crt_loss_weight if crt_loss_weight > 0.0 else torch.tensor(0.0, device=device)

                if weights_loss_weight > 0.0:
                    input_loss, h_loss, crt_loss, o_loss = weight_loss(
                        input_layer_weights,
                        hidden_layer_weights,
                        crt_layer_weights,
                        output_weights,
                        data,
                        weight_loss_fn,
                    )

                    loss_weights = (input_loss + h_loss + o_loss + crt_loss) * weights_loss_weight
                else:
                    loss_weights = torch.tensor(0.0, device=device)

                loss = loss_mesh + loss_crt + loss_edge + loss_weights + loss_laplacian + loss_normal
                loss_dict = {
                    'loss': loss,
                }
                if mesh_loss_weight > 0.0:
                    loss_dict[mesh_loss_key] = loss_mesh
                if laplacian_loss_weight > 0.0:
                    loss_dict[laplacian_loss_key] = loss_laplacian
                if crt_loss_weight > 0.0:
                    loss_dict[crt_loss_key] = loss_crt
                if edge_loss_weight > 0.0:
                    loss_dict[edge_loss_key] = loss_edge
                if normal_loss_weight > 0.0:
                    loss_dict[normal_loss_key] = loss_normal
                if weights_loss_weight > 0.0:
                    loss_dict['weights'] = loss_weights
                def get_mesh():
                    diff, _ = runtime_model(
                        input_arr=input_pose,
                        input_layer=input_layer_weights,
                        hidden_layer=hidden_layer_weights,
                        mesh_layer=output_weights,
                        crt_layer=crt_layer_weights,
                    )
                    return diff + base_vertices_xy[:, None, ...].repeat(1, diff.shape[1], 1, 1)

                losses.update(loss_dict)
                loss.backward()
                optimizer.step()
                scheduler.step()

                iteration += 1
                pbar.update()
                visualizer.plot_current_losses(iteration, loss_dict)
                pbar_description(pbar, loss_dict, 'Training ')
                if milestones is not None and iteration in milestones:
                    new_lr = scheduler.get_last_lr()
                    visualizer.print_current_losses(
                        f'New learning rate: {new_lr}',
                        pbar=pbar,
                    )

                if iteration % args.log_iterations == 0 and (get_mesh is not None or mesh_output is not None):
                    if mesh_output is None:
                        mesh_output = get_mesh()

                    losses_mean = losses.mean()
                    mean = mesh_output.reshape(-1, 2).mean(0).detach().cpu().numpy()
                    base_mean = gt_verts.reshape(-1, 2).mean(0).detach().cpu().numpy()
                    pbar.write(f'Mesh output mean: {mean}, Base mean: {base_mean}')
                    if diff is not None:
                        diff_mean = diff.reshape(-1, 2).mean(0).detach().cpu().numpy()
                        pbar.write(f'Diff mean: {diff_mean}')
                    visualizer_print(visualizer, pbar, losses_mean, f'Iteration: {iteration}, ')

                if iteration % args.save_iterations == 0 and (get_mesh is not None or mesh_output is not None):
                    if mesh_output is None:
                        mesh_output = get_mesh()

                    visual = {}

                    if diff is not None:
                        ims = plot_batch_vertex_and_pose(
                            label_path=Path(args.data_dir) / 'labels',
                            names=path,
                            vertices=mesh_output[0],
                            poses=gt_verts[0],
                        )
                        for i, item in enumerate(ims.values()):
                            visual[f'diff_{i}'] = item

                    if len(mesh_output.shape) == 4:
                        mesh_output = mesh_output[0, 0]
                    gt_verts = gt_verts[0, 0]
                    triangles = triangles[0]

                    wire = render_mesh(
                        vertices=mesh_output,
                        triangles=triangles,
                    )
                    wire_gt = render_mesh(
                        vertices=gt_verts,
                        triangles=triangles,
                    )
                    if wire is not None and wire_gt is not None:
                        vis = np.concatenate([wire, wire_gt], axis=1)
                        visual['vis'] = vis[None, ...]

                    if len(visual) > 0:
                        visualizer.display_current_results(
                            visuals=visual,
                            total_iters=iteration,
                            epoch=-1,
                            dataset='train',
                            save_results=True,
                        )
                    save_debug_mesh(
                        vertices=mesh_output,
                        triangles=triangles,
                        uvs=uvs,
                        path=debug_path / f'iter_{iteration}_pred.obj',
                        pbar=pbar,
                    )
                    save_debug_mesh(
                        vertices=gt_verts,
                        triangles=triangles,
                        uvs=uvs,
                        path=debug_path / f'iter_{iteration}_gt.obj',
                        pbar=pbar,
                    )

                if is_nan(loss):
                    raise ValueError(f'NaN loss detected at file: {path}')

                if iteration >= total_size:
                    break

            if iteration >= total_size:
                break
            save_model(trainingModel, checkpoint_path / f'model_last.pth')

    save_model(trainingModel, checkpoint_path / f'model_fin.pth')


if __name__ == '__main__':
    main()
