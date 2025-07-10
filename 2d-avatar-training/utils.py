
from pathlib import Path
from typing import Dict, Union
import numpy as np
import torch
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import dijkstra


vertex_count_max = 6000
triangle_count_max = 12000

class AvgLosses():
    def __init__(self) -> None:
        self.losses = {}

    def update(self, loss_dict: Dict[str, Union[torch.Tensor, float]]) -> None:
        for key, loss in loss_dict.items():
            self.update_single(key, loss)

    def update_single(self, key: str, loss: Union[torch.Tensor, float]) -> None:
        if key not in self.losses:
            self.losses[key] = []
        if isinstance(loss, torch.Tensor):
            self.losses[key].append(loss.detach().item())
        elif isinstance(loss, float):
            self.losses[key].append(loss)
        else:
            raise TypeError(f'Unsupported loss type: {type(loss)} for key: {key}')

    def mean(self) -> Dict[str, float]:
        mean_losses = {}
        for k, v in self.losses.items():
            if len(v) > 0:
                mean_losses[k] = np.array(v).mean()
                self.losses[k] = []
            else:
                mean_losses[k] = 0.0
        return mean_losses


class AvgLoss():
    def __init__(self) -> None:
        self.losses = []

    def update(self, loss: torch.Tensor) -> None:
        if isinstance(loss, torch.Tensor):
            self.losses.append(loss.detach().item())
        elif isinstance(loss, float):
            self.losses.append(loss)

    def mean(self) -> float:
        if len(self.losses) > 0:
            loss = np.array(self.losses).mean()
            self.losses = []
            return loss
        return 0


def to_np(
    array_like: Union['torch.Tensor', np.array, list, tuple],
    squeeze: bool = False,
) -> np.ndarray:
    '''
    Helper method to convert array_like object to numpy array.

    Parameters
    ----------
        array_like (torch.Tensor, np.ndarray, list, tuple) : array object to convert.
        squeeze (bool): squeeze batch dim.
    '''
    if isinstance(array_like, torch.Tensor):
        if squeeze and array_like.shape[0] == 1:
            array_like = array_like.squeeze(0)
        array_like = array_like.cpu().detach().numpy()
    elif isinstance(array_like, np.ndarray):
        if squeeze and array_like.shape[0] == 1:
            array_like = array_like.squeeze(0)
    elif isinstance(array_like, list) or isinstance(array_like, tuple) or isinstance(array_like, int):
        array_like = np.array(array_like)
    elif array_like is None:
        array_like = np.array([])
    elif array_like is not None:
        print(array_like)
        print(type(array_like))
        raise Exception('array_like cannot convert to numpy array')
    return array_like


def write_obj(file: Union[str, Path], mesh: Dict) -> None:
    '''
    Save mesh as wavefront(.obj) file.

    Parameters
    ----------
        file (str or Path): file path.
            Supported file extension are:
                .obj: wavefront txt file.
                .npz: save file into a np file.

        mesh (dict): mesh data stored in dict object.
            mesh keys:
                (Data type: torch.tensor/numpy array)
                verts (N, 3): Vertices.
                faces (N, 3): Indices.
                uvs (N, 2): Vertex UV.
                norms (N, 3): Vertex normal.
                vcs (N, 3): Vertex color.
                uv_faces (N, 3): UV indices.
                norm_faces (N, 3): Normal indices.
                mtl_index: material data.
    '''
    if 'verts' not in mesh:
        print('write_obj Error: Cannot find key verts in mesh. file :', file)
        return

    file = Path(file)
    if file.suffix not in ['.obj', '.npz']:
        print(f'write_obj Error: ext type {file.suffix} is not supported. (".obj", ".npz")')
        return

    verts = to_np(mesh['verts'], True).astype(np.float32)
    indices = to_np(mesh['faces']) if 'faces' in mesh else []
    uvs = to_np(mesh['uvs'], True) if 'uvs' in mesh else []
    normals = to_np(mesh['norms'], True) if 'norms' in mesh else []
    colors = to_np(mesh['vcs'], True) if 'vcs' in mesh else []
    uv_faces = to_np(mesh['uv_faces']) if 'uv_faces' in mesh else []
    norm_faces = to_np(mesh['norm_faces']) if 'norm_faces' in mesh else []

    mtl_index = mesh['mtl_index'] if 'mtl_index' in mesh else {}
    mtllib = mesh['mtllib'] if 'mtllib' in mesh else []

    if file.suffix == '.npz':
        obj_dict = {
            'verts': verts,
            'indices': indices,
            'uvs': uvs,
            'normals': normals,
            'vcs': colors,
            'uv_faces': uv_faces,
            'norm_faces': norm_faces,
            'mtl_index': mtl_index,
            'mtllib': mtllib,
        }

        np.savez(file, obj_dict=obj_dict)
        return

    def write_face(i, mlt_dict, arr, arr2=None, arr3=None):
        res = 'f'
        if i in mlt_dict:
            res = 'usemtl {}\nf'.format(mlt_dict[i])
        if arr2 is not None and arr3 is not None:
            for i in range(len(arr)):
                res += ' {}/{}/{}'.format(arr[i] + 1, arr2[i] + 1, arr3[i] + 1)
        elif arr2 is not None:
            for i in range(len(arr)):
                res += ' {}/{}/'.format(arr[i] + 1, arr2[i] + 1)
        elif arr3 is not None:
            for i in range(len(arr)):
                res += ' {}//{}'.format(arr[i] + 1, arr3[i] + 1)
        else:
            for i in range(len(arr)):
                res += ' {}'.format(arr[i] + 1)
        return res + '\n'

    uv_indices = uv_faces if len(uv_faces) > 0 else indices
    norm_indices = norm_faces if len(norm_faces) > 0 else indices

    s = ''

    if len(mtllib) > 0:
        for mtlfile in mtllib:
            s += f'mtllib {mtlfile}\n'
        s += '\n'

    if len(colors) == len(verts):
        for i in range(len(verts)):
            s += 'v {} {} {} {} {} {}\n'.format(
                verts[i][0], verts[i][1], verts[i][2], colors[i][0],
                colors[i][1], colors[i][2])
    else:
        for v in verts:
            s += 'v {} {} {}\n'.format(v[0], v[1], v[2])
    s += '\n'
    withuv = False
    withnor = False
    if len(uvs) > 0:
        for uv in uvs:
            s += 'vt {} {}\n'.format(uv[0], uv[1])
        withuv = True
        s += '\n'
    if len(normals) > 0:
        for n in normals:
            s += 'vn {} {} {}\n'.format(n[0], n[1], n[2])
        withnor = True
        s += '\n'

    if not withuv and not withnor:
        for i in range(len(indices)):
            s += write_face(i, mtl_index, indices[i])
    else:
        if withuv and withnor:
            for i in range(len(indices)):
                s += write_face(i, mtl_index, indices[i], uv_indices[i], norm_indices[i])
        elif withuv:
            for i in range(len(indices)):
                s += write_face(i, mtl_index, indices[i], uv_indices[i])
        else:
            for i in range(len(indices)):
                s += write_face(i, mtl_index, indices[i], None, norm_indices[i])

    with open(file, 'w+') as f:
        f.write(s)


def copy_src(checkpoint: Path) -> None:
    import glob
    import shutil
    import sys

    ls = glob.glob('*.py')
    out = checkpoint / 'src'
    out.mkdir(exist_ok=True)

    cmd = ' '.join(sys.argv)
    with open(checkpoint / 'cmd.txt', 'w') as f:
        f.write(cmd)

    for f in ls:
        f = Path(f)
        name = f.name
        shutil.copy(f, out / name)


def compute_vertex_normals(verts_packed: torch.Tensor, faces_packed: torch.Tensor) -> torch.Tensor:
    ''' reference: https://github.com/facebookresearch/pytorch3d/issues/736
    '''
    verts_normals = torch.zeros_like(verts_packed)
    vertices_faces = verts_packed[faces_packed]

    faces_normals = torch.cross(
        vertices_faces[:, 2] - vertices_faces[:, 1],
        vertices_faces[:, 0] - vertices_faces[:, 1],
        dim=1,
    )

    verts_normals.index_add_(0, faces_packed[:, 0], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 1], faces_normals)
    verts_normals.index_add_(0, faces_packed[:, 2], faces_normals)
    
    return torch.nn.functional.normalize(
        verts_normals, eps=1e-6, dim=1
    )


def get_tpl_edges(remesh_obj_v, remesh_obj_f):
    edge_index = []
    for v in range(len(remesh_obj_v)):
        face_ids = np.argwhere(remesh_obj_f == v)[:, 0]
        neighbor_ids = []
        for face_id in face_ids:
            for v_id in range(3):
                if remesh_obj_f[face_id, v_id] != v:
                    neighbor_ids.append(remesh_obj_f[face_id, v_id])
        neighbor_ids = list(set(neighbor_ids))
        neighbor_ids = [np.array([v, n])[np.newaxis, :] for n in neighbor_ids]
        if len(neighbor_ids) == 0:
            continue
        neighbor_ids = np.concatenate(neighbor_ids, axis=0)
        edge_index.append(neighbor_ids)
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def calc_surface_geodesic(mesh):
    # We denselu sample 4000 points to be more accuracy.
    samples = mesh.sample_points_poisson_disk(number_of_points=4000)
    pts = np.asarray(samples.points)
    pts_normal = np.asarray(samples.normals)

    N = len(pts)
    verts_dist = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    verts_nn = np.argsort(verts_dist, axis=1)
    conn_matrix = lil_matrix((N, N), dtype=np.float32)

    for p in range(N):
        nn_p = verts_nn[p, 1:6]
        norm_nn_p = np.linalg.norm(pts_normal[nn_p], axis=1)
        norm_p = np.linalg.norm(pts_normal[p])
        cos_similar = np.dot(pts_normal[nn_p], pts_normal[p]) / (norm_nn_p * norm_p + 1e-10)
        nn_p = nn_p[cos_similar > -0.5]
        conn_matrix[p, nn_p] = verts_dist[p, nn_p]
    [dist, predecessors] = dijkstra(conn_matrix, directed=False, indices=range(N),
                                    return_predecessors=True, unweighted=False)

    # replace inf distance with euclidean distance + 8
    # 6.12 is the maximal geodesic distance without considering inf, I add 8 to be safer.
    inf_pos = np.argwhere(np.isinf(dist))
    if len(inf_pos) > 0:
        euc_distance = np.sqrt(np.sum((pts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
        dist[inf_pos[:, 0], inf_pos[:, 1]] = 8.0 + euc_distance[inf_pos[:, 0], inf_pos[:, 1]]

    verts = np.array(mesh.vertices)
    vert_pts_distance = np.sqrt(np.sum((verts[np.newaxis, ...] - pts[:, np.newaxis, :]) ** 2, axis=2))
    vert_pts_nn = np.argmin(vert_pts_distance, axis=0)
    surface_geodesic = dist[vert_pts_nn, :][:, vert_pts_nn]
    return surface_geodesic


def get_geo_edges(surface_geodesic, remesh_obj_v):
    edge_index = []
    surface_geodesic += 1.0 * np.eye(len(surface_geodesic))  # remove self-loop edge here
    for i in range(len(remesh_obj_v)):
        geodesic_ball_samples = np.argwhere(surface_geodesic[i, :] <= 0.06).squeeze(1)
        if len(geodesic_ball_samples) > 10:
            geodesic_ball_samples = np.random.choice(geodesic_ball_samples, 10, replace=False)
        edge_index.append(np.concatenate((np.repeat(i, len(geodesic_ball_samples))[:, np.newaxis],
                                          geodesic_ball_samples[:, np.newaxis]), axis=1))
    edge_index = np.concatenate(edge_index, axis=0)
    return edge_index


def to_tensor(data: object | list | dict | tuple, device: torch.device) -> object | list | dict | tuple:
    if isinstance(data, list) or isinstance(data, tuple):
        ls = []
        for item in data:
            ls.append(to_tensor(item, device))
        return ls
    elif isinstance(data, dict):
        d = {}
        for k, v in data.items():
            d[k] = to_tensor(v, device)
        return d
    elif isinstance(data, torch.Tensor):
        return data.to(device)
    return data


def is_nan(data: object | list | dict | tuple) -> bool:
    if isinstance(data, list) or isinstance(data, tuple):
        for item in data:
            if torch.isnan(item).any():
                return True
        return False
    elif isinstance(data, dict):
        for k, v in data.items():
            if torch.isnan(v).any():
                return True
        return False
    elif isinstance(data, torch.Tensor):
        return torch.isnan(data).any()
    return False

