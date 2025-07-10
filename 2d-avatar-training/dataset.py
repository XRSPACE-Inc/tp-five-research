from pathlib import Path
import numpy as np
import torch
import json
from torch.utils.data import Dataset
from utils import vertex_count_max, triangle_count_max


def load_npz(path: Path) -> dict:
    '''
        verts (N, 3) float32
        triangles (F, 3) int32
        vertex_normals (N, 3) float32
        input_poses (256, 15, 2) float32
        output_verts (256, N, 2) float32
        output_crts (256, 15, 5) float32
        & model weights
    '''
    if not path.exists():
        raise FileNotFoundError(f'Path {path} does not exist.')
    zdict = np.load(path, allow_pickle=True)
    d = {}
    for k in zdict:
        d[k] = zdict[k].astype(np.float32) if isinstance(zdict[k], np.ndarray) else zdict[k]

    return d


def make_training_vertices(verts: np.ndarray) -> np.ndarray:
    if len(verts.shape) == 2:
        verts = verts[np.newaxis, :, :]
    with_empty = np.zeros([verts.shape[0], vertex_count_max, verts.shape[-1]], dtype=np.float32) 
    with_empty[:, :verts.shape[1], :] = verts
    return with_empty


def load_joints(json_dict: dict) -> np.ndarray:
    morph_joint_names = [
        'Bip001_Pelvis',        # 0
        'Bip001_Spine2',        # 1
        'Bip001_Head',          # 2
        'Bip001_RUpArmTwist',   # 3
        'Bip001_R_ForeTwist',   # 4
        'Bip001_R_Hand',        # 5
        'Bip001_LUpArmTwist',   # 6
        'Bip001_L_ForeTwist',   # 7
        'Bip001_L_Hand',        # 8
        'Bip001_RThighTwist',   # 9
        'Bip001_RCalfTwist',    # 10
        'Bip001_R_Foot',        # 11
        'Bip001_LThighTwist',   # 12
        'Bip001_LCalfTwist',    # 13
        'Bip001_L_Foot'         # 14
    ]
    res = [None] * len(morph_joint_names)
    for key in json_dict['joints']:
        if key in morph_joint_names:
            index = morph_joint_names.index(key)
            res[index] = np.array(json_dict['joints'][key]['pos'], dtype=np.float32)
    return np.array(res)


def load_joint_file(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f'Path {path} does not exist.')
    with open(path, 'r') as f:
        json_dict = json.load(f)
    return load_joints(json_dict['joints'])


class MDataset(Dataset):
    def __init__(self, data_path: Path) -> None:
        self.data_files = []
        for file in data_path.glob('*.npz'):
            self.data_files.append(file)

    def __len__(self):
        return len(self.data_files)

    @staticmethod
    def load_npz(path: Path) -> dict:
        '''
            verts (2278, 3) float32
            triangles (2187, 3) int32
            vertex_normals (2278, 3) float32
            tpl_edge_index (2, 8764) int64
            geo_edge_index (2, 7510) int64
            input_poses (51, 15, 2) float32
            output_verts (51, 2278, 2) float32
            output_crts (51, 15, 5) float32
        '''
        if not path.exists():
            raise FileNotFoundError(f'Path {path} does not exist.')
        zfile = np.load(path, allow_pickle=True)
        
        vertices = zfile['verts'].astype(np.float32)
        triangles = zfile['triangles'].astype(np.int32)
        # vertex_normals = zfile['vertex_normals'].astype(np.float32)
        tpl_e = zfile['tpl_edge_index'].astype(np.int64)
        geo_e = zfile['geo_edge_index'].astype(np.int64)

        input_poses = zfile['input_poses'].astype(np.float32)
        output_verts = zfile['output_verts'].astype(np.float32)
        output_crts = zfile['output_crts'].astype(np.float32)

        # data_ls = list(
        #     zip(
        #         input_poses,
        #         output_verts,
        #         output_crts,
        #     )
        # )

        return vertices, triangles, tpl_e, geo_e, input_poses, output_verts, output_crts

    def __getitem__(self, index) -> tuple:
        path = self.data_files[index]
        npz = MDataset.load_npz(path)

        return str(path), npz


class PthDataset(Dataset):
    def __init__(self, data_path: Path) -> None:
        self.data_files = []
        for file in data_path.glob('*.pth'):
            self.data_files.append(file)

    def __len__(self):
        return len(self.data_files)

    @staticmethod
    def load_pth(path: Path) -> dict:
        '''
            input.weight (64, 30)
            input.bias (64,)
            input_bn.alpha (64,)
            input_bn.beta (64,)
            h.weight (64, 64)
            h.bias (64,)
            h_bn.alpha (64,)
            h_bn.beta (64,)
            output.weight (2040, 64)
            output.bias (2040,)
            crt.weight (75, 64)
            crt.bias (75,)
            base_vert (1020, 3)
            triangles (1121, 3)
            uvs (1020, 2)
        '''
        if not path.exists():
            raise FileNotFoundError(f'Path {path} does not exist.')
        pth = torch.load(path, map_location='cpu', weights_only=False)
        if 'state_dict' in pth:
            del pth['state_dict']
        pth['base_vert'] = pth['base_vert'].astype(np.float32)
        pth['uvs'] = pth['uvs'].astype(np.float32)
        return pth

    def __getitem__(self, index) -> tuple:
        path = self.data_files[index]
        pth = PthDataset.load_pth(path)

        return str(path), pth


class CombinedDataset(Dataset):
    def __init__(self, dataset_path: Path) -> None:
        np_data_files = []
        for file in (dataset_path / 'training_data_trimed').glob('*.npz'):
            np_data_files.append(file.stem)
        pth_data_files = []
        for file in (dataset_path / 'pth').glob('*.pth'):
            pth_data_files.append(file.stem)
        json_data_files = []
        for file in (dataset_path / 'labels').glob('*.json'):
            json_data_files.append(file.stem)
        self.data_files = list(set(np_data_files) & set(pth_data_files) & set(json_data_files))
        self.dataset_path = dataset_path

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index) -> tuple:
        name = self.data_files[index]
        npz_path = self.dataset_path / 'training_data_trimed' / f'{name}.npz'
        pth_path = self.dataset_path / 'pth' / f'{name}.pth'
        pose_path = self.dataset_path / 'labels' / f'{name}.json'

        npz = MDataset.load_npz(npz_path)
        pth = PthDataset.load_pth(pth_path)
        pose = load_joint_file(pose_path)

        verts = pth['base_vert']
        with_empty = np.zeros([vertex_count_max, verts.shape[1]], dtype=np.float32)
        with_empty[:min(verts.shape[0], vertex_count_max), :] = verts[:min(verts.shape[0], vertex_count_max), :]
        pth['vert_count'] = verts.shape[0]
        pth['base_vert'] = with_empty
        uvs = pth['uvs']
        with_empty = np.zeros([vertex_count_max, uvs.shape[1]], dtype=np.float32)
        with_empty[:min(uvs.shape[0], vertex_count_max), :] = uvs[:min(uvs.shape[0], vertex_count_max), :]
        pth['uvs'] = with_empty

        triangles = pth['triangles']
        with_empty = np.zeros([triangle_count_max, triangles.shape[1]], dtype=np.int32)
        with_empty[:min(triangles.shape[0], triangle_count_max), :] = triangles[:min(triangles.shape[0], triangle_count_max), :]
        pth['triangles'] = with_empty
        pth['tri_count'] = triangles.shape[0]

        output_weights = pth['output.weight']
        with_empty = np.zeros([triangle_count_max, output_weights.shape[1]], dtype=np.float32)
        with_empty[:output_weights.shape[0], :] = output_weights[:triangle_count_max, :]
        pth['output.weight'] = with_empty
        output_bias = pth['output.bias']
        with_empty = np.zeros([triangle_count_max], dtype=np.float32)
        with_empty[:output_bias.shape[0]] = output_bias[:triangle_count_max]
        pth['output.bias'] = with_empty

        (
            _,
            _,
            tpl_e,
            geo_e,
            input_poses,
            output_verts,
            output_crts,
        ) = npz

        with_empty = np.zeros([output_verts.shape[0], vertex_count_max, output_verts.shape[2]], dtype=np.float32) 
        with_empty[:, :output_verts.shape[1], :] = output_verts[:, :vertex_count_max, :]
        output_verts = with_empty

        npz = (
            input_poses,
            output_verts,
            output_crts,
        )

        return name, npz, pth, pose


class ARAPDataset(Dataset):
    def __init__(self, dataset_path: Path, with_pose: bool) -> None:
        np_data_files = []
        for file in (dataset_path / 'data').glob('*.npz'):
            np_data_files.append(file.stem)
        json_data_files = []
        for file in (dataset_path / 'labels').glob('*.json'):
            json_data_files.append(file.stem)
        self.data_files = list(set(np_data_files) & set(json_data_files))
        self.dataset_path = dataset_path
        self.with_pose = with_pose

    def __len__(self):
        return len(self.data_files)
    
    def __getitem__(self, index) -> tuple:
        name = self.data_files[index]
        npz_path = self.dataset_path / 'data' / f'{name}.npz'
        if self.with_pose:
            pose_path = self.dataset_path / 'labels' / f'{name}.json'

        npz = load_npz(npz_path)
        pose = load_joint_file(pose_path) if self.with_pose else None

        if self.with_pose:
            return name, npz, pose

        return name, npz
