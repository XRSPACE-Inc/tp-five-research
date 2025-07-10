from typing import Dict
import numpy as np
from PIL import Image
from pyrender import RenderFlags

try:
    import trimesh
    import trimesh.visual
    from pyrender import (DirectionalLight, Mesh, MetallicRoughnessMaterial,
                          Node, OffscreenRenderer, PerspectiveCamera, Scene)
except Exception:
    print('pyrender not installed, pyrender mode will not work')


def translate(pos: np.ndarray) -> np.ndarray:
    '''
    Vector to translate matrix.

    Args:
    -----
        pos: np.ndarray, translate vector. [3]

    Returns:
        translate_matrix: np.ndarray, translate matrix. [4, 4]
    '''
    mat = np.eye(4, dtype=np.float32)
    mat[3, :3] = pos
    return mat


def rotate(euler: np.ndarray) -> np.ndarray:
    '''
    Euler to rotation matrix.

    Args:
    -----
        euler: np.array, euler angles. [3]

    Returns:
    --------
        rotation_matrix: np.array, rotation matrix. [4, 4]
    '''

    yaw = euler[1]
    pitch = euler[0]
    roll = euler[2]

    ch = np.cos(yaw)
    sh = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cb = np.cos(roll)
    sb = np.sin(roll)

    one = np.ones_like(yaw)
    zero = np.zeros_like(yaw)

    mat = np.eye(4, dtype=np.float32)

    mat[0, 0] = ch * cb + sh * sp * sb
    mat[0, 1] = sb * cp
    mat[0, 2] = -sh * cb + ch * sp * sb
    mat[0, 3] = zero
    mat[1, 0] = -ch * sb + sh * sp * cb
    mat[1, 1] = cb * cp
    mat[1, 2] = sb * sh + ch * sp * cb
    mat[1, 3] = zero
    mat[2, 0] = sh * cp
    mat[2, 1] = -sp
    mat[2, 2] = ch * cp
    mat[2, 3] = zero
    mat[3, 0] = zero
    mat[3, 1] = zero
    mat[3, 2] = zero
    mat[3, 3] = one

    return mat


def transform_mat(pos: np.ndarray, euler: np.ndarray) -> np.ndarray:
    '''
    Get transform matrix from translate and euler angle.

    Args:
    -----
        pos: np.ndarray, translate vector. [3,]
        euler: np.ndarray, euler angles. [3,]

    Returns:
    --------
        matrix: np.ndarray, transform matrix. [4, 4]
    '''
    t = translate(pos)
    r = rotate(euler)

    return np.matmul(r, t)


class Pyrenderer():
    def __init__(self, config: Dict = None, texture: np.ndarray = None, overrides: Dict = None) -> None:
        if config is not None:
            camera = config['camera']
            w = camera['width']
            h = camera['height']
            self.fov = camera['fov']
            self.near = camera['near']
            self.far = camera['far']
            self.width = int(w)
            self.height = int(h)
            self.resolution = (self.width, self.height)

            self.camera_pos = np.array(camera['pos'])
            self.camera_rot = np.deg2rad(np.array(camera['rot']))
            if 'pose' in camera and camera['pose'] is not None:
                self.camera_pose = np.array(camera['pose']).reshape(4, 4)
            else:
                self.camera_pose = None
        else:
            self.fov = 45
            self.near = 0.1
            self.far = 100.0
            self.width = 512
            self.height = 512
            self.resolution = (self.width, self.height)
            self.camera_pos = np.array([0.0, 0.0, 3.0])
            self.camera_rot = np.array([0.0, 0.0, 0.0])
            self.camera_pose = None

        if overrides is not None:
            if 'fov' in overrides and overrides['fov'] is not None:
                self.fov = float(overrides['fov'])
            if 'height' in overrides and overrides['height'] is not None:
                self.height = int(overrides['height'])
            if 'width' in overrides and overrides['width'] is not None:
                self.width = int(overrides['width'])

        self.renderer = OffscreenRenderer(viewport_width=self.width, viewport_height=self.height)
        aspect = self.width / self.height

        self.camera = PerspectiveCamera(np.deg2rad(self.fov), self.near, self.far, aspect)

        if texture is not None:
            texture = Image.fromarray(texture)

        self.mat = MetallicRoughnessMaterial(metallicFactor=0.0, roughnessFactor=1.0, baseColorTexture=texture)

    def get_camera_data(self):
        cam_pose = self.camera_pose.tolist() if isinstance(self.camera_pose, np.ndarray) else None
        cam_pos = self.camera_pos.tolist() if self.camera_pos is not None else None
        cam_rot = self.camera_rot.tolist() if self.camera_rot is not None else None
        return {
            'fov': self.fov,
            'width': self.width,
            'height': self.height,
            'near': self.near,
            'far': self.far,
            'pose': cam_pose,
            'pos': cam_pos,
            'rot': cam_rot,
        }

    def setup_scene(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: np.ndarray = None,
    ) -> 'Scene':
        if len(vertices.shape) == 3:
            vertices = vertices[0]
        tmesh = trimesh.Trimesh(
            vertices=vertices,
            faces=faces,
            visual=None if uvs is None else trimesh.visual.TextureVisuals(uv=uvs.copy()),
        )
        mesh = Mesh.from_trimesh(tmesh, self.mat)

        if self.camera_pose is not None:
            cam_pose = self.camera_pose
        else:
            cam_pose = np.eye(4)
            cam_pose[:3] = transform_mat(
                self.camera_pos,
                self.camera_rot,
            ).T[:3]

        scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]), bg_color=(0, 0, 0, 0))
        node = Node(name='mesh', mesh=mesh)
        scene.add_node(node)
        direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
        scene.add(direc_l, name='directional light', pose=cam_pose)
        scene.add(self.camera, name='camera', pose=cam_pose)
        return scene

    def setup_scene_from_file(
        self,
        obj_path: str,
    ) -> 'Scene':
        tmesh = trimesh.load(obj_path)
        mesh = Mesh.from_trimesh(tmesh, self.mat)

        if self.camera_pose is not None:
            cam_pose = self.camera_pose
        else:
            cam_pose = np.eye(4)
            cam_pose[:3] = transform_mat(
                self.camera_pos,
                self.camera_rot,
            ).T[:3]

        scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]), bg_color=(0, 0, 0, 0))
        node = Node(name='mesh', mesh=mesh)
        scene.add_node(node)
        direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
        scene.add(direc_l, name='directional light', pose=cam_pose)
        scene.add(self.camera, name='camera', pose=cam_pose)
        return scene

    def render(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        uvs: np.ndarray = None,
        wireframe: bool = False,
        normal: bool = False
    ):
        assert not (wireframe and normal), "Wireframe and normal rendering cannot be used together."
        scene = self.setup_scene(vertices, faces, uvs)
        flags = RenderFlags.FACE_NORMALS if normal else RenderFlags.ALL_WIREFRAME if wireframe else RenderFlags.NONE

        fake, depth = self.renderer.render(scene, flags=flags)

        return fake, depth

    def render_from_file(
        self,
        obj_path: str,
        wireframe: bool = False,
        normal: bool = False
    ):
        assert not (wireframe and normal), "Wireframe and normal rendering cannot be used together."
        scene = self.setup_scene_from_file(obj_path)
        flags = RenderFlags.FACE_NORMALS if normal else RenderFlags.ALL_WIREFRAME if wireframe else RenderFlags.NONE
        fake, depth = self.renderer.render(scene, flags=flags)

        return fake, depth
