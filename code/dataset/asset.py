from dataclasses import dataclass
import numpy as np
from numpy import ndarray
from typing import List, Union, Tuple
from collections import defaultdict
import os
import trimesh
from scipy.spatial.transform import Rotation as R
from .sampler import Sampler
from .exporter import Exporter
import jittor as jt
import math

def axis_angle_to_matrix(axis_angle: ndarray) -> ndarray:
    res = np.pad(R.from_rotvec(axis_angle).as_matrix(), ((0, 0), (0, 1), (0, 1)), 'constant', constant_values=((0, 0), (0, 0), (0, 0)))
    assert res.ndim == 3
    res[:, -1, -1] = 1
    return res

def axis_angle_to_matrix_jt(axis_angle: jt.Var) -> jt.Var:
    """
    axis_angle: (J, 3) 每个关节的旋转向量 (Rodrigues 格式)
    return: (J, 4, 4) 齐次旋转矩阵
    """
    # (J, 1) 旋转角度
    theta = jt.norm(axis_angle, dim=1, keepdim=True) + 1e-8
    # (J, 3) 旋转轴
    axis = axis_angle / theta

    x, y, z = axis[:,0], axis[:,1], axis[:,2]
    c = jt.cos(theta).squeeze(-1)
    s = jt.sin(theta).squeeze(-1)
    one_c = 1 - c

    # (J, 3, 3) Rodrigues' rotation formula
    rot = jt.stack([
        c + x*x*one_c,     x*y*one_c - z*s, x*z*one_c + y*s,
        y*x*one_c + z*s,   c + y*y*one_c,   y*z*one_c - x*s,
        z*x*one_c - y*s,   z*y*one_c + x*s, c + z*z*one_c
    ], dim=-1).reshape(-1, 3, 3)

    # pad to (J, 4, 4)
    pad = jt.zeros((rot.shape[0], 4, 4), dtype=rot.dtype)
    pad[:, :3, :3] = rot
    pad[:, 3, 3] = 1.0
    return pad

def linear_blend_skinning(
    vertex: ndarray,
    matrix_local: ndarray,
    matrix: ndarray,
    skin: ndarray,
    pad: int=0,
    value: float=0.,
):
    '''
    Args:
        vertex: (N, 3+pad)
        matrix_local: (J, 4, 4)
        matrix: (J, 4, 4)
        skin: (N, J), value of pseudo bones should be 0
    Returns:
        (N, 3)
    '''
    J = matrix_local.shape[0]
    padded = np.pad(vertex, ((0, 0), (0, pad)), 'constant', constant_values=(0, value))
    offset = (
        np.linalg.inv(matrix_local) @
        np.tile(padded.transpose(), (J, 1, 1))
    )
    per_bone_matrix = matrix @ offset
    weighted_per_bone_matrix = skin.T[:, np.newaxis, :] * per_bone_matrix
    g = np.sum(weighted_per_bone_matrix, axis=0)
    final = g[:3, :] / (np.sum(skin, axis=1) + 1e-8)
    return final.T

def linear_blend_skinning_jt(vertex: jt.Var,
                             matrix_local: jt.Var,
                             matrix: jt.Var,
                             skin: jt.Var,
                             pad: int = 0,
                             value: float = 0.) -> jt.Var:
    """
    Stable Linear Blend Skinning (Jittor)
    vertex: (N,3+pad)
    matrix_local: (J,4,4)
    matrix: (J,4,4)
    skin: (N,J)
    """
    N = vertex.shape[0]
    J = matrix_local.shape[0]

    # 1) pad vertex
    if pad > 0:
        pad_tensor = jt.full((N,pad), value, dtype=vertex.dtype)
        padded = jt.concat([vertex, pad_tensor], dim=1)  # (N,3+pad)
    else:
        padded = vertex  # (N,3)

    # 2) inverse local matrices
    matrix_local_inv = jt.linalg.inv(matrix_local)  # (J,4,4)

    # 3) prepare points for batch matmul: (J,4,N)
    padded_T = padded.transpose(1,0).unsqueeze(0).repeat(J,1,1)  # (J,4,N)

    # 4) offset
    offset = jt.bmm(matrix_local_inv, padded_T)  # (J,4,N)

    # 5) apply per-bone matrix
    per_bone = jt.bmm(matrix, offset)  # (J,4,N)

    # 6) take xyz and transpose to (J,N,3)
    per_bone_xyz = per_bone[:, :3, :].transpose(0, 2, 1)  # (J,N,3)

    # 7) apply skin weights
    skin_t = jt.transpose(skin, (1,0)).unsqueeze(2)  # (J,N,1)
    weighted = per_bone_xyz * skin_t  # (J,N,3)

    # 8) sum over bones and divide by weight sum
    summed = jt.sum(weighted, dim=0)  # (N,3)
    weight_sum = jt.sum(skin, dim=1, keepdims=True)  # (N,1)
    weight_sum = jt.where(weight_sum==0, jt.ones_like(weight_sum)*1e-6, weight_sum)
    final = summed / weight_sum  # (N,3)

    return final


@dataclass
class Asset(Exporter):
    '''
    A simple asset for loading mesh, skeleton and skinning.
    '''
    
    # cls of data
    cls: str
    
    # data id
    id: int
    
    # vertices of mesh, shape (N, 3), float32
    vertices: ndarray
    
    # normal of vertices of mesh, shape (N, 3), float32
    vertex_normals: Union[ndarray, None]
    
    # faces of mesh, shape (F, 3), face id starts from 0 to F-1, int64
    faces: Union[ndarray, None]
    
    # face normal of mesh, shape (F, 3), float32
    face_normals: Union[ndarray, None]
    
    # joints of bones, shape (J, 3), float32
    joints: Union[ndarray, None] = None
    
    # skinning of joints, shape (N, J), float32
    skin: Union[ndarray, None] = None
    
    # parents of joints, None represents no parent(a root joint)
    # make sure parent[k] < k
    parents: Union[List[Union[int, None]], None] = None
    
    # names of joints
    names: Union[List[str], None] = None
    
    # local coordinate of bones
    matrix_local: Union[ndarray, None] = None
    
    def check_order(self) -> bool:
        for i in range(self.J):
            if self.parents[i] is not None and self.parents[i] >= i:
                return False
        return True
    
    @property
    def N(self):
        '''
        number of vertices
        '''
        return self.vertices.shape[0]
    
    @property
    def F(self):
        '''
        number of faces
        '''
        return self.faces.shape[0]
    
    @property
    def J(self):
        '''
        number of joints
        '''
        return self.joints.shape[0]
    
    @staticmethod
    def load(path: str) -> 'Asset':
        data = np.load(path, allow_pickle=True)
        d = {n: v[()] for (n, v) in data.items()}
        return Asset(**d)
    
    def set_order_by_names(self, new_names: List[str]):
        assert len(new_names) == len(self.names)
        name_to_id = {name: id for (id, name) in enumerate(self.names)}
        new_name_to_id = {name: id for (id, name) in enumerate(new_names)}
        perm = []
        new_parents = []
        for (new_id, name) in enumerate(new_names):
            perm.append(name_to_id[name])
            pid = self.parents[name_to_id[name]]
            if new_id == 0:
                assert pid is None, 'first bone is not root bone'
            else:
                pname = self.names[pid]
                pid = new_name_to_id[pname]
                assert pid < new_id, 'new order does not form a tree'
            new_parents.append(pid)
        
        if self.joints is not None:
            self.joints = self.joints[perm]
        self.parents = new_parents
        if self.skin is not None:
            self.skin = self.skin[:, perm]
        if self.matrix_local is not None:
            self.matrix_local = self.matrix_local[perm]
        self.names = new_names
    
    def get_random_matrix_basis(self, random_pose_angle: float) -> ndarray:
        '''
        return a random pose matrix_basis
        '''
        matrix_basis = axis_angle_to_matrix((np.random.rand(self.J, 3) - 0.5) * random_pose_angle / 180 * np.pi * 2).astype(np.float32)
        return matrix_basis

    def get_random_matrix_basis_jt(self, random_pose_angle: float) -> jt.Var:
        """
        返回随机姿态的 matrix_basis
        输出: (J, 4, 4)  齐次旋转矩阵
        """
        # 生成 (J, 3) 的随机旋转向量：范围约在 [-2*angle, 2*angle]（弧度）
        rotvec = (jt.rand(self.J, 3) - 0.5) * (random_pose_angle / 180.0) * math.pi * 2.0
        matrix_basis = axis_angle_to_matrix_jt(rotvec).float32()
        return matrix_basis
    
    def apply_matrix_basis(self, matrix_basis: ndarray):
        '''
        apply a pose to armature
        
        matrix_basis: (J, 4, 4)
        '''
        matrix_local = self.matrix_local
        if matrix_local is None:
            matrix_local = np.zeros((self.J, 4, 4))
            matrix_local[:, 0, 0] = 1.
            matrix_local[:, 1, 1] = 1.
            matrix_local[:, 2, 2] = 1.
            matrix_local[:, 3, 3] = 1.
            for i in range(self.J):
                matrix_local[i, :3, 3] = self.joints[i]
        
        matrix = np.zeros((self.J, 4, 4))
        for i in range(self.J):
            if i==0:
                matrix[i] = matrix_local[i] @ matrix_basis[i]
            else:
                pid = self.parents[i]
                matrix_parent = matrix[pid]
                matrix_local_parent = matrix_local[pid]
                
                matrix[i] = (
                    matrix_parent @
                    (np.linalg.inv(matrix_local_parent) @ matrix_local[i]) @
                    matrix_basis[i]
                )
        self.joints = matrix[:, :3, 3]
        vertices = linear_blend_skinning(self.vertices, matrix_local, matrix, self.skin, pad=1, value=1.)
        # update matrix_local
        self.matrix_local = matrix
        
        # in accordance with trimesh's normals
        mesh = trimesh.Trimesh(vertices=vertices, faces=self.faces, process=False)
        self.vertices = vertices
        self.vertex_normals = mesh.vertex_normals.copy()
        self.face_normals = mesh.face_normals.copy()

    def apply_matrix_basis_jt(self, matrix_basis):
        """
        Apply pose to armature (Jittor version)
        matrix_basis: (J,4,4) np.ndarray or jt.Var
        """
        J = self.J

        # ------------------------------
        # Ensure matrix_basis is jt.Var
        if not isinstance(matrix_basis, jt.Var):
            matrix_basis = jt.array(matrix_basis).float32()

        # ------------------------------
        # Prepare matrix_local
        matrix_local = self.matrix_local
        if matrix_local is None:
            matrix_local_np = np.zeros((J,4,4), dtype=np.float32)
            matrix_local_np[:,0,0] = 1.
            matrix_local_np[:,1,1] = 1.
            matrix_local_np[:,2,2] = 1.
            matrix_local_np[:,3,3] = 1.
            for i in range(J):
                matrix_local_np[i,:3,3] = self.joints[i]
            matrix_local = jt.array(matrix_local_np).float32()
        else:
            if not isinstance(matrix_local, jt.Var):
                matrix_local = jt.array(matrix_local).float32()

        # ------------------------------
        # Initialize global matrices
        matrix = jt.zeros((J,4,4), dtype=matrix_local.dtype)

        for i in range(J):
            if i == 0:
                matrix[i] = jt.matmul(matrix_local[i], matrix_basis[i])
            else:
                pid = self.parents[i]
                matrix_parent = matrix[pid]
                matrix_local_parent = matrix_local[pid]

                inv_local_parent = jt.linalg.inv(matrix_local_parent)
                matrix[i] = jt.matmul(matrix_parent, jt.matmul(inv_local_parent @ matrix_local[i], matrix_basis[i]))

        # ------------------------------
        # Update joints
        self.joints = matrix[:, :3, 3].numpy()

        # ------------------------------
        # Apply Linear Blend Skinning
        verts_jt = linear_blend_skinning_jt(
            jt.array(self.vertices).float32(),
            matrix_local,
            matrix,
            jt.array(self.skin).float32() if self.skin is not None else jt.zeros((len(self.vertices), J)),
            pad=1,
            value=1.
        )

        # ------------------------------
        # Update matrix_local
        self.matrix_local = matrix

        # ------------------------------
        # Update vertices and normals
        self.vertices = verts_jt.numpy()
        mesh = trimesh.Trimesh(vertices=self.vertices, faces=self.faces, process=False)
        self.vertex_normals = mesh.vertex_normals.copy()
        self.face_normals = mesh.face_normals.copy()


    def sample(self, sampler: Sampler) -> 'SampledAsset':
        '''
        return sampled asset for model input
        '''
        vertex_groups = {}
        if self.skin is not None:
            vertex_groups['skin'] = self.skin.copy()
        sampled_vertices, sampled_normal, vertex_groups = sampler.sample(
            vertices=self.vertices,
            vertex_normals=self.vertex_normals,
            face_normals=self.face_normals,
            vertex_groups=vertex_groups,
            faces=self.faces,
        )
        
        eps = 1e-6
        sampled_normal = sampled_normal / (np.linalg.norm(sampled_normal, axis=1, keepdims=True) + eps)
        sampled_normal = np.nan_to_num(sampled_normal, nan=0., posinf=0., neginf=0.)
        
        return SampledAsset(
            cls=self.cls,
            id=self.id,
            vertices=sampled_vertices,
            normals=sampled_normal,
            joints=self.joints,
            skin=vertex_groups.get('skin', None),
            parents=self.parents,
            names=self.names,
        )
    
    def export_pc(self, path: str, with_normal: bool=True, size=0.01):
        '''
        export point cloud
        '''
        if with_normal:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=self.vertex_normals, size=size)
        else:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=None, size=size)
    
    def export_mesh(self, path: str):
        '''
        export mesh
        '''
        self._export_mesh(vertices=self.vertices, faces=self.faces, path=path)
    
    def export_skeleton(self, path: str):
        '''
        export spring
        '''
        self._export_skeleton(joints=self.joints, parents=self.parents, path=path)
    
    def export_fbx(
        self,
        path: str,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect: bool=True,
        extrude_from_parent: bool=True,
    ):
        '''
        export the whole model with skining
        '''
        self._export_fbx(
            path=path,
            vertices=self.vertices,
            joints=self.joints,
            skin=self.skin,
            parents=self.parents,
            names=self.names,
            faces=self.faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            try_connect=try_connect,
            extrude_from_parent=extrude_from_parent,
        )
    
    def export_animation(
        self,
        path: str,
        matrix_basis: ndarray,
        offset: ndarray,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect=True,
    ):
        self._export_animation(
            path=path,
            matrix_basis=matrix_basis,
            offset=offset,
            vertices=self.vertices,
            joints=self.joints,
            skin=self.skin,
            parents=self.parents,
            names=self.names,
            faces=self.faces,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            try_connect=try_connect,
        )
        
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            cls=self.cls,
            id=self.id,
            vertices=self.vertices,
            vertex_normals=self.vertex_normals,
            faces=self.faces,
            face_normals=self.face_normals,
            joints=self.joints,
            skin=self.skin,
            parents=self.parents,
            names=self.names,
            matrix_local=self.matrix_local
        )

@dataclass
class SampledAsset(Exporter):
    '''
    A simple sampled asset for model input.
    '''
    
    # cls of data
    cls: str
    
    # data id
    id: int
    
    # vertices of mesh, shape (N, 3), float32
    vertices: ndarray
    
    # normal of vertices of mesh, shape (N, 3), float32
    normals: ndarray
    
    # joints of bones, shape (J, 3), float32
    joints: Union[ndarray, None] = None
    
    # skinning of joints, shape (N, J), float32
    skin: Union[ndarray, None] = None
    
    # parents of joints, None represents no parent(a root joint)
    # make sure parent[k] < k
    parents: Union[List[Union[int, None]], None] = None
    
    # names of joints
    names: Union[List[str], None] = None
    
    @property
    def N(self):
        '''
        number of vertices
        '''
        return self.vertices.shape[0]
    
    @property
    def J(self):
        '''
        number of joints
        '''
        return self.joints.shape[0]
    
    def export_pc(self, path: str, with_normal: bool=True, size=0.01):
        '''
        export point cloud
        '''
        if with_normal:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=self.normals, size=size)
        else:
            self._export_pc(vertices=self.vertices, path=path, vertex_normals=None, size=size)
    
    def export_skeleton(self, path: str):
        '''
        export skeleton
        '''
        self._export_skeleton(joints=self.joints, parents=self.parents, path=path)
    
    def export_fbx(
        self,
        path: str,
        extrude_size: float=0.03,
        group_per_vertex: int=-1,
        add_root: bool=False,
        do_not_normalize: bool=False,
        try_connect: bool=True
    ):
        '''
        export the pc cloud with skining
        '''
        self._export_fbx(
            path=path,
            vertices=self.vertices,
            joints=self.joints,
            skin=self.skin,
            parents=self.parents,
            names=self.names,
            faces=None,
            extrude_size=extrude_size,
            group_per_vertex=group_per_vertex,
            add_root=add_root,
            do_not_normalize=do_not_normalize,
            try_connect=try_connect
        )