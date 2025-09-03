import jittor as jt
import numpy as np
import os
from jittor.dataset import Dataset

import os
from typing import List, Dict, Callable, Union

from .asset import Asset
from .sampler import Sampler

# def transform(asset: Asset):
#     """
#     Transform the asset data into [-1, 1]^3.
#     """
#     # Find min and max values for each dimension of points
#     min_vals = np.min(asset.vertices, axis=0)
#     max_vals = np.max(asset.vertices, axis=0)
    
#     # Calculate the center of the bounding box
#     center = (min_vals + max_vals) / 2
    
#     # Calculate the scale factor to normalize to [-1, 1]
#     # We take the maximum range across all dimensions to preserve aspect ratio
#     scale = np.max(max_vals - min_vals) / 2
    
#     # Normalize points to [-1, 1]^3
#     normalized_vertices = (asset.vertices - center) / scale
    
#     # Apply the same transformation to joints
#     if asset.joints is not None:
#         normalized_joints = (asset.joints - center) / scale
#     else:
#         normalized_joints = None
    
#     asset.vertices  = normalized_vertices
#     asset.joints    = normalized_joints
#     # remember to change matrix_local !
#     asset.matrix_local[:, :3, 3] = normalized_joints

def transform(asset: Asset):
    """
    Transform the asset data into [-1, 1]^3 using Jittor tensors.
    """
    # 转成 Jittor tensor（避免 numpy 和 jt 来回拷贝）
    vertices = jt.array(asset.vertices)
    
    # Find min and max values for each dimension of points
    min_vals = jt.min(vertices, dim=0)
    max_vals = jt.max(vertices, dim=0)
    
    # Calculate the center of the bounding box
    center = (min_vals + max_vals) / 2.0
    
    # Calculate the scale factor to normalize to [-1, 1]
    scale = jt.max(max_vals - min_vals) / 2.0
    
    # Normalize points to [-1, 1]^3
    normalized_vertices = (vertices - center) / scale
    
    # Apply the same transformation to joints
    if asset.joints is not None:
        joints = jt.array(asset.joints)
        normalized_joints = (joints - center) / scale
    else:
        normalized_joints = None
    
    # 更新 asset 数据（保持 numpy 格式以免后续出错）
    asset.vertices = normalized_vertices.numpy()
    asset.joints = normalized_joints.numpy() if normalized_joints is not None else None
    
    # 更新 matrix_local 的平移部分
    if asset.joints is not None:
        asset.matrix_local[:, :3, 3] = asset.joints

class RigDataset(Dataset):
    '''
    A simple dataset class.
    '''
    def __init__(
        self,
        data_root: str,
        paths: List[str],
        train: bool,
        batch_size: int,
        shuffle: bool,
        sampler: Sampler,
        transform: Union[Callable, None] = None,
        return_origin_vertices: bool = False,
        random_pose: bool = False,
    ):
        super().__init__()
        self.data_root  = data_root
        self.paths      = paths.copy()
        self.batch_size = batch_size
        self.train      = train
        self.shuffle    = shuffle
        self._sampler   = sampler # do not use `sampler` to avoid name conflict
        self.transform  = transform
        
        self.random_pose = random_pose
        
        self.return_origin_vertices = return_origin_vertices
        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.paths),
            shuffle=self.shuffle,
        )
    
    def __getitem__(self, index) -> Dict:
        """
        Get a sample from the dataset
        
        Args:
            index (int): Index of the sample
            
        Returns:
            data (Dict): Dictionary containing the following keys:
                - vertices: jt.Var, (B, N, 3) point cloud data
                - normals: jt.Var, (B, N, 3) point cloud normals
                - joints: jt.Var, (B, J, 3) joint positions
                - skin: jt.Var, (B, J, J) skinning weights
        """
        # if np.random.rand() < 0.5:
        #     data_root = "data_aug"
        # else:
        data_root = self.data_root

        path = self.paths[index]
        asset = Asset.load(os.path.join(self.data_root, path))
        # print(f"load from {os.path.join(self.data_root, path)}")
        # asset.apply_matrix_basis_jt(asset.get_random_matrix_basis_jt(30.0))
        # data_root = "data_aug"
        # asset.save(os.path.join(data_root, path))  # save the asset after applying random matrix basis
        # print(f"save to {os.path.join(data_root, path)}")
        
        # ===========================
        if self.random_pose:
            if self.train:
                p_motion = 0.8
                p_rotate = 0.2
                r = np.random.rand()
                # print(1)
                if r < p_motion:
                    # 动捕增强
                    idx = np.random.randint(0, 9)
                    motion_data = np.load(f"code/data/track/{idx}.npz")['matrix_basis']
                    frame_idx = np.random.randint(0, motion_data.shape[0])
                    
                    asset.apply_matrix_basis_jt(motion_data[frame_idx])
                elif r < p_motion + p_rotate:
                    # 随机旋转增强
                    asset.apply_matrix_basis_jt(asset.get_random_matrix_basis_jt(30.0))
                else:
                    # 保持 A/T pose，不做增强
                    raise RuntimeError("no")
                    pass
            else :
                idx = np.random.randint(0, 9)
                motion_data = np.load(f"code/data/track/{idx}.npz")['matrix_basis']
                frame_idx = np.random.randint(0, motion_data.shape[0])
                    
                asset.apply_matrix_basis_jt(motion_data[frame_idx]) 
                
        if self.transform is not None:
            self.transform(asset)
        origin_vertices = jt.array(asset.vertices.copy()).float32()
        
        sampled_asset = asset.sample(sampler=self._sampler)

        vertices    = jt.array(sampled_asset.vertices).float32()
        normals     = jt.array(sampled_asset.normals).float32()

        if sampled_asset.joints is not None:
            joints      = jt.array(sampled_asset.joints).float32()
        else:
            joints      = None

        if sampled_asset.skin is not None:
            skin        = jt.array(sampled_asset.skin).float32()
        else:
            skin        = None

        res = {
            'vertices': vertices,
            'normals': normals,
            'cls': asset.cls,
            'id': asset.id,
        }
        if joints is not None:
            res['joints'] = joints
        if skin is not None:
            res['skin'] = skin
        if self.return_origin_vertices:
            res['origin_vertices'] = origin_vertices
        return res
    
    def collate_batch(self, batch):
        if self.return_origin_vertices:
            max_N = 0
            for b in batch:
                max_N = max(max_N, b['origin_vertices'].shape[0])
            for b in batch:
                N = b['origin_vertices'].shape[0]
                b['origin_vertices'] = np.pad(b['origin_vertices'], ((0, max_N-N), (0, 0)), 'constant', constant_values=0.)
                b['N'] = N
        return super().collate_batch(batch)


# Example usage of the dataset
def get_dataloader(
    data_root: str,
    data_list: str,
    train: bool,
    batch_size: int,
    shuffle: bool,
    sampler: Sampler,
    transform: Union[Callable, None] = None,
    return_origin_vertices: bool = False,
    random_pose: bool = False,
):
    """
    Create a dataloader for point cloud data
    
    Args:
        data_root (str): Root directory for the data files
        data_list (str): Path to the file containing list of data files
        train (bool): Whether the dataset is for training
        batch_size (int): Batch size
        shuffle (bool): Whether to shuffle the dataset
        sampler (Sampler): Sampler to use for point cloud sampling
        transform (callable, optional): Optional post-transform to be applied on a sample
        return_origin_vertices (bool): Whether to return original vertices
        
    Returns:
        dataset (RigDataset): The dataset
    """
    with open(data_list, 'r') as f:
        paths = [line.strip() for line in f.readlines()]
    dataset = RigDataset(
        data_root=data_root,
        paths=paths,
        train=train,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        transform=transform,
        return_origin_vertices=return_origin_vertices,
        random_pose=random_pose,
    )
    
    return dataset