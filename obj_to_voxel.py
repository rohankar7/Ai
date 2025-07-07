import trimesh
import torch
import os
import numpy as np
from shapenetcore import get_random_models
from tqdm import tqdm
import config
from viz import viz_voxel, viz_mesh

target_res = config.voxel_res

def save_voxels():
    os.makedirs(config.voxel_dir, exist_ok=True)
    for path in tqdm(sorted(get_random_models()), desc=f"Progress"):
        if not os.path.isfile(f"{config.voxel_dir}/{"_".join(path.split("/"))}.pth"): create_voxels(path)

def create_voxels(path):
    try:
        mesh = trimesh.load(f"{config.directory}/{path}/{config.suffix_dir}", force="mesh")
        # Normalize: center & scale to fit inside unit cube
        mesh.apply_translation(-mesh.centroid)  # center it
        scale = 1.0 / np.max(mesh.extents)
        mesh.apply_scale(scale * 0.9)  # shrink a bit to avoid touching edges
        pitch = 1.0 / target_res # Voxelize to fixed resolution
        voxelized = mesh.voxelized(pitch=pitch)
        voxel_grid = voxelized.matrix.astype(np.float32) # Convert voxel matrix to dense tensor
        # viz_voxel(voxel_grid)
        grid = np.zeros((target_res, target_res, target_res), dtype=np.float32) # Pad or crop to fixed size
        min_shape = np.minimum(grid.shape, voxel_grid.shape)
        grid[:min_shape[0], :min_shape[1], :min_shape[2]] = voxel_grid[:min_shape[0], :min_shape[1], :min_shape[2]]
        voxel_tensor = torch.tensor(grid).unsqueeze(0)  # [1, D, H, W]
        torch.save(voxel_tensor, f"{config.voxel_dir}/{"_".join(path.split("/"))}.pt")
    except (IndexError, AttributeError, np._core._exceptions._ArrayMemoryError) as e:
        # print(path) # Missing two files from class: 03337140
        return

# save_voxels()