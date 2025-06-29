import trimesh
import torch
import os
import numpy as np

target_res = 64
def get_voxels() -> torch.Tensor:
    directory = "C:/ShapeNetCore"
    save_dir = "./voxel"
    os.makedirs(save_dir, exist_ok=True)
    for shapenet_class in os.listdir(directory)[:1]:
        os.makedirs(f"{save_dir}/{shapenet_class}", exist_ok=True)
        for category in os.listdir(f"{directory}/{shapenet_class}")[:10]:
            for model in os.listdir(f"{directory}/{shapenet_class}/{category}/models"):
                if model.endswith(".obj"):
                    mesh = trimesh.load(f"{directory}/{shapenet_class}/{category}/models/{model}", force="mesh")
                    # Normalize: center & scale to fit inside unit cube
                    mesh.apply_translation(-mesh.centroid)  # center it
                    scale = 1.0 / np.max(mesh.extents)
                    mesh.apply_scale(scale * 0.9)  # shrink a bit to avoid touching edges
                    # Voxelize to fixed resolution
                    pitch = 1.0 / target_res
                    voxelized = mesh.voxelized(pitch=pitch)
                    # Convert voxel matrix to dense tensor
                    voxel_grid = voxelized.matrix.astype(np.float32)
                    # Pad or crop to fixed size
                    grid = np.zeros((target_res, target_res, target_res), dtype=np.float32)
                    min_shape = np.minimum(grid.shape, voxel_grid.shape)
                    grid[:min_shape[0], :min_shape[1], :min_shape[2]] = voxel_grid[:min_shape[0], :min_shape[1], :min_shape[2]]
                    voxel_tensor = torch.tensor(grid).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                    print(voxel_tensor.size())
                    torch.save(voxel_tensor, f"{save_dir}/{shapenet_class}/{category}.pth")
get_voxels()