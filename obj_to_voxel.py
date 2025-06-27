import trimesh
import torch
import os

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
                    voxel = mesh.voxelized(pitch=0.02)
                    voxel_tensor = torch.tensor(voxel.matrix, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
                    print(voxel_tensor.size())
                    torch.save(voxel_tensor, f"{save_dir}/{shapenet_class}/{category}.pth")
get_voxels()