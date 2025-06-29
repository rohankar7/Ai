from skimage import measure
import trimesh
import matplotlib.pyplot as plt
import numpy as np

def viz_mesh(voxel_pred):
    # Assume voxel_pred shape = [64, 64, 64]
    verts, faces, normals, values = measure.marching_cubes(voxel_pred.numpy(), level=0.5)
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    mesh.show()

def viz_voxel(voxel_data, threshold=0):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    print(np.ndim(voxel_data))
    if np.ndim(voxel_data) == 4:
        if voxel_data.shape[-1] == 4: voxel_data = voxel_data[..., :3]
        if voxel_data.max() > 1: voxel_data  = voxel_data / 255.0 # Normalizing the voxel colors for visualization
        mask = np.any(voxel_data > threshold, axis=-1) # Masking for non-zero voxels with color intensity > 0
        x, y, z = np.indices(voxel_data.shape[:-1])  # Getting the grid coordinates
        ax.scatter(x[mask], y[mask], z[mask], c=voxel_data[mask].reshape(-1, 3), marker='o', s=20)
        ax.set_box_aspect([1, 1, 1])  # Aspect ratio is 1:1:1
    else: ax.voxels(voxel_data, edgecolor='k')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()