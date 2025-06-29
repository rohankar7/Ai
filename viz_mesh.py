from skimage import measure
import trimesh

# Assume voxel_pred shape = [64, 64, 64]
verts, faces, normals, values = measure.marching_cubes(voxel_pred.numpy(), level=0.5)
mesh = trimesh.Trimesh(vertices=verts, faces=faces)
mesh.show()