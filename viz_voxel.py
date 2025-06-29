voxel = normalize_and_voxelize("your.obj", target_res=64)
import matplotlib.pyplot as plt

plt.imshow(voxel[0,0,32])  # visualize center slice
plt.title("Middle slice of voxel grid")
plt.show()