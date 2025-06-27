from voxel_to_latent import VoxelEncoder
from latent_to_triplane import TriplaneDecoder
from obj_to_voxel import get_voxels
import torch

encoder = VoxelEncoder(latent_dim=512)
decoder = TriplaneDecoder(latent_dim=512, channels=32, res=64)

def main():
    print('Main function: VAE')
    get_voxels()
    t = torch.load("./voxel/02691156/10155655850468db78d106ce0a280f87.pth", weights_only=False)
    print(t.size())
    # latent = encoder(t)     # [1, 512]
    # triplanes = decoder(latent)        # [1, 3, 32, 64, 64]
    # print(triplanes.size())

if __name__ == '__main__':
    main()