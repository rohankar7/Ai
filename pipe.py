from voxel_to_latent import VoxelEncoder
from latent_to_triplane import TriplaneDecoder
from obj_to_voxel import get_voxels
import torch
import torch.nn.functional as F
from triplane_to_voxel import sample_feature_from_planes, FeatureDecoderMLP
import torch.optim as optim

encoder = VoxelEncoder(latent_dim=1024)
triplane_gen = TriplaneDecoder(latent_dim=1024, channels=32, res=64)
decoder_mlp = FeatureDecoderMLP(in_dim=32)
import torch.optim as optim

# Assume model includes: encoder, triplane_generator, mlp_decoder
params = list(encoder.parameters()) + \
         list(triplane_gen.parameters()) + \
         list(decoder_mlp.parameters())

optimizer = optim.Adam(params, lr=1e-4, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
# OR:
# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

def main():
    print('Main function: VAE')
    get_voxels()
    voxel_gt = torch.load("./voxel/02691156/10155655850468db78d106ce0a280f87.pth", weights_only=False)
    print(voxel_gt.size())
    # Encode to latent
    latent = encoder(voxel_gt)     # [1, 512]
    # Decode to triplanes
    triplanes = triplane_gen(latent)        # [1, 3, 32, 64, 64]
    # Sample 3D coordinates
    N = 32 ** 3
    coords = torch.rand(1, N, 3) * 2 - 1  # in [-1, 1]

    # Sample GT voxel values at coords
    gt_values = F.grid_sample(voxel_gt, coords.view(1, 1, N, 1, 3), align_corners=True).squeeze(-1)

    # Sample features from triplanes
    feats = sample_feature_from_planes(triplanes, coords)

    # Decode occupancy values
    pred = decoder_mlp(feats).squeeze(-1)

    # Loss
    loss = F.binary_cross_entropy(pred, gt_values.squeeze())

    # Backprop
    loss.backward()
    optimizer.step()
    print(triplanes.size())
    scheduler.step()

if __name__ == '__main__':
    main()