import torch.nn as nn

class VoxelEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * 8 * 8 * 8, latent_dim)  # adjust if your voxel size changes
        )

    def forward(self, x):  # x: [B, 1, D, H, W]
        return self.encoder(x)  # [B, latent_dim]