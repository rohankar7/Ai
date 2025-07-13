import torch.nn as nn

class VoxelEncoder(nn.Module):
    def __init__(self, latent_dim=1024):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=4, stride=2, padding=1), # [B, 32, 32, 32, 32]
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 64, kernel_size=4, stride=2, padding=1), # [B, 64, 16, 16, 16]
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1), # [B, 128, 8, 8, 8] -> 65,536
            nn.BatchNorm3d(128),
            nn.ReLU(),
            nn.Conv3d(128, 256, kernel_size=4, stride=2, padding=1), # [B, 256, 4, 4, 4] -> 16,384
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(256, 512, kernel_size=4, stride=2, padding=1), # [B, 512, 2, 2, 2] -> 4,096
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(512, 1024, kernel_size=4, stride=2, padding=1), # [B, 1024, 1, 1, 1] -> 1,024
            nn.BatchNorm3d(1024),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, latent_dim)  # adjust if your voxel size changes
        )
    def forward(self, x):  # x: [B, 1, D, H, W]
        return self.encoder(x)  # [B, latent_dim]