import torch
import torch.nn as nn
import torch.nn.functional as F

class TriplaneDecoder(nn.Module):
    def __init__(self, latent_dim=1024, out_channels=32, out_size=128):
        super(TriplaneDecoder, self).__init__()
        self.init_h = 8
        self.init_w = 8
        self.out_channels = out_channels

        # Project latent to a flattened tensor representing 3 planes
        self.fc = nn.Linear(latent_dim, 3 * out_channels * self.init_h * self.init_w)

        # Define a shared decoder for all three planes
        def shared_decoder():
            return nn.Sequential(
                nn.ConvTranspose2d(out_channels, 128, kernel_size=4, stride=2, padding=1),  # 8 → 16
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 16 → 32
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),  # 32 → 64
                nn.ReLU(),
                nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1),  # 64 → 128
            )

        # Either use 3 different decoders OR 1 shared decoder
        self.decoder = shared_decoder()  # shared decoder
        # self.decoder = nn.ModuleList([make_decoder() for _ in range(3)]) # Different decoders

    def forward(self, z):  # z: [B, latent_dim]
        B = z.shape[0]
        x = self.fc(z)  # [B, 3*C*8*8]
        x = x.view(B, 3, self.out_channels, self.init_h, self.init_w)  # [B, 3, C, 8, 8]

        # Decode each plane separately using the shared decoder
        planes = []
        for i in range(3):
            feat = x[:, i]  # [B, C, 8, 8]
            decoded = self.decoder(feat)  # [B, C, 128, 128]
            planes.append(decoded)

        out = torch.stack(planes, dim=1)  # [B, 3, C, 128, 128]
        return out