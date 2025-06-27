import torch.nn as nn

class TriplaneDecoder(nn.Module):
    def __init__(self, latent_dim=512, channels=32, res=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * channels * res * res)
        )
        self.channels = channels
        self.res = res

    def forward(self, z):  # z: [B, latent_dim]
        B = z.shape[0]
        out = self.fc(z)
        return out.view(B, 3, self.channels, self.res, self.res)  # [B, 3, C, H, W]