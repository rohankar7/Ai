import torch
import torch.nn as nn
import torch.nn.functional as F
def sample_feature_from_planes(triplanes, coords):
    """
    triplanes: [B, 3, C, H, W]
    coords: [B, N, 3] in [-1, 1]
    returns: [B, N, C] combined features
    """
    B, _, C, H, W = triplanes.shape
    coords = coords.view(B, -1, 3)

    xy_feat = F.grid_sample(triplanes[:, 0], coords[..., [0, 1]].unsqueeze(2), align_corners=True).squeeze(-1)
    yz_feat = F.grid_sample(triplanes[:, 1], coords[..., [1, 2]].unsqueeze(2), align_corners=True).squeeze(-1)
    xz_feat = F.grid_sample(triplanes[:, 2], coords[..., [0, 2]].unsqueeze(2), align_corners=True).squeeze(-1)

    return (xy_feat + yz_feat + xz_feat) / 3  # simple average fusion

class FeatureDecoderMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, features):
        return self.mlp(features)  # [B, N, 1]
