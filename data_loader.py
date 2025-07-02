import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
shuffle_condition = False
import config
from sklearn.model_selection import train_test_split

class VoxelDataset(Dataset):
    def __init__(self, voxel_paths):
        self.voxel_paths = voxel_paths
    def __len__(self): return len(self.voxel_paths)
    def __getitem__(self, index):
        file_path = self.voxel_paths[index]
        voxel_data = torch.load(f"{config.voxel_dir}/{file_path}", weights_only=False)
        return voxel_data
def voxel_dataloader():
    voxel_paths = [os.path.join(config.voxel_dir, path) for path in os.listdir(config.voxel_dir)]
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    dataset = VoxelDataset(voxel_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=10, shuffle=False)
    return dataloader