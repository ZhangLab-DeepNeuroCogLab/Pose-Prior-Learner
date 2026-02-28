import os

import h5py
import numpy as np
import torch
import torch.utils.data
from matplotlib import colors

DATA_DIR = ""
class TrainSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        data_file = 'cub_three.h5'
        with h5py.File(os.path.join(DATA_DIR, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])
            self.keypoints = torch.from_numpy(hf['train_kp'][...])  # [0, 1]
            self.visibility = torch.from_numpy(hf['train_vis'][...])  # 1 for visible and 0 for invisible

        self.transform = transform

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255), 'keypoints': self.keypoints[idx], 'visibility': self.visibility[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        data_file = 'cub_three.h5'
        with h5py.File(os.path.join(DATA_DIR, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['train_img'][...])
            self.keypoints = torch.from_numpy(hf['train_kp'][...])  # [0, 1]
            self.visibility = torch.from_numpy(hf['train_vis'][...])  # 1 for visible and 0 for invisible

        self.transform = transform

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255), 'keypoints': self.keypoints[idx], 'visibility': self.visibility[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]


class TestSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        data_file = 'cub_three.h5'
        with h5py.File(os.path.join(DATA_DIR, data_file), 'r') as hf:
            self.imgs = torch.from_numpy(hf['test003_img'][...])
            self.keypoints = torch.from_numpy(hf['test003_kp'][...])  # [0, 1]
            self.visibility = torch.from_numpy(hf['test003_vis'][...])  # 1 for visible and 0 for invisible

        self.transform = transform

    def __getitem__(self, idx):
        sample = {'img': self.transform(self.imgs[idx] / 255), 'keypoints': self.keypoints[idx], 'visibility': self.visibility[idx]}
        return sample

    def __len__(self):
        return self.imgs.shape[0]