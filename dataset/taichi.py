import os

import numpy as np
import pandas
import torch
from torch.utils.data import Dataset
import torchvision
from PIL import Image

DATA_DIR = ""
class TrainSet(Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform
        self.imgs = torchvision.datasets.ImageFolder(root=os.path.join(DATA_DIR, 'train'), transform=self.transform)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx][0]}
        return sample

    def __len__(self):
        return len(self.imgs)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.imgs = []
        self.poses = []

        with open(os.path.join(DATA_DIR, 'landmark', 'taichi_train_gt.pkl'), 'rb') as f:
            pose_file = pandas.read_pickle(f)

        for i in range(len(pose_file)):
            image_file = pose_file.file_name[i]
            img = Image.open(os.path.join(DATA_DIR, 'eval_images', 'taichi-256', 'train', image_file))
            img = img.resize((image_size, image_size), resample=Image.BILINEAR)
            self.imgs.append(np.asarray(img) / 255)
            self.poses.append(pose_file.value[i])  # [0, 255]

        self.transform = transform

        self.imgs = torch.tensor(np.array(self.imgs)).float().permute(0, 3, 1, 2)
        for i in range(len(self.imgs)):
            self.imgs[i] = self.transform(self.imgs[i])
        self.imgs = self.imgs.contiguous()
        self.poses = torch.tensor(self.poses).float()
        self.poses = torch.cat([self.poses[:, :, 1:2], self.poses[:, :, 0:1]], dim=2)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx], 'keypoints': self.poses[idx]}
        return sample

    def __len__(self):
        return len(self.imgs)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.imgs = []
        self.segs = []
        self.poses = []

        with open(os.path.join(DATA_DIR, 'landmark', 'taichi_test_gt.pkl'), 'rb') as f:
            pose_file = pandas.read_pickle(f)

        for i in range(len(pose_file)):
            image_file = pose_file.file_name[i]
            img = Image.open(os.path.join(DATA_DIR, 'eval_images', 'taichi-256', 'test', image_file))
            img = img.resize((image_size, image_size), resample=Image.BILINEAR)
            seg = Image.open(os.path.join(DATA_DIR, 'taichi-test-masks', image_file))
            seg = seg.resize((image_size, image_size), resample=Image.BILINEAR)
            self.imgs.append(np.asarray(img) / 255)
            self.segs.append(np.asarray(seg) / 255)
            self.poses.append(pose_file.value[i])  # [0, 255]

        self.transform = transform

        self.imgs = torch.tensor(np.array(self.imgs)).float().permute(0, 3, 1, 2)
        for i in range(len(self.imgs)):
            self.imgs[i] = self.transform(self.imgs[i])
        self.imgs = self.imgs.contiguous()
        self.segs = torch.tensor(self.segs).int()
        self.poses = torch.tensor(self.poses).float()
        self.poses = torch.cat([self.poses[:, :, 1:2], self.poses[:, :, 0:1]], dim=2)

    def __getitem__(self, idx):
        sample = {'img': self.imgs[idx], 'seg': self.segs[idx], 'keypoints': self.poses[idx]}
        return sample

    def __len__(self):
        return len(self.imgs)