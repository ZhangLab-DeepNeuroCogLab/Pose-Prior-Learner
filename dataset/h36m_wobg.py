import os

import numpy as np
import scipy.io
import torch
import torch.utils.data
from PIL import Image

DATA_DIR = ""

class TrainSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

        self.samples = []

        for subject_index in [1, 5, 6, 7, 8, 9]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(DATA_DIR, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        mask = Image.open(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'BackgroudMask',
                                       folder_names, '{}.png'.format(frame_index)))
        return {'img': self.transform(img) * self.to_tensor(mask)}

    def __len__(self):
        return len(self.samples)


class TrainRegSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

        self.samples = []

        for subject_index in [1, 5, 6, 7, 8, 9]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(DATA_DIR, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        mask = Image.open(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'BackgroudMask',
                                       folder_names, '{}.png'.format(frame_index)))
        keypoints = scipy.io.loadmat(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'Landmarks',
                                      folder_names, '{}.mat'.format(frame_index)))['keypoints_2d'].astype(np.float32)

        return {'img': self.transform(img) * self.to_tensor(mask), 'keypoints': keypoints}

    def __len__(self):
        return len(self.samples)


class TestSet(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        super().__init__()
        self.transform = transform

        self.samples = []

        for subject_index in [11]:
            for action in ['Directions', 'Discussion', 'Posing', 'Waiting', 'Greeting', 'Walking']:
                for folder_names in os.listdir(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'WithBackground')):
                    if folder_names.startswith(action):
                        for frame_index in os.listdir(os.path.join(DATA_DIR, 'S{}'.format(subject_index),
                                                                   'WithBackground', folder_names)):
                            self.samples.append((subject_index, folder_names, frame_index.split('.')[0]))

    def __getitem__(self, idx):
        subject_index, folder_names, frame_index = self.samples[idx]
        img = Image.open(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'WithBackground',
                                      folder_names, '{}.jpg'.format(frame_index)))
        mask = Image.open(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'BackgroudMask',
                                       folder_names, '{}.png'.format(frame_index)))
        keypoints = scipy.io.loadmat(os.path.join(DATA_DIR, 'S{}'.format(subject_index), 'Landmarks',
                                                  folder_names, '{}.mat'.format(frame_index)))['keypoints_2d'].astype(np.float32)

        return {'img': self.transform(img) * self.to_tensor(mask), 'keypoints': keypoints}

    def __len__(self):
        return len(self.samples)