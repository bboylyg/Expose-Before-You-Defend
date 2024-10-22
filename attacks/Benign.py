import torch
import numpy as np
import os
from torch.utils.data.dataset import Dataset


class BenignCIFAR10(Dataset):
    def __init__(self, root, transform=None, **kwargs):
        super().__init__()
        data_path = os.path.join(root, 'cifar10.1_v4_data.npy')
        targets_path = os.path.join(root, 'cifar10.1_v4_labels.npy')

        self.data = np.load(data_path)
        self.targets = np.load(targets_path)

        self.data = torch.from_numpy(self.data).type('torch.FloatTensor')
        self.targets = torch.from_numpy(self.targets).type('torch.LongTensor')
        self.data = self.data.permute(0, 3, 1, 2) / 255.0
        self.transform = transform
        print(self.data.shape, self.targets.shape)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.targets[index]
        if self.transform is not None:
            data = self.transform(data)
        return data, target

    def __len__(self):
        return len(self.targets)
