import torch
import numpy as np
import pickle
from tqdm import tqdm
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class BlendCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.1, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.targets = np.array(self.targets)
        b, w, h, c = self.data.shape

        with open('trigger/hello_kitty_pattern.npy', 'rb') as f:
            pattern = np.load(f)

        # Add Backdoor Trigers
        if train == False:
            idx = np.array(self.targets) != target_label
            self.data = self.data[idx]
            self.targets = np.array(self.targets)[idx]
            poison_rate = 1.0

        a = 0.2
        w, h, c = self.data.shape[1:]
        targets = list(range(0, len(self)))
        s = len(self)
        self.poison_idx = np.random.permutation(s)[0: int(s * poison_rate)]
        pattern = np.tile(pattern, (len(self.poison_idx), 1, 1, 1))
        print(pattern.shape)
        self.data[self.poison_idx] = (1-a)*self.data[self.poison_idx] + a * pattern
        self.targets[self.poison_idx] = target_label
        self.data = np.clip(self.data, 0, 255)
