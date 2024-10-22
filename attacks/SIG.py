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


class SIGCIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=0.3, target_label=0, **kwargs):
        super().__init__(root=root, train=train, download=download,
                         transform=transform, target_transform=target_transform)
        self.targets = np.array(self.targets)
        # https://github.com/bboylyg/NAD
        # build backdoor pattern
        alpha = 0.2
        b, w, h, c = self.data.shape
        pattern = np.load('trigger/signal_cifar10_mask.npy').reshape((w, h, 1))
        print(pattern.shape)

        # Add triger
        class_idx = [np.where(np.array(self.targets) == i)[0] for i in range(10)]
        if poison_rate > 0.0 and train:
            size = int(len(class_idx[target_label])*poison_rate)
            self.poison_idx = np.random.choice(class_idx[target_label],
                                             size=size, replace=False)

            pattern = np.array([pattern] * size)
            print(pattern.shape)
            self.data[self.poison_idx] = (1 - alpha) * (self.data[self.poison_idx]) +\
            alpha * pattern
            print("Injecting SIG pattern to %d Samples, Poison Rate (%.2f)" %
                  (size, poison_rate))
        elif poison_rate > 0.0 and not train:
            # Add to Test set for Backdoor Test
            for c, idx in enumerate(class_idx):
                if c == target_label:
                    continue
                p = np.array([pattern] * len(idx))
                print(p.shape, pattern.shape)
                self.data[idx] = alpha * (self.data[idx]) + (1 - alpha) * p
                self.targets[idx] = target_label
            self.data = np.delete(self.data, class_idx[target_label], 0)
            self.targets = np.delete(self.targets, class_idx[target_label], 0)
            print('Backdoor test size: %d' % len(self.data))
