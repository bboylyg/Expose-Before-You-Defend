import os
import numpy as np
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, MNIST, ImageNet
from torchvision.datasets.folder import ImageFolder
from taibackdoor.attacks import *

transform_options = {
    "None": {
        "train_transform": None,
        "test_transform": None},
    "NoAug": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CMNIST": {
            "train_transform": [
                transforms.ToPILImage(),
                transforms.RandomCrop(28, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ],
            "test_transform": None},

    "CIFAR10": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
         "train_transform": [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(15),
                             transforms.ToTensor()],
         "test_transform": [transforms.ToTensor()]},
    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]},
    "ISBBAImageNet": {
            # https://github.com/yuezunli/ISSBA/blob/main/train.py
            "train_transform": [
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor()
            ],
            "test_transform": [
                transforms.ToTensor()
            ]},
}

dataset_options = {
    "CIFAR10": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
    "CIFAR10NoAug": lambda path, transform, is_test, kwargs:
        CIFAR10(root=path, train=not is_test, download=True,
                transform=transform),
    "BenignCIFAR10": lambda path, transform, is_test, kwargs:
        BenignCIFAR10(root=path, train=not is_test, download=True,
                      transform=transform, **kwargs),
    "BadNetCIFAR10": lambda path, transform, is_test, kwargs:
        BadNetCIFAR10(root=path, train=not is_test, download=True,
                      transform=transform, **kwargs),
    "SIGCIFAR10": lambda path, transform, is_test, kwargs:
        SIGCIFAR10(root=path, train=not is_test, download=True,
                   transform=transform, **kwargs),
    "TrojanCIFAR10": lambda path, transform, is_test, kwargs:
        TrojanCIFAR10(root=path, train=not is_test, download=True,
                      transform=transform, **kwargs),
    "BlendCIFAR10": lambda path, transform, is_test, kwargs:
        BlendCIFAR10(root=path, train=not is_test, download=True,
                     transform=transform, **kwargs),
    "CLCIFAR10": lambda path, transform, is_test, kwargs:
        CLCIFAR10(root=path, train=not is_test, download=True,
                  transform=transform, **kwargs),
    # "CIFAR_10_1": lambda path, transform, is_test, kwargs:
    #     CIFAR_10_1(root=path, transform=transform, **kwargs),
    "DynamicCIFAR10": lambda path, transform, is_test, kwargs:
        DynamicCIFAR10(root=path, train=not is_test, download=True,
                       transform=transform, **kwargs),
    "CIFAR100": lambda path, transform, is_test, kwargs:
        CIFAR100(root=path, train=not is_test, download=True,
                 transform=transform),
    "SVHN": lambda path, transform, is_test, kwargs:
        SVHN(root=path, split='test' if is_test else 'train', download=True,
             transform=transform),
    "MNIST": lambda path, transform, is_test, kwargs:
        MNIST(root=path, train=not is_test, download=True,
              transform=transform),
    "ImageNet": lambda path, transform, is_test, kwargs:
        ImageNet(root=path, split='val' if is_test else 'train',
                 transform=transform),
    "ImageFolder": lambda path, transform, is_test, kwargs:
        ImageFolder(root=os.path.join(path, 'train') if not is_test else
                    os.path.join(path, 'val'),
                    transform=transform),
    # "CMNIST": lambda path, transform, is_test, kwargs:
    #     CMNIST(root=path, transform=transform, train=not is_test, **kwargs),
    # "CMNISTSIG": lambda path, transform, is_test, kwargs:
    #     CMNISTSIG(root=path, transform=transform, train=not is_test, **kwargs),
    # "IBAImageNet": lambda path, transform, is_test, kwargs:
    #     IBAImageNet(root=os.path.join(path, 'train') if not is_test else
    #                   os.path.join(path, 'val'),
    #                   mode='val' if is_test else 'train',
    #                   transform=transform,  **kwargs),
    # "ISBBAImageNetClean": lambda path, transform, is_test, kwargs:
    #     ISBBAImageNetClean(root=os.path.join(path, 'train') if not is_test else
    #                        os.path.join(path, 'val'),
    #                        mode='val' if is_test else 'train',
    #                        transform=transform,  **kwargs),
}


def get_classidx(dataset_type, dataset):
    if 'CIFAR100' in dataset_type:
        return [
            np.where(np.array(dataset.targets) == i)[0] for i in range(100)
        ]
    elif 'CIFAR10' in dataset_type:
        return [np.where(np.array(dataset.targets) == i)[0] for i in range(10)]
    elif 'SVHN' in dataset_type:
        return [np.where(np.array(dataset.labels) == i)[0] for i in range(10)]
    else:
        error_msg = 'dataset_type %s not supported' % dataset_type
        raise(error_msg)
