from __future__ import absolute_import
from __future__ import print_function

import argparse
import os
import time
from copy import deepcopy

import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import sys
import os

print(os.getcwd())
sys.path.append("..")
print(os.getcwd())
from Multi_Trigger_Backdoor_Attacks.model_for_cifar.model_select import select_model

# from Multi_Trigger_Backdoor_Attacks.datasets import mixed_backdoor_cifar
# from Multi_Trigger_Backdoor_Attacks.datasets.mixed_backdoor_cifar import add_Mtrigger_cifar, split_dataset
sys.path.append("Backdoor_detection")


#
# class DatasetTF(Dataset):
#     def __init__(self, full_dataset=None, transform=None):
#         self.dataset = full_dataset
#         self.transform = transform
#         self.dataLen = len(self.dataset)
#
#     def __getitem__(self, index):
#         image = self.dataset[index][0]
#         label = self.dataset[index][1]
#
#         if self.transform:
#             image = self.transform(image)
#         # print(type(image), image.shape)
#         return image, label
#
#     def __len__(self):
#         return self.dataLen

def split_dataset(dataset, frac=0.1, perm=None):
    """
    :param dataset: The whole dataset which will be split.
    """
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_split = int(frac * len(dataset))

    # generate the training set
    train_set = deepcopy(dataset)
    train_set.data = train_set.data[perm[nb_split:]]
    train_set.targets = np.array(train_set.targets)[perm[nb_split:]].tolist()

    # generate the test set
    split_set = deepcopy(dataset)
    split_set.data = split_set.data[perm[:nb_split]]
    split_set.targets = np.array(split_set.targets)[perm[:nb_split]].tolist()

    print('total data size: %d images, split test size: %d images, split ratio: %f' % (
        len(train_set.targets), len(split_set.targets), frac))

    return train_set, split_set


def data():
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10),
    ])

    clean_train = CIFAR10(root="./data", train=True, download=True, transform=transform_train)  # ,collate_fn=coffate
    clean_test = CIFAR10(root="./data", train=False, download=True, transform=transform_test)

    clean_train, clean_val = split_dataset(clean_train, frac=0.01)

    return clean_train, clean_val, clean_test


# def data():
#     MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
#     STD_CIFAR10 = (0.2023, 0.1994, 0.2010)
#
#     tf_train = torchvision.transforms.Compose([
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(),
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
#     ])
#
#     tf_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
#     ])
#
#     clean_train = CIFAR10(root='./data', train=True, download=True, transform=None)
#     clean_test = CIFAR10(root='./data', train=False, download=True, transform=None)
#
#
#     if args.attack_type == 'single_trigger':
#         # only one backdoor trigger injected
#
#
#         poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types,
#                                           poison_rate=args.poison_rate,
#                                           mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
#         poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
#
#         # split a small test subset
#         _, split_set = split_dataset(clean_test, frac=0.1)
#         poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=args.trigger_types, poison_rate=1.0,
#                                          mode='test', poison_target=args.poison_target, attack_type=args.attack_type)
#         poison_test_tf = DatasetTF(full_dataset=poison_test, transform=tf_test)
#         clean_test_tf = DatasetTF(full_dataset=split_set, transform=tf_test)
#
#         poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=8)
#         clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=8)
#         poison_test_loader = DataLoader(poison_test_tf, batch_size=args.batch_size, num_workers=8)
#
#
#     elif args.attack_type == 'multi_triggers_all2all':
#         # 10 backdoor triggers injected
#
#
#         poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types,
#                                           poison_rate=args.poison_rate,
#                                           mode='train', poison_target=args.poison_target,
#                                           attack_type=args.attack_type)
#         poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
#         # split a small test subset
#         _, split_set_clean = split_dataset(clean_test, frac=0.5)
#         _, split_set = split_dataset(clean_test, frac=0.05)
#         # get each of poison data
#         onePixelTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['onePixelTrigger'],
#                                                          poison_rate=1.0,
#                                                          mode='test', poison_target=args.poison_target,
#                                                          attack_type=args.attack_type)
#         gridTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['gridTrigger'],
#                                                      poison_rate=1.0,
#                                                      mode='test', poison_target=args.poison_target,
#                                                      attack_type=args.attack_type)
#         wanetTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['wanetTrigger'],
#                                                       poison_rate=1.0,
#                                                       mode='test', poison_target=args.poison_target,
#                                                       attack_type=args.attack_type)
#         trojanTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['trojanTrigger'],
#                                                        poison_rate=1.0,
#                                                        mode='test', poison_target=args.poison_target,
#                                                        attack_type=args.attack_type)
#         blendTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['blendTrigger'],
#                                                       poison_rate=1.0,
#                                                       mode='test', poison_target=args.poison_target,
#                                                       attack_type=args.attack_type)
#         signalTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['signalTrigger'],
#                                                        poison_rate=1.0,
#                                                        mode='test', poison_target=args.poison_target,
#                                                        attack_type=args.attack_type)
#         CLTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['CLTrigger'], poison_rate=1.0,
#                                                    mode='test', poison_target=args.poison_target,
#                                                    attack_type=args.attack_type)
#         smoothTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['smoothTrigger'],
#                                                        poison_rate=1.0,
#                                                        mode='test', poison_target=args.poison_target,
#                                                        attack_type=args.attack_type)
#         dynamicTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['dynamicTrigger'],
#                                                         poison_rate=1.0,
#                                                         mode='test', poison_target=args.poison_target,
#                                                         attack_type=args.attack_type)
#         nashTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['nashTrigger'],
#                                                      poison_rate=1.0,
#                                                      mode='test', poison_target=args.poison_target,
#                                                      attack_type=args.attack_type)
#         # add data transforms
#         clean_test_tf = DatasetTF(full_dataset=split_set_clean, transform=tf_test)
#         one_poison_test_tf = DatasetTF(full_dataset=onePixelTrigger_poison_test, transform=tf_test)
#         grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
#         wa_poison_test_tf = DatasetTF(full_dataset=wanetTrigger_poison_test, transform=tf_test)
#         tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
#         ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
#         sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
#         cl_test_tf = DatasetTF(full_dataset=CLTrigger_poison_test, transform=tf_test)
#         sm_test_tf = DatasetTF(full_dataset=smoothTrigger_poison_test, transform=tf_test)
#         dy_test_tf = DatasetTF(full_dataset=dynamicTrigger_poison_test, transform=tf_test)
#         nash_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)
#
#         # dataloader
#         poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=0)
#         clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=0)
#         one_poison_test_loader = DataLoader(one_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         wa_poison_test_loader = DataLoader(wa_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         cl_poison_test_loader = DataLoader(cl_test_tf, batch_size=args.batch_size, num_workers=0)
#         sm_poison_test_loader = DataLoader(sm_test_tf, batch_size=args.batch_size, num_workers=0)
#         dy_poison_test_loader = DataLoader(dy_test_tf, batch_size=args.batch_size, num_workers=0)
#         nash_poison_test_loader = DataLoader(nash_test_tf, batch_size=args.batch_size, num_workers=0)
#
#         poison_data_loaders_dct = {'onePixelTrigger': one_poison_test_loader,
#                                    'gridTrigger': grid_poison_test_loader,
#                                    'wanetTrigger': wa_poison_test_loader,
#                                    'trojanTrigger': tro_poison_test_loader,
#                                    'blendTrigger': ble_poison_test_loader,
#                                    'signalTrigger': sig_poison_test_loader,
#                                    'CLTrigger': cl_poison_test_loader,
#                                    'smoothTrigger': sm_poison_test_loader,
#                                    'dynamicTrigger': dy_poison_test_loader,
#                                    'nashTrigger': nash_poison_test_loader}
#
#     elif args.attack_type == 'multi_triggers_all2one':
#         # 10 backdoor triggers injected
#         logger.info('Trigger types: {}'.format(args.trigger_types))
#
#         poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types,
#                                           poison_rate=args.poison_rate,
#                                           mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
#         poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
#         # split a small test subset
#         _, split_set_clean = split_dataset(clean_test, frac=0.5)
#         _, split_set = split_dataset(clean_test, frac=0.05)
#         # get each of poison data
#         onePixelTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['onePixelTrigger'],
#                                                          poison_rate=1.0,
#                                                          mode='test', poison_target=args.poison_target,
#                                                          attack_type=args.attack_type)
#         gridTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['gridTrigger'], poison_rate=1.0,
#                                                      mode='test', poison_target=args.poison_target,
#                                                      attack_type=args.attack_type)
#         wanetTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['wanetTrigger'], poison_rate=1.0,
#                                                       mode='test', poison_target=args.poison_target,
#                                                       attack_type=args.attack_type)
#         trojanTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['trojanTrigger'], poison_rate=1.0,
#                                                        mode='test', poison_target=args.poison_target,
#                                                        attack_type=args.attack_type)
#         blendTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['blendTrigger'], poison_rate=1.0,
#                                                       mode='test', poison_target=args.poison_target,
#                                                       attack_type=args.attack_type)
#         signalTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['signalTrigger'], poison_rate=1.0,
#                                                        mode='test', poison_target=args.poison_target,
#                                                        attack_type=args.attack_type)
#         CLTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['CLTrigger'], poison_rate=1.0,
#                                                    mode='test', poison_target=args.poison_target,
#                                                    attack_type=args.attack_type)
#         smoothTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['smoothTrigger'], poison_rate=1.0,
#                                                        mode='test', poison_target=args.poison_target,
#                                                        attack_type=args.attack_type)
#         dynamicTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['dynamicTrigger'],
#                                                         poison_rate=1.0,
#                                                         mode='test', poison_target=args.poison_target,
#                                                         attack_type=args.attack_type)
#         nashTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['nashTrigger'], poison_rate=1.0,
#                                                      mode='test', poison_target=args.poison_target,
#                                                      attack_type=args.attack_type)
#         # add data transforms
#         clean_test_tf = DatasetTF(full_dataset=split_set_clean, transform=tf_test)
#         one_poison_test_tf = DatasetTF(full_dataset=onePixelTrigger_poison_test, transform=tf_test)
#         grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
#         wa_poison_test_tf = DatasetTF(full_dataset=wanetTrigger_poison_test, transform=tf_test)
#         tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
#         ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
#         sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
#         cl_test_tf = DatasetTF(full_dataset=CLTrigger_poison_test, transform=tf_test)
#         sm_test_tf = DatasetTF(full_dataset=smoothTrigger_poison_test, transform=tf_test)
#         dy_test_tf = DatasetTF(full_dataset=dynamicTrigger_poison_test, transform=tf_test)
#         nash_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)
#
#         # dataloader
#         poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=8)
#         clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=8)
#         one_poison_test_loader = DataLoader(one_poison_test_tf, batch_size=args.batch_size, num_workers=8)
#         grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=8)
#         wa_poison_test_loader = DataLoader(wa_poison_test_tf, batch_size=args.batch_size, num_workers=8)
#         tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=8)
#         ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=8)
#         sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=8)
#         cl_poison_test_loader = DataLoader(cl_test_tf, batch_size=args.batch_size, num_workers=8)
#         sm_poison_test_loader = DataLoader(sm_test_tf, batch_size=args.batch_size, num_workers=8)
#         dy_poison_test_loader = DataLoader(dy_test_tf, batch_size=args.batch_size, num_workers=8)
#         nash_poison_test_loader = DataLoader(nash_test_tf, batch_size=args.batch_size, num_workers=8)
#
#         poison_data_loaders_dct = {'onePixelTrigger': one_poison_test_loader,
#                                    'gridTrigger': grid_poison_test_loader,
#                                    'wanetTrigger': wa_poison_test_loader,
#                                    'trojanTrigger': tro_poison_test_loader,
#                                    'blendTrigger': ble_poison_test_loader,
#                                    'signalTrigger': sig_poison_test_loader,
#                                    'CLTrigger': cl_poison_test_loader,
#                                    'smoothTrigger': sm_poison_test_loader,
#                                    'dynamicTrigger': dy_poison_test_loader,
#                                    'nashTrigger': nash_poison_test_loader}
#
#     elif args.attack_type == 'multi_triggers_all2rand':
#         # generate random label list
#         poison_target_list = np.arange(10)
#         np.random.shuffle(poison_target_list)
#
#         # 10 backdoor triggers injected
#         logger.info('Trigger types: {}'.format(args.trigger_types))
#         logger.info('Poison_target_list: {}'.format(poison_target_list))
#         # print('main:', poison_target_list)
#
#         poison_train = add_Mtrigger_cifar(data_set=clean_train, trigger_types=args.trigger_types,
#                                           poison_rate=args.poison_rate,
#                                           mode='train', poison_target=args.poison_target, attack_type=args.attack_type)
#         poison_train_tf = DatasetTF(full_dataset=poison_train, transform=tf_train)
#         # split a small test subset
#         _, split_set_clean = split_dataset(clean_test, frac=0.1)
#         _, split_set = split_dataset(clean_test, frac=0.01)
#         # get each of poison data
#         onePixelTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['onePixelTrigger'],
#                                                          poison_rate=1.0,
#                                                          mode='test', poison_target=poison_target_list[0],
#                                                          attack_type=args.attack_type)
#         gridTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['gridTrigger'], poison_rate=1.0,
#                                                      mode='test', poison_target=poison_target_list[1],
#                                                      attack_type=args.attack_type)
#         wanetTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['wanetTrigger'], poison_rate=1.0,
#                                                       mode='test', poison_target=poison_target_list[2],
#                                                       attack_type=args.attack_type)
#         trojanTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['trojanTrigger'], poison_rate=1.0,
#                                                        mode='test', poison_target=poison_target_list[3],
#                                                        attack_type=args.attack_type)
#         blendTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['blendTrigger'], poison_rate=1.0,
#                                                       mode='test', poison_target=poison_target_list[4],
#                                                       attack_type=args.attack_type)
#         signalTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['signalTrigger'], poison_rate=1.0,
#                                                        mode='test', poison_target=poison_target_list[5],
#                                                        attack_type=args.attack_type)
#         CLTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['CLTrigger'], poison_rate=1.0,
#                                                    mode='test', poison_target=poison_target_list[6],
#                                                    attack_type=args.attack_type)
#         smoothTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['smoothTrigger'], poison_rate=1.0,
#                                                        mode='test', poison_target=poison_target_list[7],
#                                                        attack_type=args.attack_type)
#         dynamicTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['dynamicTrigger'],
#                                                         poison_rate=1.0,
#                                                         mode='test', poison_target=poison_target_list[8],
#                                                         attack_type=args.attack_type)
#         nashTrigger_poison_test = add_Mtrigger_cifar(data_set=split_set, trigger_types=['nashTrigger'], poison_rate=1.0,
#                                                      mode='test', poison_target=poison_target_list[9],
#                                                      attack_type=args.attack_type)
#         # add data transforms
#         clean_test_tf = DatasetTF(full_dataset=split_set_clean, transform=tf_test)
#         one_poison_test_tf = DatasetTF(full_dataset=onePixelTrigger_poison_test, transform=tf_test)
#         grid_poison_test_tf = DatasetTF(full_dataset=gridTrigger_poison_test, transform=tf_test)
#         wa_poison_test_tf = DatasetTF(full_dataset=wanetTrigger_poison_test, transform=tf_test)
#         tro_poison_test_tf = DatasetTF(full_dataset=trojanTrigger_poison_test, transform=tf_test)
#         ble_poison_test_tf = DatasetTF(full_dataset=blendTrigger_poison_test, transform=tf_test)
#         sig_poison_test_tf = DatasetTF(full_dataset=signalTrigger_poison_test, transform=tf_test)
#         cl_test_tf = DatasetTF(full_dataset=CLTrigger_poison_test, transform=tf_test)
#         sm_test_tf = DatasetTF(full_dataset=smoothTrigger_poison_test, transform=tf_test)
#         dy_test_tf = DatasetTF(full_dataset=dynamicTrigger_poison_test, transform=tf_test)
#         nash_test_tf = DatasetTF(full_dataset=nashTrigger_poison_test, transform=tf_test)
#
#         # dataloader
#         poison_train_loader = DataLoader(poison_train_tf, batch_size=args.batch_size, shuffle=True, num_workers=0)
#         clean_test_loader = DataLoader(clean_test_tf, batch_size=args.batch_size, num_workers=0)
#         one_poison_test_loader = DataLoader(one_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         grid_poison_test_loader = DataLoader(grid_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         wa_poison_test_loader = DataLoader(wa_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         tro_poison_test_loader = DataLoader(tro_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         ble_poison_test_loader = DataLoader(ble_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         sig_poison_test_loader = DataLoader(sig_poison_test_tf, batch_size=args.batch_size, num_workers=0)
#         cl_poison_test_loader = DataLoader(cl_test_tf, batch_size=args.batch_size, num_workers=0)
#         sm_poison_test_loader = DataLoader(sm_test_tf, batch_size=args.batch_size, num_workers=0)
#         dy_poison_test_loader = DataLoader(dy_test_tf, batch_size=args.batch_size, num_workers=0)
#         nash_poison_test_loader = DataLoader(nash_test_tf, batch_size=args.batch_size, num_workers=0)
#
#         poison_data_loaders_dct = {'onePixelTrigger': one_poison_test_loader,
#                                    'gridTrigger': grid_poison_test_loader,
#                                    'wanetTrigger': wa_poison_test_loader,
#                                    'trojanTrigger': tro_poison_test_loader,
#                                    'blendTrigger': ble_poison_test_loader,
#                                    'signalTrigger': sig_poison_test_loader,
#                                    'CLTrigger': cl_poison_test_loader,
#                                    'smoothTrigger': sm_poison_test_loader,
#                                    'dynamicTrigger': dy_poison_test_loader,
#                                    'nashTrigger': nash_poison_test_loader}
#     else:
#         raise ValueError('Please use valid backdoor attacks')


def train_step_unlearning(args, model, criterion, optimizer, data_loader):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)

        loss = criterion(output, labels)

        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
        (-loss).backward()
        optimizer.step()

    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test(model, criterion, data_loader):
    model.eval()
    total_correct = 0
    total_loss = 0.0

    count_classes = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]

            for predict in pred:
                count_classes[predict] += 1

            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc, count_classes


def unlearning(model, args):
    batch_size = args.batch_size
    clean_train, clean_val, clean_test = data()

    clean_train_loader = DataLoader(clean_train, batch_size=batch_size, shuffle=True, num_workers=0)
    # poison_train_loader = DataLoader(poison_train, batch_size=batch_size, shuffle=True, num_workers=0)
    clean_val_loader = DataLoader(clean_val, batch_size=batch_size, num_workers=0)
    # poison_val_loader = DataLoader(poison_val, batch_size=batch_size, num_workers=0)
    clean_test_loader = DataLoader(clean_test, batch_size=batch_size, num_workers=0)
    # poison_test_loader = DataLoader(poison_test, batch_size=batch_size, num_workers=0)

    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.unlearning_lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.schedule, gamma=0.1)

    print('----------- Model Unlearning --------------')
    print('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t CleanLoss \t CleanACC')  # \t PoisonLoss \t PoisonACC

    cl_classes = [[] for i in range(10)]
    po_classes = [[] for i in range(10)]
    train_acc_list = []
    cl_test_acc_list = []
    po_test_acc_list = []

    for epoch in range(0, args.unlearning_epochs + 1):
        start = time.time()
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step_unlearning(args=args, model=model, criterion=criterion, optimizer=optimizer,
                                                      data_loader=clean_val_loader)
        cl_test_loss, cl_test_acc, cl_count_classes = test(model=model, criterion=criterion,
                                                           data_loader=clean_test_loader)
        # po_test_loss, po_test_acc, po_count_classes = test(model=model, criterion=criterion,
        #                                                    data_loader=poison_test_loader)
        scheduler.step()
        end = time.time()
        print("{} \t {:.3f} \t {:.1f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f}  ".format(epoch, lr,
                                                                                        end - start,
                                                                                        train_loss,
                                                                                        train_acc,
                                                                                        # po_test_loss,
                                                                                        # po_test_acc,
                                                                                        cl_test_loss,
                                                                                        cl_test_acc))
        # '\t {:.4f} \t {:.4f}'
        train_acc_list.append(train_acc)
        cl_test_acc_list.append(cl_test_acc)
        # po_test_acc_list.append(po_test_acc)

        for k in range(len(cl_count_classes)):
            cl_classes[k].append(cl_count_classes[k])
            # po_classes[k].append(po_count_classes[k])

    input = np.array(cl_count_classes)
    VAR = np.var(input / len(clean_test))
    if VAR > 0.01:
        print("This is a backdoor model\t", 'VAR ={:.5f}'.format(VAR))
        return 1
    else:
        print("Not a backdoor model\t", 'VAR ={:.5f}'.format(VAR))
        return 0


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--NSTEP', default=3000)
    parser.add_argument('--model_path',
                        default='../model/PreActResNet18_CIFAR10_multi_triggers_all2all_poison_rate0.01_model_last.tar')
    parser.add_argument('--unlearning_lr', type=float, default=0.005, help='the learning rate for neuron unlearning')
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--unlearning_epochs', type=int, default=20, help='the number of epochs for unlearning')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--model_name', default='PreActResNet18',
                        choices=['ResNet18', 'VGG16', 'PreActResNet18', 'MobileNetV2'])
    parser.add_argument('--pretrained', default=True, help='read model weight')

    parser.add_argument('--multi_model', default=False, help="Detect the model in the Model folder")
    parser.add_argument('--model_name_list', '--arg',default=[], nargs='+', help="model architecture")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.multi_model == False:
        model_path = os.path.join(args.model_path)
        pretrained_models_path = model_path
        model = select_model(args=args, dataset=args.dataset, model_name=args.model_name, pretrained=args.pretrained,
                             pretrained_models_path=pretrained_models_path)
        print(model)
        unlearning(args=args, model=model)
    else:
        model_name_list = args.model_name_list
        out = []
        path = os.path.split(os.path.realpath(__file__))[0]  # 当前路径
        models = os.listdir(path + '/../model')  # 模型路径

        for i in range(len(models)):
            for j in model_name_list:
                if models[i].startswith(j) and args.dataset in models[i]:
                    args.model_path = '../model/' + models[i]
                    model_path = os.path.join(args.model_path)
                    pretrained_models_path = model_path
                    print("~~~~~~~~~~~~~~ Unlearning ~~~~~~~~~~~~~~")
                    print(models[i])
                    print()
                    model = select_model(args=args, dataset=args.dataset, model_name=j, pretrained=args.pretrained,
                                         pretrained_models_path=pretrained_models_path)
                    out.append(unlearning(args=args, model=model))
        print("~~~~~~~~~~~~~~ Unlearning ~~~~~~~~~~~~~~")
        print(out)
