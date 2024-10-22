from __future__ import absolute_import
from __future__ import print_function
from torchvision.datasets import CIFAR10
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import argparse
import sys
import os

print(os.getcwd())
sys.path.append("..")
print(os.getcwd())
from Multi_Trigger_Backdoor_Attacks.model_for_cifar.model_select import select_model
from copy import deepcopy
import time
import os
import sys
import math
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import PIL
import random
import copy as cp
import numpy as np
from torch.utils.data import DataLoader


def mmbd(model, args):
    random.seed()

    # Detection parameters
    NC = args.num_classes
    NI = 150
    PI = 0.9
    NSTEP = args.NSTEP
    TC = 6
    batch_size = 20
    model = model.to(device)

    def lr_scheduler(iter_idx):
        lr = 1e-2

        return lr

    res = []
    for t in range(10):

        images = torch.rand([30, 3, 32, 32]).to(device)
        images.requires_grad = True

        last_loss = 1000
        labels = t * torch.ones((len(images),), dtype=torch.long).to(device)
        onehot_label = F.one_hot(labels, num_classes=NC)
        for iter_idx in range(NSTEP):

            optimizer = torch.optim.SGD([images], lr=lr_scheduler(iter_idx), momentum=0.2)
            optimizer.zero_grad()
            outputs = model(torch.clamp(images, min=0, max=1))

            loss = -1 * torch.sum((outputs * onehot_label)) \
                   + torch.sum(torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0])
            loss.backward(retain_graph=True)
            optimizer.step()
            if abs(last_loss - loss.item()) / abs(last_loss) < 1e-5:
                break
            last_loss = loss.item()

        res.append(torch.max(torch.sum((outputs * onehot_label), dim=1) \
                             - torch.max((1 - onehot_label) * outputs - 1000 * onehot_label, dim=1)[0]).item())
        print(t, res[-1])

    stats = res
    from scipy.stats import median_abs_deviation as MAD
    from scipy.stats import gamma

    mad = MAD(stats, scale='normal')
    abs_deviation = np.abs(stats - np.median(stats))
    score = abs_deviation / mad
    print(score)

    np.save('results.npy', np.array(res))
    ind_max = np.argmax(stats)
    r_eval = np.amax(stats)
    r_null = np.delete(stats, ind_max)

    shape, loc, scale = gamma.fit(r_null)
    pv = 1 - pow(gamma.cdf(r_eval, a=shape, loc=loc, scale=scale), len(r_null) + 1)
    print(pv)

    if pv > 0.05:
        print("Not a backdoor mode")
        return 0
    else:
        print("This is a backdoor model")
        print('There is attack with target class {}'.format(np.argmax(stats)))
        return 1


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--NSTEP', default=3000)
    parser.add_argument('--model_path',
                        default='../model/ResNet18_CIFAR10_multi_triggers_all2all_poison_rate0.01_model_last(1).tar')
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--model_name', default='ResNet18',
                        choices=['ResNet18', 'VGG16', 'PreActResNet18', 'MobileNetV2'])
    parser.add_argument('--pretrained', default=True,help='read model weight')

    parser.add_argument('--multi_model', default=False,help="Detect the model in the Model folder")
    parser.add_argument('--model_name_list', '--arg',default=[], nargs='+', help="model architecture")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.multi_model == False:
        model_path = os.path.join(args.model_path)
        pretrained_models_path = model_path
        model = select_model(args=args, dataset=args.dataset, model_name=args.model_name, pretrained=args.pretrained,
                             pretrained_models_path=pretrained_models_path)

        mmbd(args=args, model=model)

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
                    print("~~~~~~~~~~~~~~ MMBD ~~~~~~~~~~~~~~")
                    print(models[i])
                    print()
                    model = select_model(args=args, dataset=args.dataset, model_name=j, pretrained=args.pretrained,
                                         pretrained_models_path=pretrained_models_path)
                    out.append(mmbd(args=args, model=model))
        print("~~~~~~~~~~~~~~ MMBD ~~~~~~~~~~~~~~")
        print(out)
