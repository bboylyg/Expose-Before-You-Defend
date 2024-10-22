import argparse
import sys
import os
print(os.getcwd())
sys.path.append("..")
# print(os.getcwd())
from Multi_Trigger_Backdoor_Attacks.model_for_cifar.model_select import select_model

import torch
import torch
import numpy as np
import torchvision
from torch.nn import CrossEntropyLoss
import tqdm
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
import json
import torch.nn as nn
from torchvision.datasets import CIFAR10


def train(model, target_label, train_loader, param):
    print("Processing label: {}".format(target_label))

    width, height = param["image_size"]
    trigger = torch.rand((3, width, height), requires_grad=True)
    trigger = trigger.to(device).detach().requires_grad_(True)
    mask = torch.rand((width, height), requires_grad=True)
    mask = mask.to(device).detach().requires_grad_(True)

    Epochs = param["Epochs"]
    lamda = param["lamda"]

    min_norm = np.inf
    min_norm_count = 0

    criterion = CrossEntropyLoss()
    optimizer = torch.optim.Adam([{"params": trigger}, {"params": mask}], lr=0.005)
    model.to(device)
    model.eval()

    for epoch in range(Epochs):
        norm = 0.0
        for images, _ in tqdm.tqdm(train_loader, desc='Epoch %3d' % (epoch + 1)):
            optimizer.zero_grad()
            images = images.to(device)
            trojan_images = (1 - torch.unsqueeze(mask, dim=0)) * images + torch.unsqueeze(mask, dim=0) * trigger
            y_pred = model(trojan_images)
            y_target = torch.full((y_pred.size(0),), target_label, dtype=torch.long).to(device)
            loss = criterion(y_pred, y_target) + lamda * torch.sum(torch.abs(mask))
            loss.backward()
            optimizer.step()

            # figure norm
            with torch.no_grad():
                # 防止trigger和norm越界
                torch.clip_(trigger, 0, 1)
                torch.clip_(mask, 0, 1)
                norm = torch.sum(torch.abs(mask))

        # to early stop
        if norm < min_norm:
            min_norm = norm
            min_norm_count = 0
        else:
            min_norm_count += 1

        if min_norm_count > 30:
            break

    print("min norm" + str(min_norm))

    return trigger.cpu(), min_norm.cpu()


class augDataset_npy(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.dataLen


def reverse_engineer(model, param):
    model = model.to(device)
    tf_train = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])

    train_dataset = CIFAR10(root="./data", train=False, download=True)

    list_ = []
    count = [0] * param["num_classes"]
    threshold = int(param["ratio"] * len(train_dataset))
    for i in train_dataset:  # 这段仅是用来调整个别类别的数量
        if count[i[1]] < threshold:
            list_.append(i)
            count[i[1]] += 1
    train_dataset = list_
    count = [0] * param["num_classes"]

    train_dataset = augDataset_npy(full_dataset=train_dataset, transform=tf_train)
    train_loader = DataLoader(
        train_dataset,
        batch_size=param["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    for label in range(param["num_classes"]):
        trigger, mask = train(model, label, train_loader, param)
        norm_list.append(mask)

        trigger = trigger.cpu().detach().numpy()
        trigger = np.transpose(trigger, (1, 2, 0))
        # plt.axis("off")
        # plt.imshow(trigger)
        # plt.savefig('mask/trigger_{}.png'.format(label), bbox_inches='tight', pad_inches=0.0)
        #
        # mask = mask.cpu().detach().numpy()
        # plt.axis("off")

    l1_norm_list = norm_list[-param["num_classes"]:]
    list_target = [0] * param["num_classes"]
    for i in range(len(list_target)):
        list_target[i] = i
    flag_label,decide = outlier_detection(l1_norm_list, list_target)

    print("flag label:" + str(flag_label))
    return flag_label,decide


def outlier_detection(l1_norm_list, idx_mapping):
    print("-" * 30)
    print("Determining whether model is backdoor")
    consistency_constant = 1.4826
    l1_norm_list = torch.tensor(l1_norm_list)
    median = torch.median(l1_norm_list)  # 返回中位数
    mad = consistency_constant * torch.median(torch.abs(l1_norm_list - median))  # 常数乘以范数减去中位数之后的中位数
    min_mad = torch.abs(torch.min(l1_norm_list) - median) / mad  # 范数中最小值减去中位数之后除以mad的绝对值

    print("Median: {}, MAD: {}".format(median, mad))
    print("Anomaly index: {}".format(min_mad))

    decide = 0

    if min_mad < 2:  # 判断是否为后门模型
        print("Not a backdoor model")
        decide = 0
    else:
        print("This is a backdoor model")
        decide = 1
    flag_list = []
    for y_label in idx_mapping:  # 记录异常的类别及其mask的范数
        if l1_norm_list[idx_mapping[y_label]] > median:
            continue
        if torch.abs(l1_norm_list[idx_mapping[y_label]] - median) / mad > 2:
            flag_list.append((y_label, l1_norm_list[idx_mapping[y_label]]))

    if len(flag_list) > 0:  # 若有不少于一个异常类别，则取异常值最小的
        flag_list = sorted(flag_list, key=lambda x: x[1])

    if flag_list == []:
        return float("inf"),decide
    else:
        return flag_list[0][0],decide


if __name__ == '__main__':
    # Prepare arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', default=10)
    parser.add_argument('--lamda', default=0.01)
    parser.add_argument('--dataset', default='CIFAR10')
    parser.add_argument('--model_name', default='PreActResNet18')
    parser.add_argument('--model_path', default='../model/PreActResNet18_CIFAR10_multi_triggers_all2rand_poison_rate0.01_model_last.tar')
    parser.add_argument('--pretrained', default=True,help='read model weight')

    parser.add_argument('--multi_model', default=False,help="Detect the model in the Model folder")
    parser.add_argument('--model_name_list', '--arg',default=[], nargs='+', help="model architecture")

    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.multi_model==False:
        # 单模型
        model_path = os.path.join(args.model_path)
        pretrained_models_path = model_path
        model = select_model(args=args, dataset=args.dataset, model_name=args.model_name, pretrained=args.pretrained,
                             pretrained_models_path=pretrained_models_path)
        norm_list = []
        param = {
            "dataset": "cifar10",
            "Epochs": 20,
            "batch_size": 64,
            "lamda": args.lamda,
            "num_classes": 10,
            "image_size": (32, 32),
            "ratio": 0.005
        }
        flag_label,decide = reverse_engineer(model=model,param=param)
        minium_norm = float("inf")
        minium_norm_index = float("inf")
        for i in range(len(norm_list)):
            if norm_list[i] < minium_norm:
                minium_norm = norm_list[i]
                minium_norm_index = i % 10
        print("minium norm:" + str(minium_norm) + " and its index:" + str(minium_norm_index))
    else:
        # 多模型
        model_name_list = args.model_name_list
        out = []
        path = os.path.split(os.path.realpath(__file__))[0]  # 当前路径
        models = os.listdir(path + '/../model')  # 模型路径
        for m in range(len(models)):
            for name in model_name_list:
                if models[m].startswith(name) and args.dataset in models[m]:
                    args.model_path = '../model/' + models[m]
                    model_path = os.path.join(args.model_path)
                    pretrained_models_path = model_path
                    print("~~~~~~~~~~~~~~ NC ~~~~~~~~~~~~~~")
                    print(models[m])
                    print()
                    model = select_model(args=args, dataset=args.dataset, model_name=name, pretrained=args.pretrained,
                                         pretrained_models_path=pretrained_models_path)

                    norm_list = []
                    param = {
                        "dataset": "cifar10",
                        "Epochs": 20,
                        "batch_size": 64,
                        "lamda": args.lamda,
                        "num_classes": 10,
                        "image_size": (32, 32),
                        "ratio": 0.005
                    }
                    flag_label, decide = reverse_engineer(model=model, param=param)
                    minium_norm = float("inf")
                    minium_norm_index = float("inf")
                    for i in range(len(norm_list)):
                        if norm_list[i] < minium_norm:
                            minium_norm = norm_list[i]
                            minium_norm_index = i % 10
                    print("minium norm:" + str(minium_norm) + " and its index:" + str(minium_norm_index))
                    out.append(decide)
        print("~~~~~~~~~~~~~~ NC ~~~~~~~~~~~~~~")
        print(out)



