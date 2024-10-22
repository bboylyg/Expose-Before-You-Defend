import time
from copy import deepcopy
from torch.utils.data import Subset
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, RandomSampler
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import pandas as pd
from tqdm import tqdm
import mixed_backdoor_cifar
import random
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parameter import Parameter
import argparse
import logging



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = norm_layer(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            self._norm_layer = nn.BatchNorm2d
        else:
            self._norm_layer = norm_layer
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = self._norm_layer(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self._norm_layer))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


class models():
    def resnet18(num_classes=10, norm_layer=nn.BatchNorm2d):
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, norm_layer)

    def resnet34(num_classes=10, norm_layer=nn.BatchNorm2d):
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, norm_layer)

    def resnet50(num_classes=10, norm_layer=nn.BatchNorm2d):
        return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, norm_layer)

    def resnet101(num_classes=10, norm_layer=nn.BatchNorm2d):
        return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, norm_layer)

    def resnet152(num_classes=10, norm_layer=nn.BatchNorm2d):
        return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, norm_layer)


class ABL():
    def __init__(self, args, logger):

        self.args = args
        self.arguments()
        self.logger = logger
        self.logger.info(self.args)



        self.model = getattr(models, self.args.arch)(num_classes=10).to(self.args.device)
        #self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device), strict=False)

    def arguments(self):
        if 'arch' not in self.args:
            self.args.arch = 'resnet18'
        if 'trigger_list' not in self.args:
            self.args.trigger_list = ['gridTrigger']
        if 'poison_rate' not in self.args:
            self.args.poison_rate = 0.1
        if 'poison_target' not in self.args:
            self.args.poison_target = 0
        if 'batch_size' not in self.args:
            self.args.batch_size = 128
        if 'print_every' not in self.args:
            self.args.print_every = 500
        if 'model_path' not in self.args:
            self.args.model_path = 'model_abl_last.th'
        if 'isolation_epochs' not in self.args:
            self.isolation_epochs = 20
        if 'isolation_ratio' not in self.args:
            self.isolation_ratio = 0.01
        if 'gradient_ascent_type' not in self.args:
            self.gradient_ascent_type = 'Flooding'
        if 'gamma' not in self.args:
            self.gamma = 0.5
        if 'flooding' not in self.args:
            self.flooding = 0.5
        if 'finetuning_ascent_model' not in self.args:
            self.finetuning_ascent_model = True
        if 'finetuning_epochs' not in self.args:
            self.finetuning_epochs = 60
        if 'unlearning_epochs' not in self.args:
            self.unlearning_epochs = 20
        if 'tuning_lr' not in self.args:
            self.tuning_lr = 0.1
        if 'lr_finetuning_init' not in self.args:
            self.lr_finetuning_init = 0.1
        if 'batch_size_isolation' not in self.args:
            self.batch_size_isolation = 64
        if 'batch_size_finetuning' not in self.args:
            self.batch_size_finetuning = 64
        if 'batch_size_unlearning' not in self.args:
            self.batch_size_unlearning = 64
        if 'lr_unlearning' not in self.args:
            self.lr_unlearning =5e-4
        if 'device' not in self.args:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'






    def compute_loss_value(self, poison_train, model_ascent):
        '''Calculate loss value per example
        args:
            Contains default parameters
        poisoned_data:
            the train dataset which contains backdoor data
        model_ascent:
            the model after the process of pretrain
        '''
        # Define loss function
        if self.args.device == 'cuda':
            criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            criterion = torch.nn.CrossEntropyLoss()

        model_ascent.eval()
        losses_record = []

        caculate_data_loader = torch.utils.data.DataLoader(dataset=poison_train, batch_size=1, shuffle=False)

        for idx, (img, target) in tqdm(enumerate(caculate_data_loader, start=0)):
            img = img.to(self.args.device)
            target = target.to(self.args.device)

            with torch.no_grad():
                output = model_ascent(img)
                loss = criterion(output, target)

            losses_record.append(loss.item())

        losses_idx = np.argsort(np.array(losses_record))  # get the index of examples by loss value in descending order

        # Show the top 10 loss values
        losses_record_arr = np.array(losses_record)
        logging.info(f'Top ten loss value: {losses_record_arr[losses_idx[:10]]}')

        return losses_idx

    def isolate_data(self, poison_train, losses_idx):
        '''isolate the backdoor data with the calculated loss
        args:
            Contains default parameters
        result:
            the attack result contain the train dataset which contains backdoor data
        losses_idx:
            the index of order about the loss value for each data
        '''
        # Initialize lists
        other_examples = []
        isolation_examples = []

        cnt = 0
        ratio = self.isolation_ratio

        example_data_loader = DataLoader(dataset=poison_train, batch_size=1, shuffle=False)
        # print('full_poisoned_data_idx:', len(losses_idx))
        perm = losses_idx[0: int(len(losses_idx) * ratio)]

        isolation_subset = Subset(poison_train, perm)

        permnot = [i for i in range(len(poison_train)) if i not in perm]
        other_subset = Subset(poison_train, permnot)

        print(len(isolation_subset))
        return isolation_subset, other_subset

    def adjust_learning_rate(self, optimizer, epoch):
        if epoch < self.isolation_epochs:
            lr = self.tuning_lr
        else:
            lr = self.tuning_lr * 0.1
        # print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def learning_rate_finetuning(self, optimizer, epoch):
        if epoch < 40:
            lr = self.lr_finetuning_init
        elif epoch < 60:
            lr = self.lr_finetuning_init * 0.1
        else:
            lr = self.lr_finetuning_init * 0.01
        # print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def learning_rate_unlearning(self, optimizer, epoch):
        if epoch < self.unlearning_epochs:
            lr = self.lr_unlearning
        else:
            lr = self.lr_unlearning * 0.2
        # print('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def train_step_isolation(self, train_loader, model_ascent, optimizer, criterion, epoch):
        args = self.args


        model_ascent.train()

        total_loss = 0
        correct = 0
        total = 0

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            if self.gradient_ascent_type == 'LGA':
                output = model_ascent(img)
                loss = criterion(output, target)
                # add Local Gradient Ascent(LGA) loss
                loss_ascent = torch.sign(loss - self.gamma) * loss
                # loss_ascent = loss

            elif self.gradient_ascent_type == 'Flooding':
                output = model_ascent(img)
                # output = student(img)
                loss = criterion(output, target)
                # add flooding loss
                loss_ascent = (loss - self.flooding).abs() + self.flooding
                # loss_ascent = loss

            else:
                raise NotImplementedError

            total_loss += loss_ascent.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            optimizer.zero_grad()
            loss_ascent.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total


        print(f'Epoch {epoch}: Average Loss: {avg_loss:.4f},Train Accuracy: {accuracy:.2f}%')

    def train_step_finetuing(self, train_loader, model_ascent, optimizer, criterion, epoch):

        model_ascent.train()
        total_loss = 0
        correct = 0
        total = 0

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            output = model_ascent(img)

            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f'Epoch {epoch}: Average Loss: {avg_loss:.4f},Train Accuracy: {accuracy:.2f}%')



    def train_step_unlearning(self, train_loader, model_ascent, optimizer, criterion, epoch):

        model_ascent.train()
        total_loss = 0
        correct = 0
        total = 0

        for idx, (img, target) in enumerate(train_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            output = model_ascent(img)

            loss = criterion(output, target)

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            optimizer.zero_grad()
            (-loss).backward()  # Gradient ascent training
            optimizer.step()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f'Epoch {epoch}: Average Loss: {avg_loss:.4f},Train Accuracy: {accuracy:.2f}%')

    def test(self, model, criterion, data_loader, device):
        model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                labels = labels.type(torch.LongTensor)
                images, labels = images.to(device), labels.to(device)
                output = model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(data_loader)
        acc = float(total_correct) / len(data_loader.dataset)
        return loss, acc

    def train(self):
        args = self.args

        # Load models
        print('----------- Network Initialization --------------')
        # arch = config.arch['abl']
        # model_ascent = arch(depth=16, num_classes=self.num_classes, widen_factor=1, dropRate=0)
        model_ascent = self.model
        model_ascent = nn.DataParallel(model_ascent)
        model_ascent = model_ascent.cuda()
        print('finished model init...')

        # initialize optimizer
        optimizer = torch.optim.SGD(model_ascent.module.parameters(),
                                    lr=self.tuning_lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)

        # define loss functions
        criterion = nn.CrossEntropyLoss().cuda()

        print('----------- Data Initialization --------------')
        poisoned_set_loader = args.bad_train_loader

        print('----------- Train Initialization --------------')
        for epoch in range(0, self.isolation_epochs):
            self.adjust_learning_rate(optimizer, epoch)


            self.train_step_isolation(poisoned_set_loader, model_ascent, optimizer, criterion, epoch)

        _, ASR = self.test(model=model_ascent, criterion=criterion, data_loader=args.bad_test_loader,
                           device=self.args.device)
        _, BA = self.test(model=model_ascent, criterion=criterion, data_loader=args.clean_test_loader,
                          device=self.args.device)
        print('ASR=', ASR)
        print('BA=', BA)

        losses_idx = self.compute_loss_value(poison_train=args.bad_data, model_ascent=model_ascent)

        data_set_isolate, data_set_other = self.isolate_data(poison_train=args.bad_data, losses_idx=losses_idx)

        isolate_other_data_loader = DataLoader(data_set_other, batch_size=self.args.batch_size, shuffle=True, num_workers=0)
        isolate_poison_data_loader = DataLoader(data_set_isolate, batch_size=self.args.batch_size, shuffle=True, num_workers=0)

        if self.finetuning_ascent_model == True:
            # this is to improve the clean accuracy of isolation model, you can skip this step
            print('----------- Finetuning isolation model --------------')
            for epoch in range(0, self.finetuning_epochs):
                self.learning_rate_finetuning(optimizer, epoch)
                self.train_step_finetuing(isolate_other_data_loader, model_ascent, optimizer, criterion, epoch + 1)
                #self.test(model=model_ascent,criterion=criterion,data_loader=self.poison_test_loader,device=self.args.device)
            _, ASR = self.test(model=model_ascent, criterion=criterion, data_loader=args.bad_test_loader,
                               device=self.args.device)
            _, BA = self.test(model=model_ascent, criterion=criterion, data_loader=args.clean_test_loader,
                              device=self.args.device)
            print('ASR=', ASR)
            print('BA=', BA)


        print('----------- Model unlearning --------------')
        for name, module in list(model_ascent.module.named_modules()):
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = 0

        for epoch in range(0, self.unlearning_epochs):
            self.learning_rate_unlearning(optimizer, epoch)
            self.train_step_unlearning(isolate_poison_data_loader, model_ascent, optimizer, criterion, epoch + 1)
        _, ASR = self.test(model=model_ascent, criterion=criterion, data_loader=args.bad_test_loader,
                    device=self.args.device)
        _, BA = self.test(model=model_ascent, criterion=criterion, data_loader=args.clean_test_loader,
                            device=self.args.device)
        print('ASR=', ASR)
        print('BA=', BA)



        torch.save(model_ascent.module.state_dict(), self.args.model_path)
        print("[Save] Unlearned model saved to %s" % self.args.model_path)


if __name__ == '__main__':
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format='[%(asctime)s] - %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S',
        level=logging.DEBUG,
        handlers=[
            logging.FileHandler('output.log'),
            logging.StreamHandler()
        ])

    parser = argparse.ArgumentParser()
    # backdoor attacks
    parser.add_argument('--cuda', type=int, default=1, help='cuda available')
    parser.add_argument('--save-every', type=int, default=5, help='save checkpoints every few epochs')
    parser.add_argument('--log_root', type=str, default='logs/', help='logs are saved here')
    parser.add_argument('--output_weight', type=str, default='weights/')
    parser.add_argument('--backdoor_model_path', type=str,
                        default='weights/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar',
                        help='path of backdoored model')
    parser.add_argument('--unlearned_model_path', type=str,
                        default=None, help='path of unlearned backdoored model')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2',
                                 'vgg19_bn'])
    parser.add_argument('--schedule', type=int, nargs='+', default=[10, 20],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')
    parser.add_argument('--ratio', type=float, default=0.01, help='ratio of defense data')

    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')

    # RNP
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--clean_threshold', type=float, default=0.20, help='threshold of unlearning accuracy')
    #parser.add_argument('--unlearning_lr', type=float, default=0.01, help='the learning rate for neuron unlearning')
    parser.add_argument('--recovering_lr', type=float, default=0.2, help='the learning rate for mask optimization')
    #parser.add_argument('--unlearning_epochs', type=int, default=20, help='the number of epochs for unlearning')
    parser.add_argument('--recovering_epochs', type=int, default=20, help='the number of epochs for recovering')
    parser.add_argument('--mask_file', type=str, default=None, help='The text file containing the mask values')
    parser.add_argument('--pruning-by', type=str, default='threshold', choices=['number', 'threshold'])
    parser.add_argument('--pruning-max', type=float, default=0.90, help='the maximum number/threshold for pruning')
    parser.add_argument('--pruning-step', type=float, default=0.05, help='the step size for evaluating the pruning')
    args = parser.parse_args()
    args.dataset = 'CIFAR10'
    args.inject_portion = 0.1


    from datasets.poison_tool_cifar import get_test_loader, get_backdoor_loader  # ??RNP??????

    bad_data, bad_train_loader = get_backdoor_loader(args)
    clean_test_loader, bad_test_loader = get_test_loader(args)

    args.bad_data = bad_data
    args.bad_train_loader = bad_train_loader
    args.clean_test_loader = clean_test_loader
    args.bad_test_loader = bad_test_loader
    args.eval = eval


    ABL = ABL(args, logger)
    ABL.train()