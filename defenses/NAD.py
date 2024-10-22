import argparse
import sys
import torch
from torch import nn
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import torch.utils.data as Data
import os
import random
from datasets.poison_tool_cifar import get_train_loader, get_test_loader

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val   = 0
		self.avg   = 0
		self.sum   = 0
		self.count = 0

	def update(self, val, n=1):
		self.val   = val
		self.sum   += val * n
		self.count += n
		self.avg   = self.sum / self.count

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
        layer1_activation = out
        out = self.layer2(out)
        layer2_activation = out
        out = self.layer3(out)
        layer3_activation = out
        out = self.layer4(out)
        layer4_activation = out
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        
        return [layer1_activation, layer2_activation, layer3_activation, layer4_activation], out


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


class NAD():
    def __init__(self, args, logger):
        self.args = args
        self.arguments()
        self.logger = logger
        self.logger.info(self.args)
    
        
    def arguments(self):
        if 'arch' not in self.args:
            self.args.arch = 'resnet18'
        if 'trigger_type' not in self.args:
            self.args.trigger_type = 'gridTrigger'
        if 'poison_target' not in self.args:
            self.args.poison_target = 0
        if 'target_type' not in self.args:
            self.args.target_type = 'all2one'
        if 'batch_size' not in self.args:
            self.args.batch_size = 128
        if 'lambda_at' not in self.args:
            self.args.lambda_at = 5000
        if 'lr' not in self.args:
            self.args.lr = 0.01
        if 'print_freq' not in self.args:
            self.args.print_freq = 400
        if 'ratio' not in self.args:
            self.args.ratio = 0.01
        if 'num_class' not in self.args:
            self.args.num_class = 10
        if 'epochs' not in self.args:
            self.args.epochs = 10
        if 'device' not in self.args:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
            
    def test_at(self, trigger_type, test_loader, model, criterion):
        top1 = AverageMeter()
        top5 = AverageMeter()
        at_losses = AverageMeter()



        model.eval()

        for idx, (img, target) in enumerate(test_loader, start=1):
            img = img.cuda()
            target = target.cuda()

            with torch.no_grad():
                _,output_s = model(img)


            prec1, prec5 = self.accuracy(output_s, target, topk=(1, 5))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

        f_l_trig = [trigger_type, top1.avg, top5.avg]
        self.logger.info('[{}]Prec@1: {:.2f}, Prec@5: {:.2f}'.format(*f_l_trig))
        return f_l_trig

    def adjust_learning_rate(self, optimizer, epoch, lr):
        if epoch < 20:
            lr = lr
        elif epoch < 30:
            lr = 0.01
        elif epoch < 40:
            lr = 0.001
        else:
            lr = 0.0001
        self.logger.info('epoch: {}  lr: {:.4f}'.format(epoch, lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


    def attention_map(self, fm, eps=1e-5, p=2):
        am = torch.pow(torch.abs(fm), p)
        am = torch.mean(am, dim=1, keepdim=True)
        norm = torch.norm(am, dim=(2, 3), keepdim=True)
        am = torch.div(am, norm + eps)

        return am
            
    def train_at(self, train_orig_loader, nets, optimizer, criterions, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cls_losses = AverageMeter()
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        at1_losses = AverageMeter()
        at2_losses = AverageMeter()
        at3_losses = AverageMeter()

        snet = nets['snet']
        tnet = nets['tnet']

        criterionCls = criterions['criterionCls']
        criterionAT = criterions['criterionAT']

        snet.train()

        for idx, (img, target) in enumerate(train_orig_loader, start=1):

            img = img.cuda()
            target = target.cuda()

            [_, activation2_s, activation3_s, activation4_s], output_s = snet(img)
            [_, activation2_t, activation3_t, activation4_t], _ = tnet(img)

            cls_loss = criterionCls(output_s, target)
            at2_loss = criterionAT(self.attention_map(activation2_s), self.attention_map(activation2_t).detach()) * self.args.lambda_at
            at3_loss = criterionAT(self.attention_map(activation3_s), self.attention_map(activation3_t).detach()) * self.args.lambda_at
            at4_loss = criterionAT(self.attention_map(activation4_s), self.attention_map(activation4_t).detach()) * self.args.lambda_at
            at_loss = at2_loss + at3_loss + at4_loss + cls_loss

            prec1, prec5 = self.accuracy(output_s, target, topk=(1, 5))
            # cls_losses.update(cls_loss.item(), img.size(0))
            at_losses.update(at_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            at_loss.backward()
            optimizer.step()

            if idx % self.args.print_freq == 0:
                self.logger.info('Epoch[{0}]:[{1:03}/{2:03}] '
                      'AT_loss:{losses.val:.4f}({losses.avg:.4f})  '
                      'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                      'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_orig_loader), losses=at_losses, top1=top1, top5=top5))
            
    def train(self, train_orig_loader, model, optimizer, criterion, epoch):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        cls_losses = AverageMeter()
        at_losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        model.train()

        for idx, (img, target) in enumerate(train_orig_loader, start=1):

            img = img.to(self.args.device)
            target = target.to(self.args.device)
            #target = torch.argmax(target, dim=1)


            _,output_s = model(img)

            at_loss = criterion(output_s, target)

            prec1, prec5 = self.accuracy(output_s, target, topk=(1, 5))
            # cls_losses.update(cls_loss.item(), img.size(0))
            at_losses.update(at_loss.item(), img.size(0))
            top1.update(prec1.item(), img.size(0))
            top5.update(prec5.item(), img.size(0))

            optimizer.zero_grad()
            at_loss.backward()
            optimizer.step()

            if idx % 20 == 0:
                self.logger.info('Epoch[{0}]:[{1:03}/{2:03}] '
                      'loss:{losses.val:.4f}({losses.avg:.4f})  '
                      'prec@1:{top1.val:.2f}({top1.avg:.2f})  '
                      'prec@5:{top5.val:.2f}({top5.avg:.2f})'.format(epoch, idx, len(train_orig_loader), losses=at_losses, top1=top1, top5=top5))       
            
    def AT(self):
        best_cleanse = 100

        # define loss functions
        criterionCls = nn.CrossEntropyLoss().cuda()
        criterionAT = nn.MSELoss().cuda()
        criterions = {'criterionCls': criterionCls, 'criterionAT': criterionAT}         

        self.logger.info('----------- Network Initialization --------------')

        # 加载模型
        self.teacher_model = getattr(models, self.args.arch)(num_classes=self.args.num_class).to(self.args.device)
        self.teacher_model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)["state_dict"])
        self.student_model = getattr(models, self.args.arch)(num_classes=self.args.num_class).to(self.args.device)
        self.student_model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device)["state_dict"])
        self.logger.info('finished teacher and student model init...')

        optimizer = torch.optim.SGD(self.teacher_model.parameters(),
                                    lr=self.args.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)
        self.logger.info('fine-tuning teacher model......')
        
        for epoch in range(0, 10 + 1):

            self.train(self.args.clean_train_loader, self.teacher_model, optimizer, criterionCls, epoch)

            # evaluate on testing set
        self.logger.info('testing fine-tuned teacher model......')

        self.test_at("CA", self.args.test_clean_loader, self.teacher_model, criterions)
        self.test_at(self.args.lr, self.args.test_bad_loader, self.teacher_model, criterions)


        self.teacher_model.eval()

        for param in self.teacher_model.parameters():
            param.requires_grad = False

        nets = {'snet': self.student_model, 'tnet': self.teacher_model}

        self.logger.info('----------- Finished load weights --------------')
        # initialize optimizer
        optimizer = torch.optim.SGD(self.student_model.parameters(),
                                    lr=self.args.lr,
                                    momentum=0.9,
                                    weight_decay=1e-4,
                                    nesterov=True)


        for epoch in range(0, self.args.epochs + 1):

            self.adjust_learning_rate(optimizer, epoch, self.args.lr)

            if epoch == 0:
                # before training
                self.logger.info('testing student model before training......')
                self.test_at("CA", self.args.test_clean_loader, self.student_model, criterions)
                self.test_at(self.args.trigger_type, self.args.test_bad_loader, self.student_model, criterions)

            self.train_at(self.args.clean_train_loader, nets, optimizer, criterions, epoch)

            # evaluate on testing set
            self.logger.info('testing student model......')
            self.test_at("CA", self.args.test_clean_loader, self.student_model, criterions)
            self.test_at(self.args.trigger_type, self.args.test_bad_loader, self.student_model, criterions)
               
    

            
    
    
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
    parser.add_argument('--log_root', type=str, default='logs/', help='logs are saved here')
#     parser.add_argument('--backdoor_model_path', type=str,
#                         default='weights/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar',
#                         help='path of backdoored model')
    parser.add_argument('--arch', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'MobileNetV2',
                                 'vgg19_bn'])
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='name of image dataset')
    parser.add_argument('--batch_size', type=int, default=128, help='The size of batch')
    parser.add_argument('--num_class', type=int, default=10, help='number of classes')

    # backdoor attacks
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')

    # NAD
    parser.add_argument("--lambda_at", type=int, default = 5000)
    parser.add_argument("--lr", type=float, default = 0.01)
    parser.add_argument("--ratio", type=float, default = 0.01)
    parser.add_argument("--print_freq", type=int, default = 400)
    parser.add_argument("--model_path", type=str, default = '/media/user/8961e245-931a-4871-9f74-9df58b1bd938/server/lyg/LfF-master(2)/checkpoint/cifar10/正常训练/ResNet18-ResNet-BadNets-target0-portion0.1-epoch80.tar')
    parser.add_argument("--epochs", type=int, default = 10)
    
    args = parser.parse_args()
    args.dataset = 'CIFAR10'
    
    args.clean_train_loader = get_train_loader(args)
    args.test_clean_loader, args.test_bad_loader = get_test_loader(args)
    
    
    NAD = NAD(args, logger)
    NAD.AT()
