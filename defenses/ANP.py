import time
import numpy as np
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


class NoisyBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(NoisyBatchNorm2d, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.neuron_mask = Parameter(torch.Tensor(num_features))
        self.neuron_noise = Parameter(torch.Tensor(num_features))
        self.neuron_noise_bias = Parameter(torch.Tensor(num_features))
        init.ones_(self.neuron_mask)
        init.zeros_(self.neuron_noise)
        init.zeros_(self.neuron_noise_bias)
        self.is_perturbed = False

    def reset(self, rand_init=False, eps=0.0):
        if rand_init:
            init.uniform_(self.neuron_noise, a=-eps, b=eps)
            init.uniform_(self.neuron_noise_bias, a=-eps, b=eps)
        else:
            init.zeros_(self.neuron_noise)
            init.zeros_(self.neuron_noise_bias)

    def include_noise(self):
        self.is_perturbed = True

    def exclude_noise(self):
        self.is_perturbed = False

    def forward(self, input: Tensor) -> Tensor:

        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore
                self.num_batches_tracked = self.num_batches_tracked + 1  # type: ignore
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if self.training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        assert self.running_mean is None or isinstance(self.running_mean, torch.Tensor)
        assert self.running_var is None or isinstance(self.running_var, torch.Tensor)

        if self.is_perturbed:
            coeff_weight = self.neuron_mask + self.neuron_noise
            coeff_bias = 1.0 + self.neuron_noise_bias
        else:
            coeff_weight = self.neuron_mask
            coeff_bias = 1.0
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not self.training or self.track_running_stats else None,
            self.running_var if not self.training or self.track_running_stats else None,
            self.weight * coeff_weight, self.bias * coeff_bias,
            bn_training, exponential_average_factor, self.eps)


class anp():
    def __init__(self, args, logger):
        self.args = args
        self.arguments()  # 对args里一些没设定的数据进行定义
        self.logger = logger

        self.logger.info(self.args)

        self.model = getattr(models, self.args.arch)(num_classes=10).to(self.args.device)
        self.model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device), strict=False)

    def arguments(self):
        if 'arch' not in self.args:
            self.args.arch = 'resnet18'
        if 'print_every' not in self.args:
            self.args.print_every = 500
        if 'model_path' not in self.args:
            self.args.model_path = 'model_last.th'
        if 'nb_iter' not in self.args:
            self.args.nb_iter = 2000
        if 'anp_eps' not in self.args:
            self.args.anp_eps = 0.4
        if 'anp_steps' not in self.args:
            self.args.anp_steps = 1
        if 'anp_alpha' not in self.args:
            self.args.anp_alpha = 0.2
        if 'mask_lr' not in self.args:
            self.args.mask_lr = 0.2
        if 'rand_init' not in self.args:
            self.args.rand_init = True
        if 'pruning_max' not in self.args:
            self.args.pruning_max = 0.95
        if 'pruning_step' not in self.args:
            self.args.pruning_step = 0.05
        if 'device' not in self.args:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def anp_mask(self):
        self.mask_model = getattr(models, self.args.arch)(num_classes=10, norm_layer=NoisyBatchNorm2d).to(
            self.args.device)
        self.mask_model.load_state_dict(torch.load(self.args.model_path, map_location=self.args.device),
                                        strict=False)

        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        parameters = list(self.mask_model.named_parameters())

        mask_params = [v for n, v in parameters if "neuron_mask" in n]
        mask_optimizer = torch.optim.SGD(mask_params, lr=self.args.mask_lr, momentum=0.9)
        noise_params = [v for n, v in parameters if "neuron_noise" in n]
        noise_optimizer = torch.optim.SGD(noise_params, lr=self.args.anp_eps / self.args.anp_steps)

        nb_repeat = int(np.ceil(self.args.nb_iter / self.args.print_every))
        self.logger.info(
            'Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        for i in range(nb_repeat):
            start = time.time()
            lr = mask_optimizer.param_groups[0]['lr']
            train_loss, train_acc = self.mask_train(args=self.args, model=self.mask_model, criterion=criterion,
                                                    data_loader=self.args.defense_data_loader,
                                                    mask_opt=mask_optimizer, noise_opt=noise_optimizer,
                                                    device=self.args.device)

            clean_out = self.args.eval(target_model=self.mask_model, epoch=i, loader=self.args.clean_test_loader)
            poison_out = self.args.eval(target_model=self.mask_model, epoch=i, loader=self.args.bad_test_loader)
            end = time.time()
            self.logger.info(
                '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                i, lr, end - start, train_loss, train_acc, poison_out[0], poison_out[1],
                clean_out[0], clean_out[1])

        self.save_mask_scores(self.mask_model.state_dict())

    def mask_train(self, args, model, criterion, mask_opt, noise_opt, data_loader, device):
        model.train()
        total_correct = 0
        total_loss = 0.0
        nb_samples = 0
        for i, (images, labels) in enumerate(data_loader):
            labels = labels.type(torch.LongTensor)
            images, labels = images.to(device), labels.to(device)
            nb_samples += images.size(0)

            # step 1: calculate the adversarial perturbation for neurons
            if args.anp_eps > 0.0:
                self.reset(model, args=args)

                for _ in range(args.anp_steps):
                    noise_opt.zero_grad()

                    self.include_noise(model)
                    output_noise = model(images)
                    loss_noise = - criterion(output_noise, labels)

                    loss_noise.backward()

                    self.sign_grad(model)
                    noise_opt.step()
                    self.clip_noise(model, self.args)

            # step 2: calculate loss and update the mask values
            mask_opt.zero_grad()
            if args.anp_eps > 0.0:
                self.include_noise(model)
                output_noise = model(images)
                loss_rob = criterion(output_noise, labels)
            else:
                loss_rob = 0.0

            self.exclude_noise(model)
            output_clean = model(images)
            loss_nat = criterion(output_clean, labels)
            loss = args.anp_alpha * loss_nat + (1 - args.anp_alpha) * loss_rob

            pred = output_clean.data.max(1)[1]
            total_correct += pred.eq(labels.view_as(pred)).sum()
            total_loss += loss.item()
            loss.backward()
            mask_opt.step()
            self.clip_mask(model)

        loss = total_loss / len(data_loader)
        acc = float(total_correct) / nb_samples
        return loss, acc

    def reset(self, model, args):
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.reset(rand_init=args.rand_init, eps=args.anp_eps)

    def include_noise(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.include_noise()

    def exclude_noise(self, model):
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                module.exclude_noise()

    def sign_grad(self, model):
        noise = [param for name, param in model.named_parameters() if 'neuron_noise' in name]
        for p in noise:
            p.grad.data = torch.sign(p.grad.data)

    def clip_noise(self, model, args):
        lower = -args.anp_eps
        upper = args.anp_eps
        params = [param for name, param in model.named_parameters() if 'noise' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)

    def clip_mask(self, model, lower=0.0, upper=1.0):
        params = [param for name, param in model.named_parameters() if 'mask' in name]
        with torch.no_grad():
            for param in params:
                param.clamp_(lower, upper)

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

    def save_mask_scores(self, state_dict):
        mask_values = []
        count = 0
        for name, param in state_dict.items():
            if 'mask' in name:
                for idx in range(param.size(0)):
                    neuron_name = '.'.join(name.split('.')[:-1])
                    mask_values.append([neuron_name, idx, param[idx].item()])
                    count += 1
        self.mask_values = mask_values

    def prune_neuron(self):
        self.mask_values = sorted(self.mask_values, key=lambda x: float(x[2]))
        criterion = torch.nn.CrossEntropyLoss().to(self.args.device)
        self.evaluate_by_threshold(self.model, self.mask_values, pruning_max=self.args.pruning_max,
                                   pruning_step=self.args.pruning_step,
                                   criterion=criterion,
                                   clean_loader=self.args.clean_test_loader, poison_loader=self.args.bad_test_loader)

    def evaluate_by_threshold(self, model, mask_values, pruning_max, pruning_step, criterion, clean_loader,
                              poison_loader):
        # print(len(mask_values))
        thresholds = np.arange(0, pruning_max + pruning_step, pruning_step)
        self.logger.info('No \t Layer Name \t Neuron Idx \t Mask \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
        start = 0
        for threshold in thresholds:
            idx = start
            for idx in range(start, len(mask_values)):
                if float(mask_values[idx][2]) <= threshold:
                    self.pruning(model, mask_values[idx])
                    start += 1
                else:
                    break
            layer_name, neuron_idx, value = mask_values[idx][0], mask_values[idx][1], mask_values[idx][2]
            clean_out = self.args.eval(target_model=model, epoch=None, loader=clean_loader)
            poison_out = self.args.eval(target_model=model, epoch=None, loader=poison_loader)

            self.logger.info(
                '%d \t %s \t %s \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                start, layer_name, neuron_idx, threshold, poison_out[0], poison_out[1], clean_out[0], clean_out[1])

    def pruning(self, net, neuron):
        state_dict = net.state_dict()
        weight_name = '{}.{}'.format(neuron[0], 'weight')
        state_dict[weight_name][int(neuron[1])] = 0.0
        net.load_state_dict(state_dict)


if __name__ == '__main__':
    def eval(target_model, epoch, loader):
        criterion = torch.nn.CrossEntropyLoss()
        target_model.eval()
        total_correct = 0
        total_loss = 0.0
        with torch.no_grad():
            for i, (images, labels) in enumerate(loader):
                labels = labels.type(torch.LongTensor)
                images, labels = images, labels
                output = target_model(images)
                total_loss += criterion(output, labels).item()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
        loss = total_loss / len(loader)
        acc = float(total_correct) / len(loader.dataset)
        return loss, acc


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
    parser.add_argument('--target_label', type=int, default=0, help='class of target label')
    parser.add_argument('--trigger_type', type=str, default='gridTrigger', help='type of backdoor trigger')
    parser.add_argument('--target_type', type=str, default='all2one', help='type of backdoor label')
    parser.add_argument('--trig_w', type=int, default=3, help='width of trigger pattern')
    parser.add_argument('--trig_h', type=int, default=3, help='height of trigger pattern')
    args = parser.parse_args()
    args.dataset = 'CIFAR10'
    args.ratio = 0.1
    args.batch_size = 128
    args.target_label = 0

    from datasets.poison_tool_cifar import get_test_loader, get_train_loader  # 仿照RNP代码读取数据

    defense_data_loader = get_train_loader(args)
    clean_test_loader, bad_test_loader = get_test_loader(args)

    args.defense_data_loader = defense_data_loader
    args.clean_test_loader = clean_test_loader
    args.bad_test_loader = bad_test_loader
    args.eval = eval

    ANP = anp(args, logger)
    ANP.anp_mask()
    ANP.prune_neuron()
