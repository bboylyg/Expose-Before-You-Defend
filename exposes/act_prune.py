import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import logging
import argparse
from torch.optim import SGD, Optimizer, lr_scheduler


if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MaskedLayer(nn.Module):
    def __init__(self, base, mask):
        super(MaskedLayer, self).__init__()
        self.base = base
        self.mask = Parameter(mask, requires_grad=True)

    def forward(self, input):
        return self.base(input) * self.mask

class ActPruning():
    """
    neuron pruning with feature activation.
    """
    def __init__(self, args, logger, net, data_loader, **kwargs) -> None:
        self.args = args
        self.arguments()
        self.logger = logger

        self.logger.info(self.args)

        self.net = net
        self.defense_loader = data_loader['defense_loader']
        self.clean_test_loader = data_loader['clean_test_loader']
        self.bad_test_loader = data_loader['bad_test_loader']

    def arguments(self):
        if 'discription' not in self.args:
            self.args.discription = 'acrPruning'
        if 'arch' not in self.args:
            self.args.arch = 'resnet18'
        if 'layer_name' not in self.args:
            self.args.layer_name = 'linear'
        if 'prune_rate' not in self.args:
            self.args.prune_rate = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if 'print_every' not in self.args:
            self.args.print_every = 500
        if 'num_classes' not in self.args:
            self.args.num_classes = 10
        if 'ft_epochs' not in self.args:
            self.args.unlearn_epochs = 20
        if 'lr' not in self.args:
            self.args.lr = 0.01
        if 'sched_gamma' not in self.args:
            self.args.sched_gamma = 0.1
        if 'sched_ms' not in self.args:
            self.args.sched_ms = [20, 20]
        if 'stop_acc' not in self.args:
            self.args.stop_acc = 0.1
        if 'device' not in self.args:
            self.args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @staticmethod
    def _fp(pnet: nn.Module, ds, layer: str, prate: float):
        with torch.no_grad():
            ctr = []

            def forward_hook(module, input, output):
                nonlocal ctr
                ctr.append(output)

            hook = getattr(pnet, layer).register_forward_hook(forward_hook)
            pnet.eval()
            for data, _ in ds:
                pnet(data.cuda())
            hook.remove()
        ctr = torch.cat(ctr, dim=0)  # conv: 500x64x32x32; linear: 500x10
        if ctr.dim() == 4:
            activation = torch.mean(ctr, dim=[0, 2, 3])
        elif ctr.dim() == 2:
            activation = torch.mean(ctr, dim=0)
        else:
            raise NotImplementedError
        seq_sort = torch.argsort(activation)
        num_channels = len(activation)
        prunned_channels = int(num_channels * prate)
        mask = torch.ones(num_channels).cuda()
        for element in seq_sort[:prunned_channels]:
            mask[element] = 0
        if len(ctr.shape) == 4:
            mask = mask.reshape(1, -1, 1, 1)
        setattr(pnet, layer, MaskedLayer(getattr(pnet, layer), mask))

    def init_defense_utils(self) -> dict:
        return dict()
    
    def do_expose(self, **kwargs) -> None:
            self.logger.info("="*20 + "Expose strategy: %s" % (self.args.discription) + "="*20)
            ds = self.defense_loader

            for prune_rate in self.args.prune_rate:
                if prune_rate == 0:
                    full_acc = self.accasr_full_test(self.net, self.defense_loader, self.bad_test_loader, prune_rate, device=device)
                    print('full_acc:', full_acc)
                else:
                    self._fp(self.net, ds, self.args.layer_name, prune_rate)
                    full_acc = self.accasr_full_test(self.net, self.defense_loader, self.bad_test_loader, prune_rate, device=device)
    
                    print('full_acc:', full_acc)
    
    def acctest(self, net, data_loader, device):
        net.eval()
        total_correct = 0
        total_samples = 0
        with torch.no_grad():
            for _, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)
                output = net(images)
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()
                total_samples += len(labels)
        acc = total_correct / total_samples
        return acc if isinstance(acc, float) else acc.item() 

    def accasr_full_test(self, net, defense_loader, bad_test_loader, prune_rate, device=device) -> dict:
        num_cls = self.args.num_classes
        net.eval()
        ret = dict()
        correct_counter = [0] * num_cls
        cls_count = [0] * num_cls
        pred_counter = [0] * num_cls
        for _, (x, y) in enumerate(defense_loader):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = net(x)
                # pred = output.argmax(dim=1)
                _, pred = output.max(1)
                for pred_ in pred:
                    pred_counter[pred_] += 1 
                for _idx, _y in enumerate(y):
                    cls_count[_y] += 1
                    # pred_counter[pred[_idx]] += 1
                    if pred[_idx] == _y:
                        correct_counter[_y] += 1
        ret["prune_rate"] = prune_rate
        ret["acc"] = round(sum(correct_counter) / sum(cls_count), 2)
        ret["asr"] = round(self.acctest(net, bad_test_loader, device), 2)
        # ret["cls_acc"] = [
        #     0 if _t == 0 else round(_c / _t,2) for _c, _t in zip(correct_counter, cls_count)
        # ]
        ret["cls_pred"] = [round(_ / sum(cls_count),2) for _ in pred_counter]
        # ret["cls_pred"] = [_ for _ in pred_counter]
        
        return ret

